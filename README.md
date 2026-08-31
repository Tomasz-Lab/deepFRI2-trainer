# deepFRI2-trainer

Training, retraining and fine-tuning of [deepFRI2](https://github.com/Tomasz-Lab/deepFRI2)
protein function predictors. Datasets come from
[FRIdata](https://github.com/Tomasz-Lab/FRIdata); this repository only consumes them.

`train.py` trains the sub-models of one ontology (MF / CC / BP) in dependency order:

| Stage | Model | Input | Loss |
|---|---|---|---|
| `sequence` | `SequenceAnalyzer` | ESM-2 embeddings | `MCLossDAG` |
| `structure` | `StructuralProber` | CA distograms | `WeightedFocalLoss` (class weights) |
| `fusion` | `FusionModel` | both — frozen sequence + frozen structure + trainable gate | `MCLossDAG` |

```bash
conda env create -f environment.yml && conda activate deepfri2_trainer
wandb login

python train.py --prepare                            # once: import the released checkpoints
python train.py --ontology MF                        # all three models
python train.py --ontology MF --stages fusion        # just the fusion gate
python train.py --ontology BP --train-on train+eval  # production models trained on everything
```

A complete model is nine runs: 3 ontologies x 3 stages. `python train.py --help` lists every
parameter; the `train.py` docstring documents them in full.

`validate.ipynb` scores trained runs with the protein-centric **CAFA evaluation** and draws
the paper's figures and tables — see [CAFA evaluation](#cafa-evaluation) below.

Not yet wired in: CAFA scores appended to `training.log` at the end of a run (they are computed
in the notebook for now), and **retraining / fine-tuning** beyond loading initial weights —
swapping the GO-term head for a regression/classification task, and restricting to a GO-term
subset.

## Architectures: owned here, checked against inference

Model definitions live in [`src/deepfri2_trainer/model.py`](src/deepfri2_trainer/model.py), the
trainer's copy of `deepFRI2/src/deepFRI2/model.py` under the same module name so it can be
diffed against, or dropped straight into, the inference repository. Architecture experiments
belong here, without editing inference first.

The cost is drift, so every run runs a **parity check**
([`parity.py`](src/deepfri2_trainer/parity.py)):

1. **Source parity** — each architecture symbol compared line by line; per-symbol verdict plus
   a unified diff.
2. **Checkpoint parity** — a trainer checkpoint loaded into the inference implementation with
   `strict=True` and the logits compared, with max and mean absolute difference. *Can deepFRI2
   run what we just trained?*

Both run **in one process, on one device, under this run's backend flags**, so the differences
they report are code differences only — a zero here says the two implementations agree, not that
the model is numerically portable. What moves between environments is measured separately (see
[TF32 and reproducibility](#tf32-and-reproducibility)).

```
architecture parity: source identical to .../deepFRI2/src/deepFRI2/model.py
  sequence  checkpoint -> inference: loads OK, logits identical on cuda:0: max|d|=0.000e+00 mean|d|=0.000e+00 (max|logit|=5.331e-01)
  structure checkpoint -> inference: loads OK, logits identical on cuda:0: max|d|=0.000e+00 mean|d|=0.000e+00 (max|logit|=6.426e-01)
  fusion    checkpoint -> inference: loads OK, logits identical on cuda:0: max|d|=0.000e+00 mean|d|=0.000e+00 (max|logit|=6.848e-01)
```

The verdict goes to the console, `log.txt`, `config_<run>.yaml`,
`source/architecture_parity.txt`, `training.log` and wandb. A divergence does not stop training
— it is your experiment — but when the checkpoint no longer fits inference the report says so:

```
architecture parity: DIVERGED from .../deepFRI2/src/deepFRI2/model.py
  changed symbols: SequenceAnalyzer
  sequence  checkpoint -> inference: FAILED to load into inference (unexpected key extra_head.weight)
  => deepFRI2 inference CANNOT run this model as-is; port the change to
     deepFRI2/src/deepFRI2/model.py before shipping the checkpoint.
```

Set `deepfri2_src: null` in `configs/paths.yaml`, or pass `--no-parity-check`, if no deepFRI2
checkout is around.

## Tests

```bash
python tests/test_model_equivalence.py     # architecture fidelity + parity
python tests/test_metrics.py               # logged P/R/F1 vs sklearn
```

- the trainer architectures produce bit-identical logits to the model classes copied verbatim
  out of the original training notebooks (`tests/reference/notebook_models.py`);
- the parity check reports them identical to inference, and reports a deliberately modified
  architecture as diverged and undeployable, so the guard is known to work;
- all nine released deepFRI2 checkpoints load into trainer-built models with `strict=True`;
- the metrics match `sklearn.metrics.precision_recall_fscore_support`.

The fusion stage re-checks fidelity against real data: the frozen branches must reproduce the
stand-alone sub-models' test predictions to 1e-6.

## Configuration

Hyperparameters and paths are not command-line arguments. Three YAML files are merged per run:

| File | Contents |
|---|---|
| `paths.yaml` | machine-specific roots (`project_location`, `deepfri2_src`, `runs_dir`) and the data-tree layout. **The only file to edit when moving hosts.** |
| `data.yaml` | dataset / GO versions, annotation threshold, `max_seq_len`, `sigma_dist`, batch size, workers |
| `sequence.yaml`, `structure.yaml`, `fusion.yaml` | architecture, optimizer, loss, epochs, initial weights |

In `paths.yaml`, `{project_location}` expands to the data-tree root; a value containing a
`{placeholder}` **must be quoted**, since an unquoted leading `{` is a YAML flow mapping.

Each model config takes a `per_ontology:` block, deep-merged for the selected ontology:

```yaml
per_ontology:
  BP:
    training:
      num_epochs: 15
```

`--set` overrides any key in the merged config (dotted paths, values parsed as YAML):

```bash
python train.py --ontology MF --set training.num_epochs=5 data.batch_size=16
```

Defaults reproduce the released checkpoints: annotation threshold 50, data version `20250908`,
GO version `20250722`; sequence 20 epochs @ 1e-4 with `MCLossDAG`, structure 20 epochs @ 2e-4
with `WeightedFocalLoss` and class weights, fusion 15 epochs @ 1e-4 with `MCLossDAG`. The one
deliberate departure is `selection: best` (see below). `run.sh` records the exact command for
every released model.

### Checkpoint selection

Training always keeps two checkpoints — `<run>_best.pth` (the optimum of
`training.selection_metric` so far) and `<run>_last.pth` (most recent epoch). **Those two are
the only epochs a run can ship.** `training.selection` decides which of them becomes
`<run>.pth` and generates the prediction TSVs:

| | |
|---|---|
| `last` | the final epoch — what the originally released models used |
| `best_strict` | the optimum of `training.selection_metric`, whenever it occurred |
| `best` (default) | the final epoch when it is within `selection_tolerance` of the optimum, the optimum otherwise |

Both files survive the run, and `config_<run>.yaml` records the shipped epoch
(`provenance.selected_epoch`, `provenance.selected_checkpoint`), so the other option can be
evaluated without retraining.

Because only those two epochs exist on disk, `select_epoch()` returns the epoch **and** the
checkpoint that holds it, and the caller loads the file it names — the epoch number alone is not
enough to identify a checkpoint.

**Selection needs a held-out split.** Under `--train-on train+eval` the eval split is part of
the training set, so `eval_fmax` and `eval_loss` are training metrics: they tend to improve
monotonically, `_best.pth` ends up equal to `_last.pth`, and every rule resolves to the final
epoch regardless of what the config asks for. Such a run is effectively `selection: last`, and
its `eval_*` numbers are not held out — judge it on the test / CAZy sets instead.

This makes a train-only vs train+eval comparison asymmetric: the train-only run can ship the
optimum of a noisy held-out curve, the train+eval run always ships its final epoch, and the
difference between those two is not a difference in training data. To compare the two fairly,
either hold out a slice for selection (train on `train` plus most of `eval`, select on the
remainder) or fix `num_epochs` from the train-only run's curve and set `selection: last` on both
sides.

### Which metric to select on

`training.selection_metric` is `eval_fmax` by default: the **protein-centric Fmax on the eval
split, with GO-DAG propagation** — the CAFA metric itself, computed in-loop. It tracks the
offline CAFA evaluation closely, up to a constant offset from the label space (the offline
evaluation uses the full GO graph, the in-loop one the run's target-matrix terms); differences
between epochs and between models are unaffected.

The alternatives disagree with each other and with CAFA, which is why the choice matters. Within
a single run, eval **loss** can bottom out many epochs before Fmax peaks, while macro and micro
F1 at a fixed threshold peak at different epochs again. Loss is a proper scoring rule, so it
punishes the overconfidence that sets in once a model starts memorising — but Fmax maximises over
the threshold and so ignores calibration entirely. Macro F1 at a fixed threshold keeps rising
because macro recall keeps rising as the model starts firing on rare terms, each weighted
equally; micro F1 falls at the same time because the bulk of predictions is degrading. None of
them is CAFA. Fmax is.

`selection_metric: eval_loss` is available if you want the conservative criterion.

`training.selection_tolerance` (default 0.002) keeps the **final** epoch when it is no more
than that below the optimum. Fmax wobbles by a few thousandths between epochs, and without it a
run can stop on an early lucky epoch while the model is still improving; the more-trained
checkpoint is the safer one inside the noise band. Once the drop exceeds the tolerance the
optimum wins. `best_strict` ignores the tolerance and always keeps the optimum; `last` ignores
it and always keeps the final epoch.

In practice most runs end with the curve flat or still rising, so `best` and `last` agree and the
tolerance changes nothing. The two differ only when the curve genuinely turns down by more than
the noise before the end. `best` and `best_strict` differ in the opposite case: on a flat curve
with an early wobble at the top, `best_strict` will ship that early epoch, which is exactly the
noise-chasing the tolerance exists to prevent — so prefer `best` unless you specifically want the
optimum irrespective of when it occurred.

Train Fmax is computed too, on a fixed subsample (`fmax_max_proteins`, default 10 000 proteins —
propagating the full train split every epoch would cost gigabytes). Both series are logged, and
each run reports how well they track each other:

```
  Fmax  train vs eval: pearson=+0.671 spearman=+0.643
  loss  train vs eval: pearson=-0.818 spearman=-0.738
```

A high correlation means the model is still learning structure that generalises; once the two
diverge, later epochs are only fitting the train split.

This matters more than it looks. Eval loss bottoms out well before the last epoch — earlier at
higher learning rates — while training runs to `num_epochs` regardless. In-distribution metrics
(eval, test) barely notice; an out-of-distribution set such as CAZy does: a model left well past
its optimum assigns lower scores to its *true* labels and its optimal threshold drifts downwards,
both of which cost more off-distribution than on.

Set `selection: last` to ship the final epoch regardless, or `best_strict` to ship the optimum
regardless of when it occurred.

### Seeding and reproducibility

`training.seed` (default 42) seeds python, numpy and torch, and the dataloaders' shuffling and
workers, so weight init and batch order are reproducible. `null` draws a seed instead and records
it in `config_<run>.yaml`, so an unpinned run is still reproducible after the fact.

The seed fixes the *inputs* to training, not how the GPU executes it. GPU libraries choose
kernels and accumulation orders from heuristics that depend on tensor shapes and on the hardware,
and some backward kernels accumulate in a non-fixed order, so two runs of the same config on the
same machine can differ in the last decimals of each update. Over a long run those differences
compound, and two runs of one recipe land on slightly different models — close in behaviour, not
identical in weights. Expect a spread of a few thousandths in Fmax between repeats of the same
configuration, and treat differences of that size between two single runs as noise rather than
signal. Where a comparison matters, repeat it across seeds and compare the spread, or evaluate
with confidence intervals over proteins.

Worth knowing: this is easy to under-estimate from a short run. A few batches per epoch can
reproduce exactly while a full epoch of the same config does not, so reproducibility should be
checked at realistic run length if you check it at all.

Runs on different GPUs or different CUDA / cuDNN / torch builds are not comparable at this
resolution regardless of seeding. `provenance.machine` in `config_<run>.yaml` records host, GPU,
capability, driver, CUDA, cuDNN and python version so a run record can answer whether two runs
shared a stack.

### `weights`: initial checkpoints and frozen sub-models

Every model config has a `weights` block, per ontology:

- **sequence / structure run** — the checkpoint to fine-tune from. `null` (the default) trains
  from scratch. The label space must match; it is loaded with `strict=True`.
- **fusion run** — the two frozen sub-models, both required. Defaults are the released deepFRI2
  run names.

Override per run instead of editing the configs:

```bash
python train.py --ontology CC --stages fusion \
    --weights-sequence wandb-name-1 --weights-structure wandb-name-2
python train.py --ontology MF --stages structure --weights-structure wandb-name-3
```

When `sequence` / `structure` are trained in the same call, their run names are passed to the
fusion stage automatically and override the config.

A reference resolves, in order, as a **wandb run name** (→
`<runs_dir>/<ontology>__<model type>__<name>/<name>.pth`), a **run directory name**, or a **path
to a `.pth`**. One that resolves to nothing raises, listing every path tried:

```
FileNotFoundError: sequence weights 'wandb-name-1' not found for CC; looked for:
  <runs_dir>/CC__sequence__wandb-name-1/wandb-name-1.pth
  <runs_dir>/wandb-name-1/wandb-name-1.pth
Train it first, run `python train.py --prepare` to import the released deepFRI2 checkpoints, ...
```

### `--prepare`

`python train.py --prepare` reads the run names the inference module declares
(`deepFRI2/src/deepFRI2/config.py :: MODEL_NAMES`) and copies
`deepFRI2/params/<ontology>/<run>.pth`, plus its labels JSON when shipped, into ordinary run
directories:

```
<runs_dir>/MF__sequence__<run>/
    <run>.pth
    config_<run>.yaml              provenance: imported, not trained here
```

That makes "fine-tune on top of the released model" and "train a fusion gate over the released
sub-models" work out of the box. Restrict to one namespace with `--ontology`; already-prepared
runs are left alone; each import is recorded in `training.log` as `IMPORT`. Imported runs have
no prediction TSVs, so the fusion branch check reports itself skipped for them.

### TF32 and reproducibility

cuDNN runs convolutions in TF32 by default, which makes the structure model's kernel bank the
one part of deepFRI2 whose outputs move between GPUs (different cuDNN versions) and against
CPU. `structure.yaml` exposes it next to `amp_dtype`:

```yaml
model:
  amp_dtype: null            # autocast dtype inside the model
  cudnn_allow_tf32: true     # false for reproducible convolutions between GPU and CPU
```

`true` reproduces the released checkpoints. It applies to structure and fusion runs (fusion
holds a frozen kernel model); the observed value of both TF32 flags is recorded in
`config_<run>.yaml` and in the `training.log` START line.

## Outputs

One directory per run, named `<ontology>__<model type>__<wandb run name>`. The wandb run name
distinguishes runs — including two of the same ontology and model differing only in a
hyperparameter — and is carried in every file name so files stay identifiable when copied out:

```
<runs_dir>/
    training.log                                  append-only, shared by all runs
    MF__sequence__<run>/
        <run>.pth                                 state dict, loadable by deepFRI2 inference
        labels_<run>.json                         {"<column index>": "<GO term>"}
        config_<run>.yaml                         merged config + provenance + parity verdict
        predictions_<run>.tsv                     eval-set predictions
        predictions_test_<run>.tsv
        predictions_cazy_<run>.tsv
        architecture_parity.diff                  only when the architectures diverged
        log.txt                                   this run's console output
        source/                                   the code that produced the run
```

`config_<run>.yaml` holds the merged config plus provenance: wandb run name, timestamp, trainer
and deepFRI2 git commits (`-dirty` when the checkout has local changes), torch version, TF32
flags, the machine the run executed on (`provenance.machine`), the shipped epoch and checkpoint,
the model's `ARCHITECTURE` dict and the parity report. `source/` snapshots `model.py`,
`load_model.py`, `data.py`, `train.py`, `pipeline.py`, `dataloader.py`, `training.py` and
`losses.py`. Config and snapshot are also logged to wandb as a `code` artifact.

`log.txt` is the run's console output, captured from the start of the stage and flushed once the
run directory name is known, so nothing printed before `wandb.init` is lost. stdout is kept
verbatim; stderr — where tqdm draws — is cleaned of progress artefacts: carriage-return
repaints, escape codes, and the bare newlines `tqdm.moveto` emits to reposition nested bars. The tee is re-installed at that point, because
`wandb.init` swaps `sys.stdout` and `wandb.finish` restores the stream *it* saved — across
several stages in one process that would otherwise send every stage after the first into the
first stage's log. The cross-stage summary is appended to the last stage's log.

`training.log` brackets every run:

```
<date> 10:34 | START  | MF__sequence__<run> | num_labels=... epochs=20 lr=0.0001 loss=MCLossDAG train_on=train train_batches=... parity=identical
<date> 11:58 | DONE   | MF__sequence__<run> | epoch=<shipped>/20 checkpoint=<run>_best.pth optimum_epoch=... selection=best train_on=train seed=42 train_loss=... eval_loss=... eval_fmax=... time=... per_epoch=... dir=...
```

A run that raises during training is recorded as `FAILED`. Training all three stages yields
three run directories, three `log.txt` files and three `START`/`DONE` pairs.

With `--no-wandb` the run name falls back to `local-<timestamp>` and nothing is uploaded.

To promote a run into deepFRI2, copy `<run>.pth` and `labels_<run>.json` into
`deepFRI2/params/<ontology>/` and add the run name to `MODEL_NAMES` in
`deepFRI2/src/deepFRI2/config.py`.

## CAFA evaluation

[`validate.ipynb`](validate.ipynb) scores runs against deepFRI v1 and the published competitors
on the evaluation, test and CAZy splits, and produces the figures and tables of the paper. It is
a notebook rather than a CLI on purpose: figures are made by looking at them.

Scores come from [CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator)
(`cafaeval`, pinned in `environment.yml`). If it is not installed in the environment but a
checkout is at hand, point `CAFA_EVALUATOR_SRC` at its `src` directory.

The mechanics are in [`utils/evaluator.py`](src/deepfri2_trainer/utils/evaluator.py) and
[`utils/figures.py`](src/deepfri2_trainer/utils/figures.py); everything specific to a machine, a
dataset or a set of competitors — the path templates, the run names, the method names and the
colours — is in the notebook's **Setup** cell, so the modules carry no local paths.

```python
paths = EvalPaths.from_configs(LAYOUT)          # versions and roots from configs/

ev = CafaEvaluation(paths)
ev.add_runs({"MF": {"deepFRI2 (fusion)": "dainty-deluge-829"}})   # wandb run names
ev.add_deepfri1(DEEPFRI1)
ev.add_competitors(COMPETITORS, keep=KEEP)      # FunFams, DeepGO-SE, eggNOG-mapper, PO2GO

curves = ev.curves        # tidy: one row per (method, ontology, split, tau)
ev.summary(weighted=True) # per method: Fmax, its threshold / precision / recall / coverage, Smin
ev.table("fmax", split="test", weighted=False)                    # methods x ontologies

figures.panel(curves, split="test", ontology="MF", weighted=True) # F1, PR, S, coverage
figures.compare(curves, "f", by="ontology", split="test", weighted=True)
figures.bars(curves, "fmax", split="cazy", weighted=False)
```

`curves` carries both the unweighted and the information-accretion weighted metrics, so
`weighted=` — an argument on every table and every figure — switches between them at read time
and never re-runs anything.

Each figure and table titles itself from what it actually shows and carries the matching file
name, so `figures.save` takes a *directory*, never a name: `bars(..., split="cazy")` can only be
written as `bars_fmax_cazy_unweighted.png`. Figures save as png + pdf, tables as csv + tex.

Every method is scored once and its curves cached as
`cafa-{eval,test,cazy}-all_<name>.pickle` next to its predictions, in the file names the
previous validation notebook used — so the scores already computed are reused as they are, and
the notebook opens in seconds. A run that has never been scored is scored on the spot;
`CafaEvaluation(paths, recompute=True)` redoes the rest.

Two conventions worth knowing:

- Metrics are weighted by information content by default. The IA table
  (`IA_<data version>_HQ.tsv`) comes from the InformationAccretion repository, which is not yet
  wired in — its location is the `ia` entry of `LAYOUT` in the notebook. Without the file, only
  the unweighted metrics are available.
- `summary()` reports `smin` as the minimum of the column it summarises. CAFA-evaluator's own
  `best` tables pick the threshold by the *unweighted* `s` and print `s_w` there, which is
  slightly higher than the minimum of `s_w`.

## Logged metrics

Per epoch, to wandb and to the console / `log.txt`:

| Metric | Averaged over |
|---|---|
| `<split>/precision` | GO terms the model predicted at least once (`tp + fp > 0`) |
| `<split>/recall` | GO terms with ground truth (`tp + fn > 0`) |
| `<split>/f1` | GO terms present in either (`tp + fp + fn > 0`) |
| `<split>/{precision,recall,f1}_micro` | all predictions pooled — mutually consistent |
| `<split>/classes_{predicted,with_support,total}` | the counts behind the macro averages |

The macro numbers each skip the terms where they are undefined, so they are averaged over
**different subsets** and `f1` is deliberately not `2PR/(P+R)` of the reported `precision` and
`recall`. That is standard macro-averaging (it matches
`sklearn.metrics.precision_recall_fscore_support(average="macro", zero_division=np.nan)`) but
easy to misread, hence the micro averages and class counts alongside.

## Sanity checks

Per model, inside every stage:

| Check | What it catches |
|---|---|
| `check_label_space` | model output width != number of GO terms |
| `report_trainable_parameters` | with `expect_only="refine_gate"`: a fusion sub-model that is not actually frozen |
| `check_fusion_branches` | a wrongly loaded or mis-configured sub-model |
| `check_prediction_file` | a truncated or out-of-range predictions TSV |

Data-level checks, for interactive use — they need a loader carrying both modalities to be
meaningful, so they are not run per stage:

```python
from deepfri2_trainer import build_loaders, load_config, load_targets, sanity

cfg = load_config("fusion", "MF")
targets = load_targets(cfg)
loaders = build_loaders(cfg, targets)
sanity.check_dataloader_consistency(loaders.test, "cuda:0", "P30679_A")
sanity.check_batch_shapes(loaders.cazy)
sanity.show_example(loaders.eval, targets, targets.protein_vectors, unfix_type="AFDB_v4")
```

`check_dataloader_consistency` asserts that embeddings, distograms and masks do not change with
which modalities are requested; `check_batch_shapes` compares mask length against non-zero
embedding/distogram rows; `show_example` plots one distogram and prints that protein's
annotations.

## Layout

```
configs/                      paths, data versions, per-model hyperparameters
environment.yml               conda environment (GPU)
train.py                      CLI entry point
validate.ipynb                CAFA evaluation: figures and tables for the paper
src/deepfri2_trainer/
    model.py                  deepFRI2 model definitions
    load_model.py             config -> model, checkpoint loading, backend flags
    parity.py                 diff + checkpoint check against deepFRI2 inference
    config.py                 config merging, path resolution, run identity
    data.py                   target-matrix loading, dataloader construction
    train.py                  config -> training loop
    pipeline.py               run_stage / run_stages orchestration
    predict.py                prediction TSV writing
    outputs.py                wandb session, run dir, artifacts, log.txt, training.log
    prepare.py                import released deepFRI2 checkpoints into runs_dir
    sanity.py                 sanity & validation checks
    utils/                    dataloader, training loop, losses
        evaluator.py          CAFA scores: ground truth, CAFA-evaluator, caching, tidy table
        figures.py            figures and tables from those scores
tests/
    test_model_equivalence.py architecture fidelity + parity
    test_metrics.py           logged P/R/F1 vs sklearn
    reference/                verbatim copy of the notebook model definitions
```
