# deepFRI2-trainer

Training, retraining and fine-tuning of [deepFRI2](https://github.com/Tomasz-Lab/deepFRI2)
protein function predictors. Datasets come from
[FRIdata](https://github.com/Tomasz-Lab/FRIdata); this repository only uses them.

The pipeline this repository sits in (last three):

| Step | Produces | Where |
|---|---|---|
| provide inputs | protein IDs, annotations, GO graph | `data/inputs/` |
| [FRIdata](https://github.com/Tomasz-Lab/FRIdata) | sequences, distograms, ESM-2 embeddings | `data/FRIdata_output/` |
| **`preprocess.py`** | target matrices, ground truth, train/eval split | `data/target_matrix/` |
| **`train.py`** | checkpoints, predictions | `runs_dir` |
| **`validate.ipynb`** | CAFA scores, figures, tables | `figures/` |
| **`calibrate.py`** | per-GO-term precision/recall curves | `runs_dir` |

`train.py` trains the sub-models of one ontology (MF / CC / BP) in dependency order:

| Stage | Model | Input | Loss |
|---|---|---|---|
| `sequence` | `SequenceAnalyzer` | ESM-2 embeddings | `MCLossDAG` |
| `structure` | `StructuralProber` | CA distograms | `WeightedFocalLoss` (class weights) |
| `fusion` | `FusionModel` | both — frozen sequence + frozen structure + trainable gate | `MCLossDAG` |

```bash
conda env create -f environment.yml && conda activate deepfri2_trainer
wandb login

python preprocess.py --dry-run                       # check every input is where configs say
python preprocess.py --ontology MF                   # target matrix, split, CAZy targets

python train.py --import-released                    # once: import the released checkpoints
python train.py --ontology MF                        # all three models
python train.py --ontology MF --stages fusion        # just the fusion gate
python train.py --ontology BP --train-on train+eval  # production models trained on everything

python calibrate.py --ontology MF                    # per-GO-term curves for the interpretability reports
```

A complete model is nine runs: 3 ontologies x 3 stages. `python train.py --help` lists every
parameter; the `train.py` docstring documents them in full.

`validate.ipynb` scores trained runs with the protein-centric **CAFA evaluation** and draws
the paper's figures and tables — see [CAFA evaluation](#cafa-evaluation) below.

To train on something other than GO — your own classification or regression labels — see
[Training on your own dataset](#training-on-your-own-dataset).

Not yet wired in: CAFA scores appended to `training.log` at the end of a run (they are computed
in the notebook for now), and restricting a GO run to a subset of terms.

## Preprocessing

[`preprocess.py`](preprocess.py) turns labels into the **target matrix** `train.py` consumes:
`go_indices` (the model's outputs), `protein_vectors*` (the label of every protein), `weights`
and `adjacency`. It is one script with two sources of labels, and the source is the only thing
that differs:

| | GO (default) | Your own labels (`--task NAME`) |
|---|---|---|
| Labels come from | the GO annotation tables + GO graph | one CSV per split |
| Train/eval split | computed here (MMseqs2), or adopted | taken as given: one FRIdata dataset per split |
| Steps | `targets`, `split`, `cazy` | one: read the CSVs |
| Code | [`preprocess.py`](src/deepfri2_trainer/preprocess.py) | [`csv_target_matrix.py`](src/deepfri2_trainer/csv_target_matrix.py) — reads the CSVs, writes the same pickles, plus an `overrides.yaml` for `train.py`; config, logging and pickle writing are shared |
| Output | `<out_dir>/<dataset>/<params>/target_matrix/` | `<custom_tasks_dir>/NAME/target_matrix/` + `overrides.yaml` |
| Train with | `train.py --ontology MF` | `train.py --task NAME` |

The pickles have the same names and shape either way, so everything from `train.py` on reads
them the same way. The rest of this section is the GO flow; for your own labels see
[Training on your own dataset](#training-on-your-own-dataset).

The GO flow has three steps, each runnable alone:

| Step | Produces | Cost |
|---|---|---|
| `targets` | the eight target-matrix pickles + the test-set FASTA | slow — reads the multi-GB annotation tables, once for all requested ontologies |
| `split` | trainval FASTA, MMseqs2 clustering, `train.tsv` / `eval.tsv` | minutes |
| `cazy` | CAZy label vectors in the model's own GO-term order | seconds |

```bash
python preprocess.py --dry-run                        # resolved config + every input checked
python preprocess.py --ontology MF                    # all three steps
python preprocess.py --ontology MF --steps split      # just re-split
python preprocess.py --set annotation_threshold=70    # a different label space
```

`targets` decides the label space: a GO term enters it only if at least
`annotation_threshold` proteins carry it, and the resulting subgraph must stay a single
connected DAG. From that come `go_indices` (the output order of the model), the sparse
`protein_vectors`, the per-term loss `weights`, the `adjacency` the hierarchy loss propagates
over, and the `grand_truth` tables the CAFA evaluation scores against. Ground truth is always
high-quality annotations only and spans the **full** GO graph, not the thresholded subgraph —
the evaluation must not be told to ignore terms the model was never given.

`split` is homology-aware: sequences are clustered with MMseqs2 at `min_seq_id` and **whole
clusters** are assigned, so no evaluation protein has a close homologue in training. Of
`num_trials` random assignments it keeps the one whose per-GO-term evaluation fraction is best
balanced, so rare terms stay represented on both sides.

Two settings in `configs/data.yaml :: preprocess` control whether a run rebuilds or reproduces:

| Setting | Null | Set |
|---|---|---|
| `go_indices_from` | derive the GO-term order from the graph | adopt an existing `go_indices.pkl` order |
| `split.adopt_from` | cluster and split with MMseqs2 (`seed`) | copy an existing `train.tsv` / `eval.tsv` |

`go_indices_from` refuses if the GO term *sets* differ: a different threshold or annotation
version is a different label space, not a reordering of one. Set both to null for a genuinely
new model.

Every run appends its whole console output to `data/data.log`, under a header giving the
date and the exact command.

## Training on your own dataset

Anything the model can be trained on comes down to structures and labels. Starting from mmCIF
files and a CSV, there are three steps.

**1. Structures -> a FRIdata dataset, one per split.** Run
[FRIdata](https://github.com/Tomasz-Lab/FRIdata) over each split's mmCIF directory. That gives
you a directory with `dataset.json`, `embeddings.idx` / `.h5` (ESM-2) and `distograms.idx` /
`.h5` — the same thing the GO flow trains on, and the only format this trainer reads.

`dataset.json` records the FRIdata root as it stood when the dataset was *built*, which goes
stale the moment the tree is moved or remounted. When that recorded path no longer exists the
loader falls back to the index's own location — the index sits at `<root>/datasets/<name>/` and
its entries are relative to `<root>`, so the directory layout gives the real root. A dataset
carrying a dead path therefore still loads, instead of failing on a missing `.h5`.

**2. Labels -> a target matrix.** The same `preprocess.py` as for GO, with `--task`: instead of
the `targets` / `split` / `cazy` steps it reads one CSV per split, keyed by protein id, and
keeps the split you give it (see [Preprocessing](#preprocessing) for how the two compare).
`--ontology`, `--steps` and `--set` do not apply here.

```csv
protein_id,label
P12345,1.7
```

```bash
python preprocess.py --task gb1 --task-type regression \
    --train-csv train.csv --train-dataset fridata/train \
    --eval-csv  valid.csv --eval-dataset  fridata/valid \
    --test-csv  test.csv  --test-dataset  fridata/test
```

`--task-type` is required and says what the labels are:

| `--task-type` | Label columns | Model output | Loss |
|---|---|---|---|
| `classification` | one, integer class ids | softmax over the classes | cross-entropy, class weights n / (K · n_class) |
| `multi-task-classification` | two or more, 0/1 | a sigmoid per label | BCE, `pos_weight` = n_neg / n_pos per label |
| `regression` | one, real values | one value | MSE |
| `multi-task-regression` | two or more, real values | a value per label | MSE |

A binary task is `classification` with classes 0 and 1. Both kinds of weight are
scikit-learn's `class_weight="balanced"`, counted on train.

Empty cells are missing labels; they count toward neither the loss nor the metrics, so a
multi-task CSV doesn't need every protein labelled for every task. The id column defaults to
`protein_id` and the label column to `label`; name others with `--label-columns y1,y2`.

What it writes, under `<custom_tasks_dir>/<task>/`: `target_matrix/` with the GO flow's pickles
— `protein_vectors{,_eval,_test}.pkl` (a dense vector per protein; one-hot for classification),
`go_indices.pkl` (class ids or column names -> output index), `weights.pkl` (as in the table)
and an all-zero `adjacency.pkl` (no label hierarchy) — plus `overrides.yaml`, which points
`train.py --task` at each split's dataset and sets the loss, class weights, the detected id
spelling, the regression target scaler and the model-selection metric (see
[Which metric to select on](#which-metric-to-select-on)). What it skips, since it only makes sense for GO: the MMseqs2 split, the CAZy set, the
CAFA ground truth and the test FASTA.

Ids in the CSV must match the ones in the dataset. Proteins found in only one of the two are
skipped, so check the `Number of proteins` lines at the start of training.

**Id spelling is detected, not configured.** FRIdata keys its indices `<id>_A` or
`AF-<id>-F1-model_v4_A` depending on how the dataset was built, while the CSV carries the bare
id. `preprocess.py` reads the dataset's own `embeddings.idx`, works out which spelling matches,
and records it as `data.<split>_unfix_type` in `overrides.yaml`; the summary line per split
prints what it found (`[ids: chain]`). Getting this wrong matches nothing and yields an *empty*
dataset rather than an error, which is why it is not left to the caller. `--set
data.trainval_unfix_type=…` still overrides it.

**Regression targets are standardised.** A target on its natural scale can be hostile to a
zero-initialised head: FLIP's Rhomax is 460-622, so the MSE starts near 290 000 and 20 epochs at
1e-4 never recover — that run ended at R2 -389. `preprocess.py` z-scores regression targets using
the **train split only**, so no eval or test statistic leaks into training, and records the
factors as `data.target_scaler` in `overrides.yaml`. Training runs in standardised space;
`output_scores()` and the regression metrics invert it, so the prediction TSVs and every reported
MSE / RMSE / MAE are in the target's own units. R2 and the correlations are invariant either
way. A constant target is left alone.

**3. Train.**

```bash
python train.py --task gb1                  # all three stages
python train.py --task gb1 --stages fusion  # just the fusion gate
```

Everything a GO run supports works here too — `--stages`, `--weights-*`, `--train-on`, the
sanity checks, the run directory and its outputs. Step 2 writes the target matrix and an
`overrides.yaml` under `preprocess.custom_tasks_dir` (see `configs/paths.yaml`), and `--task`
picks it up;
`--set` still wins over it, e.g. `--set training.use_class_weights=false` to train without
class weights.

What differs from a GO run: the best epoch is picked on the task's own metric — accuracy for
classification, Spearman for regression, rather than Fmax — there's no Fmax and no CAZy set
(both GO-specific), and the metrics follow the task — accuracy / balanced accuracy /
macro F1 for classification, P/R/F1 and AUROC for multi-task classification, MSE / RMSE / MAE /
R2 / Pearson / Spearman for regression. Predictions are softmax or sigmoid probabilities, or
the predicted values for regression.

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
python tests/test_calibration.py           # calibration sweep vs sklearn, F-max vs brute force
```

- the trainer architectures produce bit-identical logits to the model classes copied verbatim
  out of the original training notebooks (`tests/reference/notebook_models.py`);
- the parity check reports them identical to inference, and reports a deliberately modified
  architecture as diverged and undeployable, so the guard is known to work;
- all nine released deepFRI2 checkpoints load into trainer-built models with `strict=True`;
- the metrics match `sklearn.metrics.precision_recall_fscore_support`;
- the calibration grid matches `sklearn.metrics.precision_score` / `recall_score` at every
  threshold, and its F-max matches a brute-force sweep over every distinct score -- including for
  terms with no ground truth, terms everybody carries, and saturated score vectors full of ties.

The fusion stage re-checks fidelity against real data: the frozen branches must reproduce the
stand-alone sub-models' test predictions to 1e-6.

## Configuration

Hyperparameters and paths are not command-line arguments. Three YAML files are merged per run:

| File | Contents |
|---|---|
| `paths.yaml` | machine-specific roots (`project_location`, `deepfri2_src`, `runs_dir`) and the data-tree layout. **The only file to edit when moving hosts.** Its `layout:` block is the single definition of where every data file lives: training resolves it through `RunConfig`, and the CAFA evaluation, `calibrate.py` and `validate.ipynb` through `EvalPaths.from_configs()`. |
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
with `WeightedFocalLoss` and class weights, fusion 20 epochs @ 1e-4 with `MCLossDAG`. The
deliberate departures are `selection: best_min` with `selection_min_epoch: 5` (see below), and
fusion running the same 20 epochs as the other two stages rather than 15. `run.sh` records the
exact command for every released model.

### Checkpoint selection

Training always keeps two checkpoints — `<run>_best.pth` (the optimum of
`training.selection_metric` so far) and `<run>_last.pth` (most recent epoch). **Those two are
the only epochs a run can ship.** `training.selection` decides which of them becomes
`<run>.pth` and generates the prediction TSVs:

| | |
|---|---|
| `last` | the final epoch — what the originally released models used |
| `best_strict` | the optimum of `training.selection_metric`, whenever it occurred |
| `best` | the final epoch when it is within `selection_tolerance` of the optimum, the optimum otherwise |
| `best_min` (default) | `best_strict` over the epochs from `training.selection_min_epoch` (default 5) onwards; the warm-up epochs can never ship |

`best_min` exists because an opening epoch can score well for the wrong reason: a classifier that
has collapsed onto the majority class, or a correlation computed on predictions that have barely
moved off their initialisation. Those are not checkpoints worth shipping, and on a metric that is
noisy early, `best`/`best_strict` will happily pick one. The guard applies to the **rolling**
`_best.pth` as well as to the final choice — it has to, because only two checkpoints exist on
disk, so letting an ineligible epoch win `_best.pth` would leave the selection naming weights
that were overwritten. Set `selection_min_epoch: 1` to recover plain `best_strict`. If training
is shorter than the guard, nothing is eligible and the final epoch ships.

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

For a **custom task** there is no Fmax, so the choice is between eval loss and the metric the
task is actually judged by. The full set, from `SELECTION_METRICS` in `config.py` — `eval_loss`
is the only one minimised, every other is maximised:

| task kind | selectable | `preprocess.py --task` default |
|---|---|---|
| any | `eval_fmax`, `eval_loss` | — |
| `classification` | `eval_accuracy`, `eval_balanced_accuracy`, `eval_f1` | `eval_accuracy` |
| `multi-task-classification` | `eval_precision`, `eval_recall`, `eval_f1` | `eval_f1` |
| `regression` | `eval_r2`, `eval_spearman_mean`, `eval_pearson_mean` | `eval_spearman_mean` |

**Do not select a many-class classification task on `eval_loss`.** Cross-entropy bottoms out
long before accuracy peaks and then rises while accuracy is still climbing, so the run ships a
badly undertrained checkpoint. On a 1195-class fold-classification task the loss minimum landed
at epoch 6/20 with accuracy 0.299, while epoch 17 reached 0.516 — roughly half the accuracy the
same run had already trained, thrown away by the selection rule. For **regression** `eval_loss`
is fine, because it *is* the MSE; but if the benchmark reports a rank correlation, select on
`eval_spearman_mean` so the thing being optimised for and the thing being reported agree.

These all come from the headline metrics that `train_model()` flattens onto each epoch record,
so anything the metrics table prints can be selected on.

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
Train it first, run `python train.py --import-released` to import the released deepFRI2 checkpoints, ...
```

### `--import-released`

`python train.py --import-released` reads the run names the inference module declares
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
`deepFRI2/src/deepFRI2/config.py`. Copy `calibration_<fusion run>.json` too if the triple has
been calibrated — see [Calibration](#calibration) below.

## CAFA evaluation

[`validate.ipynb`](validate.ipynb) scores runs against deepFRI v1 and the published competitors
on the evaluation, test and CAZy splits, and produces the figures and tables of the paper. It is
a notebook rather than a CLI on purpose: figures are made by looking at them.

Scores come from [CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator)
(`cafaeval`, pinned in `environment.yml`). If it is not installed in the environment but a
checkout is at hand, point `CAFA_EVALUATOR_SRC` at its `src` directory.

The mechanics are in [`utils/evaluator.py`](src/deepfri2_trainer/utils/evaluator.py) and
[`utils/figures.py`](src/deepfri2_trainer/utils/figures.py); everything specific to a machine, a
dataset or a set of competitors — the run names, the method names and the colours — is in the
notebook's **Setup** cell; the paths themselves come from `configs/paths.yaml`, so neither the
notebook nor the modules carry a local path.

```python
paths = EvalPaths.from_configs()                # layout, roots and versions from configs/

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
  wired in — its location is the `ia` entry of `layout:` in `configs/paths.yaml`. Without the
  file, only the unweighted metrics are available.
- `summary()` reports `smin` as the minimum of the column it summarises. CAFA-evaluator's own
  `best` tables pick the threshold by the *unweighted* `s` and print `s_w` there, which is
  slightly higher than the minimum of `s_w`.

## Calibration

CAFA reports one number per ontology: how good the model is on average. It does not say whether a
given score on a given GO term is high, and that is what a reader of a deepFRI2 interpretability
report actually needs — a fusion score of 0.92 can mean 39% precision on one term and 95% on
another, because a term carried by 40% of the training proteins and one carried by 0.5% peak at
completely different thresholds.

[`calibrate.py`](calibrate.py) sweeps the decision threshold **per GO term, per sub-model and per
split** over the prediction TSVs a run already wrote, and stores the curves next to the fusion
checkpoint, named the way `labels_<run>.json` is:

```bash
python calibrate.py --dry-run            # resolved paths + every input checked
python calibrate.py                      # the released models (deepFRI2 MODEL_NAMES), all ontologies
python calibrate.py --ontology MF --sequence <run_sequence> --structure <run_structure> --fusion <run_fusion>
```

```
<runs_dir>/MF__fusion__<run_fusion>/calibration_<run_fusion>.json
```

Copy that file into `deepFRI2/params/<ontology>/` along with the checkpoints. deepFRI2's
`interpret.py` picks it up automatically and opens every report with a row of three panels —
fusion, sequence, structure — showing precision, recall and F1 against the threshold, with a
vertical line at the score *this* protein got. Without the file the reports are written exactly as
before, minus that row.

`eval` and `test` disagreeing on a term is almost always sample size, not a real gap. 

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
preprocess.py                 CLI entry point: GO inputs or labels CSVs -> target matrix
train.py                      CLI entry point: training
calibrate.py                  CLI entry point: per-GO-term calibration for the reports
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
    import_released.py        import released deepFRI2 checkpoints into runs_dir
    preprocess.py             target matrix from GO: targets / split / CAZy steps, shared config + logging
    csv_target_matrix.py      target matrix from labels CSVs instead of GO (preprocess.py --task)
    calibrate.py              per-GO-term threshold sweep -> calibration_<run>.json
    sanity.py                 sanity & validation checks
    utils/                    dataloader, training loop, losses
        target_matrix.py      protein -> GO-term supervision from the annotation tables
        split.py              homology-aware train/eval split (MMseqs2)
        evaluator.py          CAFA scores: ground truth, CAFA-evaluator, caching, tidy table
        figures.py            figures and tables from those scores
tests/
    test_model_equivalence.py architecture fidelity + parity
    test_metrics.py           logged P/R/F1 vs sklearn
    reference/                verbatim copy of the notebook model definitions
```
