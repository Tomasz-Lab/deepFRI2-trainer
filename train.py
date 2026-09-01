#!/usr/bin/env python
"""Train the deepFRI2 sub-models for one ontology.

Trained in dependency order; the fusion gate consumes the frozen sequence and structure
models:

    sequence    SequenceAnalyzer    ESM-2 embeddings   MCLossDAG
    structure   StructuralProber    CA distograms      WeightedFocalLoss (class weights)
    fusion      FusionModel         both               MCLossDAG

Hyperparameters, dataset versions and paths are not arguments -- they live in `configs/`
(`paths.yaml`, `data.yaml`, `sequence.yaml`, `structure.yaml`, `fusion.yaml`). The arguments
below select what to train and where to run it.

Parameters
----------
--ontology {MF,CC,BP}
    Required, except with --import-released.
--stages {sequence,structure,fusion} [...]
    Which models to train; default all three, always in sequence -> structure -> fusion order.
    Training `fusion` alone takes its frozen sub-models from `weights.<ontology>` in
    `configs/fusion.yaml`; training them together chains the freshly trained ones.
--weights-sequence NAME, --weights-structure NAME
    Checkpoint references, overriding `weights.<ontology>` without editing the configs. For a
    fusion run these are the frozen sub-models; for a sequence or structure run it is the
    checkpoint to fine-tune from. Accepts a wandb run name (resolved inside
    `<runs_dir>/<ontology>__<model type>__<name>/`), a run directory name, or a .pth path:
        python train.py --ontology CC --stages fusion \\
            --weights-sequence wandb-name-1 --weights-structure wandb-name-2
        python train.py --ontology MF --stages structure --weights-structure wandb-name-3
    Ignored for sub-models trained in the same call (those win).
--import-released
    Import the released deepFRI2 checkpoints into `runs_dir` and exit. (`--prepare` is kept as
    a deprecated alias.) Reads the run names from
    the inference module (`deepFRI2/src/deepFRI2/config.py :: MODEL_NAMES`) and copies
    `deepFRI2/params/<ontology>/<run>.pth` (plus its labels JSON, when shipped) into a normal
    run directory `<ontology>__<model type>__<run>/`. Those are what `configs/fusion.yaml`
    refers to by default. Restrict to one namespace with --ontology. Imported runs have no
    prediction TSVs, so the fusion branch sanity check reports itself skipped for them.
--train-on {train,train+eval}
    `train` (default) evaluates on the held-out eval split. `train+eval` trains on the union of
    both, as the released production models were trained; the eval metrics then include
    training data and are optimistic. With no held-out split left, `training.selection` has no
    signal to act on and the run ships its final epoch whatever the config says.
--device DEVICE
    Torch device, default `cuda:0`.
--no-wandb
    Disable wandb. Runs are then named `local-<timestamp>` and nothing is uploaded.
--config-dir DIR
    Alternative directory of YAML configs (default `<repo>/configs`).
--set KEY=VALUE [...]
    Config overrides, dotted keys into the merged config, values parsed as YAML. Repeatable,
    and accepts several assignments per flag; both forms below are equivalent:
        --set training.num_epochs=5 training.learning_rate=2.0e-4
        --set training.num_epochs=5 --set training.learning_rate=2.0e-4
    The applied overrides are echoed at startup.
--no-parity-check
    Skip comparing `src/deepfri2_trainer/model.py` against the deepFRI2 inference model
    definitions. Skip it only when no deepFRI2 checkout is around.
--max-steps-per-epoch N
    Cap train and eval batches per epoch. For smoke-testing the pipeline.
--keep-models
    Keep each stage's model and dataloaders in memory after it finishes.

Outputs
-------
One directory per run, `<runs_dir>/<ontology>__<model type>__<run name>/`, where `run name` is
the wandb run name:

    <run name>.pth                  state dict, loadable by deepFRI2 inference
    labels_<run name>.json          {"<column index>": "<GO term>"}
    config_<run name>.yaml          merged config + provenance (git commits, backend flags,
                                    parity verdict)
    predictions_<run name>.tsv      eval-set predictions
    predictions_test_<run name>.tsv
    predictions_cazy_<run name>.tsv
    <run name>_best.pth             optimum of selection_metric } training.selection picks which
    <run name>_last.pth             final epoch                 } becomes <run name>.pth
                                    (selection: last | best_strict | best)
    architecture_parity.diff        only when the architectures have diverged
    log.txt                         this run's console output
    source/                         snapshot of the code that produced the run

The config and the source snapshot are also logged to wandb as a `code` artifact. Every run
appends START / DONE (or FAILED) lines to the shared `<runs_dir>/training.log`, so training all
three stages yields three run directories, three log.txt files and three START/DONE pairs.

Examples
--------
    python train.py --import-released
    python train.py --ontology MF
    python train.py --ontology BP --train-on train+eval
    python train.py --ontology CC --stages fusion
    python train.py --ontology MF --max-steps-per-epoch 2 --set training.num_epochs=1 --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from deepfri2_trainer import (  # noqa: E402
    ONTOLOGIES,
    STAGE_ORDER,
    TRAIN_ON,
    import_released_runs,
    run_stages,
)


def _parse_overrides(assignments: list[str]) -> dict:
    """Turn ``training.num_epochs=5`` into ``{"training": {"num_epochs": 5}}``."""
    overrides: dict = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise SystemExit(f"--set expects KEY=VALUE, got {assignment!r}")
        key, _, raw_value = assignment.partition("=")
        value = yaml.safe_load(raw_value)
        node = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring in train.py for the full parameter list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ontology", choices=ONTOLOGIES,
                        help="Gene Ontology namespace to train (required unless --import-released)")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=list(STAGE_ORDER),
                        help="models to train (always run in sequence, structure, fusion order)")
    parser.add_argument("--train-on", choices=TRAIN_ON, default="train",
                        help="'train' for development, 'train+eval' for production models")
    parser.add_argument("--device", default="cuda:0", help="torch device")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--config-dir", default=None, help="alternative configs/ directory")
    # action="extend": with plain nargs="+", a repeated --set OVERWRITES the previous one, so
    # `--set a=1 --set b=2` silently dropped a=1. default=None avoids argparse's shared-mutable
    # -default trap.
    parser.add_argument("--set", nargs="+", action="extend", default=None, metavar="KEY=VALUE",
                        dest="overrides",
                        help="config overrides; repeatable, e.g. --set training.num_epochs=5 "
                             "--set data.batch_size=16")
    parser.add_argument("--weights-sequence", default=None, metavar="NAME",
                        help="sequence checkpoint: frozen sub-model (fusion run) or fine-tuning "
                             "starting point (sequence run)")
    parser.add_argument("--weights-structure", default=None, metavar="NAME",
                        help="structure checkpoint: frozen sub-model (fusion run) or fine-tuning "
                             "starting point (structure run)")
    parser.add_argument("--no-parity-check", action="store_true",
                        help="skip the architecture parity check against deepFRI2 inference")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="cap batches per epoch (smoke tests only)")
    parser.add_argument("--keep-models", action="store_true",
                        help="keep each stage's model in memory after it finishes")
    parser.add_argument("--import-released", "--prepare", dest="import_released",
                        action="store_true",
                        help="import the released deepFRI2 checkpoints into runs_dir and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = _parse_overrides(args.overrides)
    if overrides:
        print(f"config overrides: {overrides}")

    if args.import_released:
        import_released_runs(
            ontologies=(args.ontology,) if args.ontology else ONTOLOGIES,
            config_dir=args.config_dir,
            overrides=overrides,
        )
        return 0

    if args.ontology is None:
        parser.error("--ontology is required (unless --import-released)")

    weights = {
        which: name
        for which, name in (("sequence", args.weights_sequence),
                            ("structure", args.weights_structure))
        if name
    }
    if weights:
        overrides.setdefault("weights", {}).setdefault(args.ontology, {}).update(weights)

    run_stages(
        ontology=args.ontology,
        stages=tuple(args.stages),
        train_on=args.train_on,
        device=args.device,
        log_wandb=not args.no_wandb,
        config_dir=args.config_dir,
        overrides=overrides,
        max_steps_per_epoch=args.max_steps_per_epoch,
        check_architecture_parity=not args.no_parity_check,
        keep_models=args.keep_models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
