#!/usr/bin/env python
"""Build the target matrices and splits the trainer consumes, from the primitive inputs.

The step between FRIdata and `train.py`:

    provide IDs, annotations, GO graph        ->  data/inputs/
    FRIdata: sequences, distograms, embeddings
    preprocess.py: target matrix, splits     ->  data/target_matrix/     <- this script
    train.py
    validate.ipynb

Three steps, each runnable on its own:

    targets   the eight target-matrix pickles (go_indices, protein_vectors, weights, adjacency,
              grand truth, ...) and the test-set FASTA. Reads the multi-gigabyte annotation
              tables, so this is the slow one; it runs once for all requested ontologies.
    split     the train/eval FASTA, MMseqs2 clustering at `min_seq_id`, and the cluster split
              whose per-GO-term eval fraction is best balanced. Needs `targets` on disk.
    cazy      label vectors for the CAZy test set, in the GO-term order `targets` produced.
              Needs `targets` on disk.

Parameters
----------
--ontology {MF,CC,BP} [...]
    Which ontologies to build; default all three.
--steps {targets,split,cazy} [...]
    Which steps to run; default all three, in that order.
--set KEY=VALUE
    Override a config value, e.g. `--set annotation_threshold=70`. Accepted keys are the fields
    of PreprocessConfig (`annotation_threshold`, `qualities`, `exclude_roots`, `dataset_name`,
    `data_version`, `go_version`), dotted for the nested blocks: `--set split.num_trials=100`,
    `--set split.seed=1`.
--dry-run
    Print the resolved configuration and check every input exists, then exit.

Custom (non-GO) classification/regression tasks
------------------------------------------------
A benchmark like PEER that ships its own train/valid/test split: one labels CSV plus one
already-built FRIdata dataset directory per split. Switches to a different path entirely
(--ontology/--steps are ignored), and requires all three splits.

--task NAME
    Task name, used for output directory naming and as `train.py --task NAME` later.
--train-csv, --eval-csv, --test-csv PATH
    Labels CSV for each split.
--train-dataset, --eval-dataset, --test-dataset DIR
    FRIdata dataset directory for each split.
--id-column NAME
    Protein id column in the CSVs (default `protein_id`).
--label-columns NAME[,NAME...]
    Label column(s); default a single column named `label`.

Examples
--------
    python preprocess.py --dry-run
    python preprocess.py --ontology MF
    python preprocess.py --ontology MF --steps split
    python preprocess.py --ontology MF CC BP --set annotation_threshold=70

    python preprocess.py --task gb1 \\
        --train-csv train.csv --train-dataset fridata_train_dir \\
        --eval-csv  valid.csv --eval-dataset  fridata_valid_dir \\
        --test-csv  test.csv  --test-dataset  fridata_test_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from deepfri2_trainer.config import ONTOLOGIES  # noqa: E402
from deepfri2_trainer.custom_preprocess import CustomPreprocessConfig  # noqa: E402
from deepfri2_trainer.custom_preprocess import run_with_predefined_splits as run_predefined  # noqa: E402
from deepfri2_trainer.preprocess import STEPS, PreprocessConfig, run  # noqa: E402


def _parse_value(raw: str):
    """YAML scalar if it parses as one, otherwise the literal string.

    Path templates such as ``{project_location}/...`` are valid YAML flow mappings as far as the
    parser is concerned, so a failed parse means "this was meant as a string".
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _parse_overrides(assignments: list[str]) -> dict:
    """``split.num_trials=100`` -> ``{"split": {"num_trials": 100}}``, merged onto the config."""
    overrides: dict = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise SystemExit(f"--set expects KEY=VALUE, got {assignment!r}")
        key, _, raw_value = assignment.partition("=")
        parts = key.strip().split(".")
        if len(parts) == 1:
            overrides[parts[0]] = _parse_value(raw_value)
        else:
            overrides.setdefault(parts[0], {})[parts[1]] = _parse_value(raw_value)
    return overrides


def _check_inputs(cfg: PreprocessConfig, ontologies: list[str], steps: list[str]) -> int:
    """Report which primitive inputs each requested step needs, and whether they are there."""
    required: list[tuple[str, Path]] = []
    if "targets" in steps:
        required.append(("unified table", cfg.unified_file))
        required += [(f"annotations {o}", cfg.annotations_dir / f"annots_{o}.pickle")
                     for o in ontologies]
        required += [(f"GO graph {o}", cfg.graphs_dir / f"graph_{o}.json") for o in ontologies]
    if "split" in steps:
        required += [("sequence index", cfg.sequences_index), ("mmseqs", cfg.mmseqs_bin)]
    if "cazy" in steps:
        required += [("cazy data", cfg.cazy_input("data")),
                     ("cazy mapping", cfg.cazy_input("mapping"))]
        required += [(f"GO graph {o}", cfg.graphs_dir / f"graph_{o}.json") for o in ontologies]

    missing = 0
    for label, path in required:
        exists = path.exists()
        missing += not exists
        print(f"  [{'ok' if exists else 'MISSING'}] {label:<20} {path}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology", nargs="+", choices=ONTOLOGIES, default=list(ONTOLOGIES))
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")

    # Custom (non-GO) classification/regression task, predefined train/eval/test split (e.g.
    # PEER): one CSV + one already-built FRIdata dataset directory per split.
    custom = parser.add_argument_group("custom task (predefined split, e.g. PEER)")
    custom.add_argument("--task", metavar="NAME", help="task name, used for output dir naming "
                        "and later as `train.py --task NAME`")
    custom.add_argument("--id-column", default="protein_id", metavar="NAME",
                        help="protein id column in the labels CSVs")
    custom.add_argument("--label-columns", default=None, metavar="NAME[,NAME...]",
                        help="comma-separated label column(s); default: a single column named "
                             "'label'. Needed for a multi-task CSV, or one whose label column "
                             "is named differently")
    for split in ("train", "eval", "test"):
        custom.add_argument(f"--{split}-csv", type=Path, metavar="PATH",
                            help=f"labels CSV for the {split} split")
        custom.add_argument(f"--{split}-dataset", type=Path, metavar="DIR",
                            help=f"already-built FRIdata dataset directory for the {split} split")
    args = parser.parse_args(argv)
    label_columns = args.label_columns.split(",") if args.label_columns else None

    splits = {
        split: (getattr(args, f"{split}_csv"), getattr(args, f"{split}_dataset"))
        for split in ("train", "eval", "test")
    }
    if any(csv_path or dataset_path for csv_path, dataset_path in splits.values()):
        if not args.task:
            parser.error("--task is required")
        missing = [split for split, (csv_path, dataset_path) in splits.items()
                  if not csv_path or not dataset_path]
        if missing:
            parser.error(f"--train/--eval/--test-csv and -dataset are all required together; "
                         f"missing {missing}")
        custom_cfg = CustomPreprocessConfig.from_configs(
            args.config_dir, **_parse_overrides(args.overrides))
        if args.dry_run:
            print(custom_cfg.describe())
            for split, (csv_path, dataset_path) in splits.items():
                print(f"{split:<8}: {csv_path} ({'ok' if csv_path.is_file() else 'MISSING'}) "
                      f"+ {dataset_path} ({'ok' if dataset_path.is_dir() else 'MISSING'})")
            print(f"{'output':<8}: {custom_cfg.task_dir(args.task)}")
            return 0
        run_predefined(
            custom_cfg, task=args.task, protein_id_col=args.id_column,
            label_columns=label_columns, splits=splits,
            command=" ".join(["python", Path(__file__).name, *(argv or sys.argv[1:])]),
        )
        return 0

    # keep the canonical order regardless of the order given on the command line
    ontologies = [o for o in ONTOLOGIES if o in args.ontology]
    steps = [s for s in STEPS if s in args.steps]

    cfg = PreprocessConfig.from_configs(args.config_dir, **_parse_overrides(args.overrides))

    if args.dry_run:
        print(cfg.describe())
        print(f"ontologies         : {ontologies}\nsteps              : {steps}\n\ninputs:")
        missing = _check_inputs(cfg, ontologies, steps)
        print("\noutputs:")
        for ontology in ontologies:
            print(f"  {ontology}: {cfg.ontology_dir(ontology)}")
            if "cazy" in steps:
                print(f"  {ontology}: {cfg.cazy_dir(ontology)}")
        if missing:
            print(f"\n{missing} input(s) missing.")
        return 1 if missing else 0

    run(cfg, ontologies=ontologies, steps=steps,
        command=" ".join(["python", Path(__file__).name, *(argv or sys.argv[1:])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
