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
--csv PATH
    A labels CSV (protein id column + a `label` column, see --label-columns) instead of the GO
    graph and annotation tables. Switches to a different path entirely: --ontology/--steps are
    ignored. Classification (0/1) vs. regression (real-valued) is detected from the label
    column's own dtype. Requires --task and one of --structures/--dataset.
--task NAME
    Task name for --csv; used for output directory naming and as `train.py --task NAME` later.
--structures DIR
    A directory of MMCIF structures; embeddings/distograms are built from it via FRIdata
    (configs/paths.yaml :: custom.fridata_src/fridata_config).
--dataset DIR
    An already-built FRIdata dataset directory, instead of --structures.
--id-column NAME
    Protein id column in --csv (default `protein_id`).
--embedder NAME
    FRIdata embedder type, overriding configs/paths.yaml :: custom.embedder.

Examples
--------
    python preprocess.py --dry-run
    python preprocess.py --ontology MF
    python preprocess.py --ontology MF --steps split
    python preprocess.py --ontology MF CC BP --set annotation_threshold=70

    python preprocess.py --csv labels.csv --structures mmcifs/ --task my_task
    python preprocess.py --csv labels.csv --dataset existing-fridata-dataset/ --task my_task
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
from deepfri2_trainer.custom_preprocess import run as run_custom  # noqa: E402
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

    # Custom (non-GO) classification/regression tasks: a CSV of labels instead of the GO graph
    # + annotation tables. --ontology/--steps are ignored in this mode.
    custom = parser.add_argument_group("custom task (CSV labels)")
    custom.add_argument("--csv", type=Path, metavar="PATH",
                        help="labels CSV (protein id column + one or more label columns); "
                             "switches to the custom classification/regression path")
    custom.add_argument("--task", metavar="NAME", help="task name, used for output dir naming "
                        "and later as `train.py --task NAME`")
    structures = custom.add_mutually_exclusive_group()
    structures.add_argument("--structures", type=Path, metavar="DIR",
                            help="directory of MMCIF structures; builds embeddings/distograms "
                                 "via FRIdata")
    structures.add_argument("--dataset", type=Path, metavar="DIR",
                            help="an already-built FRIdata dataset directory, skipping FRIdata")
    custom.add_argument("--id-column", default="protein_id", metavar="NAME",
                        help="protein id column in --csv")
    custom.add_argument("--label-columns", default=None, metavar="NAME[,NAME...]",
                        help="comma-separated label column(s); default: a single column named "
                             "'label'. Needed for a multi-task CSV, or one whose label column "
                             "is named differently")
    custom.add_argument("--embedder", default=None, metavar="NAME",
                        help="FRIdata embedder type, overriding configs/paths.yaml :: custom")

    # A benchmark (PEER, FLIP, ...) that already ships its own train/valid/test split: one CSV
    # + one already-built FRIdata dataset directory per split, instead of --csv/--structures and
    # MMseqs2 clustering. All three splits are required together.
    predefined = parser.add_argument_group("custom task, predefined split (e.g. PEER)")
    for split in ("train", "eval", "test"):
        predefined.add_argument(f"--{split}-csv", type=Path, metavar="PATH",
                                help=f"labels CSV for the {split} split")
        predefined.add_argument(f"--{split}-dataset", type=Path, metavar="DIR",
                                help=f"already-built FRIdata dataset directory for the {split} split")
    args = parser.parse_args(argv)
    label_columns = args.label_columns.split(",") if args.label_columns else None

    predefined_splits = {
        split: (getattr(args, f"{split}_csv"), getattr(args, f"{split}_dataset"))
        for split in ("train", "eval", "test")
        if getattr(args, f"{split}_csv") or getattr(args, f"{split}_dataset")
    }
    if predefined_splits:
        if args.csv:
            parser.error("--csv and --train-csv/--eval-csv are mutually exclusive")
        if not args.task:
            parser.error("--task is required")
        missing = {"train", "eval", "test"} - set(predefined_splits)
        if missing:
            parser.error(f"--{{train,eval,test}}-csv and --{{train,eval,test}}-dataset are all "
                         f"required together for a predefined split; missing {sorted(missing)}")
        for split, (csv_path, dataset_path) in predefined_splits.items():
            if not csv_path or not dataset_path:
                parser.error(f"--{split}-csv and --{split}-dataset must be given together")
        custom_cfg = CustomPreprocessConfig.from_configs(
            args.config_dir, **_parse_overrides(args.overrides))
        if args.dry_run:
            print(custom_cfg.describe())
            for split, (csv_path, dataset_path) in predefined_splits.items():
                print(f"{split:<19}: {csv_path} ({'ok' if csv_path.is_file() else 'MISSING'}) "
                      f"+ {dataset_path} ({'ok' if dataset_path.is_dir() else 'MISSING'})")
            print(f"{'output':<19}: {custom_cfg.task_dir(args.task)}")
            return 0
        run_predefined(
            custom_cfg, task=args.task, protein_id_col=args.id_column,
            label_columns=label_columns,
            splits={split: (csv_path, dataset_path)
                   for split, (csv_path, dataset_path) in predefined_splits.items()},
            command=" ".join(["python", Path(__file__).name, *(argv or sys.argv[1:])]),
        )
        return 0

    if args.csv:
        if not args.task:
            parser.error("--task is required with --csv")
        if not args.structures and not args.dataset:
            parser.error("--csv needs one of --structures or --dataset")
        custom_cfg = CustomPreprocessConfig.from_configs(
            args.config_dir, **_parse_overrides(args.overrides))
        if args.dry_run:
            print(custom_cfg.describe())
            print(f"{'csv':<19}: {args.csv} ({'ok' if args.csv.is_file() else 'MISSING'})")
            structure_label = "structures" if args.structures else "dataset"
            structure_source = args.structures or args.dataset
            print(f"{structure_label:<19}: {structure_source} "
                  f"({'ok' if structure_source.exists() else 'MISSING'})")
            print(f"{'output':<19}: {custom_cfg.task_dir(args.task)}")
            return 0
        run_custom(
            custom_cfg, csv_path=args.csv, task=args.task, protein_id_col=args.id_column,
            label_columns=label_columns,
            mmcif_dir=args.structures, dataset_dir=args.dataset, embedder=args.embedder,
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
