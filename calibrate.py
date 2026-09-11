#!/usr/bin/env python
"""Calibrate trained runs per GO term, for the interpretability reports.

The step after `train.py`, and the one that turns "the model score for this term is ..." into
something a reader can act on:

    train.py           checkpoints, prediction TSVs     ->  runs_dir/
    calibrate.py       per-GO-term precision / recall   ->  runs_dir/<ont>__fusion__<run>/   <- this
    (copy by hand)                                      ->  deepFRI2/params/<ontology>/
    deepFRI2 interpret.py: the curve, with this protein's score marked on it

CAFA reports one number per ontology: how good the model is on average. It does not say whether
a given score on a given GO term is high. That depends entirely on the term -- a term carried by
40% of proteins and one carried by 0.5% peak at completely different thresholds -- so the sweep
here is per term, per sub-model (sequence / structure / fusion) and per split.

Splits (`eval`, the homology-separated held-out split, and `test`, the experimental-structure
set) are calibrated separately and never pooled: where they disagree is exactly what tells a
reader how far to trust a term. CAZy is skipped -- it is a CAZy-specific protein set covering a
handful of MF terms. Scores are calibrated **raw**, not GO-DAG propagated, because that is what
the report displays; ground truth is propagated as always.

Output is one JSON per ontology, named after the fusion run like `labels_<run>.json` is:

    <runs_dir>/<ontology>__fusion__<run>/calibration_<run>.json

Copy it into `deepFRI2/params/<ontology>/` along with the checkpoints. See
`src/deepfri2_trainer/calibrate.py` for the file's contents and what is exact in it.

Parameters
----------
--ontology {MF,CC,BP} [...]
    Which ontologies to calibrate; default all three.
--sequence / --structure / --fusion RUN
    wandb run names of the triple to calibrate. Only valid with a single `--ontology`; without
    them the released models declared in `deepFRI2/src/deepFRI2/config.py :: MODEL_NAMES` are used.
--splits {eval,test} [...]
    Which splits to sweep; default both.
--step FLOAT
    Threshold grid spacing, default 0.05. Must divide 1.0. F-max is exact regardless.
--output-dir DIR
    Write the files here instead of into the fusion run directories.
--config-dir DIR
    Read `paths.yaml` / `data.yaml` from here instead of `configs/`.
--dry-run
    Print what would be calibrated and check every input exists, then exit.

Examples
--------
    python calibrate.py --dry-run
    python calibrate.py                                  # the released models, all three ontologies
    python calibrate.py --ontology MF
    python calibrate.py --ontology MF --sequence <run_sequence> \\
                        --structure <run_structure> --fusion <run_fusion>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfri2_trainer.calibrate import SPLITS, THRESHOLD_STEP, calibration_path, run  # noqa: E402
from deepfri2_trainer.config import ONTOLOGIES, MODEL_TYPES, load_config  # noqa: E402
from deepfri2_trainer.import_released import released_model_names  # noqa: E402
from deepfri2_trainer.utils.evaluator import EvalPaths  # noqa: E402


def _resolve_runs(args, ontologies: list[str], config_dir: str | None) -> dict[str, dict[str, str]]:
    """``{ontology: {model type: run name}}`` from the command line, or the released models."""
    explicit = {model_type: getattr(args, model_type) for model_type in MODEL_TYPES}
    if any(explicit.values()):
        if len(ontologies) != 1:
            raise SystemExit(
                "--sequence / --structure / --fusion name the runs of one ontology; "
                "pass a single --ontology with them, or none of them to use the released models.")
        missing = [model_type for model_type, run in explicit.items() if not run]
        if missing:
            raise SystemExit(f"missing run name for {missing}; give all three or none")
        return {ontologies[0]: explicit}

    # No runs given: calibrate whatever deepFRI2 currently ships.
    probe = load_config("fusion", ontologies[0], config_dir=config_dir)
    if probe.deepfri2_src is None:
        raise SystemExit(
            "`deepfri2_src` is null in configs/paths.yaml, so the released run names cannot be "
            "read; name the runs with --sequence / --structure / --fusion instead.")
    model_names = released_model_names(probe)
    return {ontology: dict(model_names[ontology]) for ontology in ontologies if ontology in model_names}


def _check_inputs(paths: EvalPaths, runs_by_ontology: dict[str, dict[str, str]],
                  splits: list[str]) -> int:
    """Report every input the sweep will open. Returns the number that is missing."""
    missing = 0
    for ontology, runs in runs_by_ontology.items():
        print(f"\n{ontology}:")
        for key in ("target_matrix", "split"):
            path = paths.path(key, ontology)
            print(f"  [{'ok' if path.exists() else 'MISSING'}] {key:<14} {path}")
            missing += not path.exists()
        if "test" in splits:
            path = paths.path("struct_test_ids")
            print(f"  [{'ok' if path.is_file() else 'MISSING'}] {'struct ids':<14} {path}")
            missing += not path.is_file()
        for model_type, run_name in runs.items():
            labels = sorted((paths.runs_dir or Path(".")).glob(f"*__{run_name}/labels_{run_name}.json"))
            print(f"  [{'ok' if labels else 'MISSING'}] {model_type + ' labels':<14} "
                  f"{labels[0] if labels else f'labels_{run_name}.json under {paths.runs_dir}'}")
            missing += not labels
            for split in splits:
                try:
                    print(f"  [ok]      {model_type + ' ' + split:<14} "
                          f"{paths.prediction_file(run_name, split)}")
                except FileNotFoundError as error:
                    print(f"  [MISSING] {model_type + ' ' + split:<14} {str(error).splitlines()[0]}")
                    missing += 1
        print(f"  -> {calibration_path(paths, ontology, runs['fusion'])}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology", nargs="+", choices=ONTOLOGIES, default=list(ONTOLOGIES))
    for model_type in MODEL_TYPES:
        parser.add_argument(f"--{model_type}", default=None, metavar="RUN",
                            help=f"wandb run name of the {model_type} model")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--step", type=float, default=THRESHOLD_STEP)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    # keep the canonical order regardless of the order given on the command line
    ontologies = [o for o in ONTOLOGIES if o in args.ontology]
    splits = [s for s in SPLITS if s in args.splits]

    paths = EvalPaths.from_configs(config_dir=args.config_dir)
    runs_by_ontology = _resolve_runs(args, ontologies, args.config_dir)

    print(f"project location   : {paths.project_location}")
    print(f"dataset            : {paths.dataset_name}")
    print(f"runs dir           : {paths.runs_dir}")
    print(f"splits             : {splits}")
    print(f"threshold grid     : 0.0 .. 1.0 step {args.step} (F-max exact)")
    for ontology, runs in runs_by_ontology.items():
        print(f"{ontology:<19}: {runs}")

    if args.dry_run:
        missing = _check_inputs(paths, runs_by_ontology, splits)
        if missing:
            print(f"\n{missing} input(s) missing.")
        return 1 if missing else 0

    written = run(
        paths, runs_by_ontology,
        ontologies=ontologies, splits=splits, step=args.step,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print("\nCopy next to the checkpoints they calibrate:")
    for ontology, path in written.items():
        print(f"  cp {path} <deepFRI2>/params/{ontology}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
