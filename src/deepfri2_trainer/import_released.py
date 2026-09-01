"""Import the released deepFRI2 checkpoints into the runs directory.

``python train.py --import-released`` copies the checkpoints the inference module declares
(``deepFRI2/src/deepFRI2/config.py :: MODEL_NAMES``, weights under
``deepFRI2/params/<ontology>/``) into run directories of the usual shape::

    <runs_dir>/MF__sequence__<run>/
        <run>.pth
        labels_<run>.json               (when the release ships one)
        config_<run>.yaml               provenance: imported, not trained here

They are then usable like any trained run, in particular as the frozen sub-models of a fusion
training -- what ``configs/fusion.yaml`` refers to by default. They have no prediction TSVs, so
the fusion branch sanity check reports itself skipped for them.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .config import MODEL_TYPES, ONTOLOGIES, RunConfig, load_config
from .outputs import append_training_log, git_commit


def released_model_names(cfg: RunConfig) -> dict[str, dict[str, str]]:
    """``MODEL_NAMES`` as declared by the deepFRI2 inference module."""
    cfg.register_deepfri2_src()
    from deepFRI2.config import MODEL_NAMES  # noqa: PLC0415

    return MODEL_NAMES


def released_params_dir(cfg: RunConfig) -> Path:
    """``deepFRI2/params`` -- where the released checkpoints live."""
    return cfg.deepfri2_src.parent / "params"


def import_released_runs(
    ontologies: tuple[str, ...] = ONTOLOGIES,
    model_types: tuple[str, ...] = MODEL_TYPES,
    config_dir: Path | str | None = None,
    overrides: dict | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Materialise a run directory per released checkpoint. Returns ``{run_id: run_dir}``."""
    prepared: dict[str, Path] = {}

    probe = load_config("sequence", ontologies[0], config_dir=config_dir, overrides=overrides)
    if probe.deepfri2_src is None:
        raise RuntimeError(
            "`deepfri2_src` is null in configs/paths.yaml, so there is no inference module to "
            "import the released checkpoints from."
        )
    model_names = released_model_names(probe)
    params_dir = released_params_dir(probe)
    commit = git_commit(probe.deepfri2_src.parent)
    print(f"importing released checkpoints from {params_dir} (deepFRI2 commit {commit})")

    for ontology in ontologies:
        if ontology not in model_names:
            print(f"{ontology}: not declared in deepFRI2 MODEL_NAMES - skipping")
            continue
        for model_type in model_types:
            run_name = model_names[ontology].get(model_type)
            if not run_name:
                print(f"{ontology}/{model_type}: not declared in MODEL_NAMES - skipping")
                continue

            weights = params_dir / ontology / f"{run_name}.pth"
            if not weights.is_file():
                print(f"{ontology}/{model_type}: {weights} missing - skipping")
                continue

            cfg = load_config(model_type, ontology, config_dir=config_dir, overrides=overrides)
            cfg.set_run_name(run_name)

            if cfg.checkpoint_path.is_file() and not overwrite:
                print(f"{cfg.run_id}: already prepared - skipping")
                prepared[cfg.run_id] = cfg.run_dir
                continue

            cfg.run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(weights, cfg.checkpoint_path)

            labels = params_dir / ontology / f"labels_{run_name}.json"
            if labels.is_file():
                shutil.copyfile(labels, cfg.labels_path)

            with open(cfg.config_path, "w") as handle:
                yaml.safe_dump(
                    {
                        "run_id": cfg.run_id,
                        "model_type": model_type,
                        "ontology": ontology,
                        "provenance": {
                            "run_name": run_name,
                            "imported_from": str(weights),
                            "imported_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                            "deepfri2_commit": commit,
                            "note": (
                                "Released deepFRI2 checkpoint imported by `train.py --import-released`; "
                                "not trained by this repository. No prediction TSVs, so the "
                                "fusion branch sanity check cannot use it."
                            ),
                        },
                    },
                    handle,
                    sort_keys=False,
                )

            append_training_log(
                cfg, "IMPORT", source=weights, labels=labels.is_file(), deepfri2_commit=commit
            )
            prepared[cfg.run_id] = cfg.run_dir

    print(f"\nprepared {len(prepared)} run directories under {probe.runs_dir}")
    for run_id in sorted(prepared):
        print(f"  {run_id}")
    return prepared
