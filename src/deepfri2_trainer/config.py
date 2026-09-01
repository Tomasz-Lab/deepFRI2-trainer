"""Configuration loading and path resolution.

Three YAML files are merged per run: ``paths.yaml`` (machine-specific roots and the data-tree
layout), ``data.yaml`` (dataset and GO versions, annotation threshold, loaders) and
``<model type>.yaml`` (architecture, optimizer, loss, initial weights).

Run outputs are named after the wandb run, which is known only after ``wandb.init``; the name
is attached to the config at that point via :meth:`RunConfig.set_run_name`.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

ONTOLOGIES = ("MF", "CC", "BP")
MODEL_TYPES = ("sequence", "structure", "fusion")
TRAIN_ON = ("train", "train+eval")
SELECTIONS = ("best", "best_strict", "last")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class RunConfig:
    """Fully resolved configuration for one training run."""

    model_type: str
    ontology: str
    train_on: str
    raw: dict = field(repr=False)

    @property
    def use_embeddings(self) -> bool:
        return self.model_type in ("sequence", "fusion")

    @property
    def use_distograms(self) -> bool:
        return self.model_type in ("structure", "fusion")

    # ---------- config blocks ----------

    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def model(self) -> dict:
        return self.raw.get("model", {})

    @property
    def training(self) -> dict:
        return self.raw.get("training", {})

    @property
    def selection(self) -> str:
        """Which epoch's weights become the run's checkpoint.

        Only two epochs are ever kept on disk -- the optimum of ``selection_metric`` so far
        (``<run>_best.pth``) and the most recent one (``<run>_last.pth``) -- so those are the
        only two a run can ship.

        ``last``        the final epoch; reproduces the originally released models.
        ``best_strict`` the optimum of ``selection_metric``, whenever it occurred.
        ``best``        the final epoch when it is within ``selection_tolerance`` of the
                        optimum, the optimum otherwise. Prefers the later checkpoint when the
                        difference is within the epoch-to-epoch noise.
        """
        selection = str(self.training.get("selection", "best"))
        if selection not in SELECTIONS:
            raise ValueError(
                f"training.selection must be one of {SELECTIONS}, got {selection!r}"
            )
        return selection

    @property
    def selection_metric(self) -> str:
        """What ``best`` means: ``eval_fmax`` (maximise) or ``eval_loss`` (minimise)."""
        metric = str(self.training.get("selection_metric", "eval_fmax"))
        if metric not in ("eval_fmax", "eval_loss"):
            raise ValueError(
                f"training.selection_metric must be 'eval_fmax' or 'eval_loss', got {metric!r}"
            )
        return metric

    @property
    def seed(self) -> int:
        """Seed for weight init and batch order; drawn once and recorded if not configured."""
        seed = self.training.get("seed")
        if seed is None:
            seed = int(torch.seed() % 2**31)
            self.raw.setdefault("training", {})["seed"] = seed
        return int(seed)

    @property
    def weights(self) -> dict:
        """Checkpoint references for this ontology.

        Sequence / structure run: the checkpoint to fine-tune from, empty to train from
        scratch. Fusion run: the two frozen sub-models, both required.
        """
        refs = {
            which: reference
            for which, reference in (self.raw.get("weights", {}).get(self.ontology) or {}).items()
            if reference
        }
        if self.model_type == "fusion":
            missing = {"sequence", "structure"} - refs.keys()
            if missing:
                raise KeyError(
                    f"configs/fusion.yaml has no `weights.{self.ontology}` entry for "
                    f"{sorted(missing)}; add the wandb run names (or checkpoint paths) of the "
                    "trained sequence and structure models, or pass --weights-sequence / "
                    "--weights-structure."
                )
        return refs

    def weights_candidates(self, reference: str, model_type: str) -> list[Path]:
        path = Path(reference)
        if path.suffix == ".pth":
            return [path if path.is_absolute() else self.runs_dir / path]
        return [
            self.runs_dir / f"{self.ontology}__{model_type}__{reference}" / f"{reference}.pth",
            self.runs_dir / reference / f"{reference.rsplit('__', 1)[-1]}.pth",
        ]

    def resolve_weights(self, reference: str, model_type: str) -> Path:
        """Resolve a checkpoint reference: wandb run name, run directory, or ``.pth`` path.

        Raises ``FileNotFoundError`` listing what was tried -- a mistyped or unprepared
        checkpoint must fail loudly, not silently train from scratch.
        """
        candidates = self.weights_candidates(reference, model_type)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{model_type} weights {reference!r} not found for {self.ontology}; looked for:\n"
            + "\n".join(f"  {candidate}" for candidate in candidates)
            + "\nTrain it first, run `python train.py --import-released` to import the released deepFRI2 "
            f"checkpoints, or fix `weights.{self.ontology}` in configs/."
        )

    # ---------- derived names ----------

    @property
    def data_version(self) -> str:
        return str(self.data["data_version"])

    @property
    def go_version(self) -> str:
        return str(self.data["go_version"])

    @property
    def annotation_threshold(self) -> int:
        return int(self.data["annotation_threshold"])

    @property
    def dataset_name(self) -> str:
        return self.data["dataset_name"].format(data_version=self.data_version)

    @property
    def trainval_dataset_name(self) -> str:
        return self.dataset_name + self.data["trainval_suffix"]

    @property
    def testset_name(self) -> str:
        return self.dataset_name + self.data["testset_suffix"]

    @property
    def cazyset_name(self) -> str:
        return self.dataset_name + self.data["cazyset_suffix"]

    @property
    def params(self) -> str:
        """Target-matrix subdirectory, e.g. ``20250722__MF__50``."""
        return f"{self.go_version}__{self.ontology}__{self.annotation_threshold}"

    # ---------- run identity ----------

    def set_run_name(self, run_name: str) -> str:
        self.raw["run_name"] = str(run_name)
        return self.run_name

    @property
    def run_name(self) -> str:
        run_name = self.raw.get("run_name")
        if not run_name:
            raise RuntimeError("run name is not set yet; outputs.start_run sets it")
        return str(run_name)

    @property
    def run_id(self) -> str:
        return f"{self.ontology}__{self.model_type}__{self.run_name}"

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    # Output file names carry the run name so they stay identifiable when copied out of the
    # run directory, e.g. into deepFRI2/params/<ontology>/.

    @property
    def checkpoint_path(self) -> Path:
        """This run's output checkpoint (not to be confused with ``weights``, its inputs)."""
        return self.run_dir / f"{self.run_name}.pth"

    def candidate_checkpoint_path(self, which: str) -> Path:
        """Rolling checkpoint kept during training: ``best`` (lowest eval loss) or ``last``.

        Only these two are kept -- one file per epoch would cost 20x the disk for weights that
        are never used. Both survive the run, so ``selection`` can be reconsidered without
        retraining; which epoch each holds is recorded in ``config_<run>.yaml``.
        """
        if which not in ("best", "last"):
            raise ValueError(f"which must be 'best' or 'last', got {which!r}")
        return self.run_dir / f"{self.run_name}_{which}.pth"

    @property
    def labels_path(self) -> Path:
        return self.run_dir / f"labels_{self.run_name}.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / f"config_{self.run_name}.yaml"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "log.txt"

    @property
    def source_dir(self) -> Path:
        return self.run_dir / "source"

    # ---------- resolved paths ----------

    @property
    def project_location(self) -> Path:
        return Path(self.raw["project_location"])

    @property
    def deepfri2_src(self) -> Path | None:
        value = self.raw.get("deepfri2_src")
        return Path(value) if value else None

    @property
    def runs_dir(self) -> Path:
        return Path(self.raw["runs_dir"])

    def _layout(self, key: str) -> Path:
        return self.project_location / self.raw["layout"][key].format(
            dataset_name=self.dataset_name,
            params=self.params,
            data_version=self.data_version,
            go_version=self.go_version,
            ontology=self.ontology,
            threshold=self.annotation_threshold,
        )

    @property
    def datasets_dir(self) -> Path:
        return self._layout("datasets")

    @property
    def target_matrix_dir(self) -> Path:
        return self._layout("target_matrix")

    @property
    def split_dir(self) -> Path:
        return self._layout("split")

    @property
    def cazy_target_matrix_dir(self) -> Path:
        return self._layout("cazy_target_matrix")

    # ---------- helpers ----------

    def describe(self) -> str:
        lines = [
            f"model type          : {self.model_type}",
            f"ontology            : {self.ontology}",
            f"train on            : {self.train_on}",
            f"dataset             : {self.dataset_name}",
            f"target matrix params: {self.params}",
            f"epochs / lr         : {self.training['num_epochs']} / {self.training['learning_rate']}",
            f"checkpoint selection: {self.selection} ({self.selection_metric})",
            f"seed                : {self.seed}",
            f"loss                : {self.training['loss'].get('name') or 'WeightedFocalLoss'}"
            f"  (class weights: {self.training['use_class_weights']})",
            f"target matrix dir   : {self.target_matrix_dir}",
            f"split dir           : {self.split_dir}",
            f"cazy target matrix  : {self.cazy_target_matrix_dir}",
            f"runs dir            : {self.runs_dir}",
            f"deepFRI2 src        : {self.deepfri2_src or '(parity check disabled)'}",
        ]
        if self.model_type == "fusion":
            lines.append(f"frozen sub-models   : {self.weights}")
        else:
            lines.append(f"initial weights     : {self.weights or 'from scratch'}")
        if self.raw.get("run_name"):
            lines.append(f"run dir             : {self.run_dir}")
        return "\n".join(lines)

    def register_deepfri2_src(self) -> None:
        """Put the deepFRI2 inference package on ``sys.path`` (for the parity check)."""
        if self.deepfri2_src is None:
            raise RuntimeError("`deepfri2_src` is not set in configs/paths.yaml")
        src = str(self.deepfri2_src)
        if not (self.deepfri2_src / "deepFRI2" / "model.py").is_file():
            raise FileNotFoundError(
                f"`deepfri2_src` ({src}) does not contain deepFRI2/model.py; point it at the "
                "`src` directory of a deepFRI2 checkout, or set it to null to skip the parity "
                "check."
            )
        if src not in sys.path:
            sys.path.insert(0, src)


def load_config(
    model_type: str,
    ontology: str,
    train_on: str = "train",
    config_dir: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Load and merge the YAML configs for one training run."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    if ontology not in ONTOLOGIES:
        raise ValueError(f"ontology must be one of {ONTOLOGIES}, got {ontology!r}")
    if train_on not in TRAIN_ON:
        raise ValueError(f"train_on must be one of {TRAIN_ON}, got {train_on!r}")

    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR

    raw = _read_yaml(config_dir / "paths.yaml")
    raw = _deep_merge(raw, _read_yaml(config_dir / "data.yaml"))
    raw = _deep_merge(raw, _read_yaml(config_dir / f"{model_type}.yaml"))
    raw = _deep_merge(raw, raw.get("per_ontology", {}).get(ontology, {}))
    raw = _deep_merge(raw, overrides or {})

    # Every run carries all three architecture blocks: a fusion run rebuilds its frozen
    # sub-models, and the parity check builds all three models from any config. `model` is the
    # block of the model being trained; `<type>_model` addresses any of them, which is how a
    # fusion run reaches e.g. structure_model.cudnn_allow_tf32. Both names stay in sync.
    for other in MODEL_TYPES:
        base = _read_yaml(config_dir / f"{other}.yaml").get("model", {})
        if other == model_type:
            base = raw.get("model", base)
        raw[f"{other}_model"] = _deep_merge(base, raw.get(f"{other}_model", {}))
    raw["model"] = raw[f"{model_type}_model"]

    project_location = str(raw["project_location"])
    for key in ("deepfri2_src", "runs_dir"):
        if isinstance(raw.get(key), str):
            raw[key] = raw[key].format(project_location=project_location)

    return RunConfig(model_type=model_type, ontology=ontology, train_on=train_on, raw=raw)
