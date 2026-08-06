"""Run outputs: wandb session, run directory, weights, labels, config, source snapshot, log.

A run is named after its wandb run, and every output file carries that name so it stays
identifiable when copied out of the run directory::

    <runs_dir>/<ontology>__<model type>__<run name>/
        <run name>.pth                    state dict, loadable by deepFRI2 inference
        labels_<run name>.json            {"<column index>": "<GO term>"}
        config_<run name>.yaml            merged config + provenance (git commits, parity)
        predictions_<run name>.tsv        eval-set predictions
        predictions_test_<run name>.tsv
        predictions_cazy_<run name>.tsv
        architecture_parity.diff          only when the architectures have diverged
        source/                           the code that produced the run

``<runs_dir>/training.log`` is a single append-only log shared by all runs. The config and
source snapshot are also logged to wandb as a ``code`` artifact.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .config import REPO_ROOT, RunConfig
from .data import Targets
from .parity import BackendSensitivity, ParityReport

LOG_NAME = "training.log"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _for_log(text: str) -> str:
    """Drop progress-bar repaints and escape codes; keep everything else verbatim."""
    if "\r" in text:
        return ""
    stripped = _ANSI.sub("", text)
    # A write that was nothing but cursor movement (tqdm's nested-bar repaints) leaves only
    # whitespace behind; keeping it would fill the log with blank lines.
    return "" if (stripped != text and not stripped.strip()) else stripped

#: Snapshotted into every run: everything that determines what was trained.
SOURCE_FILES = (
    "src/deepfri2_trainer/model.py",           # the model definitions
    "src/deepfri2_trainer/load_model.py",      # config block -> model
    "src/deepfri2_trainer/data.py",            # target matrix + dataloaders
    "src/deepfri2_trainer/train.py",           # config -> training arguments
    "src/deepfri2_trainer/pipeline.py",        # stage orchestration
    "src/deepfri2_trainer/utils/dataloader.py",
    "src/deepfri2_trainer/utils/training.py",
    "src/deepfri2_trainer/utils/losses.py",
    "train.py",                                # the entry point
)


class _Tee:
    """Write to the real stream, and a cleaned-up copy to a sink."""

    def __init__(self, stream, sink, strip_progress: bool = False):
        self._stream = stream
        self._sink = sink
        self._strip_progress = strip_progress

    def mute(self) -> None:
        """Stop copying to the sink, but stay in the stream chain.

        Used when a newer tee is installed on top: the old one must keep forwarding (wandb's
        wrapper holds a reference to it) without duplicating every line into the log.
        """
        self._sink = None

    def write(self, text):
        written = self._stream.write(text)
        if self._sink is not None:
            self._sink.write(_for_log(text) if self._strip_progress else text)
        return written

    def flush(self):
        self._stream.flush()
        if self._sink is not None:
            self._sink.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _Sink:
    """Buffers until a file is attached, then writes through to it."""

    def __init__(self):
        self._buffer = io.StringIO()
        self._handle = None
        self._closed = False

    def attach(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "w")
        self._handle.write(self._buffer.getvalue())
        self._handle.flush()
        self._buffer = io.StringIO()

    def write(self, text):
        # A tee from a finished stage can still be reachable through wandb's stream stack;
        # writes arriving after close belong to another run and must not reopen this log.
        if self._closed or not text:
            return
        (self._handle or self._buffer).write(text)

    def flush(self):
        if self._handle is not None:
            self._handle.flush()

    def close(self):
        self._closed = True
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class RunLogger:
    """Tee the console output of a run into ``<run_dir>/log.txt``.

    The run directory is only known once the wandb run exists, so output is buffered from the
    start of the stage and flushed by :meth:`attach`.

    :meth:`attach` also re-installs the tee over the *current* streams. That matters because
    ``wandb.init`` swaps ``sys.stdout`` to capture the console, and ``wandb.finish`` restores
    the stream it saved at init -- which, across several stages in one process, is a tee from
    an earlier stage. Without re-installing, every stage after the first would log nothing past
    ``wandb.init``, and its output would land in the first stage's log.
    """

    def __init__(self):
        self._sink = _Sink()
        self._saved: tuple = ()
        self._tees: list[_Tee] = []
        self.path: Path | None = None

    def _install(self) -> None:
        for tee in self._tees:
            tee.mute()
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._sink, strip_progress=True)
        sys.stderr = _Tee(sys.stderr, self._sink, strip_progress=True)
        self._tees.extend([sys.stdout, sys.stderr])

    def __enter__(self) -> RunLogger:
        self._install()
        return self

    def attach(self, path: Path) -> Path:
        self._install()
        self._sink.attach(path)
        self.path = path
        print(f"console log -> {path}")
        return path

    def append(self, text: str) -> None:
        """Append to this run's log file after the stage has finished."""
        if self.path is None:
            return
        with open(self.path, "a") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")

    def __exit__(self, *exc_info) -> None:
        try:
            self._sink.flush()
        finally:
            if self._saved:
                sys.stdout, sys.stderr = self._saved
            self._sink.close()


def _yaml_safe(value):
    """Coerce a value into something ``yaml.safe_dump`` accepts.

    Config and provenance carry str subclasses (``torch.__version__``), Paths and tuples; a
    dump failure here would lose the record of an otherwise finished run.
    """
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_safe(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (str, Path)):
        return str(value)
    if hasattr(value, "item"):          # numpy / torch scalars
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float)):
        return type(value).__bases__[0](value)
    return str(value)


def git_commit(repo: Path | str) -> str | None:
    """Current commit of ``repo``, with ``-dirty`` appended when there are local changes."""
    repo = Path(repo)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return None


def provenance(cfg: RunConfig, model: nn.Module | None = None, **extra) -> dict:
    """Everything about a run that is not part of the config itself."""
    info = {
        "run_name": cfg.raw.get("run_name"),
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "trainer_commit": git_commit(REPO_ROOT),
        "deepfri2_commit": git_commit(cfg.deepfri2_src.parent) if cfg.deepfri2_src else None,
        "torch_version": torch.__version__,
        "torch_backends": {
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        **extra,
    }
    if model is not None:
        info["architecture"] = getattr(model, "ARCHITECTURE", None)
    return info


def start_run(
    cfg: RunConfig,
    model: nn.Module,
    targets: Targets,
    log_wandb: bool = True,
    wandb_entity: str = "deepfri",
    wandb_project: str = "deepfri2",
    wandb_tags: tuple[str, ...] = ("non_subset",),
) -> str:
    """Open the wandb run, fix the run name, and return it.

    Called before training: the run name determines the run directory and every output file
    name. Without wandb the name falls back to ``local-<timestamp>``.
    """
    import wandb  # noqa: PLC0415

    if log_wandb:
        try:
            wandb.finish()
        except Exception:
            pass
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config={
                "learning_rate": cfg.training["learning_rate"],
                "epochs": cfg.training["num_epochs"],
                "loss_function": cfg.training["loss"].get("name"),
                "use_distograms": cfg.use_distograms,
                "use_embeddings": cfg.use_embeddings,
                "class_weights": cfg.training["use_class_weights"],
                "grad_clip_max_norm": cfg.training["grad_clip_max_norm"],
                "model_type": cfg.model_type,
                "ontology": cfg.ontology,
                "train_on": cfg.train_on,
                "dataset_name": cfg.dataset_name,
                "target_matrix_params": cfg.params,
                "annotation_threshold": cfg.annotation_threshold,
                "num_labels": targets.num_labels,
                "batch_size": int(cfg.data["batch_size"]),
                "weights": cfg.weights or None,
                **(getattr(model, "ARCHITECTURE", None) or {}),
            },
            tags=list(wandb_tags),
        )
        # An offline run has no friendly name; fall back so the run directory is still usable.
        run_name = wandb.run.name or f"offline-{wandb.run.id}"
    else:
        run_name = f"local-{datetime.now():%Y%m%d-%H%M%S}"

    cfg.set_run_name(run_name)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run name: {run_name}\nrun dir : {cfg.run_dir}")
    return run_name


def save_weights(model: nn.Module, cfg: RunConfig) -> Path:
    """Save the state dict as ``<run_dir>/<run name>.pth``."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"saved weights -> {cfg.checkpoint_path}")
    return cfg.checkpoint_path


def save_labels(targets: Targets, cfg: RunConfig) -> Path:
    """Save the label map as ``<run_dir>/labels_<run name>.json``.

    Format matches what deepFRI2 inference expects: ``{"<column index>": "<GO term>"}``.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.labels_path, "w") as handle:
        json.dump({index: go_term for go_term, index in targets.go_indices.items()}, handle)
    print(f"saved labels  -> {cfg.labels_path}")
    return cfg.labels_path


def save_config(cfg: RunConfig, model: nn.Module | None = None, parity: ParityReport | None = None,
                sensitivity: BackendSensitivity | None = None, **extra) -> Path:
    """Save the merged config plus provenance as ``<run_dir>/config_<run name>.yaml``."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": cfg.run_id,
        "model_type": cfg.model_type,
        "ontology": cfg.ontology,
        "train_on": cfg.train_on,
        "provenance": provenance(cfg, model, **extra),
        "architecture_parity": parity.as_dict() if parity is not None else None,
        "backend_sensitivity": sensitivity.as_dict() if sensitivity is not None else None,
        "config": cfg.raw,
    }
    with open(cfg.config_path, "w") as handle:
        yaml.safe_dump(_yaml_safe(payload), handle, sort_keys=False, default_flow_style=False)
    print(f"saved config  -> {cfg.config_path}")
    return cfg.config_path


def save_source_snapshot(cfg: RunConfig, parity: ParityReport | None = None) -> list[Path]:
    """Copy the code that produced the run into ``<run_dir>/source/``.

    With the saved config and the recorded git commits, enough to rebuild the run after the
    repository has moved on.
    """
    cfg.source_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for relative in SOURCE_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        target = cfg.source_dir / Path(relative).name
        shutil.copyfile(source, target)
        written.append(target)

    if parity is not None:
        (cfg.source_dir / "architecture_parity.txt").write_text(parity.summary() + "\n")
        written.append(cfg.source_dir / "architecture_parity.txt")
        if parity.diff:
            diff_path = cfg.run_dir / "architecture_parity.diff"
            diff_path.write_text(parity.diff)
            written.append(diff_path)

    print(f"saved source  -> {cfg.source_dir} ({len(written)} files)")
    return written


def log_run_artifacts(cfg: RunConfig, extra_files: tuple[Path, ...] = ()) -> None:
    """Log the config and the source snapshot to wandb as one ``code`` artifact."""
    import wandb  # noqa: PLC0415

    if wandb.run is None:
        print("wandb run is not active - skipping artifacts")
        return

    artifact = wandb.Artifact(f"run-{cfg.run_name}", type="code")
    if cfg.config_path.is_file():
        artifact.add_file(str(cfg.config_path))
    if cfg.source_dir.is_dir():
        artifact.add_dir(str(cfg.source_dir), name="source")
    for path in extra_files:
        if Path(path).is_file():
            artifact.add_file(str(path))
    wandb.run.log_artifact(artifact)
    print(f"logged wandb artifact: run-{cfg.run_name} (config + source snapshot)")


def append_training_log(cfg: RunConfig, event: str, **fields) -> Path:
    """Append ``<timestamp> | <EVENT> | <run id> | key=value ...`` to ``training.log``.

    ``event`` is ``START``, ``DONE``, ``FAILED`` or ``IMPORT``; the CAFA evaluation will append
    its own.
    """
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.runs_dir / LOG_NAME
    run_id = cfg.run_id if cfg.raw.get("run_name") else f"{cfg.ontology}__{cfg.model_type}__?"
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {event:<6} | {run_id} | {details}"
    with open(path, "a") as handle:
        handle.write(line + "\n")
    print(line)
    return path
