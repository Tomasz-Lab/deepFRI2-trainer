"""Architecture parity: trainer model definitions vs. the deepFRI2 inference repository.

Two questions per run:

1. **Source parity** -- does each architecture symbol still read exactly as it does in
   ``deepFRI2/src/deepFRI2/model.py``? Per-symbol verdict plus a unified diff.
2. **Checkpoint parity** -- can the inference implementation load a checkpoint from the
   trainer's model (``strict=True``) and produce the same logits? This decides whether a
   trained model is deployable as-is.

Both compare the *two implementations* against each other, in one process, on one device, with
whatever backend flags the run set. They deliberately say nothing about how the numbers move
between environments -- that is what :func:`probe_backend_sensitivity` measures.

Divergence is reported, not punished.
"""

from __future__ import annotations

import difflib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch

from . import model as model_defs
from .config import RunConfig

#: Symbols that make up the deepFRI2 architecture, in dependency order.
SYMBOLS = (
    "prepare_template_kernel",
    "grid_starts_from_shape",
    "effective_length_from_mask",
    "build_diag_band_mask",
    "build_upper_triangle_mask",
    "build_kernel_bank",
    "KernelParam",
    "Pooling",
    "StructuralProber",
    "SequenceAnalyzer",
    "FusionModel",
)

_PARITY_NUM_LABELS = 7
_PARITY_RESIDUES = 80


def _probe_inputs(emb_size: int, device: str | torch.device = "cpu", seed: int = 0):
    """Random embeddings / distogram / mask shaped like a real batch."""
    generator = torch.Generator().manual_seed(seed)
    embed = torch.randn(2, _PARITY_RESIDUES + 1, emb_size, generator=generator)
    dist = torch.rand(2, _PARITY_RESIDUES, _PARITY_RESIDUES, generator=generator)
    dist = 0.5 * (dist + dist.transpose(1, 2))
    mask = torch.zeros(2, _PARITY_RESIDUES)
    mask[0] = 1
    mask[1, : _PARITY_RESIDUES - 17] = 1
    return embed.to(device), dist.to(device), mask.to(device)


def _difference(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    delta = (a.double() - b.double()).abs()
    return float(delta.max()), float(delta.mean())


@dataclass
class CheckpointVerdict:
    """Can the inference implementation load and reproduce one model's checkpoint?"""

    loads: bool
    device: str
    max_abs: float | None = None    # over the probe batch's logits
    mean_abs: float | None = None
    logit_scale: float | None = None  # max |logit|, so the differences can be read relatively
    error: str | None = None

    @property
    def identical(self) -> bool:
        return self.loads and self.max_abs == 0.0

    def describe(self) -> str:
        if not self.loads:
            return f"FAILED to load into inference ({self.error})"
        verdict = "logits identical" if self.identical else "logits differ"
        return (
            f"loads OK, {verdict} on {self.device}: "
            f"max|d|={self.max_abs:.3e} mean|d|={self.mean_abs:.3e} "
            f"(max|logit|={self.logit_scale:.3e})"
        )

    def as_dict(self) -> dict:
        return {
            "loads_into_inference": self.loads,
            "device": self.device,
            "logits_max_abs_diff": self.max_abs,
            "logits_mean_abs_diff": self.mean_abs,
            "logits_max_abs_value": self.logit_scale,
            "error": self.error,
        }


@dataclass
class ParityReport:
    """Outcome of comparing the trainer architectures with the inference ones."""

    status: str                                  # "identical" | "diverged" | "unavailable"
    inference_module: str | None = None
    identical: list[str] = field(default_factory=list)
    differing: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    diff: str = ""
    checkpoint: dict[str, CheckpointVerdict] = field(default_factory=dict)
    note: str = ""

    @property
    def deployable(self) -> bool:
        """Every model's checkpoint loads into inference and reproduces its logits exactly."""
        return bool(self.checkpoint) and all(v.identical for v in self.checkpoint.values())

    @property
    def loadable(self) -> bool:
        """Every model's checkpoint at least loads into inference."""
        return bool(self.checkpoint) and all(v.loads for v in self.checkpoint.values())

    def summary(self) -> str:
        if self.status == "unavailable":
            return f"architecture parity: NOT CHECKED ({self.note})"
        lines = []
        if self.status == "identical":
            lines.append(f"architecture parity: source identical to {self.inference_module}")
        else:
            lines.append(f"architecture parity: source DIVERGED from {self.inference_module}")
            if self.differing:
                lines.append(f"  changed symbols: {', '.join(self.differing)}")
            if self.missing:
                lines.append(f"  symbols absent from inference: {', '.join(self.missing)}")
        for model_type, verdict in self.checkpoint.items():
            lines.append(f"  {model_type:<9} checkpoint -> inference: {verdict.describe()}")
        if self.checkpoint:
            lines.append(
                "  (both implementations run in this process on one device with this run's "
                "backend flags, so these differences are code differences only)"
            )
        if not self.loadable:
            lines.append(
                "  => deepFRI2 inference CANNOT run this model as-is; port the change to "
                "deepFRI2/src/deepFRI2/model.py before shipping the checkpoint."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "inference_module": self.inference_module,
            "identical_symbols": list(self.identical),
            "differing_symbols": list(self.differing),
            "symbols_absent_from_inference": list(self.missing),
            "checkpoint_parity": {k: v.as_dict() for k, v in self.checkpoint.items()},
            "loadable_by_inference": self.loadable,
            "deployable_by_inference": self.deployable,
            "note": self.note or None,
        }


def _import_inference_models(cfg: RunConfig):
    cfg.register_deepfri2_src()
    import deepFRI2.model as inference_model  # noqa: PLC0415

    return inference_model


def check_source_parity(
    cfg: RunConfig, trainer_module: ModuleType = model_defs
) -> ParityReport:
    """Diff every architecture symbol against the inference module.

    ``trainer_module`` defaults to :mod:`deepfri2_trainer.model`; pass another module to
    compare an alternative implementation.
    """
    try:
        inference = _import_inference_models(cfg)
    except Exception as error:  # missing / misconfigured deepFRI2 checkout
        return ParityReport(status="unavailable", note=f"{type(error).__name__}: {error}")

    identical, differing, missing, diff_chunks = [], [], [], []
    for name in SYMBOLS:
        ours = getattr(trainer_module, name, None)
        theirs = getattr(inference, name, None)
        if ours is None:
            missing.append(f"{name} (absent from the trainer)")
            continue
        if theirs is None:
            missing.append(name)
            continue
        our_src, their_src = inspect.getsource(ours), inspect.getsource(theirs)
        if our_src == their_src:
            identical.append(name)
            continue
        differing.append(name)
        diff_chunks.append(
            "".join(
                difflib.unified_diff(
                    their_src.splitlines(keepends=True),
                    our_src.splitlines(keepends=True),
                    fromfile=f"deepFRI2.model.{name}",
                    tofile=f"{trainer_module.__name__}.{name}",
                )
            )
        )

    return ParityReport(
        status="identical" if not differing and not missing else "diverged",
        inference_module=str(Path(inference.__file__)),
        identical=identical,
        differing=differing,
        missing=missing,
        diff="\n".join(diff_chunks),
    )


def check_checkpoint_parity(
    cfg: RunConfig,
    report: ParityReport,
    trainer_module: ModuleType = model_defs,
    device: str | torch.device = "cpu",
) -> ParityReport:
    """Load a trainer checkpoint into the inference implementation and compare logits.

    One verdict per model type, on small random inputs with a reduced label count. Runs on
    ``device`` so the comparison exercises the same kernels the run will use.
    """
    from .load_model import build_model_from  # noqa: PLC0415  (circular at module import time)

    try:
        inference = _import_inference_models(cfg)
    except Exception as error:
        report.checkpoint = {}
        report.note = report.note or f"{type(error).__name__}: {error}"
        return report

    emb_size = int(cfg.raw["sequence_model"].get("emb_size", 1280))
    embed, dist, mask = _probe_inputs(emb_size, device)

    for model_type in ("sequence", "structure", "fusion"):
        try:
            ours = build_model_from(
                trainer_module, cfg, model_type, _PARITY_NUM_LABELS, emb_size
            ).eval().to(device)
            theirs = build_model_from(
                inference, cfg, model_type, _PARITY_NUM_LABELS, emb_size
            ).eval().to(device)
        except Exception as error:
            report.checkpoint[model_type] = CheckpointVerdict(
                loads=False, device=str(device), error=f"cannot build: {error}"
            )
            continue

        try:
            theirs.load_state_dict(ours.state_dict(), strict=True)
        except Exception as error:
            report.checkpoint[model_type] = CheckpointVerdict(
                loads=False, device=str(device), error=str(error)
            )
            continue

        with torch.no_grad():
            our_logits = ours(embed, dist, mask)
            their_logits = theirs(embed, dist, mask)
        max_abs, mean_abs = _difference(our_logits, their_logits)
        report.checkpoint[model_type] = CheckpointVerdict(
            loads=True,
            device=str(device),
            max_abs=max_abs,
            mean_abs=mean_abs,
            logit_scale=float(our_logits.abs().max()),
        )

    return report


def check_parity(
    cfg: RunConfig,
    trainer_module: ModuleType = model_defs,
    device: str | torch.device = "cpu",
) -> ParityReport:
    """Full parity check: source diff plus checkpoint loadability."""
    report = check_source_parity(cfg, trainer_module)
    if report.status == "unavailable":
        return report
    return check_checkpoint_parity(cfg, report, trainer_module, device)


# ----------------------------------------------------------------------------------------
# How much do the numbers move between backends?
# ----------------------------------------------------------------------------------------

@contextmanager
def _cudnn_tf32(enabled: bool):
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


@dataclass
class BackendSensitivity:
    """Logit differences for one model across backend configurations.

    Each entry is ``label -> (max abs diff, mean abs diff)`` against the run's own
    configuration. This is the number to look at when logits disagree between environments:
    the parity check above cannot see it, because it runs both implementations under one
    configuration.
    """

    model_type: str
    device: str
    cudnn_allow_tf32: bool
    logit_scale: float
    comparisons: dict[str, tuple[float, float]] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.comparisons:
            return ""
        lines = [
            f"backend sensitivity ({self.model_type} model on {self.device}, "
            f"cudnn.allow_tf32={self.cudnn_allow_tf32}, max|logit|={self.logit_scale:.3e}):"
        ]
        for label, (max_abs, mean_abs) in self.comparisons.items():
            lines.append(f"  vs {label:<22} max|d|={max_abs:.3e} mean|d|={mean_abs:.3e}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "device": self.device,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "logits_max_abs_value": self.logit_scale,
            "comparisons": {
                label: {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs}
                for label, (max_abs, mean_abs) in self.comparisons.items()
            },
        }


def probe_backend_sensitivity(
    cfg: RunConfig, device: str | torch.device = "cpu", seed: int = 0
) -> BackendSensitivity | None:
    """Measure how far this run's logits move under the other backend configurations.

    Runs one model (the one this run trains) on random inputs under the run's own settings,
    then again with cuDNN's TF32 flag flipped and on CPU, and reports the differences. Cheap:
    a handful of forward passes on a two-protein batch.

    Returns ``None`` for a run whose model has no convolutions (the sequence model), where
    there is nothing for cuDNN to change.
    """
    from .load_model import build_model_from  # noqa: PLC0415

    if not cfg.use_distograms:
        return None

    emb_size = int(cfg.raw["sequence_model"].get("emb_size", 1280))
    model_type = cfg.model_type
    model = build_model_from(model_defs, cfg, model_type, _PARITY_NUM_LABELS, emb_size).eval()

    configured = bool(torch.backends.cudnn.allow_tf32)
    device = str(device)
    on_cuda = torch.device(device).type == "cuda"

    def _run(target_device: str) -> torch.Tensor:
        embed, dist, mask = _probe_inputs(emb_size, target_device, seed)
        with torch.no_grad():
            return model.to(target_device)(embed, dist, mask).cpu()

    reference = _run(device)
    sensitivity = BackendSensitivity(
        model_type=model_type,
        device=device,
        cudnn_allow_tf32=configured,
        logit_scale=float(reference.abs().max()),
    )

    if on_cuda:
        with _cudnn_tf32(not configured):
            flipped = _run(device)
        sensitivity.comparisons[f"cudnn tf32={not configured}"] = _difference(reference, flipped)

        cpu_logits = _run("cpu")
        sensitivity.comparisons["cpu (fp32)"] = _difference(reference, cpu_logits)
        model.to(device)

    return sensitivity
