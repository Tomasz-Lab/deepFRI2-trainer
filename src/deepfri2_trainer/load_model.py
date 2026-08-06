"""Config block -> constructed model, plus checkpoint loading and backend settings.

Architectures come from :mod:`deepfri2_trainer.model`, imported here as ``model_defs`` because
``model`` names model *instances* throughout. The builders take the definitions module as an
argument so :mod:`deepfri2_trainer.parity` can construct the same model from either the
trainer's or the inference implementation.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

from . import model as model_defs
from .config import RunConfig


def apply_backend_settings(cfg: RunConfig) -> dict:
    """Apply the torch backend flags a run declares. Returns what was set, for the record.

    Only ``cudnn.allow_tf32`` so far: cuDNN runs convolutions in TF32 by default, which makes
    the structure model's kernel bank the one part of deepFRI2 whose outputs move between GPUs
    and against CPU. It is a global flag, so it is set per run rather than inside the model.
    """
    applied: dict[str, object] = {}
    if cfg.use_distograms:
        allow_tf32 = bool(cfg.raw["structure_model"].get("cudnn_allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = allow_tf32
        applied["cudnn_allow_tf32"] = allow_tf32
        print(f"torch.backends.cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}")
    return applied


def build_sequence_model(
    cfg: RunConfig, num_labels: int, emb_size: int, module: ModuleType = model_defs
) -> nn.Module:
    """``SequenceAnalyzer``: pooled ESM embeddings -> LayerNorm -> MLP -> logits."""
    spec = cfg.raw["sequence_model"]
    return module.SequenceAnalyzer(
        num_labels=num_labels,
        hidden_dim=int(spec["hidden_dim"]),
        pooling_method=str(spec["pooling_method"]),
        emb_size=emb_size,
        attn_hidden=int(spec["attn_hidden"]),
        attn_temperature=float(spec["attn_temperature"]),
    )


def build_structure_model(
    cfg: RunConfig, num_labels: int, module: ModuleType = model_defs
) -> nn.Module:
    """``StructuralProber``: distogram -> two learnable kernel banks -> stats -> logits."""
    spec = cfg.raw["structure_model"]

    m_diag, m_anti = int(spec["m_diag"]), int(spec["m_anti"])
    amp_dtype = spec.get("amp_dtype")
    if isinstance(amp_dtype, str):
        amp_dtype = getattr(torch, amp_dtype)
    hidden_dim = int(spec["hidden_dim"])

    return module.StructuralProber(
        num_labels=num_labels,
        arch_to_size_diag={f"diag_{i}": m_diag for i in range(1, int(spec["num_diag"]) + 1)},
        arch_to_size_anti={f"anti_{i}": m_anti for i in range(1, int(spec["num_anti"]) + 1)},
        canonical_diag_ms=m_diag,
        diag_stride=int(spec["diag_stride"]),
        canonical_anti_ms=m_anti,
        anti_stride=int(spec["anti_stride"]),
        diag_feats=tuple(spec["diag_feats"]),
        anti_feats=tuple(spec["anti_feats"]),
        peak_thresh=float(spec["peak_thresh"]),
        topk_k=int(spec["topk_k"]),
        frozen_kernels=False,
        amp_dtype=amp_dtype,
        enforce_symmetry_diag=bool(spec["enforce_symmetry_diag"]),
        enforce_symmetry_anti=bool(spec["enforce_symmetry_anti"]),
        enforce_positivity_diag=bool(spec["enforce_positivity_diag"]),
        enforce_positivity_anti=bool(spec["enforce_positivity_anti"]),
        hidden_dim_in=hidden_dim,
        hidden_dim_out=hidden_dim,
    )


def load_weights(model: nn.Module, checkpoint: Path | str, strict: bool = True):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint {checkpoint} not found")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    result = model.load_state_dict(state_dict, strict=strict)
    print(
        f"loaded {checkpoint}: missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}"
    )
    return model


def build_fusion_model(
    cfg: RunConfig, num_labels: int, emb_size: int, module: ModuleType = model_defs,
    load_submodels: bool = True,
) -> nn.Module:
    """``FusionModel``: frozen structure + frozen sequence sub-model + trainable gate.

    ``load_submodels=False`` builds the same shape with random sub-model weights, for the
    parity check.
    """
    structure_model = build_structure_model(cfg, num_labels, module)
    sequence_model = build_sequence_model(cfg, num_labels, emb_size, module)

    if load_submodels:
        refs = cfg.weights
        load_weights(structure_model, cfg.resolve_weights(refs["structure"], "structure"))
        load_weights(sequence_model, cfg.resolve_weights(refs["sequence"], "sequence"))

    return module.FusionModel(
        structure_model=structure_model,
        esm_model=sequence_model,
        gate_input=str(cfg.raw["fusion_model"]["gate_input"]),
        gate_init_bias=float(cfg.raw["fusion_model"]["gate_init_bias"]),
    )


def build_model_from(
    module: ModuleType,
    cfg: RunConfig,
    model_type: str,
    num_labels: int,
    emb_size: int,
    load_submodels: bool = False,
) -> nn.Module:
    """Build ``model_type`` from an explicit definitions module, on CPU."""
    if model_type == "sequence":
        return build_sequence_model(cfg, num_labels, emb_size, module)
    if model_type == "structure":
        return build_structure_model(cfg, num_labels, module)
    if model_type == "fusion":
        return build_fusion_model(cfg, num_labels, emb_size, module, load_submodels=load_submodels)
    raise ValueError(f"unknown model_type {model_type!r}")


def build_model(
    cfg: RunConfig, num_labels: int, emb_size: int, device: str | torch.device
) -> nn.Module:
    """Build the model this run trains, load any initial weights, move it to ``device``.

    A sequence or structure run with a ``weights`` reference is fine-tuned on top of that
    checkpoint; without one it starts from scratch. A fusion run always loads its two frozen
    sub-models.
    """
    apply_backend_settings(cfg)

    if cfg.model_type == "fusion":
        return build_fusion_model(cfg, num_labels, emb_size).to(device=device)

    if cfg.model_type == "sequence":
        model = build_sequence_model(cfg, num_labels, emb_size)
    elif cfg.model_type == "structure":
        model = build_structure_model(cfg, num_labels)
    else:
        raise ValueError(f"unknown model_type {cfg.model_type!r}")

    reference = cfg.weights.get(cfg.model_type)
    if reference:
        print(f"fine-tuning on top of {reference}")
        load_weights(model, cfg.resolve_weights(reference, cfg.model_type))

    return model.to(device=device)
