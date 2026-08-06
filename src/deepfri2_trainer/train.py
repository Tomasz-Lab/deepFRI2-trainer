"""Config-driven wrapper around the training loop."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .config import RunConfig
from .data import Loaders, Targets
from .utils.training import count_trainable_parameters, train_model


def build_loss_kwargs(cfg: RunConfig, targets: Targets, device: str | torch.device) -> dict[str, Any]:
    """Translate the ``training.loss`` config block into ``train_model`` arguments."""
    loss_cfg = cfg.training["loss"]
    name = loss_cfg.get("name")

    if name is None:
        loss_fn_kwargs = None          # train_model falls back to WeightedFocalLoss
    elif name == "MCMLossDAG":
        loss_fn_kwargs = {"A": targets.adjacency.to(device), **(loss_cfg.get("kwargs") or {})}
    else:
        raise ValueError(
            f"unsupported loss {name!r}; configs may use 'MCMLossDAG' or null "
            "(null selects WeightedFocalLoss)"
        )

    return {
        "loss_fn_name": name,
        "loss_fn_kwargs": loss_fn_kwargs,
        "weights": targets.weights if cfg.training["use_class_weights"] else None,
    }


def run_training(
    cfg: RunConfig,
    model: nn.Module,
    loaders: Loaders,
    targets: Targets,
    log_wandb: bool = True,
    max_steps_per_epoch: int | None = None,
):
    """Train ``model`` following the configuration; returns ``(model, metrics)``.

    Assembles ``train_model``'s arguments from the config; the wandb run must already be open
    (see ``outputs.start_run``).
    """
    device = next(model.parameters()).device
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    loss_args = build_loss_kwargs(cfg, targets, device)

    return train_model(
        model=model,
        train_dataloader=loaders.train,
        eval_dataloader=loaders.eval,
        num_epochs=int(cfg.training["num_epochs"]),
        learning_rate=float(cfg.training["learning_rate"]),
        threshold=float(cfg.training["metric_threshold"]),
        use_embeddings=cfg.use_embeddings,
        use_distograms=cfg.use_distograms,
        grad_clip_max_norm=cfg.training["grad_clip_max_norm"],
        max_steps_per_epoch=max_steps_per_epoch,
        log_wandb=log_wandb,
        **loss_args,
    )
