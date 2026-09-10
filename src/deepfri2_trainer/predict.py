"""Prediction writing: one ``<protein>\\t<GO term>\\t<probability>`` line per pair.

``predictions_<run>.tsv`` (eval), ``predictions_test_<run>.tsv``, ``predictions_cazy_<run>.tsv``
-- the input format of the CAFA evaluation, to be wired in later.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
from torch.utils.data import DataLoader

from .config import RunConfig
from .data import Loaders, Targets
from .utils.training import process_batch

SPLIT_PREFIX = {"eval": "predictions", "test": "predictions_test", "cazy": "predictions_cazy"}
SPLITS = tuple(SPLIT_PREFIX)


def prediction_path(cfg: RunConfig, split: str, run_dir: Path | str | None = None) -> Path:
    """``<run_dir>/predictions[_test|_cazy]_<run name>.tsv``."""
    if split not in SPLIT_PREFIX:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    directory = Path(run_dir) if run_dir is not None else cfg.run_dir
    return directory / f"{SPLIT_PREFIX[split]}_{cfg.run_name}.tsv"


def write_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: RunConfig,
    targets: Targets,
    split: str,
) -> Path:
    """Run inference over ``dataloader`` and write the predictions TSV.

    A classification model's raw logits are squashed through a sigmoid into probabilities; a
    regression model's outputs are the prediction already, so they are written as-is.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_path(cfg, split)

    device = next(model.parameters()).device
    go_terms = targets.go_terms

    model.eval()
    rows: list[tuple[str, str, float]] = []
    with torch.no_grad():
        for batch in dataloader:
            embed, dist, _target, mask = process_batch(
                batch, device=device, use_embeddings=cfg.use_embeddings, use_distograms=cfg.use_distograms
            )
            raw = model(embed, dist, mask).detach().cpu().numpy()
            preds = raw if targets.task_kind == "regression" else sigmoid(raw)
            for prot_id, pred in zip(batch[0], preds):
                for go_id, value in zip(go_terms, pred):
                    rows.append((prot_id, go_id, value))

    with open(path, "w") as handle:
        for prot_id, go_id, value in rows:
            handle.write(f"{prot_id}\t{go_id}\t{value}\n")

    print(f"{split}: {len(rows):,} predictions -> {path}")
    return path


def write_all_predictions(
    model: nn.Module,
    loaders: Loaders,
    cfg: RunConfig,
    targets: Targets,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Path]:
    """Write predictions for whichever of eval / test / CAZy have a loader.

    A custom (non-GO) target matrix typically has no held-out test or CAZy set, in which case
    ``loaders.test`` / ``loaders.cazy`` are ``None`` and those files are simply not written.
    """
    loader_by_split = {"eval": loaders.eval, "test": loaders.test, "cazy": loaders.cazy}
    return {
        split: write_predictions(model, loader_by_split[split], cfg, targets, split)
        for split in splits
        if loader_by_split[split] is not None
    }
