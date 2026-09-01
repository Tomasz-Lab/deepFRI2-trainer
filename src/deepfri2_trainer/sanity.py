"""Sanity and validation checks.

Cheap relative to a training run, and they catch the failure modes that silently corrupt a
model: mis-aligned masks, a wrong label space, an unfrozen or wrongly loaded sub-model, a
truncated predictions file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RunConfig
from .data import Loaders, Targets
from .utils.training import count_trainable_parameters, process_batch


def _present(tensor: torch.Tensor | None) -> bool:
    """False for an absent modality (returned as an empty tensor by the dataset)."""
    return tensor is not None and tensor.numel() > 0


def check_dataloader_consistency(
    dataloader: DataLoader,
    device: str | torch.device,
    protein_id: str,
    max_batches: int | None = None,
) -> None:
    """Embeddings, distograms and masks must not depend on what else is requested.

    Iterates the loader three times with different ``use_embeddings`` / ``use_distograms``
    combinations and asserts the tensors for ``protein_id`` agree.

    A modality the dataset was not built with comes back empty, so its comparison is
    trivially satisfied; the check has teeth only on a loader carrying both.
    """

    def _find(use_embeddings: bool, use_distograms: bool):
        for idx, batch in enumerate(dataloader):
            embed, dist, target, mask = process_batch(
                batch, device=device, use_embeddings=use_embeddings, use_distograms=use_distograms
            )
            if protein_id in batch[0]:
                positions = [i for i, el in enumerate(batch[0]) if el == protein_id]
                return embed, dist, target, mask, positions
            if max_batches is not None and idx + 1 >= max_batches:
                break
        raise AssertionError(
            f"protein {protein_id!r} not found in the dataloader "
            f"(use_embeddings={use_embeddings}, use_distograms={use_distograms})"
        )

    _, dist1, _, mask1, id1 = _find(use_embeddings=False, use_distograms=True)
    embed2, _, _, mask2, id2 = _find(use_embeddings=True, use_distograms=False)
    embed3, dist3, _, mask3, id3 = _find(use_embeddings=True, use_distograms=True)

    assert (mask1[id1] == mask2[id2]).all(), "mask differs between distogram-only and embedding-only"
    assert (mask1[id1] == mask3[id3]).all(), "mask differs between distogram-only and combined"
    assert (dist1[id1] == dist3[id3]).all(), "distogram differs between distogram-only and combined"
    assert (embed2[id2] == embed3[id3]).all(), "embedding differs between embedding-only and combined"

    modalities = [
        name for name, tensor in (("embeddings", embed3), ("distograms", dist3)) if _present(tensor)
    ]
    print(f"dataloader consistency OK (probe protein: {protein_id}, present: {', '.join(modalities)})")


def check_batch_shapes(dataloader: DataLoader, num_proteins: int = 3) -> None:
    """Print per-protein valid lengths so padding can be eyeballed.

    A protein's mask length, the number of non-zero embedding rows and the number of
    non-zero distogram rows should agree (up to the <cls> token in the embeddings).
    """
    prot_id, embed, dist, _target, mask = next(iter(dataloader))[:5]
    for idx in range(min(num_proteins, len(prot_id))):
        n_mask = int((mask[idx] == 1).sum())
        n_embed = int((embed[idx][:, 0] != 0).sum()) if _present(embed) else "-"
        n_dist = int((dist[idx][:, 0] != 0).sum()) if _present(dist) else "-"
        print(f"{prot_id[idx]}: mask={n_mask} embedding rows={n_embed} distogram rows={n_dist}")


def show_example(dataloader: DataLoader, targets: Targets, protein_vectors, unfix_type: str) -> str:
    """Plot one similarity matrix (when distograms are loaded) and print its annotations."""
    prot_id, _embed, dist, _target, _mask = next(iter(dataloader))[:5]
    if _present(dist):
        plt.imshow(dist.numpy()[0])
        plt.title(prot_id[0])
        plt.show()

    raw_id = prot_id[0]
    if unfix_type == "AFDB_v4":
        key = raw_id.replace("AF-", "").replace("-F1-model_v4_A", "")
    elif unfix_type == "chain":
        key = raw_id.removesuffix("_A")
    else:
        key = raw_id

    vector = protein_vectors[key]
    print(f"\nExample protein {key}:")
    print(f"Number of annotations: {vector.sum()}")
    print(
        "Annotated GO terms: "
        f"{[go_id for go_id, idx in targets.go_indices.items() if vector[idx] == 1]}"
    )
    return raw_id


def check_label_space(targets: Targets, model: nn.Module) -> None:
    """The model's output width must equal the number of GO terms in the target matrix."""
    out_features = None
    for attr in ("output_layer", "out"):
        layer = getattr(model, attr, None)
        if isinstance(layer, nn.Linear):
            out_features = layer.out_features
    if out_features is None:
        out_features = getattr(model, "num_labels", None)
    assert out_features == targets.num_labels, (
        f"model outputs {out_features} labels but the target matrix has {targets.num_labels}"
    )
    print(f"label space OK: {targets.num_labels} GO terms")


def report_trainable_parameters(model: nn.Module, expect_only: str | None = None) -> None:
    """List trainable parameters; optionally assert they are all under one prefix.

    For the fusion model, ``expect_only="refine_gate"`` verifies that both sub-models
    really are frozen and only the gate is being trained.
    """
    trainable = [(name, tuple(p.shape)) for name, p in model.named_parameters() if p.requires_grad]
    print("trainable:")
    for name, shape in trainable:
        print(" -", name, shape)
    print(f"{count_trainable_parameters(model):,} trainable parameters")

    if expect_only is not None:
        unexpected = [name for name, _ in trainable if not name.startswith(expect_only)]
        assert not unexpected, f"expected only `{expect_only}.*` to be trainable, also got {unexpected}"


def _submodel_test_predictions(cfg: RunConfig, reference: str, model_type: str) -> Path | None:
    """Locate the test predictions written when a fusion sub-model was trained.

    Returns ``None`` when there are none -- the case for checkpoints imported by
    ``train.py --import-released``, which have weights but no predictions.
    """
    weights = cfg.resolve_weights(reference, model_type)
    # Predictions sit next to the weights and carry the same run name.
    candidates = [weights.parent / f"predictions_test_{weights.stem}.tsv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    print(
        f"  SKIPPED for {reference}: no test predictions found (looked for "
        + " and ".join(str(c) for c in candidates)
        + ")"
    )
    return None


def check_fusion_branches(
    model: nn.Module,
    loaders: Loaders,
    cfg: RunConfig,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """The frozen branches must reproduce their stand-alone models' test predictions.

    Compares ``logits_struct`` / ``logits_esm`` from one test batch against the test
    predictions of the sub-models this fusion model was built from. This is the check that
    catches a wrongly loaded or mis-configured sub-model.

    A branch whose sub-model has no prediction file next to its weights -- i.e. a checkpoint
    not produced by this trainer -- is reported as skipped instead of failing.
    """
    device = next(model.parameters()).device
    refs = cfg.weights

    model.eval()
    batch = next(iter(loaders.test))
    embed, dist, _target, mask = process_batch(
        batch, device=device, use_embeddings=True, use_distograms=True
    )
    with torch.no_grad():
        _logits, logits_struct, logits_esm, _gate = model(embed, dist, mask, return_branches=True)

    prot_idx = int(np.random.randint(len(batch[0])))
    protein = batch[0][prot_idx]
    print(f"comparing branch outputs for {protein}")

    def _compare(branch: str, reference: str, logits: torch.Tensor) -> bool:
        path = _submodel_test_predictions(cfg, reference, branch)
        if path is None:
            return False
        table = pd.read_csv(path, delimiter="\t", header=None, names=["protein", "go_term", "score"])
        expected = table[table["protein"] == protein]["score"].to_numpy()
        assert expected.size, f"{protein} has no predictions in {path}"
        np.testing.assert_allclose(
            logits[prot_idx].detach().sigmoid().cpu().numpy(), expected, rtol=rtol, atol=atol
        )
        print(f"  {branch} branch matches {path.name}")
        return True

    checked = [
        _compare("structure", refs["structure"], logits_struct),
        _compare("sequence", refs["sequence"], logits_esm),
    ]
    if all(checked):
        print("fusion branch outputs match the stand-alone sub-model predictions")
    else:
        print("WARNING: fusion branch check incomplete - some sub-models had no predictions")


def check_prediction_file(path: str | Path, targets: Targets, expected_proteins: int | None = None):
    """Re-read a written predictions TSV and check its shape and value range."""
    table = pd.read_csv(path, delimiter="\t", header=None, names=["protein", "go_term", "score"])
    n_proteins = table["protein"].nunique()
    assert len(table) == n_proteins * targets.num_labels, (
        f"{path}: expected {n_proteins} x {targets.num_labels} rows, got {len(table)}"
    )
    assert table["score"].between(0.0, 1.0).all(), f"{path}: scores outside [0, 1]"
    if expected_proteins is not None:
        assert n_proteins == expected_proteins, (
            f"{path}: {n_proteins} proteins, expected {expected_proteins}"
        )
    print(f"{Path(path).name}: {n_proteins:,} proteins x {targets.num_labels} GO terms OK")
    return table
