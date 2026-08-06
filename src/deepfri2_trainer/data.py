"""Target-matrix loading and dataloader construction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import RunConfig
from .utils.dataloader import DeepFRIDataset, create_data_loaders, create_test_loader, get_data_config


@dataclass
class Targets:
    """Everything the loss and the prediction writers need about the label space."""

    go_indices: dict[str, int]           # GO term -> column index
    protein_vectors: dict[str, Any]      # train/eval targets (sparse rows)
    protein_vectors_test: dict[str, Any]
    protein_vectors_cazy: dict[str, Any]
    weights: Any                         # per-GO-term class weights
    adjacency: torch.Tensor              # direct GO adjacency, child -> parent

    @property
    def num_labels(self) -> int:
        return len(self.go_indices)

    @property
    def go_terms(self) -> list[str]:
        """GO terms ordered by column index (the order model logits come in)."""
        return list(self.go_indices.keys())


def _load_pickle_for_ontology(path: Path, ontology: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)[ontology]


def load_targets(cfg: RunConfig) -> Targets:
    """Load the target matrix (train/eval + test + CAZy) for ``cfg.ontology``.

    ``adjacency_prop.pkl`` (the transitive closure) is not loaded: ``MCMLossDAG`` works on the
    direct edges in ``adjacency.pkl``.
    """
    tm = cfg.target_matrix_dir
    ont = cfg.ontology

    targets = Targets(
        go_indices=_load_pickle_for_ontology(tm / "go_indices.pkl", ont),
        protein_vectors=_load_pickle_for_ontology(tm / "protein_vectors.pkl", ont),
        protein_vectors_test=_load_pickle_for_ontology(tm / "protein_vectors_test.pkl", ont),
        protein_vectors_cazy=_load_pickle_for_ontology(
            cfg.cazy_target_matrix_dir / "protein_vectors.pkl", ont
        ),
        weights=_load_pickle_for_ontology(tm / "weights.pkl", ont),
        adjacency=_load_pickle_for_ontology(tm / "adjacency.pkl", ont),
    )

    # The CAZy target matrix is built independently; its label space must match.
    go_indices_cazy = _load_pickle_for_ontology(cfg.cazy_target_matrix_dir / "go_indices.pkl", ont)
    assert go_indices_cazy == targets.go_indices, (
        "CAZy go_indices differ from the train/eval go_indices -- the two target "
        "matrices were built with different GO versions or annotation thresholds."
    )

    return targets


@dataclass
class Loaders:
    """Dataloaders for one training run."""

    train: DataLoader          # honours cfg.train_on ("train" or "train+eval")
    eval: DataLoader
    test: DataLoader
    cazy: DataLoader
    emb_size: int


def build_loaders(cfg: RunConfig, targets: Targets) -> Loaders:
    """Build the train / eval / test / CAZy dataloaders.

    ``unfix_type`` restores the protein-id spelling used by the embedding and distogram
    indices: AFDB ids in the train/eval set, ``<id>_A`` in the test and CAZy sets.
    """
    dataset_kwargs = dict(
        use_embeddings=cfg.use_embeddings,
        use_distograms=cfg.use_distograms,
        MAX_SEQ_LEN=int(cfg.data["max_seq_len"]),
        sigma_dist=int(cfg.data["sigma_dist"]),
    )
    batch_size = int(cfg.data["batch_size"])
    num_workers = int(cfg.data["num_workers"])

    # `DeepFRIDataset.__init__` prints "Number of proteins: N"; the label goes on the same
    # line so the three datasets are distinguishable without touching the class.
    print("train/eval set: ", end="")
    emb_config = get_data_config(cfg.trainval_dataset_name, cfg.datasets_dir)
    dataset = DeepFRIDataset(
        emb_config["data_path"],
        protein_vectors=targets.protein_vectors,
        emb_size=emb_config["emb_size"],
        unfix_type="AFDB_v4",  # AF-<id>-F1-model_v4_A
        **dataset_kwargs,
    )
    train_dataloader, eval_dataloader = create_data_loaders(
        dataset, cfg.split_dir, batch_size=batch_size, num_workers=num_workers
    )

    # --- test ---
    print("test set:       ", end="")
    emb_config_test = get_data_config(cfg.testset_name, cfg.datasets_dir)
    dataset_test = DeepFRIDataset(
        emb_config_test["data_path"],
        protein_vectors=targets.protein_vectors_test,
        emb_size=emb_config_test["emb_size"],
        unfix_type="chain",  # <id>_A
        **dataset_kwargs,
    )
    test_dataloader = create_test_loader(dataset_test, batch_size=batch_size, num_workers=num_workers)

    # --- CAZy test ---
    print("cazy set:       ", end="")
    emb_config_cazy = get_data_config(cfg.cazyset_name, cfg.datasets_dir)
    dataset_cazy = DeepFRIDataset(
        emb_config_cazy["data_path"],
        protein_vectors=targets.protein_vectors_cazy,
        emb_size=emb_config_cazy["emb_size"],
        unfix_type="chain",
        **dataset_kwargs,
    )
    cazy_dataloader = create_test_loader(dataset_cazy, batch_size=batch_size, num_workers=num_workers)

    # production variant: train on train+eval
    train_loader = train_dataloader
    if cfg.train_on == "train+eval":
        train_loader = DataLoader(
            ConcatDataset([train_dataloader.dataset, eval_dataloader.dataset]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataloader.collate_fn,
        )

    return Loaders(
        train=train_loader,
        eval=eval_dataloader,
        test=test_dataloader,
        cazy=cazy_dataloader,
        emb_size=emb_config["emb_size"],
    )
