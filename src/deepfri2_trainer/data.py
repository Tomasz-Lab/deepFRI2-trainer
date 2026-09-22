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

    go_indices: dict[str, int]           # GO term (or custom label) -> column index
    protein_vectors: dict[str, Any]      # trainval targets
    protein_vectors_eval: dict[str, Any] | None   # custom task only; a GO run splits trainval
    protein_vectors_test: dict[str, Any]
    protein_vectors_cazy: dict[str, Any] | None   # GO only
    weights: Any                         # per-GO-term class weights
    adjacency: torch.Tensor              # direct GO adjacency, child -> parent
    task_kind: str | None = None         # None for a GO run; see RunConfig.task_kind

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
    """Load the target matrix for ``cfg.ontology``.

    ``adjacency_prop.pkl`` (the transitive closure) is not loaded: ``MCLossDAG`` works on the
    direct edges in ``adjacency.pkl``.

    Two of the pickles are optional and tell the flows apart: a GO run has the CAZy set but no
    separate eval one (it splits trainval), a custom task the other way round.
    """
    tm = cfg.target_matrix_dir
    ont = cfg.ontology

    cazy_go_indices_path = cfg.cazy_target_matrix_dir / "go_indices.pkl"
    targets = Targets(
        go_indices=_load_pickle_for_ontology(tm / "go_indices.pkl", ont),
        protein_vectors=_load_pickle_for_ontology(tm / "protein_vectors.pkl", ont),
        protein_vectors_eval=(
            _load_pickle_for_ontology(tm / "protein_vectors_eval.pkl", ont)
            if (tm / "protein_vectors_eval.pkl").is_file() else None
        ),
        protein_vectors_test=_load_pickle_for_ontology(tm / "protein_vectors_test.pkl", ont),
        protein_vectors_cazy=(
            _load_pickle_for_ontology(cfg.cazy_target_matrix_dir / "protein_vectors.pkl", ont)
            if cazy_go_indices_path.is_file() else None
        ),
        weights=_load_pickle_for_ontology(tm / "weights.pkl", ont),
        adjacency=_load_pickle_for_ontology(tm / "adjacency.pkl", ont),
        task_kind=cfg.task_kind,
    )

    if targets.protein_vectors_cazy is not None:
        # The CAZy target matrix is built independently; its label space must match.
        go_indices_cazy = _load_pickle_for_ontology(cazy_go_indices_path, ont)
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
    cazy: DataLoader | None    # None for a custom task (CAZy is a GO-only benchmark)
    emb_size: int


def build_loaders(cfg: RunConfig, targets: Targets) -> Loaders:
    """Build the train / eval / test / CAZy dataloaders.

    A GO run splits one trainval dataset by the MMseqs2 assignment in ``cfg.split_dir``. A
    custom task comes with an eval dataset of its own, so it is loaded like the test set and
    the MMseqs2 split is never touched.

    ``unfix_type`` restores the id spelling the embedding and distogram indices use -- AFDB
    ids in the GO trainval set, ``<id>_A`` in the test and CAZy ones. A custom task overrides
    it per split via ``data.<split>_unfix_type``.
    """
    dataset_kwargs = dict(
        use_embeddings=cfg.use_embeddings,
        use_distograms=cfg.use_distograms,
        MAX_SEQ_LEN=int(cfg.data["max_seq_len"]),
        sigma_dist=int(cfg.data["sigma_dist"]),
    )
    loader_kwargs = dict(
        batch_size=int(cfg.data["batch_size"]),
        num_workers=int(cfg.data["num_workers"]),
        seed=cfg.seed,
    )

    def dataset(split: str, protein_vectors, default_unfix: str | None) -> DeepFRIDataset:
        # DeepFRIDataset prints "Number of proteins: N"; this puts a label on the same line.
        print(f"{split:<9}: ", end="")
        config = get_data_config(cfg.dataset_for(split), cfg.datasets_dir)
        return DeepFRIDataset(
            config["data_path"],
            protein_vectors=protein_vectors,
            emb_size=config["emb_size"],
            unfix_type=cfg.data.get(f"{split}_unfix_type", default_unfix),
            **dataset_kwargs,
        )

    trainval = dataset("trainval", targets.protein_vectors, "AFDB_v4")  # AF-<id>-F1-model_v4_A
    if targets.protein_vectors_eval is None:
        train_loader, eval_loader = create_data_loaders(trainval, cfg.split_dir, **loader_kwargs)
    else:
        train_loader = create_test_loader(trainval, shuffle=True, **loader_kwargs)
        eval_loader = create_test_loader(
            dataset("evalset", targets.protein_vectors_eval, None), **loader_kwargs
        )

    test_loader = create_test_loader(
        dataset("testset", targets.protein_vectors_test, "chain"), **loader_kwargs  # <id>_A
    )
    cazy_loader = None if targets.protein_vectors_cazy is None else create_test_loader(
        dataset("cazyset", targets.protein_vectors_cazy, "chain"), **loader_kwargs
    )

    # production variant: train on train+eval
    if cfg.train_on == "train+eval":
        train_loader = DataLoader(
            ConcatDataset([train_loader.dataset, eval_loader.dataset]),
            batch_size=loader_kwargs["batch_size"],
            shuffle=True,
            num_workers=loader_kwargs["num_workers"],
            pin_memory=True,
            collate_fn=train_loader.collate_fn,
            generator=train_loader.generator,
            worker_init_fn=train_loader.worker_init_fn,
        )

    return Loaders(
        train=train_loader,
        eval=eval_loader,
        test=test_loader,
        cazy=cazy_loader,
        emb_size=trainval.emb_size,
    )
