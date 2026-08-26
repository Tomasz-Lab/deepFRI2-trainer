"""Dataset and DataLoader construction.

Embeddings and distograms are read from the FRIdata HDF5 stores, distograms go through the
Gaussian similarity transform, everything is padded to ``MAX_SEQ_LEN`` and the mask marks valid
residues.
"""

import copy
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data


def get_data_config(dataset_name: str, datasets_dir: Path | str):
    """Locate an embedding/distogram index directory and read its embedding size."""
    datasets_dir = Path(datasets_dir)

    with open(datasets_dir / dataset_name / "dataset.json") as file:
        config = json.load(file)

    embedding_size = config["embedding_size"]

    return {"data_path": datasets_dir / dataset_name, "emb_size": embedding_size}


def process_distogram(distogram, sigma_dist):
    sim_matrix = torch.exp(-torch.square(distogram) / (2 * sigma_dist**2))
    sim_matrix.masked_fill_(torch.isinf(distogram) | torch.isnan(distogram), 0)
    return sim_matrix


def _unfix_protein_name(protein_id: str, unfix_type: str | None = None) -> str:
    if unfix_type == "AFDB_v4":
        return f"AF-{protein_id}-F1-model_v4_A"
    elif unfix_type == "chain":
        return f"{protein_id}_A"
    return protein_id


def _load_id_mapping(config: dict, idx_path: Path) -> dict:
    with open(idx_path, "r") as f:
        mapping = json.load(f)
    base = config["config"]["data_path"] + "/"
    return {k.replace(".pdb", ""): base + v for k, v in mapping.items()}


def pad_embedding(embedding: torch.Tensor, max_len: int, emb_size: int) -> torch.Tensor:
    padded = torch.zeros(max_len, emb_size)
    seq_len = min(embedding.shape[0], max_len)
    padded[:seq_len, :].copy_(embedding[:seq_len, :])
    return padded


def pad_distogram(distogram: torch.Tensor, max_len: int) -> torch.Tensor:
    padded = torch.full((max_len, max_len), float("inf"))
    seq_len = min(distogram.shape[0], max_len)
    padded[:seq_len, :seq_len].copy_(distogram[:seq_len, :seq_len])
    return padded


def _load_h5_tensor(fname_loc: str, *keys: str, max_retries: int = 3, retry_delay: int = 1) -> torch.Tensor:
    for attempt in range(max_retries):
        try:
            with h5py.File(fname_loc, "r") as f:
                node = f
                for key in keys:
                    node = node[key]
                return torch.tensor(node[()])
        except BlockingIOError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def _drop_eof_token(embedding: torch.Tensor | None) -> torch.Tensor | None:
    if embedding is None or embedding.shape[0] == 0:
        return embedding
    return embedding[:-1]


class DeepFRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        idx_dir: Path,
        protein_vectors: np.array,
        emb_size=1280,
        *,
        use_embeddings: bool = True,
        use_distograms: bool = True,
        unfix_type: str = None,  # 'AFDB_v4' or 'chain'
        MAX_SEQ_LEN: int = 1020,
        sigma_dist=10,
    ):
        self.emb_size = emb_size
        self.protein_vectors = protein_vectors
        self.use_embeddings = use_embeddings
        self.use_distograms = use_distograms
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.sigma_dist = sigma_dist

        if not self.use_embeddings and not self.use_distograms:
            raise ValueError("DeepFRIDataset requires at least one of use_embeddings or use_distograms to be True.")

        self.protein_vectors = {
            _unfix_protein_name(k, unfix_type): v for k, v in protein_vectors.items()
        }

        with open(idx_dir / "dataset.json", "r") as f:
            self.config = json.load(f)

        if self.use_embeddings:
            self.embeddings_mapping = _load_id_mapping(self.config, idx_dir / "embeddings.idx")
        else:
            self.embeddings_mapping = {}

        if use_distograms:
            self.distogram_mapping = _load_id_mapping(self.config, idx_dir / "distograms.idx")
        else:
            self.distogram_mapping = {}

        available_ids = set(self.protein_vectors.keys())
        if self.use_embeddings:
            available_ids &= set(self.embeddings_mapping.keys())
        if self.use_distograms:
            available_ids &= set(self.distogram_mapping.keys())
        # Sorted, not `list(set(...))`: set iteration order over strings varies between
        # processes (hash randomization), which made the protein -> dataset index mapping
        # differ from run to run.
        self.protein_ids = sorted(available_ids)
        print(f"Number of proteins: {len(self.protein_ids)}")

    def __len__(self):
        return len(self.protein_ids)

    def _get_embedding(self, protein_id):
        return _load_h5_tensor(self.embeddings_mapping[protein_id], protein_id)

    def _get_distogram(self, protein_id):
        return _load_h5_tensor(self.distogram_mapping[protein_id], protein_id, "distogram")

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]

        distogram = None
        if self.use_distograms and protein_id in self.distogram_mapping:
            distogram = self._get_distogram(protein_id)

        embedding = self._get_embedding(protein_id) if self.use_embeddings else None
        embedding = _drop_eof_token(embedding)

        embedding_seq_len = embedding.shape[0] if embedding is not None else None
        distogram_seq_len = distogram.shape[0] if distogram is not None else None

        if embedding_seq_len is None and distogram_seq_len is None:
            raise ValueError(f"Protein '{protein_id}' has neither embedding nor distogram data.")

        # Mask semantics should follow the residue/distogram length.
        # ESM embeddings keep <cls> but drop <eof>; mask tracks residue validity only.
        mask_seq_len = (
            distogram_seq_len
            if distogram_seq_len is not None
            else max(0, embedding_seq_len - 1)
        )
        valid_len = min(mask_seq_len, self.MAX_SEQ_LEN)

        # An absent modality is returned as an empty tensor: `process_batch` replaces it
        # with None anyway, and the default collate stacks it into a (B, 0) tensor.
        padded_embedding = (
            pad_embedding(embedding, self.MAX_SEQ_LEN, self.emb_size)
            if embedding is not None
            else torch.zeros(0)
        )
        if distogram is not None:
            dist_input = process_distogram(pad_distogram(distogram, self.MAX_SEQ_LEN), self.sigma_dist)
        else:
            dist_input = torch.zeros(0)

        mask = torch.zeros(self.MAX_SEQ_LEN, dtype=torch.float32)
        mask[:valid_len] = 1

        return (
            protein_id,
            padded_embedding,
            dist_input,
            self.protein_vectors[protein_id].to_dense(),
            mask,
        )


def _worker_init(worker_id: int) -> None:
    """Give every worker a deterministic, distinct seed for python/numpy."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _generator(seed: int | None):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def create_data_loaders(dataset, split_dir: Path | str, batch_size=32, num_workers=4, seed=None):
    """Split the dataset by the mmseqs train/eval assignment and build both loaders."""
    # TODO: should be an output of TargetMatrix
    split_dir = Path(split_dir)
    TRAIN_IDX = split_dir / "train.tsv"
    VAL_IDX = split_dir / "eval.tsv"

    kwargs = {'delimiter': '\t', 'names': ['protein_id', 'function_id']}
    train_df = pd.read_csv(TRAIN_IDX, **kwargs)
    val_df = pd.read_csv(VAL_IDX, **kwargs)

    train_proteins = train_df["protein_id"].unique()
    val_proteins = val_df["protein_id"].unique()

    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    train_dataset.protein_ids = train_proteins
    val_dataset.protein_ids = val_proteins

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=False: with several DataLoaders constructed upfront, persistent
        # workers for loaders that aren't currently being iterated still hold pinned prefetch
        # buffers in RAM indefinitely, causing steady memory growth -> swap thrashing ->
        # stall/OOM at epoch boundaries. Workers are spawned fresh per epoch instead.
        persistent_workers=False,
        generator=_generator(seed),
        worker_init_fn=_worker_init if seed is not None else None,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_worker_init if seed is not None else None,
    )

    return train_dataloader, eval_dataloader


def create_test_loader(dataset, batch_size=32, num_workers=4, seed=None):
    """Build a loader over the whole dataset."""
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        generator=_generator(seed),
        worker_init_fn=_worker_init if seed is not None else None,
    )

    return test_dataloader
