"""Target matrix for a custom (non-GO) classification or regression task, built from CSV
labels and a predefined train/eval/test split -- for a benchmark like PEER that ships its own
split and its own precomputed structures, trained with the same sequence/structure/fusion
architectures as the GO flow.

    protein_id,label
    P12345,1
    P67890,0

The label column defaults to ``label``; pass ``--label-columns`` for a different name or
several columns (multi-task). Classification vs. regression is detected from the label
column's dtype: 0/1 is classification, real-valued is regression.

Structures come from an already-built FRIdata dataset directory per split. FRIdata numbers
proteins per split (``train/0.cif``, ``valid/0.cif``, ... are different proteins), so ids are
prefixed by split name and the datasets are merged before training sees them.

    python preprocess.py --task my_task \\
        --train-csv train.csv --train-dataset fridata_train_dir \\
        --eval-csv  valid.csv --eval-dataset  fridata_valid_dir \\
        --test-csv  test.csv  --test-dataset  fridata_test_dir
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from .config import CONFIG_DIR, _deep_merge
from .preprocess import Transcript, note, rule
from .utils.dataloader import _load_id_mapping
from .utils.split import fasta_to_dict
from .utils.target_matrix import BaseTargetMatrix

# how a CSV protein id turns into a FRIdata dataset's own id, tried in this order
ID_CONVENTIONS = {
    None: lambda protein_id: protein_id,
    "chain": lambda protein_id: f"{protein_id}_A",
    "AFDB_v4": lambda protein_id: f"AF-{protein_id}-F1-model_v4_A",
}


def detect_task_kind(frame: pd.DataFrame, label_columns: list[str]) -> str:
    """classification if every label column is a 0/1 indicator, regression if any is
    real-valued. A mix raises -- split into two CSVs and train two runs."""
    kinds = set()
    for column in label_columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        is_binary = pd.api.types.is_bool_dtype(values) or (
            pd.api.types.is_numeric_dtype(values) and values.isin([0, 1]).all()
        )
        kinds.add("classification" if is_binary else "regression")
    if len(kinds) > 1:
        raise ValueError(
            f"label columns {label_columns} mix classification and regression values -- "
            "split them into two CSVs and train two runs."
        )
    return kinds.pop() if kinds else "classification"


def build_targets_from_csv(
    csv_path: Path | str, protein_id_col: str = "protein_id", label_columns: list[str] | None = None,
):
    """CSV -> (go_indices, protein_vectors, task_kind).

    Classification labels become sparse multi-hot vectors, the same representation
    ``TargetMatrix`` uses for GO; regression labels are dense per-task values, which
    ``DeepFRIDataset`` reads just as well since ``.to_dense()`` is a no-op on a dense tensor.

    ``label_columns`` defaults to a column named ``label`` -- a CSV can carry other columns
    that aren't labels (PEER's also has a ``sequence`` column), so "every column but the id"
    isn't a safe default.
    """
    frame = pd.read_csv(csv_path)
    frame[protein_id_col] = frame[protein_id_col].astype(str)
    if label_columns:
        label_columns = list(label_columns)
    elif "label" in frame.columns:
        label_columns = ["label"]
    else:
        raise ValueError(
            f"{csv_path} has no 'label' column and no --label-columns was given "
            f"(columns: {list(frame.columns)})"
        )
    missing = set(label_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} has no column(s) {sorted(missing)}")

    task_kind = detect_task_kind(frame, label_columns)
    go_indices = {name: i for i, name in enumerate(label_columns)}

    protein_vectors: dict[str, torch.Tensor] = {}
    for _, row in frame.iterrows():
        protein_id = row[protein_id_col]
        if task_kind == "classification":
            present = sorted(go_indices[name] for name in label_columns if row[name] == 1)
            protein_vectors[protein_id] = BaseTargetMatrix._make_sparse_vector(
                present, len(label_columns))
        else:
            protein_vectors[protein_id] = torch.tensor(
                [float(row[name]) for name in label_columns], dtype=torch.float32)

    return go_indices, protein_vectors, task_kind


def _detect_id_convention(sequence_keys: set[str], protein_ids: list[str]) -> str | None:
    """Which of :data:`ID_CONVENTIONS` a FRIdata dataset uses, by sampling a few CSV ids."""
    sample = protein_ids[:20] or protein_ids
    for convention, transform in ID_CONVENTIONS.items():
        hits = sum(transform(protein_id) in sequence_keys for protein_id in sample)
        if hits >= max(1, len(sample) // 2):
            return convention
    raise RuntimeError(
        "could not match the CSV's protein ids to the FRIdata dataset's; CSV ids look like "
        f"{sorted(protein_ids)[:3]}, dataset ids look like {sorted(sequence_keys)[:3]}."
    )


def read_dataset_sequences(dataset_dir: Path) -> dict[str, str]:
    """Every protein's sequence out of a FRIdata dataset directory.

    A small dataset keeps one flat ``sequences.fasta``; a bigger FRIdata dataset only has
    ``sequences.idx``, mapping each protein to one of several shared FASTA files -- read once
    each.
    """
    flat = dataset_dir / "sequences.fasta"
    if flat.is_file():
        return fasta_to_dict(flat)

    config = json.loads((dataset_dir / "dataset.json").read_text())
    mapping = _load_id_mapping(config, dataset_dir / "sequences.idx")
    sequences: dict[str, str] = {}
    for fasta_path in sorted(set(mapping.values())):
        sequences.update(fasta_to_dict(fasta_path))
    return sequences


# DeepFRIDataset reads an embedding as h5[id][()] and a distogram as h5[id]["distogram"][()]
_H5_IDX_KINDS = {"embeddings.idx": None, "distograms.idx": "distogram"}


def merge_datasets(sources: dict[str, Path], out_dir: Path) -> Path:
    """Merge several FRIdata dataset directories into one.

    Every protein id is prefixed with its source name so the same numbering in different
    splits (``train/0.cif``, ``valid/0.cif``, ...) doesn't collide. This has to be a real copy,
    not just an index rewrite: ``DeepFRIDataset`` looks an embedding up by protein id *inside*
    the HDF5 file too, not just by file path, so each embedding/distogram is copied under its
    new id into one merged HDF5 file per kind.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_size = None
    for idx_name, nested_key in _H5_IDX_KINDS.items():
        merged_idx: dict[str, str] = {}
        h5_name = idx_name.removesuffix(".idx") + ".h5"
        with h5py.File(out_dir / h5_name, "w") as merged_h5:
            for name, source in sources.items():
                idx_path = source / idx_name
                if not idx_path.is_file():
                    continue
                config = json.loads((source / "dataset.json").read_text())
                embedding_size = embedding_size or config.get("embedding_size")

                by_file: dict[str, list[str]] = {}
                for protein_id, path in _load_id_mapping(config, idx_path).items():
                    by_file.setdefault(path, []).append(protein_id)

                for path, protein_ids in by_file.items():
                    with h5py.File(path, "r") as source_h5:
                        for protein_id in protein_ids:
                            new_key = f"{name}_{protein_id}"
                            node = source_h5[protein_id]
                            data = node[nested_key][()] if nested_key else node[()]
                            if nested_key:
                                merged_h5.create_group(new_key).create_dataset(
                                    nested_key, data=data)
                            else:
                                merged_h5.create_dataset(new_key, data=data)
                            merged_idx[new_key] = h5_name
        (out_dir / idx_name).write_text(json.dumps(merged_idx))
    (out_dir / "dataset.json").write_text(
        json.dumps({"embedding_size": embedding_size, "config": {"data_path": str(out_dir)}}))
    return out_dir


@dataclass
class CustomPreprocessConfig:
    """Paths for a custom target matrix, from ``configs/paths.yaml :: custom``."""

    project_location: Path
    out_dir_relative: str
    datasets_dir_relative: str
    log_file_template: str = "{project_location}/data.log"

    @classmethod
    def from_configs(cls, config_dir: Path | str | None = None, **overrides) -> "CustomPreprocessConfig":
        config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
        paths = yaml.safe_load((config_dir / "paths.yaml").read_text()) or {}
        custom = paths.get("custom") or {}

        defaults = {
            "project_location": Path(paths["project_location"]),
            "out_dir_relative": custom.get("out_dir", "custom_target_matrix"),
            "datasets_dir_relative": paths["layout"]["datasets"],
            # kept as an unformatted template (not a Path), so a `--set project_location=...`
            # override is picked up by `log_file` below, read lazily at run start.
            "log_file_template": str(paths["preprocess"]["log_file"]),
        }
        merged = _deep_merge(defaults, overrides)
        if merged["project_location"] is not None:
            merged["project_location"] = Path(merged["project_location"])
        return cls(**merged)

    @property
    def datasets_dir(self) -> Path:
        """Where `RunConfig.datasets_dir` looks for a dataset -- unaffected by `out_dir`."""
        return self.project_location / self.datasets_dir_relative

    @property
    def log_file(self) -> Path:
        return Path(self.log_file_template.format(project_location=self.project_location))

    @property
    def out_dir(self) -> Path:
        return self.project_location / self.out_dir_relative

    def task_dir(self, task: str) -> Path:
        return self.out_dir / task

    def run_overrides(self, task: str, dataset_name: str, task_kind: str) -> dict:
        """The ``--set-file``-able overrides a training run needs to find this target matrix.

        ``trainval_unfix_type``/``testset_unfix_type`` are ``null``: the merge above already
        renamed every protein to the exact key the training/eval/test dataset uses, so no
        further id transform is needed on load.
        """
        root = f"{self.out_dir_relative}/{task}"
        training = {"loss": {"name": "MSE" if task_kind == "regression" else None}}
        if task_kind == "regression":
            # eval_fmax doesn't exist for regression -- select on loss instead.
            training["selection_metric"] = "eval_loss"
        return {
            "data": {
                "dataset_name": dataset_name,
                "trainval_suffix": "_trainval", "testset_suffix": "_test", "cazyset_suffix": "",
                "go_version": "custom", "annotation_threshold": 0,
                "trainval_unfix_type": None, "testset_unfix_type": None,
            },
            "training": training,
            "layout": {"target_matrix": f"{root}/target_matrix", "split": f"{root}/mmseqs_output"},
        }

    def describe(self) -> str:
        return f"out dir : {self.out_dir}\nlog     : {self.log_file}"


def run_with_predefined_splits(
    cfg: CustomPreprocessConfig,
    task: str,
    splits: dict[str, tuple[Path, Path]],
    protein_id_col: str = "protein_id",
    label_columns: list[str] | None = None,
    command: str | None = None,
) -> Path:
    """Build a target matrix from an existing train/eval/test split, instead of computing one
    with MMseqs2 -- for a benchmark (like PEER) that already ships one and already has its
    own precomputed structures.

    ``splits`` maps each of ``"train"``, ``"eval"`` and ``"test"`` (all three required -- a
    benchmark split is not something to guess at or partially honour) to that split's
    ``(labels csv, FRIdata dataset directory)``. Every split's CSV must agree on label columns
    and task kind. Train and eval are merged into one trainval dataset and then split back
    apart by ``train.tsv``/``eval.tsv`` -- exactly the ids each split came in with, never
    reshuffled; test gets its own, separate dataset.
    """
    if set(splits) != {"train", "eval", "test"}:
        raise ValueError("splits needs exactly 'train', 'eval' and 'test'")

    out_dir = cfg.task_dir(task)
    command = command or f"custom_preprocess.run_with_predefined_splits(task={task!r})"

    with Transcript(cfg.log_file, command):
        print(cfg.describe())
        rule(f"target matrix ({task})")

        go_indices = task_kind = None
        per_split: dict[str, dict] = {}
        for name, (csv_path, dataset_dir) in splits.items():
            split_go_indices, vectors, split_kind = build_targets_from_csv(
                csv_path, protein_id_col, label_columns)
            if go_indices is None:
                go_indices, task_kind = split_go_indices, split_kind
            elif split_go_indices != go_indices:
                raise ValueError(
                    f"'{name}' label columns {list(split_go_indices)} differ from "
                    f"'train' {list(go_indices)} -- every split must label the same task"
                )
            elif split_kind != task_kind:
                raise ValueError(f"'{name}' is {split_kind}, 'train' is {task_kind}")

            sequences = read_dataset_sequences(dataset_dir)
            transform = ID_CONVENTIONS[_detect_id_convention(set(sequences), list(vectors))]
            per_split[name] = {
                f"{name}_{transform(pid)}": vector for pid, vector in vectors.items()
                if transform(pid) in sequences
            }
            note(task, f"{name}: {split_kind}, {len(per_split[name])}/{len(vectors)} proteins "
                       f"have a structure in {dataset_dir}")

        rule("merging structures")
        merge_datasets({n: splits[n][1] for n in ("train", "eval")},
                       cfg.datasets_dir / f"{task}_trainval")
        merge_datasets({"test": splits["test"][1]}, cfg.datasets_dir / f"{task}_test")
        protein_vectors = {**per_split["train"], **per_split["eval"]}
        note(task, f"trainval: {len(protein_vectors)} proteins, "
                   f"test: {len(per_split['test'])} proteins")

        rule("split")
        split_dir = out_dir / "mmseqs_output"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "eval"):
            with open(split_dir / f"{split_name}.tsv", "w") as handle:
                for protein_id in per_split[split_name]:
                    handle.write(f"{protein_id}\tdummy\n")
        note(task, f"{len(per_split['train'])} train / {len(per_split['eval'])} eval proteins")

        rule("target matrix pickles")
        target_matrix_dir = out_dir / "target_matrix"
        target_matrix_dir.mkdir(parents=True, exist_ok=True)
        pickles = {
            "go_indices.pkl": go_indices,
            "protein_vectors.pkl": protein_vectors,
            "protein_vectors_test.pkl": per_split["test"],
            "weights.pkl": np.ones(len(go_indices), dtype=np.float32),
            "adjacency.pkl": torch.zeros((len(go_indices), len(go_indices))),
        }
        for name, artefact in pickles.items():
            with open(target_matrix_dir / name, "wb") as handle:
                pickle.dump({task: artefact}, handle)
        (target_matrix_dir / "task.json").write_text(json.dumps({"task_kind": task_kind}))

        overrides_path = out_dir / "overrides.yaml"
        overrides_path.write_text(
            yaml.safe_dump(cfg.run_overrides(task, task, task_kind), sort_keys=False))

        rule("summary")
        note(task, f"wrote target matrix -> {out_dir}")
        note(task, f"train: python train.py --task {task} --set-file {overrides_path}")

    return out_dir
