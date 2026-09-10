"""Target matrix from a plain CSV of labels -- a custom (non-GO) classification or regression
task, trained with the same ``sequence`` / ``structure`` / ``fusion`` architectures as the GO
flow, just given a different label space.

    protein_id,label             <- a 0/1 column: classification
    P12345,1
    P67890,0

    protein_id,label             <- a real-valued column: regression
    P12345,3.14
    P67890,1.02

The label column is assumed to be named ``label``; pass ``--label-columns`` for a different
name, or several columns (multi-task) -- a CSV can carry other columns that are not labels (a
benchmark like PEER also has a ``sequence`` column), so "every column but the id" is not a safe
default.

Structures come from FRIdata: a raw MMCIF directory is handed to its ``generate_data`` command,
which writes the same ``dataset.json`` / ``embeddings.idx`` / ``distograms.idx`` shape the GO
flow already reads through ``get_data_config`` / ``DeepFRIDataset``. An already-built FRIdata
dataset can be pointed at directly instead, skipping that step.

The train/eval split reuses ``utils.split.run_split_pipeline`` (MMseqs2 homology clustering)
exactly as the GO flow does: classification balances the split by how many proteins carry each
label, regression -- where "balance per label" is meaningless -- balances the split as one group
covering every protein.

    python preprocess.py --csv labels.csv --structures /path/to/mmcifs --task my_task
    python preprocess.py --csv labels.csv --dataset /existing/fridata/dataset --task my_task
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from .config import CONFIG_DIR, _deep_merge
from .preprocess import Transcript, note, rule
from .utils.dataloader import _load_id_mapping
from .utils.split import fasta_to_dict, run_split_pipeline, write_fasta
from .utils.target_matrix import BaseTargetMatrix

#: How a raw CSV protein id might appear in FRIdata's output, tried in this order.
ID_CONVENTIONS = {
    None: lambda protein_id: protein_id,
    "chain": lambda protein_id: f"{protein_id}_A",
    "AFDB_v4": lambda protein_id: f"AF-{protein_id}-F1-model_v4_A",
}


def detect_task_kind(frame: pd.DataFrame, label_columns: list[str]) -> str:
    """"classification" if every label column looks like a 0/1 indicator, "regression" if any
    holds a real-valued number. A mix of both raises -- train them as two separate CSVs/runs.
    """
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
            f"label columns {label_columns} mix classification-like (0/1) and regression-like "
            "(real-valued) values -- split them into two CSVs and train two runs."
        )
    return kinds.pop() if kinds else "classification"


def build_targets_from_csv(
    csv_path: Path | str, protein_id_col: str = "protein_id", label_columns: list[str] | None = None,
):
    """CSV -> ``(go_indices, protein_vectors, proteins_by_label, weights, adjacency, task_kind)``.

    Named after the GO target matrix's own fields, so the rest of the trainer (dataloaders,
    model, checkpoint labels) does not need to know the difference. Classification labels are
    sparse multi-hot vectors, the same representation ``TargetMatrix`` uses; regression labels
    are dense per-task values, which ``DeepFRIDataset`` reads just as well -- ``.to_dense()`` is
    a no-op on an already-dense tensor.

    ``label_columns`` defaults to a single column named ``label`` -- a CSV can carry other
    columns that are not labels (a benchmark like PEER also has a ``sequence`` column), so
    "every column but the id" is not a safe default. Pass ``label_columns`` explicitly for a
    multi-task CSV, or when the label column is not called ``label``.
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
    proteins_by_label: dict[str, list[str]] = {name: [] for name in label_columns}
    for _, row in frame.iterrows():
        protein_id = row[protein_id_col]
        if task_kind == "classification":
            present = [name for name in label_columns if row[name] == 1]
            protein_vectors[protein_id] = BaseTargetMatrix._make_sparse_vector(
                sorted(go_indices[name] for name in present), len(label_columns)
            )
            for name in present:
                proteins_by_label[name].append(protein_id)
        else:
            protein_vectors[protein_id] = torch.tensor(
                [float(row[name]) for name in label_columns], dtype=torch.float32
            )

    if task_kind == "regression":
        proteins_by_label = {"__all__": frame[protein_id_col].tolist()}

    weights = np.ones(len(label_columns), dtype=np.float32)
    adjacency = torch.zeros((len(label_columns), len(label_columns)))
    return go_indices, protein_vectors, proteins_by_label, weights, adjacency, task_kind


def _detect_id_convention(sequence_keys: set[str], protein_ids: list[str]) -> str | None:
    """Which of :data:`ID_CONVENTIONS` FRIdata's output ids follow, by sampling a few CSV ids."""
    sample = protein_ids[:20] or protein_ids
    for convention, transform in ID_CONVENTIONS.items():
        hits = sum(transform(protein_id) in sequence_keys for protein_id in sample)
        if hits >= max(1, len(sample) // 2):
            return convention
    raise RuntimeError(
        "could not match the CSV's protein ids to FRIdata's output; CSV ids look like "
        f"{sorted(protein_ids)[:3]}, FRIdata ids look like {sorted(sequence_keys)[:3]}. Check "
        "--id-column, or that --structures / --dataset covers these proteins."
    )


def read_dataset_sequences(dataset_dir: Path) -> dict[str, str]:
    """Every protein's sequence out of a FRIdata dataset directory.

    A small/simple dataset keeps one flat ``sequences.fasta``; FRIdata's own multi-batch
    datasets only have ``sequences.idx``, mapping each protein to one of several shared FASTA
    files -- read once each, the same way ``preprocess.py::build_trainval_fasta`` does for the
    GO flow.
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


#: (idx file, nested key) -- `DeepFRIDataset` reads an embedding as `h5[protein_id][()]` and a
#: distogram as `h5[protein_id]["distogram"][()]`; the merge below has to reproduce that shape.
_H5_IDX_KINDS = {"embeddings.idx": None, "distograms.idx": "distogram"}


def merge_datasets(sources: dict[str, Path], out_dir: Path) -> Path:
    """Merge several already-built FRIdata dataset directories into one.

    For a predefined split (PEER's separate ``train``/``valid``/``test`` directories, each with
    its own embeddings/distograms), every protein id is prefixed with its source name so the
    same numbering in different sources (``train/0.cif``, ``valid/0.cif``, ...) does not
    collide. A plain index rewrite is not enough here: ``DeepFRIDataset`` looks an embedding up
    by protein id *inside* its HDF5 file too, not just by file path, so each embedding/distogram
    is copied -- under its prefixed id -- into one merged HDF5 file per kind. Only what training
    actually reads (embeddings, distograms) is merged; ``sequences.idx`` is not needed after the
    id-convention check in ``run_with_predefined_splits``.
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

                # group ids by their source HDF5 file, so each file is opened once
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


def run_fridata(
    mmcif_dir: Path, ids: list[str], task: str, fridata_src: Path, fridata_config: Path,
    fridata_python: str, embedder: str, out_dir: Path,
) -> Path:
    """Shell out to FRIdata's ``generate_data``: sequences + distograms + embeddings for ``ids``
    out of the MMCIF files in ``mmcif_dir``. Returns the produced dataset directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_file = out_dir / f"{task}_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")

    command = [
        fridata_python, str(fridata_src / "fridata.py"), "generate_data",
        "-t", "sequences,coordinates,distograms,embeddings",
        "-d", "other", "-c", "subset",
        "--version", task,
        "-i", str(ids_file),
        "--input-path", str(mmcif_dir),
        "-e", embedder,
        "--config", str(fridata_config),
    ]
    note(task, f"running FRIdata: {' '.join(command)}")
    result = subprocess.run(
        command, cwd=fridata_src, env={**os.environ, "PYTHONPATH": str(fridata_src)},
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"FRIdata failed (exit {result.returncode}):\n{result.stderr}")

    config = json.loads(fridata_config.read_text())
    matches = list((Path(config["data_path"]) / "datasets").glob(f"*--{task}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one dataset directory named '*--{task}' under "
            f"{Path(config['data_path']) / 'datasets'}, found {matches}. FRIdata's naming may "
            "have changed -- point --dataset at the right directory directly."
        )
    return matches[0]


@dataclass
class CustomPreprocessConfig:
    """Paths for the CSV/MMCIF preprocessing path, from ``configs/paths.yaml :: custom``."""

    project_location: Path
    out_dir_relative: str
    datasets_dir_relative: str
    mmseqs_bin: Path
    fridata_src: Path | None
    fridata_config: Path | None
    fridata_python: str = "python3"
    embedder: str = "esm2_t33_650M_UR50D"
    split: dict = field(default_factory=dict)
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
            "mmseqs_bin": Path(paths["preprocess"]["mmseqs_bin"]),
            "fridata_src": Path(custom["fridata_src"]) if custom.get("fridata_src") else None,
            "fridata_config": Path(custom["fridata_config"]) if custom.get("fridata_config") else None,
            "fridata_python": custom.get("fridata_python", "python3"),
            "embedder": custom.get("embedder", "esm2_t33_650M_UR50D"),
            "split": custom.get("split", {}),
            # kept as an unformatted template (not a Path) so a `--set project_location=...`
            # override still lands in `log_file` below -- it is read once, at run start.
            "log_file_template": str(paths["preprocess"]["log_file"]),
        }
        merged = _deep_merge(defaults, overrides)
        # `--set project_location=...` (etc.) hands back a plain string; the path-typed fields
        # must stay actual Paths for `/` to work in `out_dir` / `datasets_dir` below.
        for key in ("project_location", "mmseqs_bin", "fridata_src", "fridata_config"):
            if merged[key] is not None:
                merged[key] = Path(merged[key])
        return cls(**merged)

    @property
    def datasets_dir(self) -> Path:
        """Where `RunConfig.datasets_dir` looks for a dataset -- unaffected by `out_dir`, since
        that layout entry is never overridden."""
        return self.project_location / self.datasets_dir_relative

    @property
    def log_file(self) -> Path:
        return Path(self.log_file_template.format(project_location=self.project_location))

    @property
    def out_dir(self) -> Path:
        return self.project_location / self.out_dir_relative

    def task_dir(self, task: str) -> Path:
        return self.out_dir / task

    def run_overrides(self, task: str, dataset_name: str, task_kind: str,
                      id_convention: str | None, trainval_suffix: str = "",
                      testset_suffix: str = "") -> dict:
        """The ``--set-file``-able overrides a training run needs to find this target matrix.

        ``id_convention`` (one of :data:`ID_CONVENTIONS`, including ``None`` for "no suffix, ids
        match directly") is always written explicitly -- ``None`` is itself a valid detected
        convention, not "unset", and ``data.trainval_unfix_type`` must say so or a training run
        would fall back to the wrong default (``AFDB_v4``). A test split (``testset_suffix``
        given) gets the same treatment: a predefined-split merge bakes each source's id
        convention into the merged protein-vector keys itself (see
        ``run_with_predefined_splits``), so both the train/eval and the test dataset need to be
        read back with no further transform -- ``testset_unfix_type: null`` -- rather than the
        GO flow's ``chain`` default.
        """
        root = f"{self.out_dir_relative}/{task}"
        data = {
            "dataset_name": dataset_name,
            "trainval_suffix": trainval_suffix, "testset_suffix": testset_suffix,
            "cazyset_suffix": "", "go_version": "custom", "annotation_threshold": 0,
            "trainval_unfix_type": id_convention,
        }
        if testset_suffix:
            data["testset_unfix_type"] = None
        training = {"loss": {"name": "MSE" if task_kind == "regression" else None}}
        if task_kind == "regression":
            # eval_fmax does not exist for a regression target matrix -- select on loss instead.
            training["selection_metric"] = "eval_loss"
        return {
            "data": data,
            "training": training,
            "layout": {"target_matrix": f"{root}/target_matrix", "split": f"{root}/mmseqs_output"},
        }

    def describe(self) -> str:
        return "\n".join([
            f"out dir            : {self.out_dir}",
            f"mmseqs             : {self.mmseqs_bin}",
            f"FRIdata checkout   : {self.fridata_src or '(none -- use --dataset)'}",
            f"FRIdata config     : {self.fridata_config}",
            f"FRIdata embedder   : {self.embedder}",
            f"log                : {self.log_file}",
        ])


def _write_target_matrix(
    target_matrix_dir: Path, task: str, go_indices: dict, protein_vectors: dict,
    weights: np.ndarray, adjacency: torch.Tensor, task_kind: str,
    protein_vectors_test: dict | None = None,
) -> None:
    target_matrix_dir.mkdir(parents=True, exist_ok=True)
    pickles = {
        "go_indices.pkl": go_indices, "protein_vectors.pkl": protein_vectors,
        "weights.pkl": weights, "adjacency.pkl": adjacency,
    }
    if protein_vectors_test is not None:
        pickles["protein_vectors_test.pkl"] = protein_vectors_test
    for name, artefact in pickles.items():
        with open(target_matrix_dir / name, "wb") as handle:
            pickle.dump({task: artefact}, handle)
    (target_matrix_dir / "task.json").write_text(json.dumps({"task_kind": task_kind}))


def run(
    cfg: CustomPreprocessConfig,
    csv_path: Path,
    task: str,
    protein_id_col: str = "protein_id",
    label_columns: list[str] | None = None,
    mmcif_dir: Path | None = None,
    dataset_dir: Path | None = None,
    embedder: str | None = None,
    command: str | None = None,
) -> Path:
    """Build a target matrix and train/eval split for ``csv_path``, writing into
    ``cfg.out_dir / task``. Returns that directory.

    Exactly one of ``mmcif_dir`` (invoke FRIdata) or ``dataset_dir`` (an already-built FRIdata
    dataset) must be given.
    """
    if (mmcif_dir is None) == (dataset_dir is None):
        raise ValueError("give exactly one of --structures or --dataset")

    out_dir = cfg.task_dir(task)
    command = command or f"custom_preprocess.run(task={task!r})"

    with Transcript(cfg.log_file, command):
        print(cfg.describe())
        rule(f"target matrix ({task})")
        go_indices, protein_vectors, proteins_by_label, weights, adjacency, task_kind = (
            build_targets_from_csv(csv_path, protein_id_col, label_columns)
        )
        note(task, f"{task_kind}: {len(go_indices)} label column(s), "
                   f"{len(protein_vectors)} proteins from {csv_path}")

        if dataset_dir is None:
            if cfg.fridata_src is None or cfg.fridata_config is None:
                raise RuntimeError(
                    "configs/paths.yaml has no `custom.fridata_src` / `custom.fridata_config` -- "
                    "set them to a FRIdata checkout and its config.json, or pass --dataset to "
                    "use an already-built FRIdata dataset instead of --structures."
                )
            dataset_dir = run_fridata(
                mmcif_dir=mmcif_dir, ids=list(protein_vectors), task=task,
                fridata_src=cfg.fridata_src, fridata_config=cfg.fridata_config,
                fridata_python=cfg.fridata_python, embedder=embedder or cfg.embedder,
                out_dir=out_dir,
            )
        note(task, f"structures/embeddings: {dataset_dir}")

        sequences = read_dataset_sequences(dataset_dir)
        csv_ids = list(protein_vectors)
        id_convention = _detect_id_convention(set(sequences), csv_ids)
        transform = ID_CONVENTIONS[id_convention]
        present = {pid for pid in csv_ids if transform(pid) in sequences}
        note(task, f"{len(present)}/{len(csv_ids)} CSV proteins have a FRIdata sequence "
                   f"(id convention: {id_convention or 'unchanged'})")
        if not present:
            raise RuntimeError(
                f"none of the CSV's {protein_id_col!r} values matched a protein in "
                f"{dataset_dir}; check FRIdata's id convention against the CSV."
            )

        rule("split")
        fasta_path = write_fasta(out_dir / "sequences_trainval.fasta",
                                 {pid: sequences[transform(pid)] for pid in present})
        split = cfg.split
        run_split_pipeline(
            fasta_file=fasta_path, proteins_by_go=proteins_by_label,
            mmseqs_bin=cfg.mmseqs_bin, output_dir=out_dir / "mmseqs_output",
            tmp_dir=out_dir / "tmp", min_seq_id=split.get("min_seq_id", 0.5),
            eval_fraction=split.get("eval_fraction", 0.1),
            num_trials=split.get("num_trials", 100), seed=split.get("seed"),
        )

        rule("target matrix pickles")
        _write_target_matrix(out_dir / "target_matrix", task, go_indices, protein_vectors,
                            weights, adjacency, task_kind)

        overrides_path = out_dir / "overrides.yaml"
        overrides_path.write_text(yaml.safe_dump(
            cfg.run_overrides(task, dataset_dir.name, task_kind, id_convention),
            sort_keys=False,
        ))

        rule("summary")
        note(task, f"wrote target matrix + split -> {out_dir}")
        note(task, f"train: python train.py --task {task} --set-file {overrides_path}")

    return out_dir


def run_with_predefined_splits(
    cfg: CustomPreprocessConfig,
    task: str,
    splits: dict[str, tuple[Path, Path]],
    protein_id_col: str = "protein_id",
    label_columns: list[str] | None = None,
    command: str | None = None,
) -> Path:
    """Build a target matrix that respects an existing train/eval/test split, instead of
    computing one with MMseqs2 -- for a benchmark (like PEER) that already ships one.

    ``splits`` maps each of ``"train"``, ``"eval"`` and ``"test"`` (all three required -- a
    benchmark split is not something to guess at, or partially honour) to that split's
    ``(labels csv, prebuilt FRIdata dataset directory)``. Every split's CSV must agree on label
    columns and task kind. Ids are prefixed by split name before merging, so the same numbering
    FRIdata's per-split directories reuse (PEER's ``train/0.cif``, ``valid/0.cif``,
    ``test/0.cif``, ...) does not collide once combined. Train and eval are merged into one
    trainval dataset and then split back apart by ``train.tsv``/``eval.tsv`` -- exactly the ids
    each one came in with, never reshuffled -- and test gets its own, separate dataset; nothing
    here ever moves a protein between splits.
    """
    if not {"train", "eval", "test"} <= set(splits):
        raise ValueError("splits needs 'train', 'eval' and 'test'")

    out_dir = cfg.task_dir(task)
    command = command or f"custom_preprocess.run_with_predefined_splits(task={task!r})"

    with Transcript(cfg.log_file, command):
        print(cfg.describe())
        rule(f"target matrix ({task}, predefined split)")

        go_indices = task_kind = None
        per_split: dict[str, dict] = {}
        for name, (csv_path, dataset_dir) in splits.items():
            split_go_indices, vectors, _, _, _, split_kind = build_targets_from_csv(
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
            id_convention = _detect_id_convention(set(sequences), list(vectors))
            transform = ID_CONVENTIONS[id_convention]
            prefixed = {f"{name}_{transform(pid)}": vector for pid, vector in vectors.items()
                       if transform(pid) in sequences}
            note(task, f"{name}: {split_kind}, {len(prefixed)}/{len(vectors)} proteins have a "
                       f"structure in {dataset_dir}")
            per_split[name] = prefixed

        rule("merging structures")
        trainval_dir = merge_datasets(
            {name: splits[name][1] for name in ("train", "eval")},
            cfg.datasets_dir / f"{task}_trainval",
        )
        protein_vectors = {**per_split["train"], **per_split["eval"]}
        note(task, f"trainval: {len(protein_vectors)} proteins -> {trainval_dir}")

        test_dir = merge_datasets({"test": splits["test"][1]}, cfg.datasets_dir / f"{task}_test")
        protein_vectors_test = per_split["test"]
        note(task, f"test: {len(protein_vectors_test)} proteins -> {test_dir}")

        rule("split")
        split_dir = out_dir / "mmseqs_output"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "eval"):
            with open(split_dir / f"{split_name}.tsv", "w") as handle:
                for protein_id in per_split[split_name]:
                    handle.write(f"{protein_id}\tdummy\n")
        note(task, f"{len(per_split['train'])} train / {len(per_split['eval'])} eval proteins "
                   "(predefined split -- no clustering)")

        rule("target matrix pickles")
        _write_target_matrix(out_dir / "target_matrix", task, go_indices, protein_vectors,
                            np.ones(len(go_indices), dtype=np.float32),
                            torch.zeros((len(go_indices), len(go_indices))), task_kind,
                            protein_vectors_test=protein_vectors_test)

        overrides_path = out_dir / "overrides.yaml"
        overrides_path.write_text(yaml.safe_dump(
            cfg.run_overrides(
                task, task, task_kind, id_convention=None, trainval_suffix="_trainval",
                testset_suffix="_test",
            ),
            sort_keys=False,
        ))

        rule("summary")
        note(task, f"wrote target matrix + predefined split -> {out_dir}")
        note(task, f"train: python train.py --task {task} --set-file {overrides_path}")

    return out_dir
