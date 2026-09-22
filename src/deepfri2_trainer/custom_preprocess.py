"""Target matrix for a custom task -- classification or regression, one label or many, read
from a CSV instead of the GO annotation tables.

Each of train / eval / test brings its own labels CSV and its own FRIdata dataset, and the
split is used exactly as it came: nothing is reclustered, merged or renamed here. What comes
out is a target matrix plus an ``overrides.yaml`` telling ``train.py`` where everything is.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import CONFIG_DIR, _read_yaml
from .preprocess import Transcript, rule

#: split -> (its `data.<key>_*` config keys, its pickle). `train` goes first: it fixes the
#: label space and the task kind that the other two have to match.
SPLITS = {
    "train": ("trainval", "protein_vectors.pkl"),
    "eval": ("evalset", "protein_vectors_eval.pkl"),
    "test": ("testset", "protein_vectors_test.pkl"),
}


def task_dir(task: str, config_dir: Path | str | None = None) -> Path:
    """Where ``preprocess.py --task NAME`` writes and ``train.py --task NAME`` reads back."""
    return _task_dir(_read_yaml(Path(config_dir or CONFIG_DIR) / "paths.yaml"), task)


def _task_dir(paths: dict, task: str) -> Path:
    root = paths["custom_tasks_dir"].format(project_location=paths["project_location"])
    return Path(root) / task


def read_labels(csv_path: Path | str, id_column: str = "protein_id",
                label_columns: list[str] | None = None):
    """``(label columns, {protein id: label vector}, "classification" | "regression")``.

    Vectors are dense, unlike the GO flow's sparse ones -- a custom task has a handful of
    labels, not thousands, and ``.to_dense()`` in the dataset is a no-op on a dense tensor.

    A CSV often carries columns that aren't labels (PEER's also has ``sequence``), so the
    default is the single column ``label`` rather than "everything but the id".
    """
    frame = pd.read_csv(csv_path)
    label_columns = list(label_columns) if label_columns else ["label"]
    missing = [c for c in (id_column, *label_columns) if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} has no column(s) {missing}; it has {list(frame.columns)}")

    values = frame[label_columns].to_numpy(dtype=np.float32)
    vectors = {str(pid): torch.from_numpy(row) for pid, row in zip(frame[id_column], values)}
    kind = "classification" if np.isin(values, (0.0, 1.0)).all() else "regression"
    return label_columns, vectors, kind


def _dump(path: Path, task: str, payload) -> None:
    """Target-matrix pickles are keyed by ontology; a custom task uses its own name."""
    with open(path, "wb") as handle:
        pickle.dump({task: payload}, handle)


def _overrides(task: str, task_kind: str, splits: dict, target_matrix_dir: Path) -> dict:
    """The config `train.py --task NAME` merges over configs/ to train this task.

    Each split points straight at its own dataset directory, in place of the GO flow's
    `dataset_name` + suffix. `unfix_type: null` means the CSV's ids are the dataset's ids; a
    dataset spelling them `<id>_A` needs `--set data.trainval_unfix_type=chain`.
    """
    data: dict = {"dataset_name": task, "task_kind": task_kind}
    for split, (key, _) in SPLITS.items():
        data[f"{key}_dataset"] = str(Path(splits[split][1]).resolve())
        data[f"{key}_unfix_type"] = None

    training: dict = {"loss": {"name": "MSE" if task_kind == "regression" else "BCE"}}
    if task_kind == "regression":
        training["selection_metric"] = "eval_loss"  # Fmax only means something for classification
    return {
        "data": data,
        "training": training,
        "layout": {"target_matrix": str(target_matrix_dir)},
    }


def run(
    task: str,
    splits: dict[str, tuple[Path, Path]],
    id_column: str = "protein_id",
    label_columns: list[str] | None = None,
    config_dir: Path | str | None = None,
    command: str | None = None,
) -> Path:
    """Build a custom task's target matrix and the config that trains it.

    ``splits`` maps ``"train"``, ``"eval"`` and ``"test"`` to that split's ``(labels CSV,
    FRIdata dataset directory)``. All three are required -- the split is the dataset's, and
    this does not try to reinvent it.
    """
    if set(splits) != set(SPLITS):
        raise ValueError(f"splits needs exactly {sorted(SPLITS)}, got {sorted(splits)}")

    paths = _read_yaml(Path(config_dir or CONFIG_DIR) / "paths.yaml")
    out_dir = _task_dir(paths, task)
    target_matrix_dir = out_dir / "target_matrix"
    log_file = paths["preprocess"]["log_file"].format(project_location=paths["project_location"])

    with Transcript(Path(log_file), command or f"preprocess.py --task {task}"):
        rule(f"custom task ({task})")
        target_matrix_dir.mkdir(parents=True, exist_ok=True)

        columns = task_kind = None
        for split, (key, pickle_name) in SPLITS.items():
            csv_path, dataset_dir = splits[split]
            split_columns, vectors, split_kind = read_labels(csv_path, id_column, label_columns)
            if columns is None:
                columns, task_kind = split_columns, split_kind
            elif (split_columns, split_kind) != (columns, task_kind):
                raise ValueError(
                    f"'{split}' labels {split_columns} ({split_kind}) differ from 'train' "
                    f"{columns} ({task_kind}) -- every split must label the same task"
                )
            _dump(target_matrix_dir / pickle_name, task, vectors)
            print(f"{split:<6}: {len(vectors)} proteins from {csv_path}, structures {dataset_dir}")

        go_indices = {name: index for index, name in enumerate(columns)}
        _dump(target_matrix_dir / "go_indices.pkl", task, go_indices)
        # Neither is used by MSE or BCE, but load_targets and DAGPropagator expect the files.
        _dump(target_matrix_dir / "weights.pkl", task, np.ones(len(go_indices), dtype=np.float32))
        _dump(target_matrix_dir / "adjacency.pkl", task,
              torch.zeros((len(go_indices), len(go_indices))))

        overrides_path = out_dir / "overrides.yaml"
        overrides_path.write_text(yaml.safe_dump(
            _overrides(task, task_kind, splits, target_matrix_dir), sort_keys=False))

        rule("summary")
        print(f"{task_kind}, {len(go_indices)} label(s): {', '.join(columns)}")
        print(f"wrote {out_dir}")
        print(f"train with: python train.py --task {task}")

    return out_dir
