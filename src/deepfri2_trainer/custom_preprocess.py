"""Target matrix for a custom task, read from a CSV instead of the GO annotation tables.

The task type is given, never guessed from the labels -- see ``TASK_TYPES``. Empty cells are
missing labels and are left out of the loss and the metrics.

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

#: --task-type -> how the trainer treats it (`data.task_kind`)
TASK_TYPES = {
    "classification": "multiclass",               # one column of class ids -> softmax
    "multi-task-classification": "multilabel",    # several 0/1 columns -> a sigmoid each
    "regression": "regression",
    "multi-task-regression": "regression",
}

#: split -> (its `data.<key>_*` config keys, its pickle)
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


def read_labels(csv_paths: dict[str, Path], task_type: str, id_column: str = "protein_id",
                label_columns: list[str] | None = None):
    """Read every split's labels and check they fit ``task_type``.

    Returns ``(go_indices, {split: {protein id: target vector}}, weights)``. Vectors are dense
    -- a custom task has a handful of labels, not GO's thousands -- and hold NaN where a label
    is missing; a protein with no labels at all is dropped. For classification they are
    one-hot over the classes of all three splits, so a class seen only in test still gets an
    output.

    ``weights`` follows scikit-learn's ``class_weight="balanced"``, counted on train: class
    weights for classification, BCE's ``pos_weight`` for multi-task classification, and ones
    for regression.

    A CSV often carries columns that aren't labels (PEER's also has ``sequence``), so the
    default is the single column ``label`` rather than "everything but the id".
    """
    task_kind = TASK_TYPES[task_type]
    label_columns = list(label_columns) if label_columns else ["label"]
    multi_task = task_type.startswith("multi-task")
    if multi_task != (len(label_columns) > 1):
        raise ValueError(
            f"{task_type} needs {'two or more label columns' if multi_task else 'one label column'}"
            f", got {label_columns}")

    ids, values = {}, {}
    for split, csv_path in csv_paths.items():
        frame = pd.read_csv(csv_path)
        missing = [c for c in (id_column, *label_columns) if c not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} has no column(s) {missing}; it has {list(frame.columns)}")
        labels = frame[label_columns].to_numpy(dtype=np.float32)
        labelled = ~np.isnan(labels).all(axis=1)
        ids[split] = frame[id_column].astype(str).to_numpy()[labelled]
        values[split] = labels[labelled]

    present = np.concatenate([v[~np.isnan(v)] for v in values.values()])
    if task_kind == "multilabel" and not np.isin(present, (0, 1)).all():
        raise ValueError("multi-task classification labels must be 0, 1 or empty")
    if task_kind == "multiclass" and not (present == np.round(present)).all():
        raise ValueError("classification labels must be integer class ids")

    if task_kind == "multiclass":
        classes = np.unique(present)
        go_indices = {str(int(c)): i for i, c in enumerate(classes)}
        for split, v in values.items():
            values[split] = np.eye(len(classes), dtype=np.float32)[np.searchsorted(classes, v[:, 0])]
    else:
        go_indices = {name: i for i, name in enumerate(label_columns)}

    weights = np.ones(len(go_indices), dtype=np.float32)
    if task_kind == "multiclass":
        # n / (K * n_c) over the K classes train has; one it never sees keeps weight 1.
        counts = values["train"].sum(axis=0)
        seen = counts > 0
        weights[seen] = counts.sum() / (seen.sum() * counts[seen])
    elif task_kind == "multilabel":
        # sklearn's class_weight="balanced" weighs positives against negatives as n_neg / n_pos,
        # which is exactly what pos_weight scales. Counted on train, over labels that are there.
        train = values["train"]
        positives = np.nansum(train, axis=0)
        negatives = (~np.isnan(train)).sum(axis=0) - positives
        weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0).astype(np.float32)

    vectors = {
        split: {pid: torch.from_numpy(row) for pid, row in zip(ids[split], values[split])}
        for split in values
    }
    return go_indices, vectors, weights


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

    training = {
        "loss": {"name": {"multiclass": "CE", "multilabel": "BCE", "regression": "MSE"}[task_kind]},
        "use_class_weights": task_kind != "regression",
        "selection_metric": "eval_loss",
    }
    return {
        "data": data,
        "training": training,
        "layout": {"target_matrix": str(target_matrix_dir)},
    }


def run(
    task: str,
    task_type: str,
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

        task_kind = TASK_TYPES[task_type]
        go_indices, vectors, weights = read_labels(
            {split: csv_path for split, (csv_path, _) in splits.items()}, task_type, id_column,
            label_columns)
        for split, (_, pickle_name) in SPLITS.items():
            _dump(target_matrix_dir / pickle_name, task, vectors[split])
            print(f"{split:<6}: {len(vectors[split])} proteins from {splits[split][0]}, "
                  f"structures {splits[split][1]}")

        _dump(target_matrix_dir / "go_indices.pkl", task, go_indices)
        _dump(target_matrix_dir / "weights.pkl", task, weights)
        # No label hierarchy, but load_targets and DAGPropagator expect the file.
        _dump(target_matrix_dir / "adjacency.pkl", task,
              torch.zeros((len(go_indices), len(go_indices))))

        overrides_path = out_dir / "overrides.yaml"
        overrides_path.write_text(yaml.safe_dump(
            _overrides(task, task_kind, splits, target_matrix_dir), sort_keys=False))

        rule("summary")
        print(f"{task_type}, {len(go_indices)} output(s): {', '.join(go_indices)}")
        if task_kind != "regression":
            print("class weights: " + ", ".join(f"{w:.2f}" for w in weights))
        print(f"wrote {out_dir}")
        print(f"train with: python train.py --task {task}")

    return out_dir
