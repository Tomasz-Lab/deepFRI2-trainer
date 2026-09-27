"""Per-GO-term calibration of a trained sequence / structure / fusion triple.

The CAFA score a run reports is one number for a whole ontology: it says how good the model is
on average, not how much a particular score on a particular GO term is worth. A user reading an
interpretability report needs the second thing -- "this protein scores 0.42 for GO:0004252, is
that a lot?" -- and the answer differs wildly between terms: a term carried by 40% of the
training proteins and a term carried by 0.5% of them reach their best F1 at completely
different thresholds.

So this module sweeps the decision threshold **per GO term, per sub-model and per split** and
writes the resulting precision / recall curves to one JSON file per ontology, named after the
fusion run the way ``labels_<run>.json`` is::

    <runs_dir>/<ontology>__fusion__<run>/calibration_<run>.json

That file is copied by hand into ``deepFRI2/params/<ontology>/`` next to the checkpoints (the
same manual step the weights already need), and the interpretability report reads it to draw the
precision / recall / F1 curves of the term it is explaining, with a vertical line at the score
this protein actually got.

What is calibrated
------------------
**Raw scores, not propagated ones.** ``interpret.py`` shows the sigmoid of the term's own logit;
propagating the maximum over descendants up the GO DAG (as the CAFA evaluation does) would put
the report's vertical line on a different scale than the curve behind it. Ground truth is the
usual one and *is* propagated -- it comes from the ``grand_truth`` tables, which span the full GO
graph -- so a protein annotated only to a child of the term still counts as a positive.

**Splits are calibrated separately and never pooled.** ``eval`` is the homology-separated
held-out split (thousands of proteins, every model term represented); ``test`` is the
experimental-structure set (far smaller, so many terms have few or no positives there). Pooling
them would hide exactly the disagreement that tells a reader how much to trust a term. CAZy is
skipped: it is a CAZy-specific protein set and covers a handful of MF terms.

Curve resolution
----------------
The stored grid is coarse on purpose -- ``0.00, 0.05, ..., 1.00`` -- because BP alone has ~4000
terms and a 0.01 grid would quintuple a file that has to be copied around by hand. The reader
interpolates between grid points. F1 is **not** stored: it is derived as ``2PR/(P+R)`` from the
interpolated precision and recall, which is both smaller on disk and more correct than
interpolating F1 itself.

Precision is ``null`` above the highest score the model ever produced for that term -- see
:func:`grid_curves` for why that is not 0. ``n_pred`` records how many proteins clear each
threshold, which is what lets the reader tell a measurement from an anecdote and fade the curve
where it thins out.

``fmax`` and ``fmax_threshold``, on the other hand, are **exact**: they come from a full sweep
over every distinct score in the split (a sort-and-cumsum pass, ties kept together), not from
the 21-point grid. So the F-max marker a report draws is a real optimum, not a grid artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ONTOLOGIES, MODEL_TYPES
from .utils.evaluator import EvalPaths, load_ground_truth

#: Splits to calibrate. CAZy is deliberately absent -- see the module docstring.
SPLITS = ("eval", "test")

#: Stored threshold grid. 21 points; the reader interpolates.
THRESHOLD_STEP = 0.05

#: GO terms per sort-and-cumsum chunk of the exact F-max sweep. 256 x ~8000 proteins of float32
#: intermediates is a few hundred MB, which keeps BP (4000 terms) comfortable.
CHUNK_TERMS = 256

#: Decimals kept in the JSON. Curves are drawn, not integrated: more digits only inflate the file.
DECIMALS = 4


def threshold_grid(step: float = THRESHOLD_STEP) -> np.ndarray:
    """``[0.0, step, ..., 1.0]``, rounded so the JSON holds ``0.35`` and not ``0.35000000000000003``."""
    count = int(round(1.0 / float(step)))
    if not np.isclose(count * float(step), 1.0):
        raise ValueError(f"threshold step must divide 1.0, got {step!r}")
    return np.round(np.linspace(0.0, 1.0, count + 1), 6)


# --------------------------------------------------------------------------- inputs


def load_label_space(paths: EvalPaths, runs: dict[str, str], ontology: str) -> list[str]:
    """The GO terms of the run triple, in the model's own output order.

    Read from ``labels_<run>.json`` in each run directory. All three sub-models of an ontology
    must share one label space -- the fusion model is built on the other two -- so a disagreement
    is a configuration error (a run from a different annotation threshold or GO version), not
    something to paper over by intersecting.
    """
    spaces: dict[str, list[str]] = {}
    for model_type, run_name in runs.items():
        matches = sorted((paths.runs_dir or Path(".")).glob(f"*__{run_name}/labels_{run_name}.json"))
        if not matches:
            raise FileNotFoundError(
                f"no labels_{run_name}.json under {paths.runs_dir} for the {model_type} run "
                f"{run_name!r} of {ontology}"
            )
        with open(matches[0]) as handle:
            mapping = json.load(handle)
        # {"0": "GO:0003674", ...} -- keys are the label indices, as strings after a JSON round trip
        spaces[model_type] = [mapping[key] for key in sorted(mapping, key=int)]

    reference_type, reference = next(iter(spaces.items()))
    for model_type, space in spaces.items():
        if space != reference:
            raise ValueError(
                f"{ontology}: the {model_type} run {runs[model_type]!r} and the {reference_type} run "
                f"{runs[reference_type]!r} have different label spaces "
                f"({len(space)} vs {len(reference)} terms). They are not a model triple; check the "
                "annotation threshold and GO version the runs were trained with."
            )
    return reference


def load_prediction_matrix(
    path: Path,
    proteins: Sequence[str] | None,
    terms: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Read a prediction TSV into a dense ``(protein, term)`` matrix of raw scores.

    ``proteins`` fixes the row order; pass ``None`` to take (sorted) whatever the file holds.
    Pairs outside the requested rows/columns are dropped, and any pair the file does not mention
    stays 0.0 -- a term the model never scored is a term it did not predict.

    The file is one line per ``(protein, term)`` pair, so it runs to tens of millions of lines for
    BP. Reading both id columns as categoricals keeps that in integer codes instead of tens of
    millions of Python strings, and the codes are then remapped straight into row/column indices.
    """
    frame = pd.read_csv(
        path, sep="\t", header=None, names=["protein", "term", "score"],
        dtype={"protein": "category", "term": "category", "score": "float32"},
    )

    file_proteins = list(frame["protein"].cat.categories)
    if proteins is None:
        proteins = sorted(file_proteins)
    row_of = {protein: index for index, protein in enumerate(proteins)}
    column_of = {term: index for index, term in enumerate(terms)}

    # category code -> destination index, or -1 for "not wanted"
    protein_rows = np.array([row_of.get(p, -1) for p in file_proteins], dtype=np.int64)
    term_columns = np.array([column_of.get(t, -1) for t in frame["term"].cat.categories], dtype=np.int64)

    rows = protein_rows[frame["protein"].cat.codes.to_numpy()]
    columns = term_columns[frame["term"].cat.codes.to_numpy()]
    keep = (rows >= 0) & (columns >= 0)

    matrix = np.zeros((len(proteins), len(terms)), dtype=np.float32)
    matrix[rows[keep], columns[keep]] = frame["score"].to_numpy()[keep]
    return list(proteins), matrix


def build_ground_truth_matrix(
    truth: pd.DataFrame,
    proteins: Sequence[str],
    terms: Sequence[str],
) -> np.ndarray:
    """``(protein, term)`` boolean matrix from CAFA-evaluator-format ``(protein, GO term)`` pairs."""
    row_of = {protein: index for index, protein in enumerate(proteins)}
    column_of = {term: index for index, term in enumerate(terms)}

    rows = truth[0].map(row_of).to_numpy()
    columns = truth[1].map(column_of).to_numpy()
    keep = pd.notna(rows) & pd.notna(columns)

    labels = np.zeros((len(proteins), len(terms)), dtype=bool)
    labels[rows[keep].astype(np.int64), columns[keep].astype(np.int64)] = True
    return labels


# --------------------------------------------------------------------------- the sweep


def grid_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and prediction count per term and threshold, as ``(term, threshold)``.

    One vectorised pass per threshold over the whole matrix -- 21 passes, which is cheaper than
    the sort the exact F-max needs and keeps the grid honest about the ``score >= threshold`` rule
    the inference CLI actually applies.

    **Precision is NaN, not 0, where the model predicted nothing at all.** The CAFA convention of
    scoring that as 0 is right for a leaderboard -- a method that predicts nothing earns nothing --
    but wrong for what this file is read for. Most terms have a ceiling: for BP GO:0006139 only 6
    of 8383 evaluation proteins reach 0.90 and none reach 0.95, all 6 correct. Reporting precision
    0 above 0.95 would tell a user whose protein scores 0.97 that the prediction is worthless,
    when the truth is that it is off the top of the calibrated range and the last measurement below
    it was perfect. NaN says "no evidence here", the curve simply ends, and the report says so.

    The **prediction count** is returned alongside, and it is what makes any single point on the
    curve trustworthy or not: precision measured over 300 predictions is a measurement, precision
    measured over 2 is an anecdote, and nothing in the precision value itself tells them apart. The
    count falls as the threshold rises, and on the small test split it is in single digits well
    before the curve ends -- so the report fades the curve where it gets thin rather than letting a
    reader take "precision 0.00", computed from two proteins, at face value. Where exactly to draw
    that line is the reader's decision, not this file's, so the raw counts are stored.
    """
    positives = labels.sum(axis=0).astype(np.float64)
    precision = np.full((scores.shape[1], thresholds.size), np.nan, dtype=np.float64)
    recall = np.zeros_like(precision)
    n_predicted = np.zeros(precision.shape, dtype=np.int64)

    for index, threshold in enumerate(thresholds):
        predicted = scores >= np.float32(threshold)
        predicted_positive = predicted.sum(axis=0).astype(np.float64)
        true_positive = (predicted & labels).sum(axis=0).astype(np.float64)
        n_predicted[:, index] = predicted_positive.astype(np.int64)
        precision[:, index] = np.divide(
            true_positive, predicted_positive,
            out=np.full_like(true_positive, np.nan), where=predicted_positive > 0)
        recall[:, index] = np.divide(
            true_positive, positives,
            out=np.zeros_like(true_positive), where=positives > 0)
    return precision, recall, n_predicted


def exact_fmax(
    scores: np.ndarray,
    labels: np.ndarray,
    chunk_terms: int = CHUNK_TERMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-term F-max and the threshold reaching it, over every distinct score.

    Sorting a term's scores descending and taking the cumulative sum of the labels gives the
    confusion counts at every possible cut in one pass. Cuts that would split a run of equal
    scores are masked out: ``score >= threshold`` cannot separate two proteins with the same
    score, so such a cut is not a threshold anyone could apply.

    Terms with no positive in this split get ``(0.0, nan)`` -- there is no optimum to report.
    """
    n_proteins, n_terms = scores.shape
    positives = labels.sum(axis=0).astype(np.float64)
    fmax = np.zeros(n_terms, dtype=np.float64)
    fmax_threshold = np.full(n_terms, np.nan, dtype=np.float64)
    ranks = np.arange(1, n_proteins + 1, dtype=np.float64)[:, None]

    for start in range(0, n_terms, chunk_terms):
        stop = min(start + chunk_terms, n_terms)
        chunk_positives = positives[start:stop]
        if not np.any(chunk_positives > 0):
            continue

        order = np.argsort(-scores[:, start:stop], axis=0, kind="stable")
        sorted_scores = np.take_along_axis(scores[:, start:stop], order, axis=0)
        sorted_labels = np.take_along_axis(labels[:, start:stop], order, axis=0)

        true_positive = np.cumsum(sorted_labels, axis=0, dtype=np.float64)
        precision = true_positive / ranks
        recall = np.divide(
            true_positive, chunk_positives[None, :],
            out=np.zeros_like(true_positive), where=chunk_positives[None, :] > 0)
        total = precision + recall
        f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)

        # Keep only the last row of each run of equal scores: the others are cuts no threshold reaches.
        applicable = np.empty(sorted_scores.shape, dtype=bool)
        applicable[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        applicable[-1] = True
        f1 = np.where(applicable, f1, -1.0)

        best = f1.argmax(axis=0)
        columns = np.arange(stop - start)
        has_positives = chunk_positives > 0
        fmax[start:stop] = np.where(has_positives, f1[best, columns], 0.0)
        fmax_threshold[start:stop] = np.where(has_positives, sorted_scores[best, columns], np.nan)

    return fmax, fmax_threshold


# --------------------------------------------------------------------------- driver


def calibrate_ontology(
    paths: EvalPaths,
    ontology: str,
    runs: dict[str, str],
    splits: Sequence[str] = SPLITS,
    step: float = THRESHOLD_STEP,
    verbose: bool = True,
) -> dict:
    """Calibrate one ontology's model triple. Returns the payload written to the JSON file."""
    def log(message: str) -> None:
        if verbose:
            print(message)

    terms = load_label_space(paths, runs, ontology)
    thresholds = threshold_grid(step)
    log(f"\n{ontology}: {len(terms)} GO terms, runs {runs}")

    # {go term: {split: {"n_gt": int, model_type: {...}}}}, filled split by split
    per_term: dict[str, dict] = {term: {} for term in terms}
    split_info: dict[str, dict] = {}

    for split in splits:
        truth = load_ground_truth(paths, ontology, split)
        truth_proteins = set(truth[0])

        # The scored universe is the proteins that have both a prediction and a ground-truth
        # entry. A protein with no ground truth is not a negative, it is an unknown, and counting
        # it as one would depress precision for every term.
        proteins: list[str] | None = None
        matrices: dict[str, np.ndarray] = {}
        for model_type, run_name in runs.items():
            path = paths.prediction_file(run_name, split)
            if proteins is None:
                predicted_proteins = sorted(
                    set(pd.read_csv(path, sep="\t", header=None, usecols=[0],
                                    names=["protein"], dtype={"protein": "category"})["protein"]
                        .cat.categories))
                proteins = sorted(truth_proteins.intersection(predicted_proteins))
                if not proteins:
                    raise ValueError(
                        f"{ontology}/{split}: no protein is both predicted by {run_name!r} and in "
                        "the ground truth -- the prediction file and the split do not match.")
            _, matrices[model_type] = load_prediction_matrix(path, proteins, terms)

        labels = build_ground_truth_matrix(truth, proteins, terms)
        counts = labels.sum(axis=0)
        split_info[split] = {
            "n_proteins": len(proteins),
            "n_terms_with_ground_truth": int((counts > 0).sum()),
        }
        log(f"  {split}: {len(proteins)} proteins scored, "
            f"{int((counts > 0).sum())}/{len(terms)} terms have at least one positive")

        for model_type, scores in matrices.items():
            precision, recall, n_predicted = grid_curves(scores, labels, thresholds)
            fmax, fmax_threshold = exact_fmax(scores, labels)
            log(f"    {model_type:<9} mean F-max over terms with ground truth: "
                f"{float(fmax[counts > 0].mean()):.3f}")

            for index, term in enumerate(terms):
                entry = per_term[term].setdefault(split, {"n_gt": int(counts[index])})
                if counts[index] == 0:
                    # No positive here: precision is 0 at every threshold and recall is undefined.
                    # Storing that for ~half of BP's terms would be pure noise on disk, so the
                    # split entry keeps `n_gt: 0` and the reader draws "no ground truth".
                    continue
                entry[model_type] = {
                    # `null`, not NaN: NaN is not valid JSON, and every reader turns null into a
                    # missing value on its own.
                    "precision": [None if np.isnan(v) else round(float(v), DECIMALS)
                                  for v in precision[index]],
                    "recall": [round(float(v), DECIMALS) for v in recall[index]],
                    # Proteins scoring >= each threshold: how much weight any point of the curve
                    # can carry. See grid_curves.
                    "n_pred": [int(v) for v in n_predicted[index]],
                    "fmax": round(float(fmax[index]), DECIMALS),
                    "fmax_threshold": round(float(fmax_threshold[index]), DECIMALS),
                }

        del matrices, labels

    return {
        "ontology": ontology,
        "runs": dict(runs),
        "propagated": False,
        "splits": split_info,
        "thresholds": [round(float(t), 6) for t in thresholds],
        "n_terms": len(terms),
        "provenance": {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_name": paths.dataset_name,
            "data_version": paths.data_version,
            "go_version": paths.go_version,
            "annotation_threshold": paths.annotation_threshold,
        },
        "terms": per_term,
    }


def calibration_path(paths: EvalPaths, ontology: str, fusion_run: str) -> Path:
    """``<runs_dir>/<ontology>__fusion__<run>/calibration_<run>.json``."""
    if paths.runs_dir is None:
        raise RuntimeError("`runs_dir` is not set in configs/paths.yaml")
    return paths.runs_dir / f"{ontology}__fusion__{fusion_run}" / f"calibration_{fusion_run}.json"


def write_calibration(payload: dict, path: Path) -> Path:
    """Write the payload compactly. It is machine input; the report is the human-readable view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def run(
    paths: EvalPaths,
    runs_by_ontology: dict[str, dict[str, str]],
    ontologies: Sequence[str] = ONTOLOGIES,
    splits: Sequence[str] = SPLITS,
    step: float = THRESHOLD_STEP,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Calibrate every requested ontology and write one file each. Returns ``{ontology: path}``."""
    written: dict[str, Path] = {}
    for ontology in ontologies:
        runs = runs_by_ontology.get(ontology)
        if not runs:
            print(f"{ontology}: no runs given - skipping")
            continue
        missing = set(MODEL_TYPES) - runs.keys()
        if missing:
            raise KeyError(f"{ontology}: no run given for {sorted(missing)}")

        payload = calibrate_ontology(paths, ontology, runs, splits=splits, step=step, verbose=verbose)
        path = calibration_path(paths, ontology, runs["fusion"])
        if output_dir is not None:
            path = Path(output_dir) / path.name
        written[ontology] = write_calibration(payload, path)
        if verbose:
            size_mb = written[ontology].stat().st_size / 1e6
            print(f"  wrote {written[ontology]}  ({size_mb:.1f} MB)")
    return written
