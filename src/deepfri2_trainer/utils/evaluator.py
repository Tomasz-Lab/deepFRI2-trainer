"""CAFA evaluation of trained runs against deepFRI v1 and the published competitors.

The scores come from `CAFA-evaluator <https://github.com/BioComputingUP/CAFA-evaluator>`_
(``pip install cafaeval``); this module feeds it the right ground truth and predictions, caches
its output next to the predictions, and returns everything as one table --
``method, ontology, split, tau, <metric columns>`` -- from which every figure and every paper
table is a groupby away. See ``utils/figures.py`` for the plotting side.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

ONTOLOGIES = ("MF", "CC", "BP")
SPLITS = ("eval", "test", "cazy")

#: GO namespace of each ontology, as CAFA-evaluator names them.
NAMESPACES = {"MF": "molecular_function", "CC": "cellular_component", "BP": "biological_process"}
ONTOLOGY_OF = {namespace: ontology for ontology, namespace in NAMESPACES.items()}


def save_pickle(path: Path | str, obj) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path | str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------- paths


@dataclass
class EvalPaths:
    """Where the ground truth, the predictions and the GO / IA files live.

    ``layout`` maps names to templates relative to ``project_location``; ``{data_version}``,
    ``{go_version}``, ``{dataset_name}``, ``{params}``, ``{ontology}`` and ``{threshold}`` are
    substituted. The keys used below are ``obo``, ``ia``, ``target_matrix``, ``split``,
    ``cazy_target_matrix``, ``struct_test_ids``, ``predictions`` and ``competitors``.
    """

    layout: dict[str, str]
    project_location: Path
    data_version: str
    go_version: str
    annotation_threshold: int
    dataset_name: str
    testset_suffix: str = "_test_AF3"
    cazyset_suffix: str = "_cazy_uniprot"
    runs_dir: Path | None = None

    @classmethod
    def from_configs(cls, layout: dict[str, str], config_dir: Path | str | None = None, **overrides):
        """Dataset versions and roots from ``configs/paths.yaml`` + ``configs/data.yaml``."""
        config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
        paths = yaml.safe_load((config_dir / "paths.yaml").read_text()) or {}
        data = (yaml.safe_load((config_dir / "data.yaml").read_text()) or {})["data"]

        location = os.getenv("PROJECT_LOCATION") or paths["project_location"]
        version = str(data["data_version"])
        runs_dir = paths.get("runs_dir")
        return cls(**{
            "layout": layout,
            "project_location": Path(location),
            "data_version": version,
            "go_version": str(data["go_version"]),
            "annotation_threshold": int(data["annotation_threshold"]),
            "dataset_name": data["dataset_name"].format(data_version=version),
            "testset_suffix": data.get("testset_suffix", "_test_AF3"),
            "cazyset_suffix": data.get("cazyset_suffix", "_cazy_uniprot"),
            "runs_dir": Path(runs_dir.format(project_location=location)) if runs_dir else None,
            **overrides,
        })

    def params(self, ontology: str) -> str:
        """Target-matrix subdirectory, e.g. ``20250722__MF__50``."""
        return f"{self.go_version}__{ontology}__{self.annotation_threshold}"

    def dataset_of(self, split: str) -> str:
        return self.dataset_name + {"eval": "", "test": self.testset_suffix, "cazy": self.cazyset_suffix}[split]

    def path(self, key: str, ontology: str = "MF") -> Path:
        return self.project_location / self.layout[key].format(
            data_version=self.data_version,
            go_version=self.go_version,
            dataset_name=self.dataset_name,
            params=self.params(ontology),
            ontology=ontology,
            threshold=self.annotation_threshold,
        )

    @property
    def obo_file(self) -> Path:
        return self.path("obo")

    @property
    def ia_file(self) -> Path | None:
        """Information-accretion table; ``None`` when absent -- weighted metrics need it."""
        path = self.path("ia")
        return path if path.is_file() else None

    def competitor_dir(self, split: str, ontology: str) -> Path:
        """Root of the third-party predictions for one split.

        The evaluation set is split per ontology (each has its own protein subset), the test and
        CAZy sets are not.
        """
        root = self.path("competitors") / self.dataset_of(split)
        return root / self.params(ontology) if split == "eval" else root

    def prediction_file(self, run_name: str, split: str) -> Path:
        """``predictions[_test|_cazy]_<run>.tsv``, in the run directory or the shared results dir."""
        prefix = {"eval": "predictions", "test": "predictions_test", "cazy": "predictions_cazy"}[split]
        name = f"{prefix}_{run_name}.tsv"
        candidates = [self.path("predictions") / name]
        if self.runs_dir is not None:
            candidates += sorted(self.runs_dir.glob(f"*__{run_name}/{name}"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"no {split} predictions for run {run_name!r}; looked for:\n"
            + "\n".join(f"  {candidate}" for candidate in candidates)
            + "\nWrite them first -- `write_all_predictions(...)`, or a full `python train.py` run."
        )


# --------------------------------------------------------------------------- inputs


def load_ground_truth(paths: EvalPaths, ontology: str, split: str) -> pd.DataFrame:
    """``(protein, GO term)`` pairs of one split, in CAFA-evaluator's headerless format.

    Protein identifiers are rewritten to match the prediction files: the training tables key on
    the bare accession, the predictions on the AlphaFold model name plus the chain.
    """
    if split == "eval":
        target_matrix = paths.path("target_matrix", ontology)
        keep = set(pd.read_csv(paths.path("split", ontology) / "eval.tsv", sep="\t", header=None)[0])
        truth = load_pickle(target_matrix / "grand_truth.pkl")[ontology]
        truth = truth[["DB_Object_ID", "GO_ID_all"]].copy()
        truth["DB_Object_ID"] = "AF-" + truth["DB_Object_ID"] + "-F1-model_v4_A"
        truth = truth[truth["DB_Object_ID"].isin(keep)]
        columns = ("DB_Object_ID", "GO_ID_all")

    elif split == "test":
        target_matrix = paths.path("target_matrix", ontology)
        keep = set(load_pickle(target_matrix / "protein_vectors_test.pkl")[ontology])
        # the test set is scored on proteins that have an experimental structure
        keep &= set(pd.read_csv(paths.path("struct_test_ids"), header=None)[0])
        truth = load_pickle(target_matrix / "grand_truth_test.pkl")[ontology]
        truth = truth[["DB_Object_ID", "GO_ID_all"]].copy()
        truth = truth[truth["DB_Object_ID"].isin(keep)]
        truth["DB_Object_ID"] += "_A"
        columns = ("DB_Object_ID", "GO_ID_all")

    elif split == "cazy":
        target_matrix = paths.path("cazy_target_matrix", ontology)
        keep = set(load_pickle(target_matrix / "protein_vectors.pkl")[ontology])
        truth = load_pickle(target_matrix / "grand_truth.pkl")[ontology]
        truth = truth[["AlphaFoldDB_Entry", "GO"]].copy()
        truth = truth[truth["AlphaFoldDB_Entry"].isin(keep)]
        truth["AlphaFoldDB_Entry"] += "_A"
        columns = ("AlphaFoldDB_Entry", "GO")

    else:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    return truth.rename(columns={columns[0]: 0, columns[1]: 1})[[0, 1]]


def load_deepfri1_predictions(directory: Path, ontology: str, one_file_per_protein: bool) -> pd.DataFrame:
    """deepFRI v1 CSV output, concatenated into ``(protein, GO term, score)``.

    The CAZy run was executed one protein at a time, so there the protein id is the file name;
    the evaluation and test sets were run in batches and carry a ``Protein`` column.
    """
    frames = []
    for file in sorted(directory.glob(f"*_{ontology}_predictions.csv")):
        frame = pd.read_csv(file, comment="#", usecols=["Protein", "GO_term/EC_number", "Score"])
        if frame.empty:  # deepFRI v1 writes a header-only file for a protein it skipped
            continue
        frame["Protein"] = (
            file.stem.removesuffix(f"_{ontology}_predictions") if one_file_per_protein
            else frame["Protein"] + "_A"
        )
        frames.append(frame.rename(columns={"Protein": 0, "GO_term/EC_number": 1, "Score": 2}))
    if not frames:
        raise FileNotFoundError(f"no *_{ontology}_predictions.csv under {directory}")
    return pd.concat(frames, ignore_index=True)[[0, 1, 2]]


# --------------------------------------------------------------------------- CAFA-evaluator


def cafa_evaluate(
    predictions: pd.DataFrame | Path | str,
    ground_truth: pd.DataFrame,
    obo_file: Path | str,
    ia_file: Path | str | None = None,
    n_cpu: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run CAFA-evaluator on one prediction table or a whole directory of them.

    Returns ``(curves, best)`` as ``cafaeval.evaluation.cafa_eval`` does: the metrics at every
    threshold, indexed by ``(filename, ns, tau)``, and the optimal row per metric.
    """
    src = os.getenv("CAFA_EVALUATOR_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from cafaeval.evaluation import cafa_eval
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "CAFA-evaluator is needed to score predictions (cached scores load without it).\n"
            "Install it with `pip install cafaeval==1.2.1`, or point CAFA_EVALUATOR_SRC at the "
            "`src` directory of a https://github.com/BioComputingUP/CAFA-evaluator checkout."
        ) from error

    with tempfile.TemporaryDirectory() as workdir:
        workdir = Path(workdir)
        gt_file = workdir / "ground_truth.tsv"
        ground_truth.to_csv(gt_file, sep="\t", index=False, header=False)
        if isinstance(predictions, pd.DataFrame):
            pred_dir = workdir / "predictions"
            pred_dir.mkdir()
            predictions.to_csv(pred_dir / "preds.csv", sep="\t", index=False, header=False)
        else:
            pred_dir = Path(predictions)
        return cafa_eval(
            obo_file=str(obo_file), pred_dir=str(pred_dir), gt_file=str(gt_file),
            ia=str(ia_file) if ia_file else None, n_cpu=n_cpu,
        )


# --------------------------------------------------------------------------- evaluation


@dataclass
class _Method:
    """One thing to score: what it is called, where its predictions are, where scores are cached."""

    name: str
    label: str | None  # None: name the methods after the prediction files (competitors)
    ontology: str
    split: str
    cache: Path
    predictions: Callable[[], pd.DataFrame | Path]


class CafaEvaluation:
    """Collects CAFA scores for deepFRI2 runs, deepFRI v1 and the competitors.

    Every ``add_*`` call scores (or loads) one family of methods and appends to :attr:`curves`.
    Nothing is recomputed unless the cache is missing or ``recompute=True``.
    """

    def __init__(
        self,
        paths: EvalPaths,
        ontologies: Sequence[str] = ONTOLOGIES,
        splits: Sequence[str] = SPLITS,
        recompute: bool = False,
        n_cpu: int = 0,
        verbose: bool = True,
    ):
        self.paths = paths
        self.ontologies = tuple(ontologies)
        self.splits = tuple(splits)
        self.recompute = recompute
        self.n_cpu = n_cpu
        self.verbose = verbose
        self._frames: list[pd.DataFrame] = []
        self._ground_truth: dict[tuple[str, str], pd.DataFrame] = {}

    def ground_truth(self, ontology: str, split: str) -> pd.DataFrame:
        """Ground truth of one (ontology, split), loaded once and kept."""
        key = (ontology, split)
        if key not in self._ground_truth:
            truth = load_ground_truth(self.paths, ontology, split)
            self._log(f"  ground truth {split}/{ontology}: {truth[0].nunique():,} proteins")
            self._ground_truth[key] = truth
        return self._ground_truth[key]

    # ---------- methods to score ----------

    def add_runs(
        self,
        runs: dict[str, Sequence[str] | dict[str, str]],
        splits: Sequence[str] | None = None,
    ) -> "CafaEvaluation":
        """Score trained deepFRI2 runs, given per ontology as wandb run names.

        Either a plain list, labelled ``deepFRI2 (<run name>)``, or a ``{label: run name}``
        mapping when the figures should say what the model is.
        """
        for ontology in self.ontologies:
            entries = runs.get(ontology) or {}
            if not isinstance(entries, dict):
                entries = {f"deepFRI2 ({run})": run for run in entries}
            for label, run_name in entries.items():
                for split in splits or self.splits:
                    predictions = self.paths.prediction_file(run_name, split)
                    self._score(_Method(
                        name=label, label=label, ontology=ontology, split=split,
                        # cached next to the predictions, under the names the previous validation
                        # notebook used, so scores already computed are reused as they are
                        cache=predictions.parent / f"cafa-{split}-all_{run_name}.pickle",
                        predictions=lambda p=predictions: pd.read_csv(p, sep="\t", header=None),
                    ))
        return self

    def add_deepfri1(self, label: str = "deepFRI", splits: Sequence[str] | None = None):
        """Score deepFRI v1 predictions."""
        for ontology in self.ontologies:
            for split in splits or self.splits:
                directory = self.paths.competitor_dir(split, ontology) / "predictions_deepfri_v1"
                # the evaluation-set directory is already per ontology, the others are not
                stem = "deepFRI" if split == "eval" else f"deepFRI-{ontology}"
                self._score(_Method(
                    name=label, label=label, ontology=ontology, split=split,
                    cache=directory / f"cafa-eval-all_{stem}.pickle",
                    predictions=lambda d=directory, o=ontology, s=split: load_deepfri1_predictions(
                        d, o, one_file_per_protein=(s == "cazy")),
                ))
        return self

    def add_competitors(
        self,
        names: dict[str, str],
        keep: Iterable[str] | None = None,
        splits: Sequence[str] | None = None,
    ) -> "CafaEvaluation":
        """Score the published competitors, which share one prediction directory per split.

        ``names`` maps prediction file names to method names; ``keep`` selects which of them
        reach :attr:`curves` (``None``: all of them).
        """
        for ontology in self.ontologies:
            for split in splits or self.splits:
                root = self.paths.competitor_dir(split, ontology)
                stem = "competitors" if split == "eval" else f"competitors-{ontology}"
                self._score(
                    _Method(
                        name="competitors", label=None, ontology=ontology, split=split,
                        cache=root / f"cafa-eval-all_{stem}.pickle",
                        predictions=lambda r=root: r / "predictions",
                    ),
                    names=names,
                    keep=None if keep is None else set(keep),
                )
        return self

    # ---------- results ----------

    @property
    def curves(self) -> pd.DataFrame:
        """Tidy metrics: one row per ``(method, ontology, split, tau)``.

        Columns are CAFA-evaluator's, unweighted and information-accretion weighted (``_w``):
        ``pr``, ``rc``, ``f``, ``s``, ``cov``, ``mi``, ``ru``, the ``_micro`` variants and the
        raw counts. Both weightings are always present; ``weighted=`` picks one at read time.
        """
        if not self._frames:
            return pd.DataFrame(columns=["method", "ontology", "split", "tau"])
        return pd.concat(self._frames, ignore_index=True)

    def summary(self, weighted: bool = True, decimals: int | None = 3) -> pd.DataFrame:
        """One row per ``(method, ontology, split)``: Fmax and where it falls, micro-F, Smin."""
        return summarize(self.curves, weighted=weighted, decimals=decimals)

    def table(
        self,
        metric: str = "fmax",
        split: str = "test",
        weighted: bool = True,
        decimals: int | None = 3,
    ) -> pd.DataFrame:
        """``summary()`` pivoted to methods x ontologies for one split -- ready to paste.

        Carries its own title and file name in ``.attrs``, so ``figures.save(table, dir)``
        cannot file it under the wrong split.
        """
        summary = self.summary(weighted=weighted, decimals=None)
        summary = summary[summary["split"] == split]
        pivot = summary.pivot(index="method", columns="ontology", values=metric).reindex(
            # methods in the order they were added, ontologies in MF / CC / BP order
            index=[m for m in dict.fromkeys(summary["method"])],
            columns=[o for o in self.ontologies if o in summary["ontology"].values],
        )
        pivot = pivot.round(decimals) if decimals is not None else pivot
        return _name_table(pivot, metric=metric, split=split, weighted=weighted)

    # ---------- internals ----------

    def _score(self, method: _Method, names: dict[str, str] | None = None, keep: set[str] | None = None):
        where = f"{method.name} [{method.ontology}/{method.split}]"
        if not self.recompute and method.cache.is_file():
            self._log(f"loading {where}")
            curves = load_pickle(method.cache)
        else:
            self._log(f"scoring {where} -> {method.cache}")
            curves, best = cafa_evaluate(
                method.predictions(), self.ground_truth(method.ontology, method.split),
                obo_file=self.paths.obo_file, ia_file=self.paths.ia_file, n_cpu=self.n_cpu,
            )
            method.cache.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(method.cache, curves)
            # the optimal rows are recomputed from the curves by summarize(), but keep writing
            # them so other tooling reading these directories still finds what it expects
            save_pickle(Path(str(method.cache).replace("-all_", "-best_")), best)

        self._frames.append(_tidy(curves, method, names or {}, keep))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


def _tidy(curves: pd.DataFrame, method: _Method, names: dict[str, str], keep: set[str] | None):
    """CAFA-evaluator's ``(filename, ns, tau)`` frame -> tidy ``method/ontology/split/tau`` rows."""
    frame = curves.reset_index()
    if method.label is not None:
        frame["method"] = method.label
    else:
        frame["method"] = frame["filename"].map(lambda file: names.get(file, file))
        if keep is not None:
            frame = frame[frame["method"].isin(keep)]
    frame["ontology"] = frame["ns"].map(ONTOLOGY_OF).fillna(method.ontology)
    frame["split"] = method.split
    frame = frame.drop(columns=["filename", "ns"])
    lead = ["method", "ontology", "split", "tau"]
    return frame[lead + [column for column in frame.columns if column not in lead]]


def summarize(curves: pd.DataFrame, weighted: bool = True, decimals: int | None = 3) -> pd.DataFrame:
    """Collapse tidy curves to one row per ``(method, ontology, split)``.

    ``fmax`` and the precision / recall / coverage beside it come from the F-max threshold,
    ``smin`` from the threshold that minimises the semantic distance -- the two CAFA headline
    numbers.

    Note on ``smin`` when ``weighted``: CAFA-evaluator's own ``best`` tables pick the threshold
    by the *unweighted* ``s`` and then report ``s_w`` there, which is not the minimum of
    ``s_w``. Here ``smin`` is the minimum of the column being summarised, so the weighted value
    comes out a little lower than the one those tables print.
    """
    if curves.empty:
        return curves
    w = "_w" if weighted else ""
    rows = []
    for (method, ontology, split), group in curves.groupby(["method", "ontology", "split"], sort=False):
        at_fmax = group.loc[group[f"f{w}"].idxmax()]
        rows.append({
            "method": method, "ontology": ontology, "split": split,
            "fmax": at_fmax[f"f{w}"], "tau": at_fmax["tau"],
            "precision": at_fmax[f"pr{w}"], "recall": at_fmax[f"rc{w}"],
            "coverage": at_fmax[f"cov{w}"],
            "f_micro": group[f"f_micro{w}"].max(),
            "smin": group[f"s{w}"].min(),
        })
    summary = pd.DataFrame(rows)
    if decimals is not None:
        numeric = summary.select_dtypes("number").columns
        summary[numeric] = summary[numeric].round(decimals)
    return _name_table(summary, metric="summary", weighted=weighted)


def _name_table(frame: pd.DataFrame, **bits) -> pd.DataFrame:
    """Attach the title and file name a table should be saved under (see ``figures.save``)."""
    from .figures import naming  # local: figures imports summarize()

    frame.attrs["title"], frame.attrs["name"] = naming("table", **bits)
    return frame
