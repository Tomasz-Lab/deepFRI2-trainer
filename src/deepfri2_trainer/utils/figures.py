"""Figures and tables from the CAFA curves produced by ``utils/evaluator.py``.

Two primitives -- :func:`curve` (a metric against the decision threshold) and :func:`pr_curve`
(precision against recall) -- draw into an axes you pass in; :func:`panel`, :func:`compare` and
:func:`bars` arrange them. Everything returns the matplotlib objects, so a figure that is nearly
right can be finished by hand::

    fig, axes = figures.panel(curves, split="test", ontology="MF", weighted=False)
    axes[0].set_title("Molecular function")
    figures.save(fig, FIGURE_DIR)

Every figure and table titles itself from what it actually plots and carries the file name to
match, so :func:`save` takes a *directory*, never a name -- a figure cannot be filed under
another figure's label. ``weighted`` is an argument everywhere and is part of both.

:data:`PALETTE` and :data:`SPLIT_LABELS` are empty here and set from the notebook, next to the
method names they belong to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluator import ONTOLOGIES, SPLITS, summarize

#: Fixed colour per method, so a method looks the same in every figure. Set from the notebook.
#: Methods not listed are assigned blues (deepFRI2 variants, keeping them together) or tab10.
PALETTE: dict[str, str] = {}
DEEPFRI2_COLORS = ("#1f77b4", "#17becf", "#08306b", "#6baed6")

#: Display name per split, e.g. ``{"test": "Test set"}``. Set from the notebook.
SPLIT_LABELS: dict[str, str] = {}

#: Axis and title labels per metric, including the ``summary()`` columns.
METRIC_LABELS = {
    "f": "F1-score", "pr": "Precision", "rc": "Recall", "cov": "Coverage",
    "s": "Semantic distance (S)", "mi": "Misinformation", "ru": "Remaining uncertainty",
    "f_micro": "micro F1-score", "pr_micro": "micro precision", "rc_micro": "micro recall",
    "pr_rc": "Precision-recall", "fmax": "F-max", "smin": "S-min", "summary": "CAFA scores",
}
#: Metrics that are not rates: their y-range is fitted to the data instead of (0, 1).
UNBOUNDED = ("s", "mi", "ru")

#: Defaults every function reads; edit once in the notebook to restyle every figure.
STYLE = {
    "linewidth": 1.4,
    "linewidth_emphasis": 1.4,
    "figsize": (5.5, 4.4),
    "legend_fontsize": 8,
    "guides": True,   # faint reference lines at 0.5
    "iso_f1": True,   # F1 contours behind the precision-recall curve
    "weighting": {True: "IA-weighted", False: "unweighted"},
}


# --------------------------------------------------------------------------- naming


def label_of(kind: str, key: str) -> str:
    """Display name of one naming bit: a split, a metric, or anything else (as it is)."""
    if kind == "split":
        return SPLIT_LABELS.get(key, key)
    if kind == "metric":
        return METRIC_LABELS.get(str(key).removesuffix("_w"), key)
    return str(key)


def naming(kind: str, weighted: bool = True, **bits) -> tuple[str, str]:
    """``(title, file name)`` for a figure or table, from what it actually shows.

    The title reads in display names, the file name in the raw keys, and both end in the
    weighting -- so ``bars(..., split="cazy")`` can only ever be saved as ``bars_fmax_cazy_*``.
    """
    values = [(name, value) for name, value in bits.items() if value is not None]
    tag = "weighted" if weighted else "unweighted"
    title = " · ".join([label_of(name, value) for name, value in values]
                       + [STYLE["weighting"][bool(weighted)]])
    return title, "_".join([kind, *(str(value) for _, value in values), tag])


def _named(figure, kind: str, weighted: bool, suptitle: bool = True, **bits):
    """Give a figure its title and the file name :func:`save` will use."""
    title, name = naming(kind, weighted, **bits)
    figure.set_label(name)
    if suptitle:
        figure.suptitle(title)
    return figure


# --------------------------------------------------------------------------- selection


def select(curves, split=None, ontology=None, methods: Sequence[str] | None = None):
    """Rows of one split / ontology, optionally restricted and ordered by ``methods``."""
    frame = curves
    if split is not None:
        frame = frame[frame["split"] == split]
    if ontology is not None:
        frame = frame[frame["ontology"] == ontology]
    if methods is not None:
        order = {method: rank for rank, method in enumerate(methods)}
        frame = frame[frame["method"].isin(order)]
        frame = frame.sort_values("method", key=lambda column: column.map(order), kind="stable")
    return frame


def method_order(frame: pd.DataFrame, methods: Sequence[str] | None = None) -> list[str]:
    present = set(frame["method"])
    return [m for m in methods if m in present] if methods else list(dict.fromkeys(frame["method"]))


def method_colors(methods: Iterable[str], colors: dict[str, str] | None = None) -> dict[str, str]:
    """Colour per method: explicit override, then :data:`PALETTE`, then blues / tab10."""
    resolved, deepfri2, other = {}, 0, 0
    for method in methods:
        if colors and method in colors:
            resolved[method] = colors[method]
        elif method in PALETTE:
            resolved[method] = PALETTE[method]
        elif method.lower().startswith("deepfri2"):
            resolved[method] = DEEPFRI2_COLORS[deepfri2 % len(DEEPFRI2_COLORS)]
            deepfri2 += 1
        else:
            resolved[method] = plt.cm.tab10(other % 10)
            other += 1
    return resolved


def metric_column(metric: str, weighted: bool) -> str:
    """``"f"`` -> ``"f_w"`` when weighted; a column that is already explicit is left alone."""
    return f"{metric}_w" if weighted and not metric.endswith("_w") else metric


def _widths(methods: Sequence[str], emphasis: Iterable[str] | None) -> dict[str, float]:
    if emphasis is None:
        emphasis = [m for m in methods if m.lower().startswith("deepfri2")]
    emphasis = set(emphasis)
    return {m: STYLE["linewidth_emphasis"] if m in emphasis else STYLE["linewidth"] for m in methods}


def _axes(ax, **naming_bits):
    """The axes to draw on; a fresh single-axes figure titles and names itself."""
    if ax is not None:
        return ax
    title, name = naming(**naming_bits)
    figure, ax = plt.subplots(figsize=STYLE["figsize"])
    figure.set_label(name)
    ax.set_title(title, fontsize=10)
    return ax


def _guides(ax, guides: bool | None, horizontal: bool = True) -> None:
    if STYLE["guides"] if guides is None else guides:
        if horizontal:
            ax.axhline(0.5, lw=0.5, ls="--", c="r", alpha=0.25)
        ax.axvline(0.5, lw=0.5, ls="--", c="r", alpha=0.25)


# --------------------------------------------------------------------------- primitives


def curve(
    curves: pd.DataFrame,
    metric: str = "f",
    *,
    split: str | None = None,
    ontology: str | None = None,
    weighted: bool = True,
    methods: Sequence[str] | None = None,
    ax=None,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    emphasis: Iterable[str] | None = None,
    legend: bool = True,
    guides: bool | None = None,
    ylim: tuple[float, float] | None = None,
    **line_kwargs,
):
    """One metric against the decision threshold, one line per method."""
    frame = select(curves, split, ontology, methods)
    column = metric_column(metric, weighted)
    order = method_order(frame, methods)
    palette, widths = method_colors(order, colors), _widths(order, emphasis)

    ax = _axes(ax, kind="curve", weighted=weighted, metric=metric, split=split, ontology=ontology)
    for method in order:
        rows = frame[frame["method"] == method].sort_values("tau")
        ax.plot(rows["tau"], rows[column], color=palette[method], lw=widths[method],
                label=(labels or {}).get(method, method), **line_kwargs)

    base = metric.removesuffix("_w")
    ax.set(xlim=(0, 1), xlabel="Threshold",
           ylabel=f"{label_of('metric', base)}{' (weighted)' if weighted else ''}")
    bounded = base not in UNBOUNDED
    ax.set_ylim(*(ylim or ((0, 1.02) if bounded else _fit_ylim(frame, column))))
    _guides(ax, guides, horizontal=bounded)
    if legend:
        ax.legend(fontsize=STYLE["legend_fontsize"], loc="best")
    return ax


def _fit_ylim(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Room for every method's optimum on an open-ended metric such as the semantic distance."""
    best = frame.groupby("method")[column].min()
    return (0, float(best.max()) * 1.25 if len(best) and best.max() > 0 else 1.0)


def pr_curve(
    curves: pd.DataFrame,
    *,
    split: str | None = None,
    ontology: str | None = None,
    weighted: bool = True,
    methods: Sequence[str] | None = None,
    ax=None,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    emphasis: Iterable[str] | None = None,
    legend: bool = True,
    guides: bool | None = None,
    iso_f1: bool | None = None,
    **line_kwargs,
):
    """Precision against recall, with F1 contours behind it.

    A method that reports one score per prediction (some competitors do) collapses to a point
    and is drawn as a marker.
    """
    frame = select(curves, split, ontology, methods)
    pr, rc = metric_column("pr", weighted), metric_column("rc", weighted)
    order = method_order(frame, methods)
    palette, widths = method_colors(order, colors), _widths(order, emphasis)

    ax = _axes(ax, kind="pr", weighted=weighted, split=split, ontology=ontology)
    if STYLE["iso_f1"] if iso_f1 is None else iso_f1:
        _iso_f1(ax)
    for method in order:
        rows = frame[frame["method"] == method].sort_values("tau")
        label = (labels or {}).get(method, method)
        if rows[pr].nunique() == 1 and rows[rc].nunique() == 1:
            ax.scatter(rows[rc].iloc[0], rows[pr].iloc[0], color=palette[method], label=label, zorder=3)
        else:
            ax.plot(rows[rc], rows[pr], color=palette[method], lw=widths[method], label=label, **line_kwargs)

    suffix = " (weighted)" if weighted else ""
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall" + suffix, ylabel="Precision" + suffix)
    _guides(ax, guides)
    if legend:
        ax.legend(fontsize=STYLE["legend_fontsize"], loc="best")
    return ax


def _iso_f1(ax, levels: Sequence[float] = np.arange(0.1, 1.0, 0.1)) -> None:
    recall = np.linspace(0.01, 1, 200)
    for level in levels:
        precision = (level * recall) / (2 * recall - level)
        valid = (precision > 0) & (precision <= 1)
        if not valid.any():
            continue
        ax.plot(recall[valid], precision[valid], color="gray", lw=0.6, alpha=0.3, zorder=0)
        last = np.where(valid)[0][-1]
        ax.text(recall[last] - 0.1, precision[last] + 0.01, f"F1={level:.1f}",
                color="gray", fontsize=6, va="bottom", ha="left", zorder=0)


def _draw(curves, metric, **kwargs):
    """Dispatch ``"pr_rc"`` to :func:`pr_curve` and everything else to :func:`curve`."""
    return pr_curve(curves, **kwargs) if metric == "pr_rc" else curve(curves, metric, **kwargs)


# --------------------------------------------------------------------------- composites


def panel(
    curves: pd.DataFrame,
    *,
    split: str,
    ontology: str,
    metrics: Sequence[str] = ("f", "pr_rc", "s", "cov"),
    weighted: bool = True,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    legend_on: int = 0,
    **kwargs,
):
    """A grid of panels for one (split, ontology); ``"pr_rc"`` draws the precision-recall curve.

    The per-ontology figure of the paper: F1, precision-recall, semantic distance and coverage
    for every method side by side.
    """
    nrows = -(-len(metrics) // ncols)
    figsize = figsize or (STYLE["figsize"][0] * ncols, STYLE["figsize"][1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for index, metric in enumerate(metrics):
        _draw(curves, metric, split=split, ontology=ontology, weighted=weighted,
              ax=axes[index], legend=(index == legend_on), **kwargs)
    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    _named(fig, "panel", weighted, split=split, ontology=ontology)
    fig.tight_layout()
    return fig, axes[: len(metrics)]


def compare(
    curves: pd.DataFrame,
    metric: str = "f",
    *,
    by: str = "ontology",
    split: str | None = None,
    ontology: str | None = None,
    weighted: bool = True,
    values: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    """One metric across ontologies (``by="ontology"``) or across splits (``by="split"``).

    The three-ontology row most papers open with::

        figures.compare(curves, "f", by="ontology", split="test")
    """
    if values is None:
        present = set(curves[by])
        values = [v for v in (ONTOLOGIES if by == "ontology" else SPLIT_LABELS or SPLITS) if v in present]
    figsize = figsize or (STYLE["figsize"][0] * len(values), STYLE["figsize"][1])
    fig, axes = plt.subplots(1, len(values), figsize=figsize, squeeze=False, sharey=True)
    axes = axes.ravel()

    for index, value in enumerate(values):
        _draw(curves, metric, weighted=weighted, ax=axes[index], legend=(index == 0),
              **{"split": split, "ontology": ontology, by: value}, **kwargs)
        axes[index].set_title(label_of(by, value), fontsize=10)
        if index:
            axes[index].set_ylabel("")

    fixed = ontology if by == "split" else split
    _named(fig, f"{metric}_by-{by}", weighted, **{("split" if by == "ontology" else "ontology"): fixed})
    fig.tight_layout()
    return fig, axes


def bars(
    curves: pd.DataFrame,
    metric: str = "fmax",
    *,
    split: str,
    weighted: bool = True,
    methods: Sequence[str] | None = None,
    ontologies: Sequence[str] | None = None,
    colors: dict[str, str] | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Grouped bars of a headline number (``fmax``, ``smin``, ``f_micro``, ...) per ontology."""
    frame = summarize(select(curves, split=split), weighted=weighted, decimals=None)
    ontologies = ontologies or [o for o in ONTOLOGIES if o in set(frame["ontology"])]
    order = method_order(frame, methods)
    palette = method_colors(order, colors)

    fig, ax = plt.subplots(figsize=figsize or (max(4.0, 0.45 * len(ontologies) * max(len(order), 1)), 4.0))
    width = 0.8 / max(len(order), 1)
    positions = np.arange(len(ontologies))
    for index, method in enumerate(order):
        values = frame[frame["method"] == method].set_index("ontology")[metric]
        heights = [values.get(ontology, np.nan) for ontology in ontologies]
        offsets = positions + (index - (len(order) - 1) / 2) * width
        ax.bar(offsets, heights, width=width, color=palette[method], label=method)
        if annotate:
            for x, height in zip(offsets, heights):
                if not np.isnan(height):
                    ax.text(x, height, f"{height:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(positions, ontologies)
    ax.set_ylabel(label_of("metric", metric))
    ax.legend(fontsize=STYLE["legend_fontsize"], loc="best")
    ax.margins(y=0.12)
    _named(fig, f"bars_{metric}", weighted, split=split)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- output


def save(obj, directory: Path | str, name: str | None = None, dpi: int = 300) -> list[Path]:
    """Write a figure (png + pdf) or a table (csv + tex) into ``directory``, named after itself.

    The name comes from the object -- ``panel_test_MF_weighted``, ``bars_fmax_cazy_unweighted``
    -- so it always describes what is actually in the file. Pass ``name=`` to override.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    is_table = isinstance(obj, pd.DataFrame)
    name = name or (obj.attrs.get("name") if is_table else obj.get_label()) or "untitled"

    written = []
    for extension in ("csv", "tex") if is_table else ("png", "pdf"):
        target = directory / f"{name}.{extension}"
        if extension == "csv":
            obj.to_csv(target, index=obj.index.name is not None or obj.index.nlevels > 1)
        elif extension == "tex":
            target.write_text(obj.to_latex(float_format="%.3f", caption=obj.attrs.get("title")))
        else:
            obj.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
    return written
