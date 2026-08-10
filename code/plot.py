"""Small, explicit plotting recipes for the Zhong analysis tables.

The tabular recipes in this module all consume a *tidy* pandas DataFrame.
Arguments such as ``x="progress"`` and ``y="mean_dprime"`` name columns; the
recipes never aggregate, estimate uncertainty, or choose scientific units.
Callers must do that work before plotting.  ``guide()`` describes the expected
row grain and column roles for every recipe.
"""

from __future__ import annotations

import inspect
import zlib
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COHORT = {
    "supervised": "g",
    "task": "g",
    "unsupervised": (0.46, 0.0, 0.23),
    "grating": "0.5",
    "naive": "k",
}
STIMULUS = {
    "leaf1": "b",
    "leaf2": "c",
    "leaf3": (0.27, 0.51, 0.71),
    "leaf1_swap": (0.0, 0.47, 0.47),
    "circle1": "r",
    "circle2": "m",
}
MOMENT = {"before": "0.6", "after": "k"}
AREA = {"V1": "#4C72B0", "mHV": "#DD8452", "lHV": "#55A868", "aHV": "#C44E52"}
SEMANTIC = {**COHORT, **STIMULUS, **MOMENT, **AREA}

ORDER = {
    "cohort": ["supervised", "unsupervised", "grating", "naive"],
    "moment": ["before", "after"],
    "area": ["V1", "mHV", "lHV", "aHV"],
    "area_group": ["V1", "mHV", "lHV", "aHV"],
    "stage": ["train1", "train2", "test1", "test2", "test3"],
}

VIEWS = {
    "window_dprime": "notebooks/results/rui/window_dprime.csv",
    "session_dprime": "notebooks/results/rui/session_dprime.csv",
    "gamm": "notebooks/results/rui/python_gamm_predictions.csv",
    "session_area_metrics": "notebooks/results/tanya/session_area_metrics.csv",
    "trial_behavior": "notebooks/results/tomoya/trial_behavior_neural.csv",
    "area_metrics": "notebooks/results/janviere/area_metrics.csv",
}

CONTRACTS = {
    "trajectory": {
        "chart": "line with an optional symmetric uncertainty band",
        "input": "pandas DataFrame",
        "grain": "one row per x × by × facet combination",
        "required": "x: numeric column; y: numeric column",
        "optional": "by/facet: category columns; band: non-negative half-width (NaN = unavailable)",
        "example": "plot.trajectory(df, x='progress', y='mean', by='cohort', band='se')",
    },
    "comparison": {
        "chart": "dodged points with optional symmetric error bars",
        "input": "pandas DataFrame",
        "grain": "one row per x × by × facet combination",
        "required": "x: category column; y: numeric column",
        "optional": "by/facet: category columns; err: non-negative half-width (NaN = unavailable)",
        "example": "plot.comparison(df, x='area', y='fraction', by='cohort', err='se')",
    },
    "slope": {
        "chart": "connected values across ordered categories",
        "input": "pandas DataFrame",
        "grain": "one row per x × by × facet combination",
        "required": "x: category column; y: numeric column",
        "optional": "by/facet: category columns",
        "example": "plot.slope(df, x='moment', y='fraction', by='cohort', facet='area')",
    },
    "distribution": {
        "chart": "step histogram with optional symmetric threshold guides",
        "input": "pandas DataFrame",
        "grain": "one row per observation",
        "required": "value: numeric column",
        "optional": "by/facet: category columns; threshold; bins",
        "example": "plot.distribution(df, value='dprime', by='cohort', threshold=0.3)",
    },
    "relationship": {
        "chart": "scatter plot with an optional linear fit per group",
        "input": "pandas DataFrame",
        "grain": "one row per paired observation",
        "required": "x: numeric column; y: numeric column",
        "optional": "by: category column; fit: bool",
        "example": "plot.relationship(df, x='run_speed', y='activity', by='cohort')",
    },
    "bars": {
        "chart": "dodged bars with optional symmetric error bars",
        "input": "pandas DataFrame",
        "grain": "one row per x × by × facet combination",
        "required": "x: category column; y: numeric column",
        "optional": "by/facet: category columns; err: non-negative half-width (NaN = unavailable)",
        "example": "plot.bars(df, x='area', y='correlation', by='cohort')",
    },
    "heatmap": {
        "chart": "two-dimensional image, row zero at the bottom",
        "input": "2D numeric array",
        "grain": "rows × columns",
        "required": "matrix",
        "optional": "reward: column coordinate; cmap; ax",
        "example": "plot.heatmap(trials_by_position, reward=12)",
    },
    "density": {
        "chart": "two-dimensional density image with optional outlines",
        "input": "2D numeric array",
        "grain": "y pixels × x pixels",
        "required": "image",
        "optional": "outlines: iterable of N×2 (x, y) arrays; cmap; vmax; ax",
        "example": "plot.density(image, outlines=atlas_edges)",
    },
    "raster": {
        "chart": "event scatter with the y axis inverted",
        "input": "pandas DataFrame",
        "grain": "one row per event",
        "required": "x: numeric column; y: numeric column",
        "optional": "by: category column; ax",
        "example": "plot.raster(events, x='time', y='trial', by='event')",
    },
    "evolution": {
        "chart": "animation of a distribution or mean curve across windows",
        "input": "2D numeric array",
        "grain": "observations × windows",
        "required": "matrix",
        "optional": "kind: 'histogram' or 'curve'; fps; threshold; bins",
        "example": "plot.evolution(neurons_by_window, kind='histogram')",
    },
}

_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def use(scale: str = "screen") -> None:
    """Apply the project Matplotlib style explicitly.

    Importing :mod:`plot` does not change global Matplotlib settings.  Call
    ``plot.use()`` once in a notebook when the shared style is desired.
    """

    if scale not in {"screen", "print"}:
        raise ValueError("scale must be 'screen' or 'print'")
    mpl.rcParams.update(
        {
            "figure.dpi": 120 if scale == "screen" else 300,
            "font.size": 9 if scale == "screen" else 5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": "medium",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def color(value: Any) -> Any:
    """Return the project color for a semantic value or a stable fallback."""

    key = str(value)
    if key in SEMANTIC:
        return SEMANTIC[key]
    return _CYCLE[zlib.crc32(key.encode()) % len(_CYCLE)]


def order(column: str, values: Iterable[Any]) -> list[Any]:
    """Return unique values in semantic order without changing their types."""

    seen = list(pd.unique(pd.Series(list(values), dtype="object").dropna()))
    known = ORDER.get(column)
    if known is not None:
        remaining = list(seen)
        ordered: list[Any] = []
        for expected in known:
            match = next(
                (index for index, value in enumerate(remaining) if str(value) == expected),
                None,
            )
            if match is not None:
                ordered.append(remaining.pop(match))
        return ordered + remaining
    try:
        return sorted(seen)
    except TypeError:
        return seen


def fmt(
    ax: Axes,
    *,
    x: str | None = None,
    y: str | None = None,
    title: Any | None = None,
    invert_y: bool = False,
) -> Axes:
    """Apply labels and the small shared axis treatment to an existing axis."""

    if x is not None:
        ax.set_xlabel(x)
    if y is not None:
        ax.set_ylabel(y)
    if title is not None:
        ax.set_title(str(title))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if invert_y:
        ax.invert_yaxis()
    return ax


def view_status(root: str | Path) -> pd.DataFrame:
    """Describe the result CSV files known to :func:`attach` without mutation."""

    root_path = Path(root)
    return pd.DataFrame(
        [
            {
                "view": name,
                "csv": str(root_path / relative),
                "exists": (root_path / relative).is_file(),
            }
            for name, relative in VIEWS.items()
        ]
    )


def attach_views(db: Any, root: str | Path, *, required: bool = False) -> pd.DataFrame:
    """Attach known result CSVs as DuckDB views and return an explicit report.

    This function only registers files.  It never calculates or aggregates
    values.  With ``required=True`` it fails early if any declared CSV is absent.
    """

    report = view_status(root)
    missing = report.loc[~report["exists"], "view"].tolist()
    if required and missing:
        raise FileNotFoundError(f"Missing plotting result CSVs for views: {missing}")

    for row in report.loc[report["exists"]].itertuples(index=False):
        path_literal = row.csv.replace("'", "''")
        db.query(
            f"CREATE OR REPLACE TEMP VIEW {row.view} "
            f"AS SELECT * FROM read_csv_auto('{path_literal}')"
        )
    report["attached"] = report["exists"]
    return report


def attach(db: Any, root: str | Path, *, required: bool = False) -> Any:
    """Attach known CSV views and return ``db`` for backward compatibility.

    Use :func:`attach_views` when the per-view attachment report is useful.
    """

    attach_views(db, root, required=required)
    return db


def _tabular(
    frame: pd.DataFrame,
    *,
    recipe: str,
    roles: dict[str, str | None],
    numeric: tuple[str, ...] = (),
    categorical: tuple[str, ...] = (),
    nullable_numeric: tuple[str, ...] = (),
    nonnegative: tuple[str, ...] = (),
    unique: tuple[str, ...] = (),
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{recipe} expects a pandas DataFrame, got {type(frame).__name__}")
    if frame.empty:
        raise ValueError(f"{recipe} received an empty DataFrame")

    selected = {role: column for role, column in roles.items() if column is not None}
    missing = [column for column in selected.values() if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{recipe} is missing columns {missing}; available columns are {list(frame.columns)}"
        )

    for role in numeric:
        column = selected.get(role)
        if column is None:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{recipe} role {role!r} needs a numeric column; {column!r} is {frame[column].dtype}")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        invalid = np.isinf(values) if role in nullable_numeric else ~np.isfinite(values)
        if invalid.any():
            qualification = "infinite" if role in nullable_numeric else "non-finite"
            raise ValueError(
                f"{recipe} column {column!r} contains {int(invalid.sum())} {qualification} values; "
                "clean or filter them before plotting"
            )

    for role in categorical:
        column = selected.get(role)
        if column is not None and frame[column].isna().any():
            count = int(frame[column].isna().sum())
            raise ValueError(
                f"{recipe} category column {column!r} contains {count} missing values; "
                "label or filter them before plotting"
            )

    for role in nonnegative:
        column = selected.get(role)
        if column is not None and (frame[column].dropna() < 0).any():
            raise ValueError(f"{recipe} uncertainty column {column!r} must be non-negative")

    keys = list(dict.fromkeys(selected[role] for role in unique if role in selected))
    if keys and frame.duplicated(keys, keep=False).any():
        sample = frame.loc[frame.duplicated(keys, keep=False), keys].head(3).to_dict("records")
        raise ValueError(
            f"{recipe} expects one row per {' × '.join(keys)} combination; "
            f"duplicate keys include {sample}. Aggregate explicitly before plotting."
        )


def _matrix(values: Any, *, recipe: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{recipe} expects a numeric 2D array") from error
    if matrix.ndim != 2:
        raise ValueError(f"{recipe} expects a 2D array, got shape {matrix.shape}")
    if 0 in matrix.shape:
        raise ValueError(f"{recipe} received an empty array with shape {matrix.shape}")
    if np.isinf(matrix).any():
        raise ValueError(f"{recipe} does not accept infinite values")
    if not np.isfinite(matrix).any():
        raise ValueError(f"{recipe} needs at least one finite value")
    return matrix


def _panels(frame: pd.DataFrame, facet: str | None) -> list[tuple[Any, pd.DataFrame]]:
    if facet is None:
        return [(None, frame)]
    return [(value, frame[frame[facet] == value]) for value in order(facet, frame[facet])]


def _grouped(frame: pd.DataFrame, by: str | None) -> list[tuple[Any, pd.DataFrame]]:
    if by is None:
        return [(None, frame)]
    return [(value, frame[frame[by] == value]) for value in order(by, frame[by])]


def _canvas(ax: Axes | None, panels: int) -> tuple[Figure, list[Axes]]:
    if panels < 1:
        raise ValueError("plotting requires at least one panel")
    if ax is not None:
        if panels != 1:
            raise ValueError(
                f"ax= supplies one axis but this plot has {panels} facets; "
                "omit ax or plot each facet explicitly"
            )
        return ax.figure, [ax]
    figure, axes = plt.subplots(1, panels, figsize=(3.4 * panels, 3.0), squeeze=False)
    return figure, list(axes[0])


def _legend(ax: Axes) -> None:
    if any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize="small")


def _series_color(key: Any, by: str | None) -> Any:
    return color(key) if by is not None else None


def _band(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    half: pd.Series | None = None,
    *,
    color_value: Any = None,
    label: Any = None,
) -> None:
    (line,) = ax.plot(x, y, color=color_value, label=label, lw=1.3)
    if half is not None:
        ax.fill_between(x, y - half, y + half, color=line.get_color(), alpha=0.2, linewidth=0)


def _dots(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray | None = None,
    *,
    color_value: Any = None,
    label: Any = None,
) -> None:
    ax.errorbar(x, y, yerr=err, fmt="o", ms=3, lw=1, capsize=2, color=color_value, label=label)


def _columns(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    width: float,
    err: np.ndarray | None = None,
    *,
    color_value: Any = None,
    label: Any = None,
) -> None:
    ax.bar(
        x,
        y,
        width=width,
        yerr=err,
        color=color_value,
        label=label,
        error_kw={"lw": 0.8},
    )


def _dodge(count: int) -> list[float]:
    return [(index - (count - 1) / 2) * (0.8 / count) for index in range(count)]


def trajectory(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    facet: str | None = None,
    band: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot one pre-aggregated line per ``by`` value, optionally faceted."""

    roles = {"x": x, "y": y, "by": by, "facet": facet, "band": band}
    _tabular(
        frame,
        recipe="trajectory",
        roles=roles,
        numeric=("x", "y", "band"),
        categorical=("by", "facet"),
        nullable_numeric=("band",),
        nonnegative=("band",),
        unique=("x", "by", "facet"),
    )
    panels = _panels(frame, facet)
    figure, axes = _canvas(ax, len(panels))
    for cell, (name, panel) in zip(axes, panels):
        for key, group in _grouped(panel, by):
            group = group.sort_values(x)
            _band(
                cell,
                group[x],
                group[y],
                group[band] if band else None,
                color_value=_series_color(key, by),
                label=key,
            )
        fmt(cell, x=x, y=y, title=name)
        _legend(cell)
    return figure


def comparison(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    facet: str | None = None,
    err: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot one pre-aggregated point per category, with optional errors."""

    roles = {"x": x, "y": y, "by": by, "facet": facet, "err": err}
    _tabular(
        frame,
        recipe="comparison",
        roles=roles,
        numeric=("y", "err"),
        categorical=("x", "by", "facet"),
        nullable_numeric=("err",),
        nonnegative=("err",),
        unique=("x", "by", "facet"),
    )
    panels = _panels(frame, facet)
    figure, axes = _canvas(ax, len(panels))
    for cell, (name, panel) in zip(axes, panels):
        categories = order(x, panel[x])
        groups = _grouped(panel, by)
        for offset, (key, group) in zip(_dodge(len(groups)), groups):
            aligned = group.set_index(x).reindex(categories)
            _dots(
                cell,
                np.arange(len(categories)) + offset,
                aligned[y].to_numpy(),
                aligned[err].to_numpy() if err else None,
                color_value=_series_color(key, by),
                label=key,
            )
        cell.set_xticks(range(len(categories)), categories)
        fmt(cell, x=x, y=y, title=name)
        _legend(cell)
    return figure


def slope(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    facet: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Connect pre-aggregated values across ordered ``x`` categories."""

    roles = {"x": x, "y": y, "by": by, "facet": facet}
    _tabular(
        frame,
        recipe="slope",
        roles=roles,
        numeric=("y",),
        categorical=("x", "by", "facet"),
        unique=("x", "by", "facet"),
    )
    panels = _panels(frame, facet)
    figure, axes = _canvas(ax, len(panels))
    for cell, (name, panel) in zip(axes, panels):
        levels = order(x, panel[x])
        for key, group in _grouped(panel, by):
            aligned = group.set_index(x).reindex(levels)
            cell.plot(
                range(len(levels)),
                aligned[y].to_numpy(),
                "-o",
                ms=3,
                lw=1,
                color=_series_color(key, by),
                label=key,
            )
        cell.set_xticks(range(len(levels)), levels)
        fmt(cell, x=x, y=y, title=name)
        _legend(cell)
    return figure


def distribution(
    frame: pd.DataFrame,
    *,
    value: str,
    by: str | None = None,
    facet: str | None = None,
    threshold: float | None = 0.3,
    bins: int = 60,
    ax: Axes | None = None,
) -> Figure:
    """Plot raw observations as step histograms; no density estimate is fitted."""

    roles = {"value": value, "by": by, "facet": facet}
    _tabular(
        frame,
        recipe="distribution",
        roles=roles,
        numeric=("value",),
        categorical=("by", "facet"),
    )
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("distribution bins must be a positive integer")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("distribution threshold must be finite and non-negative, or None")

    panels = _panels(frame, facet)
    figure, axes = _canvas(ax, len(panels))
    for cell, (name, panel) in zip(axes, panels):
        for key, group in _grouped(panel, by):
            cell.hist(
                group[value],
                bins=bins,
                density=True,
                histtype="step",
                color=_series_color(key, by),
                label=key,
            )
        if threshold is not None:
            for edge in (-threshold, threshold):
                cell.axvline(edge, color="0.6", ls="--", lw=0.6)
        fmt(cell, x=value, y="density", title=name)
        _legend(cell)
    return figure


def relationship(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    fit: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot paired observations and optionally fit one straight line per group."""

    roles = {"x": x, "y": y, "by": by}
    _tabular(
        frame,
        recipe="relationship",
        roles=roles,
        numeric=("x", "y"),
        categorical=("by",),
    )
    if fit:
        for key, group in _grouped(frame, by):
            if len(group) > 2 and group[x].nunique() < 2:
                label = f" for group {key!r}" if by is not None else ""
                raise ValueError(f"relationship fit needs two distinct x values{label}")

    figure, (cell,) = _canvas(ax, 1)
    for key, group in _grouped(frame, by):
        series_color = _series_color(key, by)
        cell.scatter(
            group[x],
            group[y],
            s=6,
            alpha=0.5,
            edgecolors="none",
            color=series_color,
            label=key,
        )
        if fit and len(group) > 2:
            gradient, intercept = np.polyfit(group[x], group[y], 1)
            span = np.array([group[x].min(), group[x].max()])
            cell.plot(span, intercept + gradient * span, lw=1, color=series_color)
    fmt(cell, x=x, y=y)
    _legend(cell)
    return figure


def bars(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    facet: str | None = None,
    err: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot pre-aggregated values as dodged bars, with optional errors."""

    roles = {"x": x, "y": y, "by": by, "facet": facet, "err": err}
    _tabular(
        frame,
        recipe="bars",
        roles=roles,
        numeric=("y", "err"),
        categorical=("x", "by", "facet"),
        nullable_numeric=("err",),
        nonnegative=("err",),
        unique=("x", "by", "facet"),
    )
    panels = _panels(frame, facet)
    figure, axes = _canvas(ax, len(panels))
    for cell, (name, panel) in zip(axes, panels):
        categories = order(x, panel[x])
        groups = _grouped(panel, by)
        width = 0.8 / len(groups)
        for offset, (key, group) in zip(_dodge(len(groups)), groups):
            aligned = group.set_index(x).reindex(categories)
            _columns(
                cell,
                np.arange(len(categories)) + offset,
                aligned[y].to_numpy(),
                width,
                aligned[err].to_numpy() if err else None,
                color_value=_series_color(key, by),
                label=key,
            )
        cell.set_xticks(range(len(categories)), categories)
        fmt(cell, x=x, y=y, title=name)
        _legend(cell)
    return figure


def heatmap(
    matrix: Any,
    *,
    reward: float | None = None,
    cmap: str = "gray_r",
    ax: Axes | None = None,
) -> Figure:
    """Plot a rows-by-columns matrix with row zero at the bottom."""

    values = _matrix(matrix, recipe="heatmap")
    if reward is not None and not np.isfinite(reward):
        raise ValueError("heatmap reward must be a finite column coordinate or None")
    figure, (cell,) = _canvas(ax, 1)
    cell.imshow(values, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")
    if reward is not None:
        cell.axvline(reward, color="b", ls="--", lw=0.8)
    fmt(cell, x="position", y="trial")
    return figure


def density(
    image: Any,
    *,
    outlines: Iterable[Any] | None = None,
    cmap: str = "magma_r",
    vmax: float | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a y-by-x density image and outlines expressed as ``(x, y)`` points."""

    values = _matrix(image, recipe="density")
    if vmax is not None and (not np.isfinite(vmax) or vmax <= 0):
        raise ValueError("density vmax must be finite and positive, or None")
    edges: list[np.ndarray] = []
    if outlines is not None:
        for index, outline in enumerate(outlines):
            edge = np.asarray(outline, dtype=float)
            if edge.ndim != 2 or edge.shape[1] != 2 or len(edge) < 2:
                raise ValueError(f"density outline {index} must have shape (N, 2) with N >= 2")
            if not np.isfinite(edge).all():
                raise ValueError(f"density outline {index} contains non-finite coordinates")
            edges.append(edge)

    figure, (cell,) = _canvas(ax, 1)
    cell.imshow(
        values,
        origin="lower",
        cmap=cmap,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    for edge in edges:
        cell.plot(edge[:, 0], edge[:, 1], color="k", lw=0.5, alpha=0.4)
    cell.axis("off")
    return figure


def raster(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    by: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot one row per event; ``y`` commonly identifies the trial."""

    roles = {"x": x, "y": y, "by": by}
    _tabular(
        frame,
        recipe="raster",
        roles=roles,
        numeric=("x", "y"),
        categorical=("by",),
    )
    figure, (cell,) = _canvas(ax, 1)
    for key, group in _grouped(frame, by):
        cell.scatter(
            group[x],
            group[y],
            s=2,
            color=_series_color(key, by),
            label=key,
        )
    fmt(cell, x=x, y=y, invert_y=True)
    _legend(cell)
    return figure


def evolution(
    matrix: Any,
    *,
    kind: str = "histogram",
    fps: float = 12,
    threshold: float | None = 0.3,
    bins: int = 60,
) -> FuncAnimation:
    """Animate an observations-by-windows matrix; NaNs represent missing data."""

    values = _matrix(matrix, recipe="evolution")
    if kind not in {"histogram", "curve"}:
        raise ValueError("evolution kind must be 'histogram' or 'curve'")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("evolution fps must be finite and positive")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("evolution bins must be a positive integer")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("evolution threshold must be finite and non-negative, or None")

    windows = values.shape[1]
    figure, cell = plt.subplots(figsize=(5.0, 3.2))
    if kind == "curve":
        counts = np.sum(np.isfinite(values), axis=0)
        totals = np.nansum(values, axis=0)
        trace = np.divide(totals, counts, out=np.full(windows, np.nan), where=counts > 0)
        finite_trace = trace[np.isfinite(trace)]
        low, high = float(finite_trace.min()), float(finite_trace.max())
        if low == high:
            padding = max(abs(low) * 0.05, 0.5)
            low, high = low - padding, high + padding

        def draw(frame: int) -> None:
            cell.clear()
            cell.plot(np.arange(frame + 1), trace[: frame + 1], color="k", lw=1)
            cell.set_xlim(0, max(windows - 1, 1))
            cell.set_ylim(low, high)
            fmt(cell, x="window", y="mean", title=f"window {frame + 1}/{windows}")

    else:
        finite_values = values[np.isfinite(values)]
        low, high = np.percentile(finite_values, [0.5, 99.5])
        if low == high:
            padding = max(abs(float(low)) * 0.05, 0.5)
            low, high = low - padding, high + padding

        def draw(frame: int) -> None:
            cell.clear()
            column = values[:, frame]
            finite_column = column[np.isfinite(column)]
            if len(finite_column):
                cell.hist(
                    finite_column,
                    bins=bins,
                    range=(low, high),
                    density=True,
                    color="#4A90E2",
                    alpha=0.5,
                )
            else:
                cell.text(0.5, 0.5, "no finite observations", ha="center", va="center", transform=cell.transAxes)
            if threshold is not None:
                for edge in (-threshold, threshold):
                    cell.axvline(edge, color="0.6", ls="--", lw=0.6)
            cell.set_xlim(low, high)
            fmt(cell, x="value", y="density", title=f"window {frame + 1}/{windows}")

    return FuncAnimation(figure, draw, frames=windows, interval=1000 / fps)


def save(figure: Figure, path: str | Path, *, target: str = "screen") -> Path:
    """Save a figure at the declared screen or print resolution."""

    if target not in {"screen", "print"}:
        raise ValueError("save target must be 'screen' or 'print'")
    output = Path(path)
    figure.savefig(output, dpi=500 if target == "print" else 150, bbox_inches="tight")
    return output


RECIPES = (
    trajectory,
    comparison,
    slope,
    distribution,
    relationship,
    bars,
    heatmap,
    density,
    raster,
    evolution,
)


def guide(recipe: str | None = None) -> pd.DataFrame:
    """Return an inspectable table of recipe data contracts and signatures."""

    if recipe is not None and recipe not in CONTRACTS:
        raise KeyError(f"Unknown recipe {recipe!r}; choose from {list(CONTRACTS)}")
    selected = [recipe] if recipe is not None else list(CONTRACTS)
    rows = []
    for name in selected:
        rows.append(
            {
                "recipe": name,
                **CONTRACTS[name],
                "signature": str(inspect.signature(globals()[name])),
            }
        )
    return pd.DataFrame(rows).set_index("recipe")


__all__ = [
    "AREA",
    "COHORT",
    "CONTRACTS",
    "MOMENT",
    "ORDER",
    "RECIPES",
    "SEMANTIC",
    "STIMULUS",
    "VIEWS",
    "attach",
    "attach_views",
    "bars",
    "color",
    "comparison",
    "density",
    "distribution",
    "evolution",
    "fmt",
    "guide",
    "heatmap",
    "order",
    "raster",
    "relationship",
    "save",
    "slope",
    "trajectory",
    "use",
    "view_status",
]
