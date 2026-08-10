"""Presentation-ready plots for the boundary population forecaster.

The plotting API deliberately consumes only the two standard result tables:

``predictions``
    ``split, area, mouse, horizon, model, metric, observed, predicted,``
    ``absolute_error``.  Per-mouse forecast slides additionally use
    ``scaled_squared_error``.

``summary``
    ``split, area, horizon, model, mean_nrmse,``
    ``improvement_vs_persistence_pct``

Model selection is leakage-safe.  Callers may provide ``selected_model``;
otherwise the literal model ``selected_nested`` is preferred, followed by the
best model on the unsupervised development split.  Supervised outcomes are
never used to select the displayed model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = frozenset(
    {
        "split",
        "area",
        "mouse",
        "horizon",
        "model",
        "metric",
        "observed",
        "predicted",
        "absolute_error",
    }
)
SUMMARY_COLUMNS = frozenset(
    {
        "split",
        "area",
        "horizon",
        "model",
        "mean_nrmse",
        "improvement_vs_persistence_pct",
    }
)

METRIC_ORDER = (
    "q05",
    "median",
    "q95",
    "sd_dprime",
    "frac_leaf_selective",
    "frac_circle_selective",
)
METRIC_LABELS = {
    "q05": "Lower tail (q05 d′)",
    "median": "Median d′",
    "q50": "Median d′",
    "q95": "Upper tail (q95 d′)",
    "sd_dprime": "d′ spread (SD)",
    "std_dprime": "d′ spread (SD)",
    "frac_leaf_selective": "Leaf-selective fraction",
    "leaf_fraction": "Leaf-selective fraction",
    "p_leaf": "Leaf-selective fraction",
    "frac_circle_selective": "Circle-selective fraction",
    "circle_fraction": "Circle-selective fraction",
    "p_circle": "Circle-selective fraction",
}
AREA_COLORS = {
    "mHV": "#15847B",
    "aHV": "#E76F51",
}
MODEL_COLORS = {
    "last_before_persistence": "#A7ADB5",
    "persistence": "#A7ADB5",
    "a_only": "#4D7EA8",
    "b_only": "#E9B949",
    "a_plus_b": "#234E52",
    "selected_nested": "#234E52",
}
_FALLBACK_COLORS = (
    "#4D7EA8",
    "#E9B949",
    "#7A5195",
    "#EF5675",
    "#234E52",
    "#2A9D8F",
    "#F28E2B",
)
_HORIZON_MARKERS = ("o", "s", "^", "D", "P", "X")
_PERSISTENCE_NAMES = frozenset(
    {"last_before_persistence", "persistence", "persistence_baseline"}
)
DIRECT_FORECASTER_LABEL = "Direct boundary forecaster"


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    table_name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{table_name} is empty")


def _clean_label(value: object) -> str:
    return (
        str(value)
        .replace("a_plus_b", "A + B")
        .replace("b_only", "B only")
        .replace("a_only", "A only")
        .replace("last_before_persistence", "Persistence")
        .replace("selected_nested", "Direct boundary forecaster")
        .replace("__", " · ")
        .replace("_", " ")
    )


def _split_label(value: object) -> str:
    mapping = {
        "unsupervised_lomo": "Unsupervised nested LOMO",
        "supervised_test": "Frozen supervised transfer",
        "supervised": "Frozen supervised transfer",
    }
    return mapping.get(str(value), _clean_label(value).title())


def _natural_key(value: object) -> tuple[int, float | str]:
    """Sort numeric horizons before arbitrary string horizons."""

    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _horizon_label(value: object, *, position: int | None = None) -> str:
    """Return a presentation label while supporting zero- and one-based input."""

    if position is not None:
        return f"After W{position + 1}"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return _clean_label(value)
    return f"After W{number + 1}"


def _metric_order(values: Iterable[object], *, limit: int = 6) -> list[str]:
    available = list(dict.fromkeys(str(value) for value in values))
    ordered = [metric for metric in METRIC_ORDER if metric in available]
    ordered.extend(sorted(set(available).difference(ordered)))
    return ordered[:limit]


def _area_color(area: object, index: int) -> str:
    return AREA_COLORS.get(
        str(area),
        _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)],
    )


def _model_color(model: str, index: int, *, selected_model: str | None) -> str:
    if selected_model is not None and model == selected_model:
        return "#234E52"
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    for prefix, color in MODEL_COLORS.items():
        if model.startswith(prefix):
            return color
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return path


def resolve_selected_model(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    selected_model: str | None = None,
    development_split: str = "unsupervised_lomo",
) -> str:
    """Resolve the display model without consulting supervised outcomes."""

    _require_columns(predictions, PREDICTION_COLUMNS, table_name="predictions")
    _require_columns(summary, SUMMARY_COLUMNS, table_name="summary")
    prediction_models = set(predictions["model"].dropna().astype(str))
    if selected_model is not None:
        if selected_model not in prediction_models:
            raise ValueError(
                f"selected_model={selected_model!r} is absent from predictions"
            )
        return selected_model
    if "selected_nested" in prediction_models:
        return "selected_nested"

    development = summary.loc[
        summary["split"].astype(str).eq(development_split)
        & summary["model"].astype(str).isin(prediction_models)
    ].copy()
    development = development.loc[
        ~development["model"].astype(str).isin(_PERSISTENCE_NAMES)
    ]
    development["mean_nrmse"] = pd.to_numeric(
        development["mean_nrmse"], errors="coerce"
    )
    development = development.loc[np.isfinite(development["mean_nrmse"])]
    if development.empty:
        raise ValueError(
            "selected_model is required: no 'selected_nested' prediction and "
            f"no selectable rows on development split {development_split!r}"
        )
    ranking = (
        development.groupby("model", as_index=False, sort=True)
        .agg(mean_nrmse=("mean_nrmse", "mean"))
        .sort_values(["mean_nrmse", "model"], kind="stable")
    )
    return str(ranking.iloc[0]["model"])


def _ordered_models(
    summary: pd.DataFrame,
    *,
    selected_model: str | None,
    models: Sequence[str] | None,
) -> list[str]:
    available = list(dict.fromkeys(summary["model"].dropna().astype(str)))
    if models is not None:
        requested = list(dict.fromkeys(str(model) for model in models))
        missing = sorted(set(requested).difference(available))
        if missing:
            raise ValueError(
                "requested comparison models are absent from summary: "
                + ", ".join(missing)
            )
        available = requested
    persistence = [
        model for model in available if model in _PERSISTENCE_NAMES
    ]
    middle = [
        model
        for model in available
        if model not in _PERSISTENCE_NAMES and model != selected_model
    ]
    selected = (
        [selected_model]
        if selected_model is not None and selected_model in available
        else []
    )
    return persistence + middle + selected


def plot_model_comparison(
    summary: pd.DataFrame,
    path: str | Path,
    *,
    selected_model: str | None = None,
    models: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> Path:
    """Plot area- and horizon-resolved NRMSE for each candidate model."""

    _require_columns(summary, SUMMARY_COLUMNS, table_name="summary")
    data = summary.copy()
    data["model"] = data["model"].astype(str)
    data["split"] = data["split"].astype(str)
    data["mean_nrmse"] = pd.to_numeric(data["mean_nrmse"], errors="coerce")
    data["improvement_vs_persistence_pct"] = pd.to_numeric(
        data["improvement_vs_persistence_pct"], errors="coerce"
    )
    if splits is None:
        preferred = ["unsupervised_lomo", "supervised_test"]
        present = list(dict.fromkeys(data["split"]))
        split_order = [split for split in preferred if split in present]
        split_order.extend(split for split in present if split not in split_order)
    else:
        split_order = [str(split) for split in splits]
    data = data.loc[data["split"].isin(split_order)]
    if data.empty:
        raise ValueError("summary has no rows for the requested splits")

    areas = list(dict.fromkeys(data["area"].astype(str)))
    horizons = sorted(data["horizon"].dropna().unique(), key=_natural_key)
    model_order = _ordered_models(
        data,
        selected_model=selected_model,
        models=models,
    )
    if not model_order:
        raise ValueError("summary has no models to compare")

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#28323C",
            "text.color": "#17212B",
            "axes.edgecolor": "#9AA2AA",
        }
    ):
        figure, axes = plt.subplots(
            len(split_order),
            len(areas),
            figsize=(
                max(7.0, 5.3 * len(areas)),
                max(4.2, 3.8 * len(split_order)),
            ),
            squeeze=False,
            sharey=True,
        )
        x = np.arange(len(horizons), dtype=float)
        width = min(0.18, 0.82 / max(len(model_order), 1))
        for row, split in enumerate(split_order):
            for column, area in enumerate(areas):
                ax = axes[row, column]
                current = data.loc[
                    data["split"].eq(split)
                    & data["area"].astype(str).eq(area)
                ]
                for model_index, model in enumerate(model_order):
                    values: list[float] = []
                    improvements: list[float] = []
                    for horizon in horizons:
                        rows = current.loc[
                            current["horizon"].eq(horizon)
                            & current["model"].eq(model)
                        ]
                        values.append(float(rows["mean_nrmse"].mean()))
                        improvements.append(
                            float(
                                rows[
                                    "improvement_vs_persistence_pct"
                                ].mean()
                            )
                        )
                    offset = (
                        model_index - (len(model_order) - 1) / 2.0
                    ) * width
                    bars = ax.bar(
                        x + offset,
                        values,
                        width=width * 0.92,
                        color=_model_color(
                            model,
                            model_index,
                            selected_model=selected_model,
                        ),
                        edgecolor="white",
                        linewidth=0.45,
                        label=_clean_label(model),
                        zorder=3,
                    )
                    if model == selected_model:
                        for bar, improvement in zip(
                            bars, improvements, strict=True
                        ):
                            if not np.isfinite(improvement):
                                continue
                            ax.annotate(
                                f"{improvement:+.0f}%",
                                (
                                    bar.get_x() + bar.get_width() / 2,
                                    bar.get_height(),
                                ),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=7.5,
                                weight="semibold",
                                color="#234E52",
                            )
                ax.set_xticks(
                    x,
                    [
                        _horizon_label(horizon, position=index)
                        for index, horizon in enumerate(horizons)
                    ],
                )
                ax.set_title(f"{area} · {_split_label(split)}")
                ax.set_ylabel("Mouse-balanced NRMSE")
                ax.grid(axis="y", alpha=0.22, linewidth=0.7, zorder=0)
                ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        figure.suptitle(
            "Boundary forecaster: direct prediction of the first "
            "post-learning windows",
            fontsize=15,
            weight="bold",
            y=1.06,
        )
        figure.tight_layout()
        return _save(figure, Path(path))


def plot_predicted_vs_actual(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    selected_model: str,
    split: str = "supervised_test",
) -> Path:
    """Draw a six-metric predicted-versus-actual grid."""

    _require_columns(predictions, PREDICTION_COLUMNS, table_name="predictions")
    data = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["model"].astype(str).eq(selected_model)
    ].copy()
    if data.empty:
        raise ValueError(
            f"no predictions for split={split!r}, model={selected_model!r}"
        )
    for column in ("observed", "predicted", "absolute_error"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    metrics = _metric_order(data["metric"])
    if not metrics:
        raise ValueError("selected predictions contain no metrics")
    areas = list(dict.fromkeys(data["area"].astype(str)))
    horizons = sorted(data["horizon"].dropna().unique(), key=_natural_key)

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#9AA2AA",
        }
    ):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(14.4, 8.6),
            squeeze=False,
        )
        for metric_index, ax in enumerate(axes.flat):
            if metric_index >= len(metrics):
                ax.axis("off")
                continue
            metric = metrics[metric_index]
            current = data.loc[data["metric"].astype(str).eq(metric)]
            for area_index, area in enumerate(areas):
                for horizon_index, horizon in enumerate(horizons):
                    points = current.loc[
                        current["area"].astype(str).eq(area)
                        & current["horizon"].eq(horizon)
                    ]
                    points = points.loc[
                        np.isfinite(points["observed"])
                        & np.isfinite(points["predicted"])
                    ]
                    ax.scatter(
                        points["observed"],
                        points["predicted"],
                        s=50,
                        marker=_HORIZON_MARKERS[
                            horizon_index % len(_HORIZON_MARKERS)
                        ],
                        color=_area_color(area, area_index),
                        edgecolor="white",
                        linewidth=0.7,
                        alpha=0.9,
                        zorder=3,
                    )
            finite = current.loc[
                np.isfinite(current["observed"])
                & np.isfinite(current["predicted"])
            ]
            if finite.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No finite values",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#66717D",
                )
            else:
                values = np.concatenate(
                    [
                        finite["observed"].to_numpy(dtype=float),
                        finite["predicted"].to_numpy(dtype=float),
                    ]
                )
                lower = float(np.min(values))
                upper = float(np.max(values))
                span = max(upper - lower, 1e-6)
                padding = max(0.07 * span, 0.015 * max(abs(lower), abs(upper), 1))
                limits = (lower - padding, upper + padding)
                ax.plot(
                    limits,
                    limits,
                    linestyle="--",
                    color="#7B8794",
                    linewidth=1.1,
                    zorder=1,
                )
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                mae = float(
                    np.nanmean(
                        np.abs(
                            finite["observed"].to_numpy(dtype=float)
                            - finite["predicted"].to_numpy(dtype=float)
                        )
                    )
                )
                observed = finite["observed"].to_numpy(dtype=float)
                predicted = finite["predicted"].to_numpy(dtype=float)
                correlation = (
                    float(np.corrcoef(observed, predicted)[0, 1])
                    if len(finite) > 1
                    and np.std(observed) > 0
                    and np.std(predicted) > 0
                    else np.nan
                )
                annotation = f"MAE {mae:.3g}"
                if np.isfinite(correlation):
                    annotation += f"\nr {correlation:.2f}"
                ax.text(
                    0.04,
                    0.96,
                    annotation,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8.5,
                    color="#46515C",
                )
                ax.set_aspect("equal", adjustable="box")
            ax.set_title(METRIC_LABELS.get(metric, _clean_label(metric)))
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.18, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)

        legend_handles: list[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=_area_color(area, index),
                markeredgecolor="white",
                markersize=8,
                label=area,
            )
            for index, area in enumerate(areas)
        ]
        legend_handles.extend(
            Line2D(
                [0],
                [0],
                marker=_HORIZON_MARKERS[index % len(_HORIZON_MARKERS)],
                linestyle="none",
                color="#4D5965",
                markersize=7,
                label=_horizon_label(horizon, position=index),
            )
            for index, horizon in enumerate(horizons)
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=max(1, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        figure.suptitle(
            f"Frozen supervised transfer · {_clean_label(selected_model)}",
            fontsize=15,
            weight="bold",
            y=1.055,
        )
        figure.tight_layout()
        return _save(figure, Path(path))


def plot_supervised_trajectory_overview(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    selected_model: str,
    split: str = "supervised_test",
) -> Path:
    """Show observed and predicted horizon trajectories for every mouse."""

    _require_columns(predictions, PREDICTION_COLUMNS, table_name="predictions")
    data = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["model"].astype(str).eq(selected_model)
    ].copy()
    if data.empty:
        raise ValueError(
            f"no predictions for split={split!r}, model={selected_model!r}"
        )
    data["observed"] = pd.to_numeric(data["observed"], errors="coerce")
    data["predicted"] = pd.to_numeric(data["predicted"], errors="coerce")
    metrics = _metric_order(data["metric"])
    horizons = sorted(data["horizon"].dropna().unique(), key=_natural_key)
    horizon_x = {horizon: index for index, horizon in enumerate(horizons)}
    areas = list(dict.fromkeys(data["area"].astype(str)))

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#9AA2AA",
        }
    ):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(14.4, 8.4),
            squeeze=False,
        )
        for metric_index, ax in enumerate(axes.flat):
            if metric_index >= len(metrics):
                ax.axis("off")
                continue
            metric = metrics[metric_index]
            current = data.loc[data["metric"].astype(str).eq(metric)]
            for area_index, area in enumerate(areas):
                area_data = current.loc[
                    current["area"].astype(str).eq(area)
                ]
                color = _area_color(area, area_index)
                for _, mouse_data in area_data.groupby("mouse", sort=False):
                    mouse_data = (
                        mouse_data.groupby("horizon", as_index=False)
                        .agg(
                            observed=("observed", "mean"),
                            predicted=("predicted", "mean"),
                        )
                        .sort_values(
                            "horizon",
                            key=lambda values: values.map(horizon_x),
                        )
                    )
                    x_values = [
                        horizon_x[horizon]
                        for horizon in mouse_data["horizon"]
                    ]
                    ax.plot(
                        x_values,
                        mouse_data["observed"],
                        color=color,
                        linewidth=0.9,
                        alpha=0.22,
                        marker="o",
                        markersize=3,
                    )
                    ax.plot(
                        x_values,
                        mouse_data["predicted"],
                        color=color,
                        linewidth=0.9,
                        alpha=0.22,
                        linestyle="--",
                        marker="x",
                        markersize=3,
                    )
                mean_trajectory = (
                    area_data.groupby("horizon", as_index=False)
                    .agg(
                        observed=("observed", "mean"),
                        predicted=("predicted", "mean"),
                    )
                    .sort_values(
                        "horizon",
                        key=lambda values: values.map(horizon_x),
                    )
                )
                mean_x = [
                    horizon_x[horizon]
                    for horizon in mean_trajectory["horizon"]
                ]
                ax.plot(
                    mean_x,
                    mean_trajectory["observed"],
                    color=color,
                    linewidth=2.6,
                    marker="o",
                    markersize=6,
                    zorder=4,
                )
                ax.plot(
                    mean_x,
                    mean_trajectory["predicted"],
                    color=color,
                    linewidth=2.6,
                    linestyle="--",
                    marker="x",
                    markersize=6,
                    zorder=4,
                )
            ax.set_xticks(
                range(len(horizons)),
                [
                    _horizon_label(horizon, position=index)
                    for index, horizon in enumerate(horizons)
                ],
            )
            ax.set_title(METRIC_LABELS.get(metric, _clean_label(metric)))
            ax.set_ylabel("Population statistic")
            ax.grid(alpha=0.2, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=_area_color(area, index),
                linewidth=3,
                label=area,
            )
            for index, area in enumerate(areas)
        ]
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#3E4A56",
                    linewidth=2,
                    marker="o",
                    label="Observed",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#3E4A56",
                    linewidth=2,
                    linestyle="--",
                    marker="x",
                    label="Predicted",
                ),
            ]
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=max(1, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        figure.suptitle(
            "Supervised mice: post-learning population trajectories "
            "(thin = mouse, thick = cohort mean)",
            fontsize=15,
            weight="bold",
            y=1.055,
        )
        figure.tight_layout()
        return _save(figure, Path(path))


def _safe_filename_component(value: object) -> str:
    """Return a stable filename component without changing common mouse IDs."""

    cleaned = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in str(value)
    ).strip("_")
    return cleaned or "unknown_mouse"


def mouse_area_two_horizon_improvement(
    predictions: pd.DataFrame,
    *,
    mouse: str,
    area: str,
    selected_model: str,
    split: str = "supervised_test",
    persistence_models: Sequence[str] = (
        "persistence",
        "last_before_persistence",
        "persistence_baseline",
    ),
) -> float:
    """Return mean two-horizon NRMSE improvement over persistence.

    NRMSE is reconstructed from the metric-level ``scaled_squared_error``:
    first take the square root of the mean scaled squared error within each
    horizon, then average the two horizon NRMSE values.  The returned value is
    ``100 * (persistence - selected) / persistence``.
    """

    required = set(PREDICTION_COLUMNS).union({"scaled_squared_error"})
    _require_columns(predictions, required, table_name="predictions")
    current = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["mouse"].astype(str).eq(str(mouse))
        & predictions["area"].astype(str).eq(str(area))
    ].copy()
    current["scaled_squared_error"] = pd.to_numeric(
        current["scaled_squared_error"], errors="coerce"
    )
    current = current.loc[np.isfinite(current["scaled_squared_error"])]
    if current.empty:
        return float("nan")

    def mean_horizon_nrmse(model_names: set[str]) -> float:
        rows = current.loc[current["model"].astype(str).isin(model_names)]
        if rows.empty:
            return float("nan")
        horizon_scores = (
            rows.groupby("horizon", as_index=False)
            .agg(
                mean_scaled_squared_error=(
                    "scaled_squared_error",
                    "mean",
                )
            )
            .sort_values("horizon", key=lambda values: values.map(_natural_key))
            .head(2)
        )
        if len(horizon_scores) < 2:
            return float("nan")
        return float(
            np.sqrt(
                horizon_scores["mean_scaled_squared_error"].to_numpy(
                    dtype=float
                )
            ).mean()
        )

    selected_nrmse = mean_horizon_nrmse({str(selected_model)})
    persistence_nrmse = mean_horizon_nrmse(
        {str(model) for model in persistence_models}
    )
    if (
        not np.isfinite(selected_nrmse)
        or not np.isfinite(persistence_nrmse)
        or persistence_nrmse <= 0
    ):
        return float("nan")
    return float(
        100.0
        * (persistence_nrmse - selected_nrmse)
        / persistence_nrmse
    )


def _improvement_text(area: str, improvement: float) -> str:
    if not np.isfinite(improvement):
        return f"{area}: n/a"
    return f"{area}: {improvement:+.1f}%"


def plot_supervised_mouse_forecast(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    mouse: str,
    selected_model: str,
    split: str = "supervised_test",
    persistence_models: Sequence[str] = (
        "persistence",
        "last_before_persistence",
        "persistence_baseline",
    ),
) -> Path:
    """Create one six-panel direct boundary forecast slide for a mouse."""

    required = set(PREDICTION_COLUMNS).union({"scaled_squared_error"})
    _require_columns(predictions, required, table_name="predictions")
    data = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["mouse"].astype(str).eq(str(mouse))
        & predictions["model"].astype(str).eq(selected_model)
    ].copy()
    if data.empty:
        raise ValueError(
            f"no selected predictions for supervised mouse {mouse!r}"
        )
    data["observed"] = pd.to_numeric(data["observed"], errors="coerce")
    data["predicted"] = pd.to_numeric(data["predicted"], errors="coerce")
    metrics = _metric_order(data["metric"])
    if not metrics:
        raise ValueError(f"mouse {mouse!r} has no forecast metrics")
    horizons = sorted(data["horizon"].dropna().unique(), key=_natural_key)[:2]
    if len(horizons) < 2:
        raise ValueError(
            f"mouse {mouse!r} does not have both forecast horizons"
        )
    horizon_x = {horizon: index for index, horizon in enumerate(horizons)}
    areas = list(dict.fromkeys(data["area"].astype(str)))
    improvements = {
        area: mouse_area_two_horizon_improvement(
            predictions,
            mouse=str(mouse),
            area=area,
            selected_model=selected_model,
            split=split,
            persistence_models=persistence_models,
        )
        for area in areas
    }

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#9AA2AA",
        }
    ):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(14.4, 8.4),
            squeeze=False,
        )
        for metric_index, ax in enumerate(axes.flat):
            if metric_index >= len(metrics):
                ax.axis("off")
                continue
            metric = metrics[metric_index]
            current = data.loc[data["metric"].astype(str).eq(metric)]
            for area_index, area in enumerate(areas):
                area_data = (
                    current.loc[current["area"].astype(str).eq(area)]
                    .groupby("horizon", as_index=False)
                    .agg(
                        observed=("observed", "mean"),
                        predicted=("predicted", "mean"),
                    )
                )
                area_data = area_data.loc[
                    area_data["horizon"].isin(horizons)
                ].sort_values(
                    "horizon",
                    key=lambda values: values.map(horizon_x),
                )
                x_values = [
                    horizon_x[horizon] for horizon in area_data["horizon"]
                ]
                color = _area_color(area, area_index)
                ax.plot(
                    x_values,
                    area_data["observed"],
                    color=color,
                    linewidth=2.6,
                    marker="o",
                    markersize=7,
                    zorder=4,
                )
                ax.plot(
                    x_values,
                    area_data["predicted"],
                    color=color,
                    linewidth=2.6,
                    linestyle="--",
                    marker="X",
                    markersize=7,
                    zorder=4,
                )
                for x_value, observed, predicted in zip(
                    x_values,
                    area_data["observed"],
                    area_data["predicted"],
                    strict=True,
                ):
                    ax.plot(
                        [x_value, x_value],
                        [observed, predicted],
                        color=color,
                        alpha=0.25,
                        linewidth=1,
                        zorder=2,
                    )
            ax.set_xticks(
                range(len(horizons)),
                [
                    _horizon_label(horizon, position=index)
                    for index, horizon in enumerate(horizons)
                ],
            )
            ax.set_title(METRIC_LABELS.get(metric, _clean_label(metric)))
            ax.set_ylabel("Population statistic")
            ax.margins(x=0.18, y=0.16)
            ax.grid(alpha=0.2, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)

        area_handles = [
            Line2D(
                [0],
                [0],
                color=_area_color(area, index),
                linewidth=3,
                label=area,
            )
            for index, area in enumerate(areas)
        ]
        style_handles = [
            Line2D(
                [0],
                [0],
                color="#3E4A56",
                linewidth=2.4,
                marker="o",
                label="Actual",
            ),
            Line2D(
                [0],
                [0],
                color="#3E4A56",
                linewidth=2.4,
                linestyle="--",
                marker="X",
                label=DIRECT_FORECASTER_LABEL,
            ),
        ]
        figure.legend(
            handles=area_handles + style_handles,
            loc="upper center",
            ncol=max(1, len(area_handles) + len(style_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        improvement_line = "  ·  ".join(
            _improvement_text(area, improvements[area]) for area in areas
        )
        figure.suptitle(
            f"Supervised mouse {mouse} · {DIRECT_FORECASTER_LABEL}\n"
            "Mean W1/W2 NRMSE improvement vs persistence · "
            f"{improvement_line}",
            fontsize=15,
            weight="bold",
            y=1.075,
        )
        figure.tight_layout()
        return _save(figure, Path(path))


def plot_ineligible_mouse_panel(
    path: str | Path,
    *,
    mouse: str = "TX109",
) -> Path:
    """Create a clean input-eligibility panel without implying model failure."""

    with plt.rc_context({"font.size": 11, "text.color": "#17212B"}):
        figure, ax = plt.subplots(figsize=(14.4, 8.4))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
            0.5,
            0.70,
            f"Supervised mouse {mouse}",
            ha="center",
            va="center",
            fontsize=25,
            weight="bold",
            color="#234E52",
        )
        ax.text(
            0.5,
            0.56,
            "Not eligible for a direct boundary forecast",
            ha="center",
            va="center",
            fontsize=19,
            weight="semibold",
            color="#3E4A56",
        )
        ax.text(
            0.5,
            0.42,
            "No forecast is shown because two valid pre-learning W20 "
            "inputs are unavailable.",
            ha="center",
            va="center",
            fontsize=14,
            color="#566370",
        )
        ax.text(
            0.5,
            0.34,
            "This is an input-eligibility constraint, not a model failure.",
            ha="center",
            va="center",
            fontsize=12.5,
            color="#6A7682",
        )
        ax.text(
            0.5,
            0.17,
            DIRECT_FORECASTER_LABEL,
            ha="center",
            va="center",
            fontsize=12,
            weight="semibold",
            color="#15847B",
        )
        return _save(figure, Path(path))


def plot_supervised_mouse_forecasts(
    predictions: pd.DataFrame,
    output: str | Path,
    *,
    selected_model: str,
    split: str = "supervised_test",
    persistence_models: Sequence[str] = (
        "persistence",
        "last_before_persistence",
        "persistence_baseline",
    ),
    explicit_ineligible_mouse: str | None = "TX109",
) -> dict[str, Path]:
    """Create one slide per eligible mouse plus the explicit TX109 panel."""

    required = set(PREDICTION_COLUMNS).union({"scaled_squared_error"})
    _require_columns(predictions, required, table_name="predictions")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    selected = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["model"].astype(str).eq(selected_model)
    ]
    eligible_mice = sorted(
        selected["mouse"].dropna().astype(str).unique(),
        key=_natural_key,
    )
    paths: dict[str, Path] = {}
    for mouse in eligible_mice:
        safe_mouse = _safe_filename_component(mouse)
        path = (
            output_path
            / f"boundary_population_supervised_{safe_mouse}.png"
        )
        paths[f"supervised_mouse_{safe_mouse}"] = (
            plot_supervised_mouse_forecast(
                predictions,
                path,
                mouse=mouse,
                selected_model=selected_model,
                split=split,
                persistence_models=persistence_models,
            )
        )
    if (
        explicit_ineligible_mouse is not None
        and str(explicit_ineligible_mouse) not in set(eligible_mice)
    ):
        safe_mouse = _safe_filename_component(explicit_ineligible_mouse)
        path = (
            output_path
            / f"boundary_population_supervised_{safe_mouse}.png"
        )
        paths[f"supervised_mouse_{safe_mouse}"] = (
            plot_ineligible_mouse_panel(
                path,
                mouse=str(explicit_ineligible_mouse),
            )
        )
    return paths


def create_all_plots(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output: str | Path,
    *,
    selected_model: str | None = None,
    development_split: str = "unsupervised_lomo",
    supervised_split: str = "supervised_test",
    comparison_models: Sequence[str] | None = None,
    explicit_ineligible_mouse: str | None = "TX109",
) -> dict[str, Path]:
    """Create the model-comparison, calibration, and mouse-overview figures."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_model = resolve_selected_model(
        predictions,
        summary,
        selected_model=selected_model,
        development_split=development_split,
    )
    paths = {
        "model_comparison": (
            output_path / "boundary_population_model_comparison.png"
        ),
        "predicted_vs_actual": (
            output_path / "boundary_population_predicted_vs_actual.png"
        ),
        "supervised_trajectories": (
            output_path / "boundary_population_supervised_trajectories.png"
        ),
    }
    plot_model_comparison(
        summary,
        paths["model_comparison"],
        selected_model=resolved_model,
        models=comparison_models,
    )
    plot_predicted_vs_actual(
        predictions,
        paths["predicted_vs_actual"],
        selected_model=resolved_model,
        split=supervised_split,
    )
    plot_supervised_trajectory_overview(
        predictions,
        paths["supervised_trajectories"],
        selected_model=resolved_model,
        split=supervised_split,
    )
    paths.update(
        plot_supervised_mouse_forecasts(
            predictions,
            output_path,
            selected_model=resolved_model,
            split=supervised_split,
            explicit_ineligible_mouse=explicit_ineligible_mouse,
        )
    )
    return paths


__all__ = [
    "AREA_COLORS",
    "METRIC_LABELS",
    "METRIC_ORDER",
    "MODEL_COLORS",
    "PREDICTION_COLUMNS",
    "SUMMARY_COLUMNS",
    "create_all_plots",
    "mouse_area_two_horizon_improvement",
    "plot_ineligible_mouse_panel",
    "plot_model_comparison",
    "plot_predicted_vs_actual",
    "plot_supervised_mouse_forecast",
    "plot_supervised_mouse_forecasts",
    "plot_supervised_trajectory_overview",
    "resolve_selected_model",
]
