#!/usr/bin/env python3
"""Summarize direct-boundary forecasts in interpretable original units.

The main experiment's NRMSE combines six differently scaled outputs.  This
companion analysis keeps the units explicit:

* q05, median, and q95 are pooled as three d-prime location statistics;
* d-prime SD can optionally be included as a fourth d-prime-unit statistic;
* leaf- and circle-selective fractions are summarized separately.

The unit of evaluation is one predicted area-level statistic for one mouse,
area, and forecast horizon.  For example, the d-prime-location summary has
three value predictions per mouse/area/horizon.  ``within_0_5_pct`` is
therefore the percentage of those value predictions whose absolute error is
at most 0.5 d-prime; it is not a neuron-level accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = (
    WORKSPACE
    / "modeling/runs/objective_b"
)
SELECTED_MODEL = "selected_nested"
PERSISTENCE_MODEL = "persistence"
MODEL_LABELS = {
    SELECTED_MODEL: "Direct boundary forecaster",
    PERSISTENCE_MODEL: "Last-before W20 persistence",
}

DPRIME_LOCATION_METRICS = ("q05", "median", "q95")
DPRIME_SHAPE_METRICS = (
    "q05",
    "median",
    "q95",
    "sd_dprime",
)
FRACTION_METRICS = (
    "frac_leaf_selective",
    "frac_circle_selective",
)
METRIC_SETS = {
    "dprime_locations": DPRIME_LOCATION_METRICS,
    "dprime_locations_and_spread": DPRIME_SHAPE_METRICS,
    "selective_fractions": FRACTION_METRICS,
}


class ActualAccuracyError(ValueError):
    """Raised when saved predictions violate the accuracy-table contract."""


def _latest_run_directory(results_root: Path) -> Path:
    latest_path = results_root / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"latest-run pointer not found: {latest_path}")
    with latest_path.open(encoding="utf-8") as source:
        latest = json.load(source)
    run_directory = Path(str(latest["run_directory"]))
    if not run_directory.exists():
        raise FileNotFoundError(
            f"latest run directory does not exist: {run_directory}"
        )
    return run_directory


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    required = {
        "split",
        "model",
        "mouse",
        "area",
        "horizon",
        "metric",
        "observed",
        "predicted",
        "scale",
        "scaled_squared_error",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ActualAccuracyError(
            f"predictions are missing columns: {sorted(missing)}"
        )

    selected = predictions.loc[
        predictions["split"].astype(str).eq(split)
        & predictions["model"].isin(
            (SELECTED_MODEL, PERSISTENCE_MODEL)
        )
    ].copy()
    if selected.empty:
        raise ActualAccuracyError(
            f"no selected-model/persistence predictions for split {split!r}"
        )

    expected_metrics = set().union(*map(set, METRIC_SETS.values()))
    missing_metrics = expected_metrics - set(selected["metric"].astype(str))
    if missing_metrics:
        raise ActualAccuracyError(
            f"predictions do not contain metrics: {sorted(missing_metrics)}"
        )
    if selected[
        ["observed", "predicted", "scale", "scaled_squared_error"]
    ].isna().any().any():
        raise ActualAccuracyError("accuracy inputs contain missing values")
    if (selected["scale"].to_numpy(dtype=float) <= 0.0).any():
        raise ActualAccuracyError("all normalization scales must be positive")

    duplicate_key = [
        "model",
        "mouse",
        "area",
        "horizon",
        "metric",
    ]
    if selected.duplicated(duplicate_key).any():
        raise ActualAccuracyError(
            "predictions contain duplicate model/mouse/area/horizon/metric rows"
        )
    return selected


def _expanded_groups(
    frame: pd.DataFrame,
    *,
    include_mouse: bool,
) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    areas = sorted(frame["area"].astype(str).unique())
    horizons = sorted(frame["horizon"].astype(int).unique())
    mice: Sequence[str | None]
    if include_mouse:
        mice = tuple(sorted(frame["mouse"].astype(str).unique()))
    else:
        mice = (None,)

    for mouse in mice:
        mouse_frame = (
            frame
            if mouse is None
            else frame.loc[frame["mouse"].astype(str).eq(mouse)]
        )
        for area_label, area in [
            *((area, area) for area in areas),
            ("overall", None),
        ]:
            area_frame = (
                mouse_frame
                if area is None
                else mouse_frame.loc[
                    mouse_frame["area"].astype(str).eq(area)
                ]
            )
            for horizon_label, horizon in [
                *((f"W{position + 1}", position) for position in horizons),
                ("both", None),
            ]:
                group = (
                    area_frame
                    if horizon is None
                    else area_frame.loc[
                        area_frame["horizon"].astype(int).eq(horizon)
                    ]
                )
                if group.empty:
                    continue
                labels: dict[str, object] = {
                    "area": area_label,
                    "horizon": horizon_label,
                }
                if include_mouse:
                    labels["mouse"] = mouse
                yield labels, group


def _score_group(
    frame: pd.DataFrame,
    *,
    labels: dict[str, object],
    metric_set: str,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    selected = frame.loc[frame["metric"].isin(metrics)]
    rows: list[dict[str, object]] = []
    for model, model_frame in selected.groupby("model", sort=True):
        residual = (
            model_frame["predicted"].to_numpy(dtype=float)
            - model_frame["observed"].to_numpy(dtype=float)
        )
        absolute_error = np.abs(residual)
        scaled_squared_error = model_frame[
            "scaled_squared_error"
        ].to_numpy(dtype=float)
        row: dict[str, object] = {
            **labels,
            "metric_set": metric_set,
            "model": model,
            "model_label": MODEL_LABELS.get(str(model), str(model)),
            "mice": int(model_frame["mouse"].nunique()),
            "value_predictions": int(len(model_frame)),
            "mae": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "robust_scale_nrmse": float(
                np.sqrt(np.mean(scaled_squared_error))
            ),
            "median_absolute_error": float(np.median(absolute_error)),
            "p90_absolute_error": float(
                np.quantile(absolute_error, 0.9)
            ),
            "bias": float(np.mean(residual)),
            "within_0_5_pct": np.nan,
            "within_0_25_pct": np.nan,
            "within_1_0_pct": np.nan,
            "within_0_05_fraction_pct": np.nan,
            "within_0_10_fraction_pct": np.nan,
        }
        if metric_set != "selective_fractions":
            row.update(
                {
                    "within_0_5_pct": float(
                        100.0 * np.mean(absolute_error <= 0.5)
                    ),
                    "within_0_25_pct": float(
                        100.0 * np.mean(absolute_error <= 0.25)
                    ),
                    "within_1_0_pct": float(
                        100.0 * np.mean(absolute_error <= 1.0)
                    ),
                }
            )
        else:
            row.update(
                {
                    "within_0_05_fraction_pct": float(
                        100.0 * np.mean(absolute_error <= 0.05)
                    ),
                    "within_0_10_fraction_pct": float(
                        100.0 * np.mean(absolute_error <= 0.10)
                    ),
                }
            )
        rows.append(row)
    return rows


def _add_persistence_comparison(
    summary: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    baseline_columns = [
        *keys,
        "mae",
        "rmse",
        "robust_scale_nrmse",
        "within_0_5_pct",
        "within_0_05_fraction_pct",
        "within_0_10_fraction_pct",
    ]
    baseline = summary.loc[
        summary["model"].eq(PERSISTENCE_MODEL),
        baseline_columns,
    ].rename(
        columns={
            "mae": "persistence_mae",
            "rmse": "persistence_rmse",
            "robust_scale_nrmse": "persistence_robust_scale_nrmse",
            "within_0_5_pct": "persistence_within_0_5_pct",
            "within_0_05_fraction_pct": (
                "persistence_within_0_05_fraction_pct"
            ),
            "within_0_10_fraction_pct": (
                "persistence_within_0_10_fraction_pct"
            ),
        }
    )
    result = summary.merge(
        baseline,
        on=list(keys),
        how="left",
        validate="many_to_one",
    )
    result["mae_improvement_vs_persistence_pct"] = 100.0 * (
        1.0 - result["mae"] / result["persistence_mae"]
    )
    result["rmse_improvement_vs_persistence_pct"] = 100.0 * (
        1.0 - result["rmse"] / result["persistence_rmse"]
    )
    result["within_0_5_percentage_point_gain_vs_persistence"] = (
        result["within_0_5_pct"]
        - result["persistence_within_0_5_pct"]
    )
    result[
        "within_0_05_fraction_percentage_point_gain_vs_persistence"
    ] = (
        result["within_0_05_fraction_pct"]
        - result["persistence_within_0_05_fraction_pct"]
    )
    result[
        "within_0_10_fraction_percentage_point_gain_vs_persistence"
    ] = (
        result["within_0_10_fraction_pct"]
        - result["persistence_within_0_10_fraction_pct"]
    )
    return result


def derive_accuracy_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return area/horizon summaries pooled over eligible mice."""

    rows: list[dict[str, object]] = []
    for labels, group in _expanded_groups(
        predictions,
        include_mouse=False,
    ):
        for metric_set, metrics in METRIC_SETS.items():
            rows.extend(
                _score_group(
                    group,
                    labels=labels,
                    metric_set=metric_set,
                    metrics=metrics,
                )
            )
    summary = pd.DataFrame.from_records(rows)
    keys = ["area", "horizon", "metric_set"]
    summary = _add_persistence_comparison(summary, keys=keys)
    return summary.sort_values(
        [*keys, "model"],
    ).reset_index(drop=True)


def derive_mouse_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the same summaries separately for each eligible mouse."""

    rows: list[dict[str, object]] = []
    for labels, group in _expanded_groups(
        predictions,
        include_mouse=True,
    ):
        for metric_set, metrics in METRIC_SETS.items():
            rows.extend(
                _score_group(
                    group,
                    labels=labels,
                    metric_set=metric_set,
                    metrics=metrics,
                )
            )
    summary = pd.DataFrame.from_records(rows)
    keys = ["mouse", "area", "horizon", "metric_set"]
    summary = _add_persistence_comparison(summary, keys=keys)
    return summary.sort_values(
        [*keys, "model"],
    ).reset_index(drop=True)


def derive_metric_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return original-unit errors for every output separately."""

    rows: list[dict[str, object]] = []
    metrics = (*DPRIME_SHAPE_METRICS, *FRACTION_METRICS)
    for labels, group in _expanded_groups(
        predictions,
        include_mouse=False,
    ):
        for metric in metrics:
            metric_set = (
                "selective_fractions"
                if metric in FRACTION_METRICS
                else "dprime_single_metric"
            )
            metric_labels = {**labels, "metric": metric}
            rows.extend(
                _score_group(
                    group,
                    labels=metric_labels,
                    metric_set=metric_set,
                    metrics=(metric,),
                )
            )
    summary = pd.DataFrame.from_records(rows)
    keys = ["area", "horizon", "metric"]
    summary = _add_persistence_comparison(summary, keys=keys)
    return summary.sort_values(
        [*keys, "model"],
    ).reset_index(drop=True)


def _format_markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(
            lambda value: (
                ""
                if not np.isfinite(value)
                else f"{float(value):.1f}"
            )
        )
    headers = list(map(str, formatted.columns))
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            str(value).replace("|", r"\|").replace("\n", " ")
            for value in row
        )
        + " |"
        for row in formatted.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _write_report(
    *,
    run_directory: Path,
    split: str,
    summary: pd.DataFrame,
) -> None:
    def accuracy_table(metric_set: str) -> pd.DataFrame:
        table = summary.loc[
            summary["metric_set"].eq(metric_set)
            & summary["model"].isin(
                (SELECTED_MODEL, PERSISTENCE_MODEL)
            ),
            [
                "area",
                "horizon",
                "model_label",
                "value_predictions",
                "mae",
                "rmse",
                "within_0_5_pct",
            ],
        ].copy()
        table["value_predictions"] = table["value_predictions"].astype(int)
        table["mae"] = table["mae"].map(lambda value: f"{value:.3f}")
        table["rmse"] = table["rmse"].map(lambda value: f"{value:.3f}")
        return table

    location_table = accuracy_table("dprime_locations")
    shape_table = accuracy_table("dprime_locations_and_spread")
    fraction_table = summary.loc[
        summary["metric_set"].eq("selective_fractions")
        & summary["model"].isin(
            (SELECTED_MODEL, PERSISTENCE_MODEL)
        ),
        [
            "area",
            "horizon",
            "model_label",
            "value_predictions",
            "mae",
            "rmse",
            "within_0_05_fraction_pct",
            "within_0_10_fraction_pct",
        ],
    ].copy()
    fraction_table["value_predictions"] = fraction_table[
        "value_predictions"
    ].astype(int)
    fraction_table["mae"] = fraction_table["mae"].map(
        lambda value: f"{value:.3f}"
    )
    fraction_table["rmse"] = fraction_table["rmse"].map(
        lambda value: f"{value:.3f}"
    )
    report = f"""# Actual-value accuracy

This is a post-hoc, descriptive summary of the frozen `{split}` predictions.
One evaluation unit is one predicted **area-level population statistic** for
one mouse, area, and forecast horizon. It is not a neuron-level accuracy.

The primary table pools only q05, median, and q95 because these are all
d-prime location statistics. `within_0_5_pct` is the percentage of those
individual statistic predictions with absolute error no greater than 0.5
d-prime. D-prime SD is reported in a second four-statistic summary, and the
two selectivity fractions are kept in their native 0–1 units.

The supervised split contains three eligible mice (TX108, TX60, and VR2).
TX109 is excluded by the model contract because it lacks two valid
before-learning W20 windows.

## q05 / median / q95 headline

{_format_markdown_table(location_table)}

## q05 / median / q95 / d-prime SD

{_format_markdown_table(shape_table)}

## Leaf- and circle-selective fractions

{_format_markdown_table(fraction_table)}

## Interpretation

`mae` and `rmse` are in d-prime units for the three-statistic and
four-statistic summaries. `robust_scale_nrmse` divides each output error by
the unsupervised-training robust scale stored with the frozen predictions;
it is dimensionless but is **not a percentage**. Selective-fraction MAE is a
fraction: for example, 0.08 means 8 percentage points.

All aggregate percentages are descriptive and have coarse resolution because
there are only three eligible supervised mice. An area × horizon cell has
9 q05/median/q95 value predictions, so one prediction changes its percentage
by 11.1 points.
"""
    (run_directory / "ACTUAL_VALUE_ACCURACY.md").write_text(
        report,
        encoding="utf-8",
    )


def run(
    *,
    run_directory: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions_path = run_directory / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"predictions file not found: {predictions_path}"
        )
    predictions = _validate_predictions(
        pd.read_csv(predictions_path),
        split=split,
    )
    summary = derive_accuracy_summary(predictions)
    by_mouse = derive_mouse_accuracy(predictions)
    by_metric = derive_metric_accuracy(predictions)

    summary.to_csv(
        run_directory / "actual_value_accuracy_summary.csv",
        index=False,
    )
    by_mouse.to_csv(
        run_directory / "actual_value_accuracy_by_mouse.csv",
        index=False,
    )
    by_metric.to_csv(
        run_directory / "actual_value_accuracy_by_metric.csv",
        index=False,
    )
    _write_report(
        run_directory=run_directory,
        split=split,
        summary=summary,
    )
    return summary, by_mouse, by_metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        help=(
            "Boundary run directory. Defaults to the run in latest.json."
        ),
    )
    parser.add_argument(
        "--split",
        default="supervised_test",
        help="Prediction split to summarize (default: supervised_test).",
    )
    arguments = parser.parse_args()
    run_directory = (
        arguments.run_directory.resolve()
        if arguments.run_directory is not None
        else _latest_run_directory(DEFAULT_RESULTS_ROOT)
    )
    summary, by_mouse, by_metric = run(
        run_directory=run_directory,
        split=str(arguments.split),
    )
    print(f"Wrote actual-value accuracy outputs to {run_directory}")
    print(
        summary.loc[
            summary["area"].eq("overall")
            & summary["horizon"].eq("both")
            & summary["metric_set"].eq("dprime_locations"),
            [
                "model_label",
                "value_predictions",
                "mae",
                "rmse",
                "within_0_5_pct",
            ],
        ].to_string(index=False)
    )
    print(
        f"Rows: summary={len(summary)}, "
        f"by_mouse={len(by_mouse)}, by_metric={len(by_metric)}"
    )


if __name__ == "__main__":
    main()
