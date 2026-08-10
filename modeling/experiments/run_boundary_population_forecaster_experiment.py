#!/usr/bin/env python3
"""Run the direct area-level forecast across the learning-session boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from modeling.experiments.boundary_population_forecaster import (
    DEFAULT_ALPHAS,
    FEATURE_VARIANTS,
    HORIZONS,
    METRICS,
    BoundaryPopulationData,
    build_boundary_population_data,
    candidate_grid,
    evaluate_boundary_population_forecaster,
)
from modeling.experiments.boundary_population_forecaster_plots import (
    create_all_plots,
)
from modeling.experiments.full_neural_objective_b import (
    load_full_neural_cache,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_FULL_CACHE = (
    WORKSPACE / "modeling/experiments/cache/full_neural_objective_b_w20.npz"
)
DEFAULT_REFINED_CACHE = (
    WORKSPACE
    / "modeling/experiments/cache/refined_neuron_transitions_w20.npz"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE
    / "modeling/runs/objective_b"
)
RESULT_SCHEMA_VERSION = 1
MODEL_VERSION = "boundary_population_forecaster_v1"
SELECTED_MODEL = "selected_nested"
PERSISTENCE_MODEL = "persistence"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_run_id() -> str:
    return _utc_now().strftime("run_%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"{path} is not empty; choose another run id or pass --force"
        )
    path.mkdir(parents=True, exist_ok=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(
            lambda value: (
                ""
                if not np.isfinite(value)
                else f"{float(value):.3f}"
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


def _horizon_label(horizon: int) -> str:
    return "first after W20" if int(horizon) == 0 else "second after W20"


def derive_model_summary(mouse_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mouse-level final NRMSE without changing the score unit."""

    summary = (
        mouse_scores.groupby(
            ["split", "area", "horizon", "model"],
            as_index=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
            median_nrmse=("nrmse", "median"),
        )
        .fillna({"sem_nrmse": 0.0})
    )
    baseline = summary.loc[
        summary["model"].eq(PERSISTENCE_MODEL),
        ["split", "area", "horizon", "mean_nrmse"],
    ].rename(columns={"mean_nrmse": "persistence_mean_nrmse"})
    summary = summary.merge(
        baseline,
        on=["split", "area", "horizon"],
        how="left",
        validate="many_to_one",
    )
    summary["improvement_vs_persistence_pct"] = 100.0 * (
        1.0
        - summary["mean_nrmse"]
        / summary["persistence_mean_nrmse"]
    )
    summary["display_model"] = summary["model"].map(
        {
            PERSISTENCE_MODEL: "Last-before W20 persistence",
            SELECTED_MODEL: "Direct boundary forecaster",
        }
    ).fillna(summary["model"])
    return summary.sort_values(
        ["split", "area", "horizon", "model"]
    ).reset_index(drop=True)


def derive_metric_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report errors against the actual six values in their original units."""

    rows: list[dict[str, object]] = []
    keys = ["split", "area", "horizon", "model", "metric"]
    for key, frame in predictions.groupby(keys, sort=True):
        residual = (
            frame["predicted"].to_numpy(dtype=float)
            - frame["observed"].to_numpy(dtype=float)
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "mice": int(frame["mouse"].nunique()),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "bias": float(np.mean(residual)),
                "mean_observed": float(frame["observed"].mean()),
                "mean_predicted": float(frame["predicted"].mean()),
                "mean_scaled_absolute_error": float(
                    frame["scaled_absolute_error"].mean()
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    baseline = result.loc[
        result["model"].eq(PERSISTENCE_MODEL),
        [
            "split",
            "area",
            "horizon",
            "metric",
            "mae",
            "rmse",
        ],
    ].rename(
        columns={
            "mae": "persistence_mae",
            "rmse": "persistence_rmse",
        }
    )
    result = result.merge(
        baseline,
        on=["split", "area", "horizon", "metric"],
        how="left",
        validate="many_to_one",
    )
    result["mae_improvement_vs_persistence_pct"] = 100.0 * (
        1.0 - result["mae"] / result["persistence_mae"]
    )
    return result.sort_values(keys).reset_index(drop=True)


def derive_mouse_comparison(mouse_scores: pd.DataFrame) -> pd.DataFrame:
    """Pair each model prediction with persistence for the same mouse."""

    paired = (
        mouse_scores.pivot_table(
            index=["split", "mouse", "cohort", "area", "horizon"],
            columns="model",
            values="nrmse",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    paired["improvement_vs_persistence_pct"] = 100.0 * (
        1.0
        - paired[SELECTED_MODEL]
        / paired[PERSISTENCE_MODEL]
    )
    return paired.sort_values(
        ["split", "mouse", "area", "horizon"]
    ).reset_index(drop=True)


def boundary_examples_table(data: BoundaryPopulationData) -> pd.DataFrame:
    """Flatten the compact model contract into one auditable CSV."""

    table = data.metadata.copy()
    blocks = {
        "whole_before": data.whole_before,
        "full_tminus1": data.previous_full,
        "full_t": data.current_full,
        "svd_tminus1": data.previous_svd,
        "svd_t": data.current_svd,
    }
    for prefix, values in blocks.items():
        for position, metric in enumerate(METRICS):
            table[f"{prefix}_{metric}"] = values[:, position]
    table["before_session_mean_run_speed"] = data.behavior[:, 0]
    table["penultimate_w20_run_speed"] = data.behavior[:, 1]
    for horizon in HORIZONS:
        for position, metric in enumerate(METRICS):
            table[f"target_h{horizon}_{metric}"] = (
                data.targets[:, horizon, position]
            )
    return table


def _headline(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.loc[
        summary["model"].isin((PERSISTENCE_MODEL, SELECTED_MODEL)),
        [
            "split",
            "area",
            "horizon",
            "display_model",
            "mice",
            "mean_nrmse",
            "sem_nrmse",
            "improvement_vs_persistence_pct",
        ],
    ].copy()
    result["split"] = result["split"].map(
        {
            "unsupervised_lomo": "Unsupervised nested LOMO",
            "supervised_test": "Frozen supervised transfer",
        }
    )
    result["horizon"] = result["horizon"].map(_horizon_label)
    return result.rename(columns={"display_model": "model"})


def _final_selections(selections: pd.DataFrame) -> pd.DataFrame:
    result = selections.loc[
        selections["scope"].eq("all_unsupervised"),
        [
            "area",
            "horizon",
            "selection_rule",
            "selected_candidate",
            "selected_feature_variant",
            "selected_alpha",
            "best_mean_nrmse",
            "best_sem_nrmse",
        ],
    ].copy()
    result["horizon_label"] = result["horizon"].map(_horizon_label)
    return result.sort_values(["area", "horizon"]).reset_index(drop=True)


def _data_counts(data: BoundaryPopulationData) -> pd.DataFrame:
    return (
        data.metadata.groupby(["cohort", "area"], as_index=False)
        .agg(
            mice=("mouse", "nunique"),
            before_sessions=("before_behavior_session_id", "nunique"),
            minimum_before_w20_windows=("before_w20_windows", "min"),
            maximum_before_w20_windows=("before_w20_windows", "max"),
        )
        .sort_values(["cohort", "area"])
        .reset_index(drop=True)
    )


def _report(
    *,
    headline: pd.DataFrame,
    selections: pd.DataFrame,
    metric_errors: pd.DataFrame,
    counts: pd.DataFrame,
    configuration: dict[str, object],
) -> str:
    area_names = tuple(map(str, configuration["areas"]))
    area_text = ", ".join(f"`{area}`" for area in area_names)
    frozen_selected = headline.loc[
        headline["split"].eq("Frozen supervised transfer")
        & headline["model"].eq("Direct boundary forecaster")
    ]
    improved_comparisons = int(
        frozen_selected["improvement_vs_persistence_pct"].gt(0).sum()
    )
    total_comparisons = int(len(frozen_selected))
    eligible_transfer_mice = int(
        counts.loc[counts["cohort"].eq("supervised"), "mice"].max()
    )
    supervised_metrics = metric_errors.loc[
        metric_errors["split"].eq("supervised_test")
        & metric_errors["model"].eq(SELECTED_MODEL),
        [
            "area",
            "horizon",
            "metric",
            "mae",
            "rmse",
            "mean_observed",
            "mean_predicted",
            "mae_improvement_vs_persistence_pct",
        ],
    ].copy()
    supervised_metrics["horizon"] = supervised_metrics["horizon"].map(
        _horizon_label
    )
    return f"""# Direct population forecast across learning

## Result in one sentence

The model uses only pre-learning information to predict the complete
six-number area-level d-prime distribution in the first and second W20
windows after learning. On the {eligible_transfer_mice} supervised mice with
the required two pre-learning W20 windows, the frozen model reduces NRMSE
versus persistence in {improved_comparisons} of {total_comparisons}
area/horizon comparisons. The result is exploratory because the supervised
test set has been inspected repeatedly.

## What is predicted

For each mouse, evaluated broad area ({area_text}), and future horizon, the
target is:

`(q05, median, q95, SD, leaf-selective fraction, circle-selective fraction)`.

- horizon 0 is the **first after-learning W20**;
- horizon 1 is the **second after-learning W20**.

This is a population forecast. It does not predict or require registration of
the same individual neuron across days.

## Model in one picture

```text
complete before session + final two before W20s (full and SVD) + behavior
                                  |
                                  v
          constraint-safe population-state representation g(.)
                                  |
                                  v
       choose one direct residual model per area and future horizon
       using unsupervised mice only (nested leave-one-mouse-out)
                                  |
                                  v
 g(predicted after W20_h) = g(last before W20) + predicted boundary residual
                                  |
                                  v
          inverse-transform to six valid, directly comparable values
```

The candidate set contains exact persistence, a horizon-specific mean
cross-day shift, and ridge regressions using increasingly rich before-only
features. Standard ridge leaves the intercept unpenalized, so it can learn the
common exposure shift while shrinking unreliable state-dependent slopes.

## Headline performance

NRMSE is computed from errors against the **actual six values**, after scaling
each metric using training mice only. The percentage is error reduction versus
carrying the final before-learning W20 forward:

{_markdown_table(headline)}

## Presentation figures

- [Model comparison](plots/boundary_population_model_comparison.png)
- [Predicted versus actual six-statistic values](plots/boundary_population_predicted_vs_actual.png)
- [All eligible supervised trajectories](plots/boundary_population_supervised_trajectories.png)
- [TX108](plots/boundary_population_supervised_TX108.png)
- [TX60](plots/boundary_population_supervised_TX60.png)
- [VR2](plots/boundary_population_supervised_VR2.png)
- [TX109 input-eligibility slide](plots/boundary_population_supervised_TX109.png)

## Frozen model choices

Candidate family and alpha are chosen independently for each area and horizon
from unsupervised-mouse LOMO only:

{_markdown_table(selections)}

The richer full-neural + SVD representation is selected only where its
unsupervised validation error warrants it. Simpler mean-shift or
whole-session candidates remain valid outcomes rather than forcing the largest
feature set into a nine-mouse regression.

## Error against the real values

These are raw-unit errors for the frozen supervised predictions:

{_markdown_table(supervised_metrics)}

## Data support

{_markdown_table(counts)}

TX109 is not silently imputed: its before-learning recording does not provide
the two valid W20 inputs required by this experiment, so no boundary prediction
is made for it. The supervised transfer result therefore covers TX108, TX60,
and VR2 (75% of the four supervised mice).

## Leakage controls

- All feature filling, centering, scaling, target scaling, and ridge fitting
  use training mice only.
- Candidate selection is nested inside each unsupervised outer LOMO fold.
- One final candidate per area/horizon is selected on unsupervised LOMO and
  frozen before supervised outcomes are scored.
- All inputs precede the predicted after-learning window.
- The behavior feature stored by the compact cache describes the penultimate
  W20, not the terminal W20; the label and documentation preserve that fact.

## Interpretation

Regional differences in transfer performance are descriptive evidence, not
causal proof. The repeatedly inspected {eligible_transfer_mice}-mouse
supervised set is not a fresh confirmatory test. A new held-out cohort is
required for a confirmatory claim.

## Reproducibility

```json
{json.dumps(configuration, indent=2, sort_keys=True)}
```
"""


def run(
    *,
    full_cache_path: Path = DEFAULT_FULL_CACHE,
    refined_cache_path: Path = DEFAULT_REFINED_CACHE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    force: bool = False,
    areas: Sequence[str] = ("mHV", "aHV"),
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    selection_rule: str = "minimum_cv",
) -> dict[str, Path]:
    run_id = run_id or _default_run_id()
    output = output_root / "runs" / run_id
    _guard_output(output, force=force)

    cache = load_full_neural_cache(
        full_cache_path,
        refined_cache_path=refined_cache_path,
    )
    data = build_boundary_population_data(cache)
    candidates = candidate_grid(
        variants=FEATURE_VARIANTS,
        alphas=alphas,
    )
    result = evaluate_boundary_population_forecaster(
        data,
        candidates=candidates,
        areas=areas,
        selection_rule=selection_rule,
    )
    summary = derive_model_summary(result.mouse_scores)
    metric_errors = derive_metric_errors(result.predictions)
    mouse_comparison = derive_mouse_comparison(result.mouse_scores)
    headline = _headline(summary)
    selections = _final_selections(result.selections)
    examples = boundary_examples_table(data)
    counts = _data_counts(data)
    plot_paths = create_all_plots(
        result.predictions,
        summary,
        output / "plots",
        selected_model=SELECTED_MODEL,
        comparison_models=(PERSISTENCE_MODEL, SELECTED_MODEL),
    )

    configuration: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "created_utc": _utc_now().isoformat(),
        "run_id": run_id,
        "window_pairs_per_role": int(cache.window_pairs),
        "areas": list(areas),
        "horizons": list(HORIZONS),
        "metrics": list(METRICS),
        "feature_variants": list(FEATURE_VARIANTS),
        "candidate_count": len(candidates),
        "alphas": [float(value) for value in alphas],
        "selection_rule": selection_rule,
        "training_cohort": "unsupervised",
        "transfer_cohort": "supervised",
        "prediction_target": (
            "six area-level d-prime distribution statistics in first and "
            "second after-learning W20"
        ),
        "anchor": "last full-derived before-learning W20 population state",
        "randomness": "none",
        "full_cache": str(full_cache_path.resolve()),
        "full_cache_sha256": _sha256(full_cache_path),
        "refined_cache": str(refined_cache_path.resolve()),
        "refined_cache_sha256": _sha256(refined_cache_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    paths = {
        "predictions": output / "predictions.csv",
        "mouse_scores": output / "mouse_scores.csv",
        "model_summary": output / "model_summary.csv",
        "metric_errors": output / "metric_errors.csv",
        "mouse_comparison": output / "mouse_comparison.csv",
        "candidate_scores": output / "candidate_lomo_scores.csv",
        "candidate_summary": output / "candidate_summary.csv",
        "selections": output / "hyperparameter_selections.csv",
        "final_selections": output / "final_selections.csv",
        "headline": output / "headline_comparison.csv",
        "examples": output / "boundary_examples.csv",
        "counts": output / "data_counts.csv",
        "models": output / "fitted_boundary_population_models.joblib",
        "configuration": output / "configuration.json",
        "report": output / "RESULTS_REPORT.md",
        **plot_paths,
    }
    frames = {
        "predictions": result.predictions,
        "mouse_scores": result.mouse_scores,
        "model_summary": summary,
        "metric_errors": metric_errors,
        "mouse_comparison": mouse_comparison,
        "candidate_scores": result.candidate_scores,
        "candidate_summary": result.candidate_summary,
        "selections": result.selections,
        "final_selections": selections,
        "headline": headline,
        "examples": examples,
        "counts": counts,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    joblib.dump(
        {
            "configuration": configuration,
            "models": result.final_models,
        },
        paths["models"],
    )
    paths["configuration"].write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    paths["report"].write_text(
        _report(
            headline=headline,
            selections=selections,
            metric_errors=metric_errors,
            counts=counts,
            configuration=configuration,
        )
    )

    latest = {
        "run_id": run_id,
        "run_directory": str(output.resolve()),
        "report": str(paths["report"].resolve()),
        "created_utc": configuration["created_utc"],
        "selected_model": SELECTED_MODEL,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(
        json.dumps(latest, indent=2, sort_keys=True) + "\n"
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_FULL_CACHE)
    parser.add_argument(
        "--refined-cache",
        type=Path,
        default=DEFAULT_REFINED_CACHE,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--areas",
        nargs="+",
        choices=("V1", "mHV", "aHV"),
        default=["mHV", "aHV"],
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--selection-rule",
        choices=("minimum_cv", "one_se"),
        default="minimum_cv",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    paths = run(
        full_cache_path=arguments.full_cache,
        refined_cache_path=arguments.refined_cache,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        force=arguments.force,
        areas=arguments.areas,
        alphas=arguments.alphas,
        selection_rule=arguments.selection_rule,
    )
    print(paths["report"])


if __name__ == "__main__":
    main()
