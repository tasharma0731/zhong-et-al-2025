"""Mouse-level uncertainty, contrasts, and robustness diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import StudyConfig


def exact_sign_flip(values: Iterable[float]) -> tuple[float, float, int]:
    values = np.asarray(tuple(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    observed = float(values.mean())
    if len(values) > 20:
        # Exact enumeration grows exponentially. A fixed-seed Monte Carlo
        # sign-flip test keeps the diagnostic bounded and reproducible.
        rng = np.random.default_rng(2025)
        signs = rng.choice((-1, 1), size=(100_000, len(values)))
    else:
        patterns = np.arange(2 ** len(values))[:, None]
        signs = 1 - 2 * ((patterns >> np.arange(len(values))) & 1)
    null = np.mean(signs * values[None, :], axis=1)
    pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, pvalue, len(null)


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    repeats: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled, [tail, 1.0 - tail])
    return float(values.mean()), float(lower), float(upper)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scores.groupby(["task", "area", "model", "split"], as_index=False)
        .agg(
            mice=("mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
            min_nrmse=("nrmse", "min"),
            max_nrmse=("nrmse", "max"),
        )
        .sort_values(["task", "area", "model", "split"])
    )
    pivot = grouped.pivot_table(
        index=["task", "area", "model"],
        columns="split",
        values="mean_nrmse",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "unsupervised_lomo": "unsupervised_cv_nrmse",
            "supervised_test": "supervised_test_nrmse",
        }
    )
    persistence = pivot.loc[
        pivot["model"].eq("persistence"),
        ["task", "area", "supervised_test_nrmse"],
    ].rename(columns={"supervised_test_nrmse": "persistence_test_nrmse"})
    pivot = pivot.merge(persistence, on=["task", "area"], how="left")
    pivot["supervised_improvement_vs_persistence_pct"] = 100.0 * (
        pivot["persistence_test_nrmse"] - pivot["supervised_test_nrmse"]
    ) / pivot["persistence_test_nrmse"]
    pivot["transfer_error_ratio"] = (
        pivot["supervised_test_nrmse"] / pivot["unsupervised_cv_nrmse"]
    )
    return pivot.sort_values(["task", "area", "model"]).reset_index(drop=True)


def score_uncertainty(scores: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    records = []
    for index, ((task, area, model, split), group) in enumerate(
        scores.groupby(["task", "area", "model", "split"], sort=True)
    ):
        mean, lower, upper = bootstrap_mean_interval(
            group["nrmse"].to_numpy(),
            repeats=config.bootstrap_repeats,
            seed=config.random_seed + index,
        )
        records.append(
            {
                "task": task,
                "area": area,
                "model": model,
                "split": split,
                "mice": group["mouse"].nunique(),
                "mean_nrmse": mean,
                "bootstrap_95_low": lower,
                "bootstrap_95_high": upper,
            }
        )
    return pd.DataFrame(records)


def paired_model_tests(scores: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """Compare every supervised model with persistence at the mouse level."""

    tested = scores.loc[scores["split"].eq("supervised_test")]
    records = []
    for index, ((task, area), group) in enumerate(tested.groupby(["task", "area"])):
        wide = group.pivot(index="mouse", columns="model", values="nrmse")
        if "persistence" not in wide:
            continue
        for model in sorted(set(wide.columns) - {"persistence"}):
            paired = wide[["persistence", model]].dropna()
            improvement = paired["persistence"] - paired[model]
            effect, pvalue, permutations = exact_sign_flip(improvement)
            _, low, high = bootstrap_mean_interval(
                improvement.to_numpy(),
                repeats=config.bootstrap_repeats,
                seed=config.random_seed + 1000 + index * 31 + len(records),
            )
            records.append(
                {
                    "task": task,
                    "area": area,
                    "model": model,
                    "contrast": "persistence_minus_model_error",
                    "mice": len(paired),
                    "mean_error_improvement": effect,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "mice_improved": int((improvement > 0).sum()),
                    "exact_two_sided_pvalue": pvalue,
                    "permutations": permutations,
                }
            )
    return pd.DataFrame(records)


def area_skill_by_mouse(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the medial-minus-anterior baseline-relative skill per mouse."""

    tested = scores.loc[scores["split"].eq("supervised_test")]
    persistence = tested.loc[tested["model"].eq("persistence")].set_index(
        ["task", "mouse", "area"]
    )["nrmse"]
    learned = tested.loc[tested["model"].ne("persistence")].copy()
    learned = learned.set_index(["task", "mouse", "area"])
    learned["persistence_nrmse"] = persistence
    learned["log_skill_vs_persistence"] = np.log(
        np.maximum(learned["persistence_nrmse"], 1e-12)
        / np.maximum(learned["nrmse"], 1e-12)
    )
    wide = learned.reset_index().pivot(
        index=["task", "model", "mouse"],
        columns="area",
        values="log_skill_vs_persistence",
    )
    if not {"mHV", "aHV"}.issubset(wide.columns):
        return pd.DataFrame()
    result = wide.reset_index()
    result["mHV_minus_aHV_log_skill"] = result["mHV"] - result["aHV"]
    return result.sort_values(["task", "model", "mouse"]).reset_index(drop=True)


def area_contrasts(scores: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """Test whether baseline-relative skill is larger in mHV than aHV.

    Raw errors are not comparable evidence for the biological hypothesis when
    one area is intrinsically noisier.  Skill is therefore defined per mouse
    as ``log(error_persistence / error_model)`` and compared across areas.
    Positive values mean the learned model improves more in mHV.
    """

    records = []
    per_mouse = area_skill_by_mouse(scores)
    for index, ((task, model), group) in enumerate(
        per_mouse.groupby(["task", "model"])
    ):
        difference = group["mHV_minus_aHV_log_skill"].dropna()
        effect, pvalue, permutations = exact_sign_flip(difference)
        _, low, high = bootstrap_mean_interval(
            difference.to_numpy(),
            repeats=config.bootstrap_repeats,
            seed=config.random_seed + 2000 + index * 31 + len(records),
        )
        records.append(
            {
                "task": task,
                "model": model,
                "contrast": "mHV_minus_aHV_log_skill_vs_persistence",
                "mice": len(difference),
                "mean_log_skill_difference": effect,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "all_mice_mHV_skill_greater": bool((difference > 0).all()),
                "exact_two_sided_pvalue": pvalue,
                "permutations": permutations,
            }
        )
    return pd.DataFrame(records)


def raw_area_error_contrasts(scores: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """Retain the old raw-error contrast as a clearly labeled diagnostic."""

    tested = scores.loc[scores["split"].eq("supervised_test")]
    records = []
    for index, ((task, model), group) in enumerate(tested.groupby(["task", "model"])):
        wide = group.pivot(index="mouse", columns="area", values="nrmse")
        if not {"mHV", "aHV"}.issubset(wide.columns):
            continue
        paired = wide[["mHV", "aHV"]].dropna()
        difference = paired["aHV"] - paired["mHV"]
        effect, pvalue, permutations = exact_sign_flip(difference)
        _, low, high = bootstrap_mean_interval(
            difference.to_numpy(),
            repeats=config.bootstrap_repeats,
            seed=config.random_seed + 2000 + index,
        )
        records.append(
            {
                "task": task,
                "model": model,
                "contrast": "aHV_minus_mHV_error",
                "mice": len(paired),
                "mean_difference": effect,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "all_mice_aHV_worse": bool((difference > 0).all()),
                "exact_two_sided_pvalue": pvalue,
                "permutations": permutations,
            }
        )
    return pd.DataFrame(records)


def reference_band(scores: pd.DataFrame) -> pd.DataFrame:
    """Descriptive LOMO range; not a calibrated prediction interval."""

    records = []
    for (task, area, model), group in scores.groupby(["task", "area", "model"]):
        calibration = group.loc[group["split"].eq("unsupervised_lomo"), "nrmse"]
        tested = group.loc[group["split"].eq("supervised_test")]
        threshold = float(calibration.max())
        records.append(
            {
                "task": task,
                "area": area,
                "model": model,
                "unsupervised_lomo_max": threshold,
                "supervised_mice": int(tested["mouse"].nunique()),
                "supervised_mice_in_reference": int(tested["nrmse"].le(threshold).sum()),
                "fraction_supervised_in_reference": float(
                    tested["nrmse"].le(threshold).mean()
                ),
            }
        )
    return pd.DataFrame(records).sort_values(["task", "area", "model"])


def metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    per_mouse = (
        predictions.groupby(
            ["task", "split", "model", "area", "metric", "mouse"],
            as_index=False,
        )
        .agg(
            scaled_mae=("scaled_absolute_error", "mean"),
            scaled_mse=("scaled_squared_error", "mean"),
        )
    )
    per_mouse["scaled_rmse"] = np.sqrt(per_mouse["scaled_mse"])
    totals = per_mouse.groupby(
        ["task", "split", "model", "area", "mouse"]
    )["scaled_mse"].transform("sum")
    per_mouse["fraction_composite_mse"] = np.divide(
        per_mouse["scaled_mse"],
        totals,
        out=np.zeros(len(per_mouse), dtype=float),
        where=totals.to_numpy() > 0,
    )
    return (
        per_mouse.groupby(["task", "split", "model", "area", "metric"], as_index=False)
        .agg(
            mice=("mouse", "nunique"),
            mean_scaled_mae=("scaled_mae", "mean"),
            mean_scaled_rmse=("scaled_rmse", "mean"),
            mean_fraction_composite_mse=("fraction_composite_mse", "mean"),
        )
        .sort_values(["task", "area", "model", "metric", "split"])
    )


def next_window_moment_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score before/after next-window forecasts without transition weighting."""

    selected = predictions.loc[
        predictions["task"].eq("next_window")
        & predictions["moment"].notna()
    ].copy()
    identity = [
        "split",
        "model",
        "area",
        "mouse",
        "moment",
        "behavior_session_id",
        "prediction_id",
    ]
    per_transition = selected.groupby(identity, as_index=False).agg(
        mse=("scaled_squared_error", "mean")
    )
    per_transition["nrmse"] = np.sqrt(per_transition["mse"])
    per_session = per_transition.groupby(identity[:-1], as_index=False).agg(
        n_predictions=("prediction_id", "size"), nrmse=("nrmse", "mean")
    )
    return per_session.groupby(
        ["split", "model", "area", "mouse", "moment"], as_index=False
    ).agg(n_sessions=("behavior_session_id", "nunique"), nrmse=("nrmse", "mean"))


def next_window_moment_summary(moment_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize each moment and add improvement over persistence."""

    summary = moment_scores.groupby(
        ["split", "model", "area", "moment"], as_index=False
    ).agg(mice=("mouse", "nunique"), mean_nrmse=("nrmse", "mean"))
    baseline = summary.loc[summary["model"].eq("persistence")].rename(
        columns={"mean_nrmse": "persistence_nrmse"}
    )[["split", "area", "moment", "persistence_nrmse"]]
    summary = summary.merge(baseline, on=["split", "area", "moment"], how="left")
    summary["improvement_vs_persistence_pct"] = 100.0 * (
        summary["persistence_nrmse"] - summary["mean_nrmse"]
    ) / summary["persistence_nrmse"]
    return summary.sort_values(["split", "area", "moment", "model"])


def baseline_support_diagnostics(
    session_pairs: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Measure supervised baseline support relative to unsupervised mice."""

    records: list[dict[str, object]] = []
    columns = [f"before_{metric}" for metric in config.metrics]
    for area, group in session_pairs.groupby("area"):
        train = group.loc[group["cohort"].eq(config.training_cohort)]
        tested = group.loc[group["cohort"].eq(config.transfer_cohort)]
        center = train[columns].mean().to_numpy(dtype=float)
        scale = train[columns].std(ddof=1).replace(0, 1).to_numpy(dtype=float)
        for row in tested.itertuples(index=False):
            values = np.asarray([getattr(row, column) for column in columns], dtype=float)
            z = (values - center) / scale
            records.append(
                {
                    "area": area,
                    "mouse": row.mouse,
                    "rms_training_z": float(np.sqrt(np.mean(np.square(z)))),
                    "max_abs_training_z": float(np.max(np.abs(z))),
                    "metrics_outside_training_range": int(
                        sum(
                            value < train[column].min() or value > train[column].max()
                            for value, column in zip(values, columns)
                        )
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["area", "mouse"])


def leave_one_metric_out(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rescore fitted predictions after omitting each target summary."""

    records = []
    metrics = tuple(sorted(predictions["metric"].unique()))
    identity = ["task", "split", "model", "area", "mouse", "prediction_id"]
    for omitted in metrics:
        kept = predictions.loc[predictions["metric"].ne(omitted)]
        per_prediction = kept.groupby(identity, as_index=False).agg(
            mean_scaled_squared_error=("scaled_squared_error", "mean")
        )
        per_prediction["nrmse"] = np.sqrt(
            per_prediction["mean_scaled_squared_error"]
        )
        per_mouse = per_prediction.groupby(
            ["task", "split", "model", "area", "mouse"], as_index=False
        )["nrmse"].mean()
        summary = per_mouse.groupby(
            ["task", "split", "model", "area"], as_index=False
        ).agg(mice=("mouse", "nunique"), mean_nrmse=("nrmse", "mean"))
        summary["omitted_metric"] = omitted
        records.append(summary)
    return pd.concat(records, ignore_index=True).sort_values(
        ["task", "area", "model", "split", "omitted_metric"]
    )


def leave_one_mouse_out_aggregate(scores: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity of supervised group means to each test mouse."""

    tested = scores.loc[scores["split"].eq("supervised_test")]
    records = []
    for omitted in sorted(tested["mouse"].unique()):
        remaining = tested.loc[tested["mouse"].ne(omitted)]
        summary = remaining.groupby(["task", "area", "model"], as_index=False).agg(
            remaining_mice=("mouse", "nunique"), mean_nrmse=("nrmse", "mean")
        )
        summary["omitted_mouse"] = omitted
        records.append(summary)
    return pd.concat(records, ignore_index=True).sort_values(
        ["task", "area", "model", "omitted_mouse"]
    )
