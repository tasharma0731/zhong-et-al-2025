#!/usr/bin/env python3
"""Run the leakage-aware exposure-transfer analysis end to end.

The implementation is split into small modules:

- :mod:`modeling.preparation` validates and reshapes the history;
- :mod:`modeling.estimators` defines constraint-preserving models;
- :mod:`modeling.validation` performs mouse-grouped nested validation;
- :mod:`modeling.diagnostics` computes uncertainty and sensitivities;
- :mod:`modeling.reporting` writes immutable artifacts and plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_METRICS as METRICS
from .config import PRIMARY_CONFIG, StudyConfig
from .diagnostics import (
    area_contrasts,
    area_skill_by_mouse,
    baseline_support_diagnostics,
    exact_sign_flip,
    leave_one_metric_out,
    leave_one_mouse_out_aggregate,
    metric_summary,
    next_window_moment_scores,
    next_window_moment_summary,
    paired_model_tests,
    reference_band,
    raw_area_error_contrasts,
    score_uncertainty,
    summarize_scores,
)
from .estimators import CROSS_DAY_MODELS, NEXT_WINDOW_MODELS
from .preparation import (
    load_history,
    load_session_summaries,
    make_repeated_session_pairs,
    make_session_pairs,
    make_window_transitions,
    nonoverlapping_windows,
)
from .reporting import (
    plot_errors,
    plot_skill_contrast,
    sha256,
    write_configuration,
    write_report,
)
from .validation import (
    choose_alpha,
    evaluate_frozen_cross_day,
    evaluate_study,
    training_mouse_deletion_sensitivity,
)


def run(
    history_path: Path,
    output: Path,
    *,
    session_summary_path: Path | None = None,
    grating_control_path: Path | None = None,
    config: StudyConfig = PRIMARY_CONFIG,
    run_training_sensitivity: bool = True,
) -> dict[str, Path]:
    history, window_pairs = load_history(history_path, metrics=config.metrics)
    windows = nonoverlapping_windows(history, window_pairs)
    if session_summary_path is None:
        raise ValueError(
            "session_summary_path is required: cross-day prediction must use "
            "the window-independent, trial-balanced session estimand"
        )
    session_summaries = load_session_summaries(
        session_summary_path, metrics=config.metrics
    )
    session_pairs = make_session_pairs(session_summaries, metrics=config.metrics)
    transitions = make_window_transitions(
        windows, window_pairs, metrics=config.metrics
    )
    predictions, scores, fitted_models = evaluate_study(
        session_pairs, transitions, config
    )
    if grating_control_path is not None:
        grating_summaries = load_session_summaries(
            grating_control_path, metrics=config.metrics
        )
        grating_pairs = make_repeated_session_pairs(
            grating_summaries, metrics=config.metrics
        )
        grating_predictions, grating_scores = evaluate_frozen_cross_day(
            grating_pairs,
            session_pairs,
            fitted_models,
            config,
            split="grating_control",
        )
        grating_summary = grating_scores.groupby(
            ["area", "model"], as_index=False
        ).agg(mice=("mouse", "nunique"), mean_nrmse=("nrmse", "mean"))
        grating_baseline = grating_summary.loc[
            grating_summary["model"].eq("persistence")
        ][["area", "mean_nrmse"]].rename(
            columns={"mean_nrmse": "persistence_nrmse"}
        )
        grating_summary = grating_summary.merge(
            grating_baseline, on="area", how="left"
        )
        grating_summary["improvement_vs_persistence_pct"] = 100 * (
            grating_summary["persistence_nrmse"] - grating_summary["mean_nrmse"]
        ) / grating_summary["persistence_nrmse"]
    else:
        grating_summaries = pd.DataFrame()
        grating_pairs = pd.DataFrame()
        grating_predictions = pd.DataFrame()
        grating_scores = pd.DataFrame()
        grating_summary = pd.DataFrame()

    summary = summarize_scores(scores)
    uncertainty = score_uncertainty(scores, config)
    paired_tests = paired_model_tests(scores, config)
    area_tests = area_contrasts(scores, config)
    area_mouse_skills = area_skill_by_mouse(scores)
    raw_area_tests = raw_area_error_contrasts(scores, config)
    bands = reference_band(scores)
    per_metric = metric_summary(predictions)
    metric_sensitivity = leave_one_metric_out(predictions)
    mouse_sensitivity = leave_one_mouse_out_aggregate(scores)
    moment_scores = next_window_moment_scores(predictions)
    moment_summary = next_window_moment_summary(moment_scores)
    support = baseline_support_diagnostics(session_pairs, config)
    if run_training_sensitivity:
        training_sensitivity = training_mouse_deletion_sensitivity(
            session_pairs, transitions, config
        )
        training_sensitivity_summary = (
            training_sensitivity.groupby(
                ["task", "area", "model", "omitted_training_mouse"], as_index=False
            )
            .agg(mean_supervised_nrmse=("nrmse", "mean"))
            .groupby(["task", "area", "model"], as_index=False)
            .agg(
                deletion_mean_nrmse=("mean_supervised_nrmse", "mean"),
                deletion_min_nrmse=("mean_supervised_nrmse", "min"),
                deletion_max_nrmse=("mean_supervised_nrmse", "max"),
            )
        )
    else:
        training_sensitivity = pd.DataFrame()
        training_sensitivity_summary = pd.DataFrame()

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output / "predictions.csv",
        "mouse_scores": output / "mouse_scores.csv",
        "summary": output / "model_summary.csv",
        "uncertainty": output / "score_uncertainty.csv",
        "paired_tests": output / "paired_model_tests.csv",
        "area_contrasts": output / "area_contrasts.csv",
        "area_mouse_skills": output / "area_skill_by_mouse.csv",
        "raw_area_contrasts": output / "raw_area_error_contrasts.csv",
        "reference_bands": output / "reference_bands.csv",
        "metric_summary": output / "metric_summary.csv",
        "metric_sensitivity": output / "leave_one_metric_out.csv",
        "mouse_sensitivity": output / "leave_one_mouse_out.csv",
        "training_sensitivity": output / "training_mouse_deletion_scores.csv",
        "training_sensitivity_summary": output / "training_mouse_deletion_summary.csv",
        "moment_scores": output / "next_window_moment_scores.csv",
        "moment_summary": output / "next_window_moment_summary.csv",
        "baseline_support": output / "baseline_support.csv",
        "session_summaries": output / "session_summaries.csv",
        "grating_summaries": output / "grating_session_summaries.csv",
        "grating_pairs": output / "grating_session_pairs.csv",
        "grating_predictions": output / "grating_predictions.csv",
        "grating_scores": output / "grating_mouse_scores.csv",
        "grating_summary": output / "grating_summary.csv",
        "session_pairs": output / "session_pairs.csv",
        "window_transitions": output / "window_transitions.csv",
        "models": output / "fitted_models.joblib",
        "configuration": output / "configuration.json",
        "report": output / "REPORT.md",
        "plot": output / "model_errors.png",
        "skill_plot": output / "mHV_aHV_skill_by_mouse.png",
    }
    frames = {
        "predictions": predictions,
        "mouse_scores": scores,
        "summary": summary,
        "uncertainty": uncertainty,
        "paired_tests": paired_tests,
        "area_contrasts": area_tests,
        "area_mouse_skills": area_mouse_skills,
        "raw_area_contrasts": raw_area_tests,
        "reference_bands": bands,
        "metric_summary": per_metric,
        "metric_sensitivity": metric_sensitivity,
        "mouse_sensitivity": mouse_sensitivity,
        "training_sensitivity": training_sensitivity,
        "training_sensitivity_summary": training_sensitivity_summary,
        "moment_scores": moment_scores,
        "moment_summary": moment_summary,
        "baseline_support": support,
        "session_summaries": session_summaries,
        "grating_summaries": grating_summaries,
        "grating_pairs": grating_pairs,
        "grating_predictions": grating_predictions,
        "grating_scores": grating_scores,
        "grating_summary": grating_summary,
        "session_pairs": session_pairs,
        "window_transitions": transitions,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)

    history_hash = sha256(history_path)
    session_summary_hash = sha256(session_summary_path)
    grating_control_hash = (
        sha256(grating_control_path) if grating_control_path is not None else None
    )
    joblib.dump(
        {
            "config": config,
            "history_sha256": history_hash,
            "session_summary_sha256": session_summary_hash,
            "grating_control_sha256": grating_control_hash,
            "window_pairs": window_pairs,
            "models": fitted_models,
        },
        paths["models"],
    )
    write_configuration(
        paths["configuration"],
        config=config,
        history_path=history_path,
        history_hash=history_hash,
        session_summary_path=session_summary_path,
        session_summary_hash=session_summary_hash,
        grating_control_path=grating_control_path,
        grating_control_hash=grating_control_hash,
        window_pairs=window_pairs,
        model_names={
            "cross_day": [spec.name for spec in CROSS_DAY_MODELS],
            "next_window": [spec.name for spec in NEXT_WINDOW_MODELS],
        },
    )
    write_report(
        paths["report"],
        summary=summary,
        uncertainty=uncertainty,
        paired_tests=paired_tests,
        area_tests=area_tests,
        raw_area_tests=raw_area_tests,
        moment_summary=moment_summary,
        training_sensitivity_summary=training_sensitivity_summary,
        support=support,
        grating_summary=grating_summary,
        config=config,
        window_pairs=window_pairs,
        source_hash=history_hash,
        session_source_hash=session_summary_hash,
        session_pairs=session_pairs,
        transitions=transitions,
    )
    plot_errors(summary, paths["plot"])
    plot_skill_contrast(area_mouse_skills, paths["skill_plot"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            "modeling/window_histories/window_dprime_chronological_w20.csv"
        ),
    )
    parser.add_argument(
        "--grating-control",
        type=Path,
        default=Path(
            "modeling/window_histories/session_dprime_trial_balanced_grating.csv"
        ),
    )
    parser.add_argument(
        "--session-summary",
        type=Path,
        default=Path(
            "modeling/window_histories/session_dprime_trial_balanced.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modeling/runs/objective_a"),
    )
    parser.add_argument("--skip-training-sensitivity", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    paths = run(
        arguments.history,
        arguments.output,
        session_summary_path=arguments.session_summary,
        grating_control_path=arguments.grating_control,
        run_training_sensitivity=not arguments.skip_training_sensitivity,
    )
    for name, path in paths.items():
        print(f"{name:>20}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "choose_alpha",
    "exact_sign_flip",
    "load_history",
    "METRICS",
    "make_session_pairs",
    "make_window_transitions",
    "nonoverlapping_windows",
    "run",
]
