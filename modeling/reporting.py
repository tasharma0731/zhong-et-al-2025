"""Artifact writing and compact human-readable reports."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zhong-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import StudyConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = frame.reindex(columns=columns).copy()
    for column in values.select_dtypes(include="number"):
        values[column] = values[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def plot_errors(summary: pd.DataFrame, path: Path) -> None:
    preferred = {
        "cross_day": ("persistence", "exposure_shift", "ridge_delta"),
        "next_window": ("persistence", "transition_shift", "ridge_window_delta"),
    }
    colors = {
        "unsupervised_cv_nrmse": "#78909c",
        "supervised_test_nrmse": "#d95f59",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for axis, task in zip(axes, preferred):
        models = preferred[task]
        current = summary.loc[
            summary["task"].eq(task) & summary["model"].isin(models)
        ].copy()
        order = {area: index for index, area in enumerate(("V1", "mHV", "lHV", "aHV"))}
        model_order = {model: index for index, model in enumerate(models)}
        current["area_order"] = current["area"].map(order)
        current["model_order"] = current["model"].map(model_order)
        current = current.sort_values(["area_order", "model_order"])
        labels = [f"{row.area}\n{row.model}" for row in current.itertuples()]
        x = range(len(current))
        width = 0.38
        for offset, column in zip((-width / 2, width / 2), colors):
            axis.bar(
                [position + offset for position in x],
                current[column],
                width,
                color=colors[column],
                label=column,
            )
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("mouse-balanced normalized RMSE")
        axis.set_title("Pre → post exposure" if task == "cross_day" else "Current → next window")
        axis.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_skill_contrast(area_mouse_skills: pd.DataFrame, path: Path) -> None:
    """Plot the primary within-mouse mHV-versus-aHV transfer contrast."""

    selected = area_mouse_skills.loc[
        area_mouse_skills["task"].eq("cross_day")
        & area_mouse_skills["model"].eq("exposure_shift")
    ].sort_values("mouse")
    if selected.empty:
        raise ValueError("no cross-day exposure-shift skills were available")
    fig, axis = plt.subplots(figsize=(6.5, 5))
    colors = plt.cm.viridis([0.15, 0.38, 0.62, 0.85])
    for color, row in zip(colors, selected.itertuples(index=False)):
        axis.plot([0, 1], [row.aHV, row.mHV], color=color, marker="o", lw=1.8)
        axis.text(1.035, row.mHV, str(row.mouse), va="center", fontsize=9)
    axis.axhline(0, color="black", lw=0.9, ls="--", alpha=0.6)
    axis.set_xticks([0, 1], ["aHV", "mHV"])
    axis.set_xlim(-0.18, 1.28)
    axis.set_ylabel("log(persistence error / exposure-shift error)")
    axis.set_title("Transfer skill is higher in mHV for all four mice")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    paired_tests: pd.DataFrame,
    area_tests: pd.DataFrame,
    raw_area_tests: pd.DataFrame,
    moment_summary: pd.DataFrame,
    training_sensitivity_summary: pd.DataFrame,
    support: pd.DataFrame,
    grating_summary: pd.DataFrame,
    config: StudyConfig,
    window_pairs: int,
    source_hash: str,
    session_source_hash: str,
    session_pairs: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    chosen = summary.loc[
        (
            summary["task"].eq("cross_day")
            & summary["model"].isin(["persistence", "exposure_shift", "ridge_delta"])
        )
        | (
            summary["task"].eq("next_window")
            & summary["model"].isin(
                ["persistence", "transition_shift", "ridge_window_delta"]
            )
        )
    ]
    columns = [
        "task",
        "area",
        "model",
        "unsupervised_cv_nrmse",
        "supervised_test_nrmse",
        "supervised_improvement_vs_persistence_pct",
        "transfer_error_ratio",
    ]
    primary_tests = paired_tests.loc[
        paired_tests["model"].isin(["exposure_shift", "ridge_delta", "ridge_window_delta"])
    ]
    primary_uncertainty = uncertainty.loc[
        uncertainty["split"].eq("supervised_test")
        & uncertainty["model"].isin(
            ["persistence", "exposure_shift", "ridge_delta", "ridge_window_delta"]
        )
    ]
    primary_area_tests = area_tests.loc[
        area_tests["model"].isin(
            ["exposure_shift", "ridge_delta", "ridge_window_delta"]
        )
    ]
    raw_primary_area_tests = raw_area_tests.loc[
        raw_area_tests["model"].isin(
            ["persistence", "exposure_shift", "ridge_delta", "ridge_window_delta"]
        )
    ]
    moments = moment_summary.loc[
        moment_summary["split"].eq("supervised_test")
        & moment_summary["model"].isin(
            ["persistence", "transition_shift", "ridge_window_delta"]
        )
    ]
    support_summary = support.groupby("area", as_index=False).agg(
        mice=("mouse", "nunique"),
        mean_rms_training_z=("rms_training_z", "mean"),
        max_abs_training_z=("max_abs_training_z", "max"),
        mean_metrics_outside_range=("metrics_outside_training_range", "mean"),
    )
    grating_primary = grating_summary.loc[
        grating_summary.get("model", pd.Series(dtype=str)).isin(
            ["persistence", "exposure_shift", "ridge_delta"]
        )
    ] if not grating_summary.empty else grating_summary
    stable_report_link = Path(
        os.path.relpath(
            Path(__file__).resolve().with_name("STABLE_MODEL_REPORTS.md"),
            start=path.resolve().parent,
        )
    ).as_posix()
    text = f"""# Exposure-transfer analysis — {window_pairs}-pair windows

> For the standalone explanations of the models, equations, training
> chronology, and scientific interpretation, start with
> [`STABLE_MODEL_REPORTS.md`]({stable_report_link}).

This exploratory reanalysis fits and tunes models using only unsupervised mice,
then reports transfer to supervised mice. The published hypothesis and cohort
outcomes are already known, so the supervised cohort is not a pristine
prospective holdout.

## Exact estimands

- **Primary cross-day target:** one trial-balanced d-prime value per neuron,
  calculated from one response per eligible trial across the entire session,
  then summarized once per area. This target is independent of window size.
- **Secondary within-session target:** the immediately following strictly
  chronological, non-overlapping balanced block. Every target trial occurs
  after every input trial.
- Cross-day unit: one mouse-area before/after pair.
- Evaluation unit: one mouse. Metrics and windows are not independent animals.

## Protocol

- Chronological-window source SHA-256: `{source_hash}`
- Trial-balanced session source SHA-256: `{session_source_hash}`
- Metrics: {", ".join(config.metrics)}
- Window size: {window_pairs} leaf and {window_pairs} circle trials per block
- Paired rows: {len(session_pairs)}
- Window transitions: {len(transitions)}
- Training mice: {session_pairs.loc[session_pairs.cohort.eq(config.training_cohort), "mouse"].nunique()}
- Transfer mice: {session_pairs.loc[session_pairs.cohort.eq(config.transfer_cohort), "mouse"].nunique()}
- Model selection: nested leave-one-unsupervised-mouse-out CV
- Fitting weights: equal mouse weight, then equal session weight within mouse
- Output constraints: ordered quantiles, positive SD, valid leaf/circle fractions

## Main results

{markdown_table(chosen, columns)}

`NRMSE=1` corresponds to approximately one unsupervised-training standard
deviation across the configured target summaries. Improvement is relative to
the persistence prediction.

The mean exposure shift is the primary small-sample learned model. Ridge is a
predeclared extension; with nine training mice it should only be preferred
when nested unsupervised validation shows a material benefit.

## Mouse-bootstrap uncertainty

{markdown_table(primary_uncertainty, ["task", "area", "model", "mice", "mean_nrmse", "bootstrap_95_low", "bootstrap_95_high"])}

## Paired model-versus-persistence diagnostics

{markdown_table(primary_tests, ["task", "area", "model", "mice", "mean_error_improvement", "bootstrap_95_low", "bootstrap_95_high", "mice_improved", "exact_two_sided_pvalue"])}

## Primary medial-versus-anterior diagnostic

{markdown_table(primary_area_tests, ["task", "model", "mice", "mean_log_skill_difference", "bootstrap_95_low", "bootstrap_95_high", "all_mice_mHV_skill_greater", "exact_two_sided_pvalue"])}

Skill is `log(persistence error / learned-model error)`. The reported contrast
is mHV skill minus aHV skill, so a positive value addresses the hypothesis
without confusing transfer with the fact that aHV may simply be noisier.

![Per-mouse mHV versus aHV transfer skill](mHV_aHV_skill_by_mouse.png)

For transparency, the confounded raw-error diagnostic is retained below and
must not be used as the primary hypothesis test.

{markdown_table(raw_primary_area_tests, ["task", "model", "mice", "mean_difference", "all_mice_aHV_worse", "exact_two_sided_pvalue"])}

## Next-window before/after stratification

{markdown_table(moments, ["area", "moment", "model", "mice", "mean_nrmse", "improvement_vs_persistence_pct"])}

## Training-mouse deletion sensitivity

{markdown_table(training_sensitivity_summary, ["task", "area", "model", "deletion_mean_nrmse", "deletion_min_nrmse", "deletion_max_nrmse"])}

## Baseline cohort support

{markdown_table(support_summary, ["area", "mice", "mean_rms_training_z", "max_abs_training_z", "mean_metrics_outside_range"])}

## Separate grating-cohort control

The frozen unsupervised model was also applied to the grating cohort without
refitting. Two of its three mice have replicated sessions per moment, which
are averaged within mouse before scoring; this is a descriptive control, not
the primary estimand.

{markdown_table(grating_primary, ["area", "model", "mice", "mean_nrmse", "improvement_vs_persistence_pct"])}

Bootstrap intervals resample mice and are descriptive with these small cohorts.
With four supervised mice, the smallest possible two-sided paired sign-flip
p-value is 0.125.

## Guardrails

- Cross-day predictions concern area-level population summaries, not a
  registered neuron's trajectory. Same-cell cross-day identities are absent.
- Supervised outcomes must not tune windows, metrics, clusters, or penalties.
- The next-window analysis establishes forecastability, not exposure causality.
- Successful transfer is consistent with an exposure-general rule, but does
  not prove exposure is causal; repeated-session, grating and naive controls
  are required for that stronger statement.
- Behavior and clustering remain explicitly labeled ablations.
- Window-size and cluster-count searches are exploratory because this released
  supervised cohort has already been inspected. Confirmation requires a
  locked analysis on a new cohort.
"""
    path.write_text(text)


def write_configuration(
    path: Path,
    *,
    config: StudyConfig,
    history_path: Path,
    history_hash: str,
    session_summary_path: Path,
    session_summary_hash: str,
    grating_control_path: Path | None,
    grating_control_hash: str | None,
    window_pairs: int,
    model_names: Mapping[str, Sequence[str]],
) -> None:
    dependencies = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "duckdb"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    payload = {
        **config.as_dict(),
        "history": str(history_path.resolve()),
        "history_sha256": history_hash,
        "session_summary": str(session_summary_path.resolve()),
        "session_summary_sha256": session_summary_hash,
        "grating_control": (
            str(grating_control_path.resolve()) if grating_control_path else None
        ),
        "grating_control_sha256": grating_control_hash,
        "window_pairs": window_pairs,
        "cv_unit": "mouse",
        "cross_day_estimand": "trial-balanced full-session population summary",
        "next_window_estimand": "strictly chronological disjoint balanced blocks",
        "target_transform": "ordered quantiles + log SD + 3-part fraction composition",
        "models": {key: list(value) for key, value in model_names.items()},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
