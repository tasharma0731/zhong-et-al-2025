from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.config import DEFAULT_METRICS, StudyConfig
from modeling.diagnostics import area_contrasts
from modeling.estimators import (
    CROSS_DAY_MODELS,
    fit_model,
    mouse_sample_weights,
)
from modeling.preparation import mouse_balanced_scale
from modeling.targets import DistributionTargetTransform
from modeling.validation import evaluate_task


METRICS = DEFAULT_METRICS


def _valid_values(rows: int = 3) -> np.ndarray:
    center = np.linspace(-0.1, 0.1, rows)
    return np.column_stack(
        [
            center - 0.8,
            center,
            center + 1.0,
            np.linspace(0.4, 0.8, rows),
            np.linspace(0.1, 0.2, rows),
            np.linspace(0.2, 0.1, rows),
        ]
    )


def _cross_day_frame(mice: list[str], cohorts: list[str]) -> pd.DataFrame:
    before = _valid_values(len(mice))
    after = before.copy()
    after[:, :3] += 0.15
    after[:, 3] *= 1.1
    result = pd.DataFrame({"mouse": mice, "cohort": cohorts, "area": "mHV"})
    result["before_mean_run_speed"] = np.arange(len(result), dtype=float) + 1
    for index, metric in enumerate(METRICS):
        result[f"before_{metric}"] = before[:, index]
        result[f"after_{metric}"] = after[:, index]
    return result


def test_target_transform_round_trip_and_constraints() -> None:
    transform = DistributionTargetTransform(METRICS)
    values = _valid_values(5)
    restored = transform.inverse_transform(transform.transform(values))
    np.testing.assert_allclose(restored, values, atol=1e-10)
    transform.validate(restored)

    unconstrained = np.full((4, len(METRICS)), 100.0)
    prediction = transform.inverse_transform(unconstrained)
    transform.validate(prediction)
    assert np.all(prediction[:, 0] <= prediction[:, 1])
    assert np.all(prediction[:, 1] <= prediction[:, 2])
    assert np.all(prediction[:, 3] > 0)
    assert np.all(prediction[:, 4:6].sum(axis=1) <= 1)


def test_mouse_weights_and_scales_ignore_row_duplication() -> None:
    base = pd.DataFrame({"mouse": ["a", "b"]})
    duplicated = pd.concat([base.iloc[[0]]] * 10 + [base.iloc[[1]]], ignore_index=True)
    weights = mouse_sample_weights(duplicated)
    totals = pd.Series(weights).groupby(duplicated["mouse"]).sum()
    np.testing.assert_allclose(totals.to_numpy(), [len(weights) / 2] * 2)

    base_values = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    duplicate_values = np.vstack([np.repeat(base_values[[0]], 10, axis=0), base_values[[1]]])
    np.testing.assert_allclose(
        mouse_balanced_scale(base_values, ["a", "b"]),
        mouse_balanced_scale(duplicate_values, ["a"] * 10 + ["b"]),
    )


def test_fitted_ridge_always_returns_valid_distribution() -> None:
    frame = _cross_day_frame(
        [f"m{index}" for index in range(6)], ["unsupervised"] * 6
    )
    spec = next(spec for spec in CROSS_DAY_MODELS if spec.name == "ridge_delta")
    model = fit_model(frame, spec, metrics=METRICS, alpha=0.01)
    extreme = frame.iloc[[0]].copy()
    extreme[[f"before_{metric}" for metric in METRICS]] *= 100
    prediction = model.predict(extreme)
    DistributionTargetTransform(METRICS).validate(prediction)


def test_supervised_rows_never_reach_fit(monkeypatch) -> None:
    frame = _cross_day_frame(
        ["u1", "u2", "u3", "s1"],
        ["unsupervised", "unsupervised", "unsupervised", "supervised"],
    )
    import modeling.validation as validation

    original = validation.fit_model
    seen: list[set[str]] = []

    def spy(train, *args, **kwargs):
        seen.append(set(train["cohort"]))
        return original(train, *args, **kwargs)

    monkeypatch.setattr(validation, "fit_model", spy)
    config = StudyConfig(areas=("mHV",), bootstrap_repeats=100)
    persistence = next(
        spec for spec in CROSS_DAY_MODELS if spec.name == "persistence"
    )
    evaluate_task(frame, specs=(persistence,), config=config)
    assert seen and all(cohorts == {"unsupervised"} for cohorts in seen)


def test_area_contrast_uses_skill_not_raw_error() -> None:
    rows = []
    for mouse in ("s1", "s2", "s3", "s4"):
        for area, persistence, model in (
            ("mHV", 2.0, 1.0),
            ("aHV", 10.0, 8.0),
        ):
            for name, error in (("persistence", persistence), ("exposure_shift", model)):
                rows.append(
                    {
                        "task": "cross_day",
                        "split": "supervised_test",
                        "model": name,
                        "area": area,
                        "mouse": mouse,
                        "nrmse": error,
                    }
                )
    result = area_contrasts(
        pd.DataFrame(rows), StudyConfig(bootstrap_repeats=100)
    ).iloc[0]
    assert result["mean_log_skill_difference"] > 0
    assert bool(result["all_mice_mHV_skill_greater"])
