from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modeling.experiments.boundary_population_forecaster import (
    METRICS,
    BoundaryCandidate,
    BoundaryPopulationData,
    build_boundary_population_data,
    candidate_grid,
    evaluate_boundary_population_forecaster,
    fit_boundary_candidate,
)
from modeling.targets import DistributionTargetTransform


def _population_state(
    center: float,
    spread: float,
    leaf: float,
    circle: float,
) -> np.ndarray:
    return np.asarray(
        [
            center - 1.6 * spread,
            center,
            center + 1.7 * spread,
            spread,
            leaf,
            circle,
        ],
        dtype=float,
    )


def _direct_data(
    *,
    unsupervised_mice: int = 5,
    supervised_mice: int = 2,
    target_is_persistence: bool = False,
) -> BoundaryPopulationData:
    records: list[dict[str, object]] = []
    whole: list[np.ndarray] = []
    previous_full: list[np.ndarray] = []
    current_full: list[np.ndarray] = []
    previous_svd: list[np.ndarray] = []
    current_svd: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    transform = DistributionTargetTransform(METRICS)

    cohorts = (
        [("unsupervised", f"U{index}") for index in range(unsupervised_mice)]
        + [("supervised", f"S{index}") for index in range(supervised_mice)]
    )
    for position, (cohort, mouse) in enumerate(cohorts):
        anchor = _population_state(
            center=-0.03 + 0.015 * position,
            spread=0.65 + 0.025 * position,
            leaf=0.24 + 0.008 * position,
            circle=0.22 - 0.004 * position,
        )
        whole_state = _population_state(
            center=anchor[1] - 0.02,
            spread=anchor[3] * 0.94,
            leaf=max(0.05, anchor[4] - 0.015),
            circle=max(0.05, anchor[5] + 0.01),
        )
        prior = _population_state(
            center=anchor[1] - 0.01,
            spread=anchor[3] * 0.96,
            leaf=max(0.05, anchor[4] - 0.01),
            circle=max(0.05, anchor[5] + 0.005),
        )
        z_anchor = transform.transform(anchor[None, :])[0]
        if target_is_persistence:
            target = np.repeat(anchor[None, :], 2, axis=0)
        else:
            residual = np.asarray(
                [
                    [0.03, 0.08, 0.10, 0.06, 0.08, -0.03],
                    [0.05, 0.12, 0.15, 0.09, 0.12, -0.04],
                ]
            )
            residual *= 1.0 + 0.03 * position
            target = transform.inverse_transform(z_anchor[None, :] + residual)

        records.append(
            {
                "mouse": mouse,
                "cohort": cohort,
                "area": "mHV",
                "before_behavior_session_id": f"{mouse}_before",
            }
        )
        whole.append(whole_state)
        previous_full.append(prior)
        current_full.append(anchor)
        previous_svd.append(prior * np.asarray([0.98, 1, 0.98, 0.98, 1, 1]))
        current_svd.append(
            anchor * np.asarray([0.98, 1, 0.98, 0.98, 1, 1])
        )
        behavior.append(np.asarray([2.0 + position, 2.5 + position]))
        targets.append(target)

    return BoundaryPopulationData(
        metadata=pd.DataFrame.from_records(records),
        whole_before=np.vstack(whole),
        previous_full=np.vstack(previous_full),
        current_full=np.vstack(current_full),
        previous_svd=np.vstack(previous_svd),
        current_svd=np.vstack(current_svd),
        behavior=np.vstack(behavior),
        targets=np.stack(targets),
    )


def _cache_for_builder() -> SimpleNamespace:
    rows: list[dict[str, object]] = []
    current_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    current_run_speed: list[float] = []
    mice = (
        [("unsupervised", f"U{index}") for index in range(4)]
        + [("supervised", "S0")]
    )
    neuron_axis = np.linspace(-1.4, 1.5, 12)

    for mouse_position, (cohort, mouse) in enumerate(mice):
        for moment_position, moment in enumerate(("before", "after")):
            session = f"{mouse}_{moment}"
            shift = 0.04 * mouse_position + 0.15 * moment_position
            windows = [
                neuron_axis * (0.75 + 0.04 * window) + shift
                for window in range(3)
            ]
            for transition in range(2):
                rows.append(
                    {
                        "behavior_session_id": session,
                        "recording_id": f"{session}_recording",
                        "mouse": mouse,
                        "cohort": cohort,
                        "moment": moment,
                        "area": "mHV",
                        "current_start_pair_id": 20 * transition,
                        "next_start_pair_id": 20 * (transition + 1),
                        "current_run_speed": (
                            2.0
                            + mouse_position
                            + 0.25 * moment_position
                            + transition
                        ),
                    }
                )
                current_rows.append(windows[transition])
                future_rows.append(windows[transition + 1])
                current_run_speed.append(rows[-1]["current_run_speed"])

    current = np.vstack(current_rows)
    future = np.vstack(future_rows)
    current_denominator = np.ones_like(current)
    next_denominator = np.ones_like(future)
    current_mean_leaf = 0.5 * current
    next_mean_leaf = 0.5 * future
    zeros = np.zeros_like(current)
    half = np.full_like(current, 0.5)
    cache = SimpleNamespace(
        metadata=pd.DataFrame.from_records(rows),
        window_pairs=20,
        current=current,
        future=future,
        current_denominator=current_denominator,
        next_denominator=next_denominator,
        current_mean_leaf=current_mean_leaf,
        current_mean_circle=zeros.copy(),
        current_sd_leaf=half.copy(),
        current_sd_circle=half.copy(),
        next_mean_leaf=next_mean_leaf,
        next_mean_circle=zeros.copy(),
        next_sd_leaf=half.copy(),
        next_sd_circle=half.copy(),
        svd_current=0.92 * current,
        svd_future=0.92 * future,
        svd_current_denominator=np.ones_like(current),
        svd_next_denominator=np.ones_like(future),
        neurons_per_transition=current.shape[1],
    )
    return cache


def test_builder_creates_one_complete_example_and_declared_feature_blocks() -> None:
    data = build_boundary_population_data(_cache_for_builder())

    assert len(data.metadata) == 5
    assert not data.metadata.duplicated(["mouse", "cohort", "area"]).any()
    assert data.targets.shape == (5, 2, 6)
    expected_columns = {
        "session_only": 6,
        "dual_full": 18,
        "dual_full_svd": 30,
        "dual_full_svd_behavior": 32,
    }
    for variant, columns in expected_columns.items():
        matrix, names = data.feature_matrix(variant)
        assert matrix.shape == (5, columns)
        assert len(names) == columns
        assert np.isfinite(matrix).all()


def test_ridge_is_future_blind_and_strong_shrinkage_is_mean_shift() -> None:
    data = _direct_data()
    training = np.arange(5)
    validation = np.asarray([5])
    ridge_candidate = BoundaryCandidate("session_only", 1e12)
    mean_candidate = BoundaryCandidate("mean_shift", None)
    fitted = fit_boundary_candidate(
        data,
        training,
        ridge_candidate,
        horizon=0,
    )
    mean_shift = fit_boundary_candidate(
        data,
        training,
        mean_candidate,
        horizon=0,
    )

    expected = fitted.predict(data, validation)
    changed = copy.deepcopy(data)
    changed.targets[validation] = changed.targets[validation, ::-1] * np.asarray(
        [0.7, 1, 0.7, 0.7, 1, 1]
    )
    observed = fitted.predict(changed, validation)

    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(
        expected,
        mean_shift.predict(data, validation),
        atol=1e-9,
        rtol=0,
    )
    assert expected.shape == (1, len(METRICS))
    assert len(fitted.heads) == len(METRICS)
    assert all(head.fit_intercept for head in fitted.heads)
    DistributionTargetTransform(METRICS).validate(expected)


def test_nested_selection_can_choose_exact_persistence_without_transfer_leakage() -> None:
    data = _direct_data(target_is_persistence=True)
    candidates = (
        BoundaryCandidate("persistence", None),
        BoundaryCandidate("mean_shift", None),
        BoundaryCandidate("session_only", 1.0),
        BoundaryCandidate("dual_full_svd_behavior", 100.0),
    )
    first = evaluate_boundary_population_forecaster(
        data,
        candidates=candidates,
        areas=("mHV",),
        selection_rule="minimum_cv",
    )
    changed = copy.deepcopy(data)
    supervised = np.flatnonzero(
        changed.metadata["cohort"].astype(str).eq("supervised").to_numpy()
    )
    transform = DistributionTargetTransform(METRICS)
    changed.targets[supervised] = transform.inverse_transform(
        transform.transform(
            changed.targets[supervised].reshape(-1, len(METRICS))
        )
        + 0.5
    ).reshape(len(supervised), 2, len(METRICS))
    second = evaluate_boundary_population_forecaster(
        changed,
        candidates=candidates,
        areas=("mHV",),
        selection_rule="minimum_cv",
    )

    assert set(first.selections["selected_candidate"]) == {"persistence"}
    pd.testing.assert_frame_equal(
        first.selections.reset_index(drop=True),
        second.selections.reset_index(drop=True),
    )
    first_nested = first.predictions.loc[
        first.predictions["split"].eq("unsupervised_lomo")
    ].reset_index(drop=True)
    second_nested = second.predictions.loc[
        second.predictions["split"].eq("unsupervised_lomo")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_nested, second_nested)
    assert set(first.final_models) == {("mHV", 0), ("mHV", 1)}
    assert set(first.predictions["model"]) == {
        "selected_nested",
        "persistence",
    }
    assert {
        "absolute_error",
        "scaled_absolute_error",
        "scaled_squared_error",
    }.issubset(first.predictions)


def test_candidate_grid_contains_exact_persistence_and_strong_ridge() -> None:
    candidates = candidate_grid()

    assert candidates[0].name == "persistence"
    assert candidates[1].name == "mean_shift"
    assert any(candidate.alpha == 10_000 for candidate in candidates[2:])
    assert {
        candidate.feature_variant
        for candidate in candidates
        if candidate.feature_variant
        not in ("persistence", "mean_shift")
    } == {
        "session_only",
        "dual_full",
        "dual_full_svd",
        "dual_full_svd_behavior",
    }
