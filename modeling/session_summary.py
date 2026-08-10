"""Full-session, trial-balanced neuronal selectivity summaries.

This module defines the primary session-level estimand used by the transfer
analysis.  A trial, rather than a frame, is the observational unit:

1. average the eligible neural frames within each trial;
2. deterministically select the same number of role-2 and role-0 trials;
3. compute d-prime across trial responses for each sampled neuron; and
4. summarize the finite neuron-level values within each retinotopic area.

A zero or non-finite denominator is never replaced by zero. The corresponding
d-prime remains missing and the invalid denominator is counted in every
area-level output row. The chronological window estimator follows this same
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


SESSION_ESTIMATOR = "full_session_trial_balanced_dprime"
SESSION_ESTIMATOR_VERSION = "1.2.0"
TRIAL_RESPONSE_ESTIMATOR = "mean_eligible_frame_activity_per_trial"
TRIAL_BALANCE_METHOD = "deterministic_evenly_spaced_without_replacement"
DENOMINATOR_ESTIMATOR = "population_sd_role2_plus_population_sd_role0_no_epsilon"
NEURON_SEED_METHOD = "sha256_base_seed_and_recording_id_first_uint32"
BEHAVIOR_ESTIMATOR = "mean_of_trial_mean_over_balanced_neural_trial_sample"


@dataclass(frozen=True)
class TrialBalancedDPrime:
    """Neuron-level result and the exact balanced trial sample that produced it."""

    values: NDArray[np.float64]
    denominators: NDArray[np.float64]
    valid_denominator: NDArray[np.bool_]
    trial_ids_by_role: Mapping[int, NDArray[np.int64]]
    eligible_trial_counts_by_role: Mapping[int, int]


def _validated_trials(trial_frames: pd.DataFrame) -> pd.DataFrame:
    required = {"frame_id", "trial_id", "stimulus_role"}
    missing = required - set(trial_frames.columns)
    if missing:
        raise ValueError(f"trial_frames is missing {sorted(missing)}")

    trials = trial_frames[["trial_id", "stimulus_role"]].drop_duplicates()
    if trials.empty:
        raise ValueError("trial_frames contains no eligible trials")
    roles_per_trial = trials.groupby("trial_id")["stimulus_role"].nunique()
    if (roles_per_trial != 1).any():
        raise ValueError("each trial must have exactly one stimulus role")
    for column in ("frame_id", "trial_id", "stimulus_role"):
        numeric = pd.to_numeric(trial_frames[column], errors="coerce")
        if (
            not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise ValueError(f"{column} must contain finite integer identifiers")
    return trials.astype({"trial_id": "int64", "stimulus_role": "int64"}).sort_values(
        "trial_id"
    )


def _evenly_spaced_indices(population_size: int, sample_size: int) -> NDArray[np.int64]:
    """Select deterministic bin midpoints without replacement."""

    if not 0 < sample_size <= population_size:
        raise ValueError("sample_size must be between one and population_size")
    indices = np.floor(
        (np.arange(sample_size, dtype=np.float64) + 0.5)
        * population_size
        / sample_size
    ).astype(np.int64)
    if len(np.unique(indices)) != sample_size:
        raise RuntimeError("evenly spaced trial selection produced duplicate indices")
    return indices


def balanced_trial_ids(
    trial_frames: pd.DataFrame,
    *,
    roles: tuple[int, int] = (2, 0),
) -> tuple[dict[int, NDArray[np.int64]], dict[int, int]]:
    """Return an equal, deterministic sample of eligible trials for two roles.

    The smaller role contributes all of its trials.  The larger role is sampled
    at evenly spaced ordinal positions, which preserves coverage across the
    session without adding a random seed or privileging only early trials.
    """

    if len(roles) != 2 or roles[0] == roles[1]:
        raise ValueError("roles must contain two distinct stimulus roles")
    trials = _validated_trials(trial_frames)
    available = {
        int(role): trials.loc[trials["stimulus_role"].eq(role), "trial_id"].to_numpy(
            dtype=np.int64
        )
        for role in roles
    }
    counts = {role: len(ids) for role, ids in available.items()}
    balanced_count = min(counts.values())
    if balanced_count < 2:
        raise ValueError(
            "at least two eligible trials per role are required for trial-level d-prime"
        )
    selected = {
        role: ids[_evenly_spaced_indices(len(ids), balanced_count)]
        for role, ids in available.items()
    }
    return selected, counts


def recording_neuron_seed(recording_id: str, base_seed: int = 2025) -> int:
    """Derive one stable uint32 sampling seed per recording and base seed."""

    if isinstance(base_seed, (bool, np.bool_)) or not isinstance(
        base_seed, (int, np.integer)
    ):
        raise ValueError("base_seed must be an integer")
    if not 0 <= int(base_seed) <= np.iinfo(np.uint32).max:
        raise ValueError("base_seed must fit in an unsigned 32-bit integer")
    payload = f"{int(base_seed)}\0{recording_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _component_trial_means(
    v_components_by_frame: NDArray,
    trial_frames: pd.DataFrame,
    trial_ids: Sequence[int],
) -> NDArray[np.float64]:
    """Average component activity over eligible frames once per selected trial."""

    frames = trial_frames[["frame_id", "trial_id"]].copy()
    frames["frame_id"] = pd.to_numeric(frames["frame_id"]).astype("int64")
    frames["trial_id"] = pd.to_numeric(frames["trial_id"]).astype("int64")
    duplicated = frames["frame_id"].duplicated()
    if duplicated.any():
        raise ValueError("trial_frames must contain every frame exactly once")

    means: list[NDArray[np.float64]] = []
    for trial_id in trial_ids:
        frame_ids = frames.loc[frames["trial_id"].eq(int(trial_id)), "frame_id"].to_numpy(
            dtype=np.int64
        )
        if len(frame_ids) == 0:
            raise ValueError(f"selected trial {trial_id} has no eligible frames")
        if frame_ids.min() < 0 or frame_ids.max() >= v_components_by_frame.shape[1]:
            raise ValueError("trial_frames contains an out-of-range frame_id")
        block = np.asarray(v_components_by_frame[:, frame_ids], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("eligible neural component activity must be finite")
        means.append(np.mean(block, axis=1, dtype=np.float64))
    return np.stack(means, axis=1)


def trial_balanced_dprime(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    trial_frames: pd.DataFrame,
    *,
    neuron_ids: ArrayLike,
    roles: tuple[int, int] = (2, 0),
) -> TrialBalancedDPrime:
    """Compute per-neuron d-prime from equally weighted trial responses.

    The standard deviations use ``ddof=0`` to retain the definition used in
    the source analyses.  Invalid denominators produce ``NaN`` d-prime values;
    no epsilon, clipping, or non-finite-to-zero conversion is applied.
    """

    selected, eligible_counts = balanced_trial_ids(trial_frames, roles=roles)
    return dprime_for_balanced_trial_ids(
        u_components_by_neuron,
        v_components_by_frame,
        trial_frames,
        neuron_ids=neuron_ids,
        trial_ids_by_role=selected,
        eligible_trial_counts_by_role=eligible_counts,
        roles=roles,
    )


def dprime_for_balanced_trial_ids(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    trial_frames: pd.DataFrame,
    *,
    neuron_ids: ArrayLike,
    trial_ids_by_role: Mapping[int, ArrayLike],
    roles: tuple[int, int] = (2, 0),
    eligible_trial_counts_by_role: Mapping[int, int] | None = None,
) -> TrialBalancedDPrime:
    """Compute trial-level d-prime for an explicit equal sample of both roles."""

    if len(roles) != 2 or roles[0] == roles[1]:
        raise ValueError("roles must contain two distinct stimulus roles")
    u = np.asarray(u_components_by_neuron)
    v = np.asarray(v_components_by_frame)
    ids = np.asarray(neuron_ids)
    if u.ndim != 2 or v.ndim != 2 or u.shape[0] != v.shape[0]:
        raise ValueError("U and V must be two-dimensional with aligned components")
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("neuron_ids must be a non-empty one-dimensional vector")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("neuron_ids must contain integers")
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= u.shape[1] or len(np.unique(ids)) != len(ids):
        raise ValueError("neuron_ids must be unique and in range")
    weights = np.asarray(u[:, ids], dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("selected neural component weights must be finite")

    trials = _validated_trials(trial_frames)
    role_by_trial = trials.set_index("trial_id")["stimulus_role"]
    selected: dict[int, NDArray[np.int64]] = {}
    for role in roles:
        if role not in trial_ids_by_role:
            raise ValueError(f"trial_ids_by_role is missing role {role}")
        role_ids = np.asarray(trial_ids_by_role[role])
        if role_ids.ndim != 1 or len(role_ids) < 2:
            raise ValueError("each role must provide at least two trial ids")
        if not np.issubdtype(role_ids.dtype, np.integer):
            raise ValueError("trial ids must be integers")
        role_ids = role_ids.astype(np.int64, copy=False)
        if len(np.unique(role_ids)) != len(role_ids):
            raise ValueError("trial ids must be unique within each role")
        observed_roles = role_by_trial.reindex(role_ids)
        if observed_roles.isna().any() or not observed_roles.eq(role).all():
            raise ValueError(f"selected trials do not all have stimulus role {role}")
        selected[role] = role_ids
    if len(selected[roles[0]]) != len(selected[roles[1]]):
        raise ValueError("the two stimulus roles must have equal trial counts")
    if np.intersect1d(selected[roles[0]], selected[roles[1]]).size:
        raise ValueError("the two stimulus roles must use disjoint trials")
    if eligible_trial_counts_by_role is None:
        eligible_counts = {role: len(selected[role]) for role in roles}
    else:
        eligible_counts = {
            role: int(eligible_trial_counts_by_role[role]) for role in roles
        }

    responses: dict[int, NDArray[np.float64]] = {}
    for role in roles:
        component_means = _component_trial_means(v, trial_frames, selected[role])
        responses[role] = weights.T @ component_means

    role_a, role_b = roles
    mean_a = np.mean(responses[role_a], axis=1, dtype=np.float64)
    mean_b = np.mean(responses[role_b], axis=1, dtype=np.float64)
    sd_a = np.std(responses[role_a], axis=1, ddof=0, dtype=np.float64)
    sd_b = np.std(responses[role_b], axis=1, ddof=0, dtype=np.float64)
    denominators = sd_a + sd_b
    valid_denominator = np.isfinite(denominators) & (denominators > 0)
    valid = valid_denominator & np.isfinite(mean_a) & np.isfinite(mean_b)
    values = np.full(len(ids), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = 2.0 * (mean_a[valid] - mean_b[valid]) / denominators[valid]

    return TrialBalancedDPrime(
        values=values,
        denominators=denominators,
        valid_denominator=valid_denominator,
        trial_ids_by_role=selected,
        eligible_trial_counts_by_role=eligible_counts,
    )


def _trial_sample_digest(trials_by_role: Mapping[int, NDArray[np.int64]]) -> str:
    digest = hashlib.sha256()
    for role in sorted(trials_by_role):
        digest.update(f"role={role}:".encode())
        digest.update(np.asarray(trials_by_role[role], dtype="<i8").tobytes())
    return digest.hexdigest()


def balanced_trial_covariate_mean(
    frame_covariate: pd.DataFrame,
    trial_frames: pd.DataFrame,
    trials_by_role: Mapping[int, NDArray[np.int64]],
    *,
    value_column: str,
) -> tuple[float, int]:
    """Average a frame covariate within trial and then equally across trials."""

    required = {"frame_id", value_column}
    missing = required - set(frame_covariate.columns)
    if missing:
        raise ValueError(f"frame_covariate is missing {sorted(missing)}")
    if frame_covariate["frame_id"].duplicated().any():
        raise ValueError("frame_covariate must contain each frame once")
    eligible = trial_frames[["frame_id", "trial_id"]].merge(
        frame_covariate[["frame_id", value_column]],
        on="frame_id",
        how="left",
        validate="one_to_one",
    )
    per_trial = eligible.groupby("trial_id")[value_column].mean()
    selected_ids = np.concatenate(
        [np.asarray(trials_by_role[role], dtype=np.int64) for role in trials_by_role]
    )
    selected = per_trial.reindex(selected_ids).to_numpy(dtype=np.float64)
    finite = selected[np.isfinite(selected)]
    return (
        float(np.mean(finite)) if len(finite) else np.nan,
        int(len(finite)),
    )


def summarize_trial_balanced_dprime(
    result: TrialBalancedDPrime,
    neurons: pd.DataFrame,
    *,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Summarize neuron-level values while retaining invalidity diagnostics."""

    required = {"neuron_id", "area_group"}
    missing = required - set(neurons.columns)
    if missing:
        raise ValueError(f"neurons is missing {sorted(missing)}")
    if len(neurons) != len(result.values):
        raise ValueError("neurons and d-prime values must be aligned")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    rows: list[dict[str, object]] = []
    areas = neurons["area_group"].to_numpy()
    for area in pd.unique(areas):
        mask = areas == area
        values = result.values[mask]
        finite = values[np.isfinite(values)]
        invalid_denominator = int(np.sum(~result.valid_denominator[mask]))
        if len(finite):
            center = float(np.mean(finite))
            scale = float(np.std(finite, ddof=0))
            normalized = (
                np.zeros_like(finite) if scale == 0 else (finite - center) / scale
            )
            metrics = {
                "mean_dprime": center,
                "mean_abs_dprime": float(np.mean(np.abs(finite))),
                "sd_dprime": scale,
                "skewness": float(np.mean(normalized**3)),
                "excess_kurtosis": float(np.mean(normalized**4) - 3.0),
                "q05": float(np.quantile(finite, 0.05)),
                "median": float(np.median(finite)),
                "q95": float(np.quantile(finite, 0.95)),
                "frac_selective": float(np.mean(np.abs(finite) >= threshold)),
                "frac_leaf_selective": float(np.mean(finite >= threshold)),
                "frac_circle_selective": float(np.mean(finite <= -threshold)),
            }
        else:
            metrics = {
                name: np.nan
                for name in (
                    "mean_dprime",
                    "mean_abs_dprime",
                    "sd_dprime",
                    "skewness",
                    "excess_kurtosis",
                    "q05",
                    "median",
                    "q95",
                    "frac_selective",
                    "frac_leaf_selective",
                    "frac_circle_selective",
                )
            }
        rows.append(
            {
                "area": str(area),
                "n_neurons": int(mask.sum()),
                "n_finite": int(len(finite)),
                "n_valid_dprime": int(len(finite)),
                "n_invalid_dprime": int(mask.sum() - len(finite)),
                "n_invalid_denominator": invalid_denominator,
                "frac_invalid_dprime": float(1.0 - len(finite) / mask.sum()),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def session_trial_balanced_summary(
    db,
    session,
    *,
    areas: Sequence[str] = ("V1", "mHV", "lHV", "aHV"),
    neurons_per_area: int = 1000,
    threshold: float = 0.3,
    min_frames: int = 3,
    roles: tuple[int, int] = (2, 0),
    neuron_seed: int = 2025,
) -> pd.DataFrame:
    """Compute the primary, full-session summary for one recording."""

    from dprime import balanced_area_neurons, eligible_trial_frames
    from joiner import Joiner, session_metadata

    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    joiner = Joiner(
        db,
        session.recording_id,
        experiment=session.experiment,
        behavior_key=session.behavior_key,
    )
    frames = eligible_trial_frames(joiner.frames, roles=roles, min_frames=min_frames)
    seed = recording_neuron_seed(session.recording_id, neuron_seed)
    neurons = balanced_area_neurons(
        joiner.neurons,
        areas=areas,
        per_area=neurons_per_area,
        seed=seed,
    )
    result = trial_balanced_dprime(
        joiner.U,
        joiner.V,
        frames,
        neuron_ids=neurons["neuron_id"].to_numpy(dtype=np.int64),
        roles=roles,
    )
    mean_run_speed, valid_run_speed_trials = balanced_trial_covariate_mean(
        joiner.frames[["frame_id", "run_speed"]],
        frames,
        result.trial_ids_by_role,
        value_column="run_speed",
    )
    summary = summarize_trial_balanced_dprime(result, neurons, threshold=threshold)
    available = joiner.neurons.groupby("area_group")["neuron_id"].size()
    summary["available_neurons"] = summary["area"].map(available).astype(int)
    balanced_count = len(result.trial_ids_by_role[roles[0]])
    summary = summary.assign(
        **session_metadata(session),
        estimator=SESSION_ESTIMATOR,
        estimator_version=SESSION_ESTIMATOR_VERSION,
        trial_response_estimator=TRIAL_RESPONSE_ESTIMATOR,
        trial_balance_method=TRIAL_BALANCE_METHOD,
        denominator_estimator=DENOMINATOR_ESTIMATOR,
        behavior_estimator=BEHAVIOR_ESTIMATOR,
        mean_run_speed=mean_run_speed,
        n_valid_run_speed_trials=valid_run_speed_trials,
        role_positive=int(roles[0]),
        role_negative=int(roles[1]),
        eligible_trials_role_positive=int(
            result.eligible_trial_counts_by_role[roles[0]]
        ),
        eligible_trials_role_negative=int(
            result.eligible_trial_counts_by_role[roles[1]]
        ),
        balanced_trials_per_role=balanced_count,
        selected_trial_sample_sha256=_trial_sample_digest(result.trial_ids_by_role),
        eligible_frame_count=len(frames),
        minimum_eligible_frames_per_trial=int(min_frames),
        selectivity_threshold=float(threshold),
        neuron_sampling_seed=int(seed),
        neuron_sampling_base_seed=int(neuron_seed),
        neuron_sampling_seed_method=NEURON_SEED_METHOD,
        requested_neurons_per_area=int(neurons_per_area),
        trial_count=int(getattr(session, "trial_count", len(joiner.trials))),
        recorded_neurons=int(joiner.U.shape[1]),
        window_size_independent=True,
    )
    return summary
