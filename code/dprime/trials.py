from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from position import align_trailing_behavior_frames, bin_trial_features, decimeters_to_meters


AREA_IDS: Mapping[str, tuple[int, ...]] = {
    "V1": (8,),
    "mHV": (0, 1, 2, 9),
    "lHV": (5, 6),
    "aHV": (3, 4),
}
AREAS = tuple(AREA_IDS)


def _positive_integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from error
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(numeric)


def _frame_indices(
    selector: ArrayLike,
    *,
    n_frames: int,
    name: str,
) -> NDArray[np.intp]:
    """Normalize one boolean mask or integer index vector."""

    values = np.asarray(selector)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional frame selector")
    if len(values) == 0:
        raise ValueError(f"{name} must select at least one frame")
    if np.issubdtype(values.dtype, np.bool_):
        if len(values) != n_frames:
            raise ValueError(f"{name} boolean mask must have one value per V frame")
        indices = np.flatnonzero(values)
    elif np.issubdtype(values.dtype, np.integer):
        indices = values.astype(np.intp, copy=False)
        if np.any(indices < 0) or np.any(indices >= n_frames):
            raise ValueError(f"{name} contains an out-of-range frame index")
    else:
        raise ValueError(f"{name} must be a boolean mask or integer frame indices")
    if len(indices) == 0:
        raise ValueError(f"{name} must select at least one frame")
    return indices


def _svd_projected_group_moments(
    u: NDArray,
    v: NDArray,
    frame_indices: NDArray[np.intp],
    *,
    neuron_chunk_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-neuron mean and population SD for one frame group."""

    block = np.asarray(v[:, frame_indices], dtype=np.float64)
    valid_frames = ~np.isnan(block).any(axis=0)
    if not np.all(valid_frames):
        block = block[:, valid_frames]
    if block.shape[1] == 0:
        return (
            np.full(u.shape[1], np.nan, dtype=np.float64),
            np.full(u.shape[1], np.nan, dtype=np.float64),
        )

    component_mean = np.mean(block, axis=1)
    block -= component_mean[:, None]
    with np.errstate(all="ignore"):
        component_covariance = (block @ block.T) / block.shape[1]

    n_neurons = u.shape[1]
    mean = np.empty(n_neurons, dtype=np.float64)
    standard_deviation = np.empty(n_neurons, dtype=np.float64)
    for start in range(0, n_neurons, neuron_chunk_size):
        stop = min(start + neuron_chunk_size, n_neurons)
        weights = np.asarray(u[:, start:stop], dtype=np.float64)
        with np.errstate(all="ignore"):
            mean[start:stop] = weights.T @ component_mean
            covariance_times_weights = component_covariance @ weights
            variance = np.sum(weights * covariance_times_weights, axis=0)
        standard_deviation[start:stop] = np.sqrt(np.maximum(variance, 0.0))
    return mean, standard_deviation


def svd_dprime_contrasts(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    frame_groups: Mapping[Any, ArrayLike],
    contrasts: Mapping[str, tuple[Any, Any]],
    *,
    neuron_chunk_size: int = 4096,
) -> dict[str, NDArray[np.float64]]:
    """Compute several per-neuron d-prime contrasts from SVD factors.

    ``U`` must be ``components x neurons`` and ``V`` must be
    ``components x frames``.  Frame groups may be boolean masks or integer
    index vectors.  Every group's mean and population standard deviation is
    computed once, so a shared reference group can be reused across contrasts.

    This is algebraically equivalent to reconstructing ``U.T @ V`` and using
    ``np.nanmean``/``np.nanstd(..., ddof=0)`` on each selected frame group, but
    it never materializes the ``neurons x frames`` matrix.  It also deliberately
    adds no denominator epsilon, preserving the original notebook's NaN/Inf
    behavior when both within-group standard deviations are zero.

    If ``C`` is the number of components, ``N`` the number of neurons, ``B``
    the neuron chunk size, and ``F`` the largest selected frame group, extra
    working memory is ``O(C*F + C**2 + C*B + N*G)`` for ``G`` unique groups,
    rather than ``O(N*T)`` for a full reconstruction.
    """

    u = np.asarray(u_components_by_neuron)
    v = np.asarray(v_components_by_frame)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("U and V must both be two-dimensional")
    if u.shape[0] != v.shape[0]:
        raise ValueError("U and V must have the same component axis")
    if (
        not np.issubdtype(u.dtype, np.number)
        or not np.issubdtype(v.dtype, np.number)
        or np.issubdtype(u.dtype, np.complexfloating)
        or np.issubdtype(v.dtype, np.complexfloating)
    ):
        raise ValueError("U and V must contain real numeric values")
    neuron_chunk_size = _positive_integer("neuron_chunk_size", neuron_chunk_size)
    if not frame_groups:
        raise ValueError("frame_groups must contain at least one group")
    if not contrasts:
        raise ValueError("contrasts must contain at least one contrast")

    required_groups: list[Any] = []
    for contrast_name, pair in contrasts.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"contrast {contrast_name!r} must name exactly two frame groups"
            )
        for group_name in pair:
            if group_name not in frame_groups:
                raise ValueError(
                    f"contrast {contrast_name!r} references unknown group {group_name!r}"
                )
            if group_name not in required_groups:
                required_groups.append(group_name)

    moments: dict[Any, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for group_name in required_groups:
        indices = _frame_indices(
            frame_groups[group_name],
            n_frames=v.shape[1],
            name=f"frame_groups[{group_name!r}]",
        )
        moments[group_name] = _svd_projected_group_moments(
            u,
            v,
            indices,
            neuron_chunk_size=neuron_chunk_size,
        )

    result: dict[str, NDArray[np.float64]] = {}
    for contrast_name, (group_a, group_b) in contrasts.items():
        mean_a, sd_a = moments[group_a]
        mean_b, sd_b = moments[group_b]
        with np.errstate(all="ignore"):
            result[contrast_name] = 2.0 * (mean_a - mean_b) / (sd_a + sd_b)
    return result


def svd_dprime(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    frames_a: ArrayLike,
    frames_b: ArrayLike,
    *,
    neuron_chunk_size: int = 4096,
) -> NDArray[np.float64]:
    """Compute one per-neuron d-prime contrast without reconstructing frames."""

    return svd_dprime_contrasts(
        u_components_by_neuron,
        v_components_by_frame,
        {"a": frames_a, "b": frames_b},
        {"dprime": ("a", "b")},
        neuron_chunk_size=neuron_chunk_size,
    )["dprime"]


def trial_responses(
    trial_features: ArrayLike,
    position_mask: ArrayLike | None = None,
    *,
    require_complete_position_coverage: bool = True,
) -> NDArray[np.float64]:
    """Average position-binned features once per trial.

    ``trial_features`` must be ``trials x position bins x features``.  The
    default requires every chosen position bin to be populated for a trial, so
    label- or time-dependent corridor coverage cannot masquerade as a neural
    response difference.  Relaxing this rule is an explicitly named
    sensitivity analysis, never the primary path.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("trial_features must have shape (trials, position, features)")
    if position_mask is None:
        mask = np.ones(features.shape[1], dtype=bool)
    else:
        mask = np.asarray(position_mask, dtype=bool)
        if mask.shape != (features.shape[1],):
            raise ValueError("position_mask must have one value per position bin")
    if not np.any(mask):
        raise ValueError("position_mask must select at least one bin")
    selected = features[:, mask, :]
    complete = np.isfinite(selected).all(axis=(1, 2))
    finite = np.isfinite(selected)
    counts = np.sum(finite, axis=1)
    response = np.full((len(selected), selected.shape[2]), np.nan, dtype=np.float64)
    np.divide(
        np.nansum(selected, axis=1),
        counts,
        out=response,
        where=counts > 0,
    )
    if require_complete_position_coverage:
        response[~complete] = np.nan
    return response


def area_transform(
    u_components_by_neuron: ArrayLike,
    area_id: ArrayLike,
    area: str,
    *,
    n_features: int = 12,
) -> NDArray[np.float64]:
    """Factor the reconstructed-neuron distance metric for one visual area."""

    if area not in AREA_IDS:
        raise ValueError(f"area must be one of {list(AREA_IDS)}")
    n_features = _positive_integer("n_features", n_features)
    u = np.asarray(u_components_by_neuron, dtype=np.float64)
    ids = np.asarray(area_id)
    if u.ndim != 2 or u.shape[1] != len(ids):
        raise ValueError("U must have shape components x neurons aligned to iarea")
    mask = np.isin(ids, AREA_IDS[area])
    if not np.any(mask):
        raise ValueError(f"no neurons found for area {area}")
    weights = u[:, mask]
    gram = weights @ weights.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    keep = order[: min(n_features, len(order))]
    positive = np.maximum(eigenvalues[keep], 0.0)
    return eigenvectors[:, keep] * np.sqrt(positive)[None, :]


def prepare_session_trials(
    behavior: Mapping[str, Any],
    svd: Mapping[str, Any],
    retinotopy: Mapping[str, Any],
    *,
    area: str = "V1",
    n_features: int = 12,
    n_position_bins: int = 18,
    max_trailing_behavior_frames: int = 3,
    movement_rule: str = "moving_only",
    mouse_id: str | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Reconstruct one verified release session as trial x position x feature."""

    n_features = _positive_integer("n_features", n_features)
    n_position_bins = _positive_integer("n_position_bins", n_position_bins, minimum=2)
    max_trailing_behavior_frames = _positive_integer(
        "max_trailing_behavior_frames", max_trailing_behavior_frames, minimum=0
    )
    if movement_rule not in {"moving_only", "all_valid_frames"}:
        raise ValueError(
            "movement_rule must be one of ['moving_only', 'all_valid_frames']"
        )
    u = np.asarray(svd["U"], dtype=np.float64)
    v_by_frame = np.asarray(svd["V"], dtype=np.float64).T
    transform = area_transform(u, retinotopy["iarea"], area, n_features=n_features)
    moving = (
        np.asarray(behavior["ft_isMoving"])
        if "ft_isMoving" in behavior
        else np.asarray(behavior["ft_move"]) > 0
    )
    behavior_fields = {
        "position_dm": behavior["ft_Pos"],
        "trial_id": behavior["ft_trInd"],
        "is_moving": moving,
        "run_speed": behavior["ft_RunSpeed"],
    }
    v_by_frame, aligned, report = align_trailing_behavior_frames(
        v_by_frame,
        behavior_fields,
        max_trailing_behavior_frames=max_trailing_behavior_frames,
    )
    position_m = decimeters_to_meters(aligned["position_dm"])
    movement_mask = (
        np.asarray(aligned["is_moving"], dtype=bool)
        if movement_rule == "moving_only"
        else np.ones(len(position_m), dtype=bool)
    )
    valid = (
        movement_mask
        & np.isfinite(position_m)
        & np.isfinite(aligned["trial_id"])
        & (position_m >= 0.0)
        & (position_m <= 6.0)
    )
    edges_m = np.linspace(0.0, 6.0, n_position_bins + 1)
    trial_id, binned_v, frame_counts = bin_trial_features(
        v_by_frame,
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    speed_trial_id, speed, speed_counts = bin_trial_features(
        aligned["run_speed"],
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    trial_bins_match = np.array_equal(trial_id, speed_trial_id)
    frame_counts_match = np.array_equal(frame_counts, speed_counts)
    if not trial_bins_match or not frame_counts_match:
        raise RuntimeError("neural and speed binning lost frame alignment")
    features = np.einsum("tbp,pk->tbk", binned_v, transform, optimize=True)
    wall_by_trial = np.asarray(behavior["WallName"])[trial_id]
    wall_to_role = {}
    for wall, role in zip(np.asarray(behavior["UniqWalls"]), np.asarray(behavior["stim_id"])):
        try:
            finite = bool(np.isfinite(role))
        except TypeError:
            finite = False
        if finite:
            wall_to_role[str(wall)] = int(role)
    labels = np.asarray(
        [wall_to_role.get(str(wall), -1) for wall in wall_by_trial],
        dtype=np.int16,
    )
    n_selected_trials = len(trial_id)

    def selected_trial_values(name: str) -> NDArray[np.float64]:
        values = np.asarray(behavior.get(name, np.array([])))
        output = np.full(n_selected_trials, np.nan, dtype=np.float64)
        if values.ndim == 1 and len(values) > int(np.max(trial_id)):
            try:
                output = np.asarray(values[trial_id], dtype=np.float64)
            except (TypeError, ValueError):
                pass
        return output

    if "SoundDelPos" in behavior:
        cue_position_dm = selected_trial_values("SoundDelPos")
        corridor_length = float(behavior.get("Corridor_Length", 60.0))
        cue_position_dm = np.mod(cue_position_dm, corridor_length)
    else:
        cue_position_dm = selected_trial_values("SoundPos")
    reward_position_dm = selected_trial_values("RewPos")
    rewarded_trial = selected_trial_values("isRew")
    first_lick_dm = np.full(n_selected_trials, np.nan, dtype=np.float64)
    if "LickPos" in behavior and "LickTrind" in behavior:
        lick_position = np.asarray(behavior["LickPos"], dtype=np.float64)
        lick_trial = np.asarray(behavior["LickTrind"])
        for offset, current_trial in enumerate(trial_id):
            current = lick_position[lick_trial == current_trial]
            if len(current):
                first_lick_dm[offset] = float(current[0])
    centers_m = (edges_m[:-1] + edges_m[1:]) / 2.0
    return {
        "trial_features": features.astype(np.float32),
        "run_speed": speed[:, :, 0].astype(np.float32),
        "frame_counts": frame_counts,
        "trial_id": trial_id,
        "labels": labels,
        "wall_name": wall_by_trial,
        "position_edges_m": edges_m,
        "position_centers_m": centers_m,
        "texture_mask": edges_m[1:] <= 4.0,
        "gray_mask": edges_m[:-1] >= 4.0,
        "area": area,
        "movement_rule": movement_rule,
        "n_features": int(features.shape[-1]),
        "alignment": report,
        "reward_mode": str(behavior.get("Reward_Mode", "unknown")),
        "rewarded_trial": np.isfinite(rewarded_trial) & (rewarded_trial > 0),
        "cue_position_m": cue_position_dm / 10.0,
        "reward_position_m": reward_position_dm / 10.0,
        "first_lick_position_m": first_lick_dm / 10.0,
        "mouse_id": None if mouse_id is None else str(mouse_id),
        "recording_id": None if recording_id is None else str(recording_id),
    }
