"""Frame alignment and trial-by-position binning without interpolation.

The release stores corridor position in decimetres (0--60 for a 6 m
corridor).  These helpers convert units explicitly and never interpolate
across a trial boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import pprint
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class AlignmentReport:
    """Description of an explicitly allowed trailing-frame mismatch."""

    neural_frames: int
    behavior_frames: int
    dropped_trailing_behavior_frames: int

    def to_dict(self) -> dict[str, int]:
        return {
            "neural_frames": self.neural_frames,
            "behavior_frames": self.behavior_frames,
            "dropped_trailing_behavior_frames": (
                self.dropped_trailing_behavior_frames
            ),
        }

    def __repr__(self) -> str:
        return f"AlignmentReport({pprint.pformat(self.to_dict(), sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(pprint.pformat(self.to_dict(), sort_dicts=False))}</pre>"


def decimeters_to_meters(position_dm: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(position_dm, dtype=np.float64)
    return values / 10.0


def align_trailing_behavior_frames(
    neural: ArrayLike,
    behavior: Mapping[str, ArrayLike],
    *,
    max_trailing_behavior_frames: int = 0,
) -> tuple[NDArray, dict[str, NDArray], AlignmentReport]:
    """Align behavior arrays only when their extra frames are trailing.

    The caller must opt into the exact maximum mismatch.  Neural data are
    never truncated, and behavior arrays must all have the same length.
    """

    neural_array = np.asarray(neural)
    if neural_array.ndim == 0:
        raise ValueError("neural must have a frame axis")
    if max_trailing_behavior_frames < 0:
        raise ValueError("max_trailing_behavior_frames must be non-negative")
    if not behavior:
        raise ValueError("at least one behavior array is required")

    arrays = {name: np.asarray(values) for name, values in behavior.items()}
    lengths = {name: len(values) for name, values in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"behavior frame lengths disagree: {lengths}")

    neural_frames = len(neural_array)
    behavior_frames = unique_lengths.pop()
    extra = behavior_frames - neural_frames
    if extra < 0:
        raise ValueError(
            f"behavior has {behavior_frames} frames but neural has {neural_frames}"
        )
    if extra > max_trailing_behavior_frames:
        raise ValueError(
            f"behavior has {extra} extra frame(s); allowed maximum is "
            f"{max_trailing_behavior_frames}"
        )

    aligned = {name: values[:neural_frames] for name, values in arrays.items()}
    report = AlignmentReport(neural_frames, behavior_frames, extra)
    return neural_array, aligned, report


def position_bin_indices(
    position_m: ArrayLike,
    edges_m: ArrayLike,
) -> NDArray[np.int64]:
    """Return left-closed, right-open bin indices; include the last endpoint.

    Non-finite or out-of-range positions are rejected rather than clipped.
    """

    positions = np.asarray(position_m, dtype=np.float64)
    edges = np.asarray(edges_m, dtype=np.float64)
    if positions.ndim != 1:
        raise ValueError("position_m must be one-dimensional")
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges_m must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("edges_m must be finite and strictly increasing")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if np.any(positions < edges[0]) or np.any(positions > edges[-1]):
        raise ValueError(
            f"positions must lie in [{edges[0]}, {edges[-1]}] metres"
        )

    indices = np.searchsorted(edges, positions, side="right") - 1
    indices[positions == edges[-1]] = len(edges) - 2
    return indices.astype(np.int64, copy=False)


def bin_trial_features(
    frame_features: ArrayLike,
    position_m: ArrayLike,
    trial_id: ArrayLike,
    edges_m: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.int32]]:
    """Average frame features within each trial and fixed position bin.

    Empty bins remain NaN with count zero.  No samples are borrowed from
    neighboring bins or trials.
    """

    features = np.asarray(frame_features)
    positions = np.asarray(position_m, dtype=np.float64)
    trials_raw = np.asarray(trial_id)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError("frame_features must have shape (frames, features)")
    if len(features) != len(positions) or len(features) != len(trials_raw):
        raise ValueError("features, position_m, and trial_id must stay frame-aligned")

    mask = np.ones(len(features), dtype=bool)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != mask.shape:
            raise ValueError("valid_mask must have one value per frame")
        mask &= supplied
    mask &= np.isfinite(positions) & np.isfinite(trials_raw)
    if not np.any(mask):
        raise ValueError("no valid frames remain")

    selected_trials = trials_raw[mask]
    if not np.allclose(selected_trials, np.round(selected_trials)):
        raise ValueError("trial_id values must be integer-like")
    selected_trials = selected_trials.astype(np.int64)
    selected_positions = positions[mask]
    selected_features = features[mask]
    bin_index = position_bin_indices(selected_positions, edges_m)

    unique_trials = np.unique(selected_trials)
    n_bins = len(np.asarray(edges_m)) - 1
    output = np.full(
        (len(unique_trials), n_bins, selected_features.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    counts = np.zeros((len(unique_trials), n_bins), dtype=np.int32)

    for trial_offset, trial in enumerate(unique_trials):
        trial_mask = selected_trials == trial
        for bin_offset in range(n_bins):
            sample_mask = trial_mask & (bin_index == bin_offset)
            count = int(np.count_nonzero(sample_mask))
            counts[trial_offset, bin_offset] = count
            if count:
                output[trial_offset, bin_offset] = np.mean(
                    selected_features[sample_mask], axis=0, dtype=np.float64
                )

    return unique_trials, output, counts


def component_trial_position(joiner, position_edges, *, stimulus_only=False):
    role_filter = "AND stimulus_role IN (2, 0)" if stimulus_only else ""
    role_column = ", stimulus_role" if stimulus_only else ""
    frames = joiner.query(f"""
        SELECT frame_id, trial_id, position_m {role_column}
        FROM frames
        WHERE valid_trial AND is_moving AND in_texture
          AND position_m >= 0 AND position_m < 4
          AND trial_id IS NOT NULL AND position_m IS NOT NULL
          {role_filter}
        ORDER BY frame_id
    """)
    frames["position_bin"] = position_bin_indices(
        frames["position_m"], position_edges
    )
    joiner.register("position_frames", frames)
    frame_ids = frames["frame_id"].to_numpy(dtype=np.int64)
    trials, binned, _ = bin_trial_features(
        joiner.V[:, frame_ids].T,
        frames["position_m"],
        frames["trial_id"],
        position_edges,
    )
    response = np.transpose(binned, (2, 0, 1))
    roles = None
    if stimulus_only:
        trial_roles = joiner.query("""
            SELECT trial_id, FIRST(stimulus_role ORDER BY frame_id) AS stimulus_role
            FROM position_frames
            GROUP BY trial_id
            ORDER BY trial_id
        """)
        np.testing.assert_array_equal(trial_roles["trial_id"], trials)
        roles = trial_roles["stimulus_role"].to_numpy(dtype=int)
    return frames, response, trials, roles
