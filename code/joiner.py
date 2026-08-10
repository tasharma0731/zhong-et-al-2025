from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

import drive
from data_access import DataFrameSQL
from database import ZhongDB
from position import AlignmentReport, align_trailing_behavior_frames


AREA_GROUPS = {
    0: "mHV",
    1: "mHV",
    2: "mHV",
    3: "aHV",
    4: "aHV",
    5: "lHV",
    6: "lHV",
    8: "V1",
    9: "mHV",
}

Selection = pd.DataFrame | pd.Series | Iterable[int] | None


def session_metadata(session: Any) -> dict[str, Any]:
    return {
        name: getattr(session, name)
        for name in (
            "behavior_session_id",
            "recording_id",
            "experiment",
            "mouse",
            "cohort",
            "stage",
            "moment",
        )
    }


class Joiner(DataFrameSQL):
    def __init__(
        self,
        database: ZhongDB,
        recording_id: str,
        *,
        experiment: str | None = None,
        behavior_key: str | None = None,
        max_gib: float = drive.DATASET["default_max_gib"],
        max_trailing_behavior_frames: int = 3,
    ) -> None:
        session = database.behavior_session(
            recording_id=recording_id,
            experiment=experiment,
            behavior_key=behavior_key,
        )
        self.recording_id = recording_id
        self.experiment = str(session["experiment"])
        self.behavior_key = str(session["behavior_key"])
        self.behavior_session_id = str(session["behavior_session_id"])

        behavior = database.load_behavior(self.behavior_session_id, max_gib=max_gib)
        reduced_neural = database.load(
            recording_id,
            "reduced_neural",
            max_gib=max_gib,
        )
        retinotopy = database.load(
            recording_id,
            "retinotopy",
            max_gib=max_gib,
        )

        self.components_by_neuron, self.components_by_frame = reduced_neural_axes(
            reduced_neural
        )
        self.U = self.components_by_neuron
        self.V = self.components_by_frame

        self.trials = trials_by_id(behavior, session)
        self.stimuli = stimuli_by_wall(behavior, session)
        self.frames, self.alignment = behavior_by_frame(
            behavior,
            self.trials,
            self.stimuli,
            session,
            frame_count=self.components_by_frame.shape[1],
            max_trailing_frames=max_trailing_behavior_frames,
        )
        self.neurons = retinotopy_by_neuron(
            retinotopy,
            recording_id,
            neuron_count=self.components_by_neuron.shape[1],
        )

        super().__init__(
            frames=self.frames,
            neurons=self.neurons,
            trials=self.trials,
            stimuli=self.stimuli,
        )

    def join(
        self,
        frames: Selection = None,
        neurons: Selection = None,
        *,
        max_cells: int = 1_000_000,
    ) -> pd.DataFrame:
        frame_ids = selected_ids(frames, "frame_id", len(self.frames))
        neuron_ids = selected_ids(neurons, "neuron_id", len(self.neurons))
        pair_count = len(frame_ids) * len(neuron_ids)
        if pair_count > max_cells:
            raise ValueError(
                f"The selection contains {pair_count:,} neuron-frame pairs; "
                f"raise max_cells or select a smaller block"
            )

        activity = reconstruct_activity(
            self.components_by_neuron,
            self.components_by_frame,
            neuron_ids,
            frame_ids,
        )
        activity_with_retinotopy = activity.merge(
            self.neurons,
            on="neuron_id",
            how="left",
            validate="many_to_one",
        )
        activity_with_behavior = activity_with_retinotopy.merge(
            self.frames,
            on=["recording_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )
        return self.register("activity", activity_with_behavior)

    def __repr__(self) -> str:
        return (
            f"Joiner({self.recording_id!r}; {len(self.neurons):,} neurons; "
            f"{len(self.frames):,} frames; {self.U.shape[0]} components)"
        )


def reduced_neural_axes(reduced_neural: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(reduced_neural, Mapping) or not {"U", "V"}.issubset(
        reduced_neural
    ):
        raise ValueError("Reduced neural data must contain U and V")

    components_by_neuron = np.asarray(reduced_neural["U"])
    components_by_frame = np.asarray(reduced_neural["V"])
    if components_by_neuron.ndim != 2 or components_by_frame.ndim != 2:
        raise ValueError("U and V must both be two-dimensional")
    if components_by_neuron.shape[0] != components_by_frame.shape[0]:
        raise ValueError(
            "U and V must have the same component axis: "
            f"{components_by_neuron.shape}, {components_by_frame.shape}"
        )
    return components_by_neuron, components_by_frame


def behavior_by_frame(
    behavior: Mapping[str, Any],
    trials: pd.DataFrame,
    stimuli: pd.DataFrame,
    session: Mapping[str, Any],
    *,
    frame_count: int,
    max_trailing_frames: int,
) -> tuple[pd.DataFrame, AlignmentReport]:
    aligned, report = behavior_aligned_to_neural_frames(
        behavior,
        frame_count,
        max_trailing_frames,
    )
    trial_id, valid_trial = valid_trial_ids(aligned["trial_id"], len(trials))
    position_dm = aligned["position_dm"].astype(float)

    columns: dict[str, Any] = {
        "behavior_session_id": session["behavior_session_id"],
        "recording_id": session["recording_id"],
        "experiment": session["experiment"],
        "frame_id": np.arange(frame_count, dtype=np.int64),
        "trial_id": trial_id,
        "valid_trial": valid_trial,
        "position_dm": position_dm,
        "position_m": position_dm / 10.0,
        "is_moving": aligned["is_moving"],
        "run_speed": aligned["run_speed"],
        "wall_at_frame": aligned["wall_at_frame"],
        "in_texture": aligned["in_texture"],
    }
    if "time" in aligned:
        columns["time"] = aligned["time"]

    frames = pd.DataFrame(columns)
    frames_with_trials = frames.merge(
        trials,
        on=["behavior_session_id", "recording_id", "experiment", "trial_id"],
        how="left",
        validate="many_to_one",
    )
    frames_with_stimuli = frames_with_trials.merge(
        stimuli,
        on=["behavior_session_id", "recording_id", "experiment", "wall_name"],
        how="left",
        validate="many_to_one",
    )
    return frames_with_stimuli, report


def retinotopy_by_neuron(
    retinotopy: Any,
    recording_id: str,
    *,
    neuron_count: int,
) -> pd.DataFrame:
    if not isinstance(retinotopy, Mapping):
        raise ValueError("Retinotopy data must be a mapping")
    areas = one_dimensional(retinotopy.get("iarea", []), "iarea").astype(np.int64)
    coordinates = np.asarray(retinotopy.get("xy_t", []))
    if len(areas) != neuron_count or coordinates.shape != (neuron_count, 2):
        raise ValueError(
            "Retinotopy must provide one iarea value and one xy_t pair per U neuron"
        )

    return pd.DataFrame(
        {
            "recording_id": recording_id,
            "neuron_id": np.arange(neuron_count, dtype=np.int64),
            "area_id": areas,
            "area_group": [AREA_GROUPS.get(int(area), "excluded") for area in areas],
            "cortical_x": -coordinates[:, 1],
            "cortical_y": coordinates[:, 0],
        }
    )


def reconstruct_activity(
    components_by_neuron: np.ndarray,
    components_by_frame: np.ndarray,
    neuron_ids: np.ndarray,
    frame_ids: np.ndarray,
) -> pd.DataFrame:
    values = components_by_neuron[:, neuron_ids].T @ components_by_frame[:, frame_ids]
    return pd.DataFrame(
        {
            "neuron_id": np.repeat(neuron_ids, len(frame_ids)),
            "frame_id": np.tile(frame_ids, len(neuron_ids)),
            "activity": values.reshape(-1),
        }
    )


def trials_by_id(
    behavior: Mapping[str, Any],
    session: Mapping[str, Any],
) -> pd.DataFrame:
    if "WallName" not in behavior:
        raise ValueError("Behavior data is missing WallName")
    walls = one_dimensional(behavior["WallName"], "WallName")
    return pd.DataFrame(
        {
            "behavior_session_id": session["behavior_session_id"],
            "recording_id": session["recording_id"],
            "experiment": session["experiment"],
            "trial_id": np.arange(len(walls), dtype=np.int64),
            "wall_name": walls.astype(str),
        }
    )


def stimuli_by_wall(
    behavior: Mapping[str, Any],
    session: Mapping[str, Any],
) -> pd.DataFrame:
    missing = [name for name in ["UniqWalls", "stim_id"] if name not in behavior]
    if missing:
        raise ValueError(f"Behavior data is missing {missing}")
    walls = one_dimensional(behavior["UniqWalls"], "UniqWalls")
    roles = one_dimensional(behavior["stim_id"], "stim_id")
    if len(walls) != len(roles):
        raise ValueError("UniqWalls and stim_id must have the same length")

    return pd.DataFrame(
        {
            "behavior_session_id": session["behavior_session_id"],
            "recording_id": session["recording_id"],
            "experiment": session["experiment"],
            "wall_name": walls.astype(str),
            "stimulus_role": pd.to_numeric(
                pd.Series(roles),
                errors="coerce",
            ).astype("Int64"),
        }
    )


def behavior_aligned_to_neural_frames(
    behavior: Mapping[str, Any],
    frame_count: int,
    max_trailing_frames: int,
) -> tuple[dict[str, np.ndarray], AlignmentReport]:
    movement = "ft_isMoving" if "ft_isMoving" in behavior else "ft_move"
    source_fields = {
        "trial_id": "ft_trInd",
        "position_dm": "ft_Pos",
        "is_moving": movement,
        "run_speed": "ft_RunSpeed",
        "wall_at_frame": "ft_WallID",
        "in_texture": "ft_CorrSpc",
    }
    if "ft" in behavior:
        source_fields["time"] = "ft"

    missing = [source for source in source_fields.values() if source not in behavior]
    if missing:
        raise ValueError(f"Behavior data is missing {missing}")

    fields = {
        name: one_dimensional(behavior[source], source)
        for name, source in source_fields.items()
    }
    fields["is_moving"] = fields["is_moving"] > 0
    fields["run_speed"] = fields["run_speed"].astype(float)
    fields["wall_at_frame"] = fields["wall_at_frame"].astype(str)
    fields["in_texture"] = fields["in_texture"].astype(bool)

    _, aligned, report = align_trailing_behavior_frames(
        np.empty((frame_count, 0)),
        fields,
        max_trailing_behavior_frames=max_trailing_frames,
    )
    return aligned, report


def valid_trial_ids(
    values: np.ndarray,
    trial_count: int,
) -> tuple[pd.arrays.IntegerArray, np.ndarray]:
    raw = values.astype(float)
    rounded = np.rint(raw)
    integer = np.isfinite(raw) & np.isclose(raw, rounded)
    valid = integer & (rounded >= 0) & (rounded < trial_count)
    return pd.array(np.where(valid, rounded, np.nan), dtype="Int64"), valid


def selected_ids(selection: Selection, column: str, size: int) -> np.ndarray:
    if selection is None:
        values = np.arange(size)
    elif isinstance(selection, pd.DataFrame):
        if column not in selection:
            raise KeyError(f"Selection must contain {column!r}")
        values = selection[column].to_numpy()
    elif isinstance(selection, pd.Series):
        values = selection.to_numpy()
    else:
        values = np.asarray(list(selection))

    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{column} values must be a one-dimensional number sequence")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{column} values must be finite integers")

    ids = pd.unique(values.astype(np.int64))
    if np.any(ids < 0) or np.any(ids >= size):
        raise IndexError(f"{column} values must be between 0 and {size - 1}")
    return np.asarray(ids, dtype=np.int64)


def one_dimensional(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


__all__ = ["Joiner"]
