from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


SESSION_FIELDS = {
    "Corridor_Length": "corridor_length",
    "Texture_Length": "texture_length",
    "Gray_Space_length": "gray_space_length",
    "Reward_Delay_ms": "reward_delay_ms",
    "Reward_Mode": "reward_mode",
    "ntrials": "reported_trial_count",
}
TRIAL_FIELDS = {
    "trInd": "source_trial_id",
    "trInd_even": "is_even",
    "trInd_odd": "is_odd",
    "WallName": "wall_name",
    "WallType": "wall_type",
    "WallIsProbe": "is_probe",
    "TrialStim": "stimulus",
    "isRew": "is_rewarded",
    "StartFr": "start_frame_id",
    "EndFr": "end_frame_id",
    "GrayFr": "gray_frame_id",
    "Gray_space_time": "gray_space_time",
    "RewardFr": "reward_frame_id",
    "RewPos": "reward_position",
    "RewTime": "reward_time",
    "SoundFr": "sound_frame_id",
    "SoundPos": "sound_position",
    "SoundTime": "sound_time",
    "SoundDelayFr": "sound_delay_frame_id",
    "SoundDelPos": "sound_delay_position",
    "SoundTimeDelay": "sound_delay_time",
    "Trial_start_time": "start_time",
    "Trial_end_time": "end_time",
}
FRAME_FIELDS = {
    "ft": "time",
    "ft_trInd": "trial_id",
    "ft_trInd_even": "is_even_trial",
    "ft_trInd_odd": "is_odd_trial",
    "ft_Pos": "position_dm",
    "ft_PosCum": "position_cumulative_dm",
    "ft_RunCum": "run_cumulative",
    "ft_RunSpeed": "run_speed",
    "ft_WallID": "wall_name",
    "ft_isMoving": "is_moving",
    "ft_move": "movement_value",
    "ft_CorrSpc": "in_texture",
    "ft_GraySpc": "in_gray",
    "RunFr": "run_frame",
    "BefCueFr": "before_cue",
    "AftCueFr": "after_cue",
}
LICK_FIELDS = {
    "LickFr": "frame_id",
    "LickPos": "position",
    "LickTime": "time",
    "LickTrind": "trial_id",
    "Lick_wallName": "wall_name",
}
NUMPY_FIELDS = {
    "StimFrame",
    "StimTrial",
    "SubjMove",
    "VRpos",
    "VRposCum",
    "VRposTime",
    "run_pos",
}
RAW_FIELDS = tuple(
    sorted(
        set(SESSION_FIELDS)
        | set(TRIAL_FIELDS)
        | set(FRAME_FIELDS)
        | set(LICK_FIELDS)
        | {"UniqWalls", "stim_id"}
        | NUMPY_FIELDS
    )
)
IDENTITY = ("behavior_session_id", "experiment", "recording_id")


def tables(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    stimuli = stimuli_table(session, behavior)
    roles = dict(zip(stimuli["wall_name"], stimuli["stimulus_role"], strict=True))
    trials = trials_table(session, behavior, roles)
    frames = frames_table(session, behavior, roles, len(trials))
    events = events_table(session, behavior, trials)
    return {
        "behavior_session_summary": session_summary(session, behavior),
        "behavior_trials": trials,
        "behavior_frames": frames,
        "behavior_stimuli": stimuli,
        "behavior_events": events,
    }


def session_summary(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> pd.DataFrame:
    row = {
        "behavior_session_id": session["behavior_session_id"],
        "trial_count": len(vector(behavior, "WallName")),
        "frame_count": len(vector(behavior, "ft")),
        "lick_count": len(vector(behavior, "LickFr")),
        "vr_sample_count": len(vector(behavior, "VRpos")),
        "subject_motion_sample_count": len(
            vector(mapping(behavior, "SubjMove"), "SubjMTime")
        ),
        "position_bin_count": matrix(behavior, "run_pos").shape[1],
    }
    row.update(
        {
            target: scalar(behavior, source)
            for source, target in SESSION_FIELDS.items()
        }
    )
    return pd.DataFrame([row])


def trials_table(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
    roles: Mapping[str, Any],
) -> pd.DataFrame:
    count = len(vector(behavior, "WallName"))
    columns: dict[str, Any] = identity(session, count)
    columns["trial_id"] = np.arange(count, dtype=np.int64)
    for source, target in TRIAL_FIELDS.items():
        values = vector(behavior, source, count)
        columns[target] = nullable_integer(values) if target.endswith("_frame_id") else values
    walls = vector(behavior, "WallName", count).astype(str)
    columns["stimulus_role"] = pd.array(
        [roles.get(wall, pd.NA) for wall in walls], dtype="Int64"
    )
    return pd.DataFrame(columns)


def frames_table(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
    roles: Mapping[str, Any],
    trial_count: int,
) -> pd.DataFrame:
    count = len(vector(behavior, "ft"))
    columns: dict[str, Any] = identity(session, count)
    columns["frame_id"] = np.arange(count, dtype=np.int64)
    trial_id, valid_trial = valid_ids(vector(behavior, "ft_trInd", count), trial_count)
    columns["trial_id"] = trial_id
    columns["valid_trial"] = valid_trial
    for source, target in FRAME_FIELDS.items():
        if source == "ft_trInd":
            continue
        columns[target] = vector(behavior, source, count)
    columns["is_moving"] = np.asarray(columns["is_moving"]) > 0
    walls = np.asarray(columns["wall_name"]).astype(str)
    columns["wall_name"] = walls
    columns["stimulus_role"] = pd.array(
        [roles.get(wall, pd.NA) for wall in walls], dtype="Int64"
    )
    return pd.DataFrame(columns)


def stimuli_table(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> pd.DataFrame:
    walls = vector(behavior, "UniqWalls").astype(str)
    roles = vector(behavior, "stim_id", len(walls))
    columns = identity(session, len(walls))
    columns.update(
        {
            "stimulus_id": np.arange(len(walls), dtype=np.int64),
            "wall_name": walls,
            "stimulus_role": pd.to_numeric(pd.Series(roles), errors="coerce").astype(
                "Int64"
            ),
        }
    )
    return pd.DataFrame(columns)


def events_table(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
    trials: pd.DataFrame,
) -> pd.DataFrame:
    events = [lick_events(session, behavior)]
    for event_type, prefix in (
        ("reward", "reward"),
        ("sound", "sound"),
        ("sound_delay", "sound_delay"),
    ):
        count = len(trials)
        frame = pd.DataFrame(identity(session, count))
        frame["event_type"] = event_type
        frame["event_id"] = np.arange(count, dtype=np.int64)
        frame["trial_id"] = trials["trial_id"].array
        frame["frame_id"] = trials[f"{prefix}_frame_id"].array
        frame["position"] = trials[f"{prefix}_position"].array
        frame["time"] = trials[f"{prefix}_time"].array
        frame["wall_name"] = trials["wall_name"].array
        frame["occurred"] = frame[["frame_id", "position", "time"]].notna().any(axis=1)
        if event_type == "reward":
            frame["occurred"] &= trials["is_rewarded"].astype(bool).to_numpy()
        events.append(frame)
    return pd.concat(events, ignore_index=True)


def lick_events(
    session: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> pd.DataFrame:
    count = len(vector(behavior, "LickFr"))
    frame = pd.DataFrame(identity(session, count))
    frame["event_type"] = "lick"
    frame["event_id"] = np.arange(count, dtype=np.int64)
    frame["trial_id"] = nullable_integer(vector(behavior, "LickTrind", count))
    frame["frame_id"] = nullable_integer(vector(behavior, "LickFr", count))
    frame["position"] = vector(behavior, "LickPos", count)
    frame["time"] = vector(behavior, "LickTime", count)
    frame["wall_name"] = vector(behavior, "Lick_wallName", count).astype(str)
    frame["occurred"] = True
    return frame


def field_surface() -> pd.DataFrame:
    rows = []
    for raw_field in RAW_FIELDS:
        surface: tuple[str, str | None, str | None, str]
        if raw_field in SESSION_FIELDS:
            surface = ("session", "behavior_sessions", SESSION_FIELDS[raw_field], "direct")
        elif raw_field in TRIAL_FIELDS:
            surface = ("trial", "behavior_trials", TRIAL_FIELDS[raw_field], "direct")
        elif raw_field in FRAME_FIELDS:
            surface = ("frame", "behavior_frames", FRAME_FIELDS[raw_field], "direct")
        elif raw_field in LICK_FIELDS:
            surface = ("event", "behavior_events", LICK_FIELDS[raw_field], "direct")
        elif raw_field in ("UniqWalls", "stim_id"):
            column = "wall_name" if raw_field == "UniqWalls" else "stimulus_role"
            surface = ("stimulus", "behavior_stimuli", column, "direct")
        elif raw_field in ("StimFrame", "StimTrial"):
            table = "behavior_frames" if raw_field == "StimFrame" else "behavior_trials"
            surface = ("derived", table, "wall_name", "derived")
        else:
            surface = ("high_rate", None, None, "numpy_only")
        rows.append(
            {
                "raw_field": raw_field,
                "grain": surface[0],
                "sql_table": surface[1],
                "sql_column": surface[2],
                "representation": surface[3],
            }
        )
    return pd.DataFrame(rows)


def identity(session: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {name: np.repeat(session[name], count) for name in IDENTITY}


def mapping(source: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = source[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def matrix(source: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(source[name])
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return value


def scalar(source: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(source[name])
    if value.size != 1:
        raise ValueError(f"{name} must contain one value")
    return value.item()


def vector(
    source: Mapping[str, Any],
    name: str,
    length: int | None = None,
) -> np.ndarray:
    value = np.asarray(source[name])
    if value.ndim == 2 and 1 in value.shape:
        value = value.reshape(-1)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} has {len(value)} rows; expected {length}")
    return value


def nullable_integer(values: Any) -> pd.arrays.IntegerArray:
    raw = np.asarray(values, dtype=float)
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & np.isclose(raw, rounded)
    return pd.array(np.where(valid, rounded, np.nan), dtype="Int64")


def valid_ids(
    values: Any,
    size: int,
) -> tuple[pd.arrays.IntegerArray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & np.isclose(raw, rounded)
    valid &= (rounded >= 0) & (rounded < size)
    return pd.array(np.where(valid, rounded, np.nan), dtype="Int64"), valid


__all__ = ["field_surface", "tables"]
