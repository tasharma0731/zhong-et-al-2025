from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_AREAS = ("V1", "mHV", "lHV", "aHV")


@dataclass(frozen=True)
class Superneurons:
    components: np.ndarray
    groups: pd.DataFrame
    membership: pd.DataFrame
    raster_order: np.ndarray

    @classmethod
    def from_recording(
        cls,
        U: np.ndarray,
        V: np.ndarray,
        neurons: pd.DataFrame,
        *,
        groups_per_area: int = 10,
        areas: Sequence[str] = DEFAULT_AREAS,
        n_clusters: int = 100,
        locality: float = 0.75,
        random_state: int = 0,
    ) -> Superneurons:
        neuron_ids = neurons.loc[
            neurons["area_group"].isin(areas), "neuron_id"
        ].to_numpy(dtype=np.int64)
        features = rastermap_features(U[:, neuron_ids], V)
        order = neuron_ids[
            rastermap_order(
                features,
                V,
                n_clusters=n_clusters,
                locality=locality,
                random_state=random_state,
            )
        ]
        return cls.from_order(
            U,
            neurons,
            order,
            groups_per_area=groups_per_area,
            areas=areas,
        )

    @classmethod
    def from_order(
        cls,
        U: np.ndarray,
        neurons: pd.DataFrame,
        raster_order: np.ndarray,
        *,
        groups_per_area: int = 10,
        areas: Sequence[str] = DEFAULT_AREAS,
    ) -> Superneurons:
        components, groups, membership = aggregate_areas(
            U,
            neurons,
            raster_order,
            groups_per_area=groups_per_area,
            areas=areas,
        )
        return cls(components, groups, membership, np.asarray(raster_order))

    def activity(self, V: np.ndarray, frame_ids: np.ndarray | None = None) -> np.ndarray:
        frames = slice(None) if frame_ids is None else frame_ids
        return self.components.T @ V[:, frames]


def rastermap_features(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    validate_factors(U, V)
    features = U.T * np.linalg.norm(V, axis=1)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    population_mean = features.mean(axis=0)
    population_mean /= np.linalg.norm(population_mean)
    features -= np.outer(features @ population_mean, population_mean)
    return features.astype(np.float32)


def validate_factors(U: np.ndarray, V: np.ndarray, tolerance: float = 1e-3) -> None:
    if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0]:
        raise ValueError("U and V must share a two-dimensional component axis")
    gram = V @ V.T
    diagonal = np.diag(np.diag(gram))
    if np.linalg.norm(gram - diagonal) > tolerance * np.linalg.norm(gram):
        raise ValueError("V rows are not orthogonal SVD component traces")
    scale = V.std(axis=1)
    if np.max(np.abs(V.mean(axis=1)) / np.maximum(scale, 1e-12)) > tolerance:
        raise ValueError("V component traces are not centered")


def rastermap_order(
    features: np.ndarray,
    V: np.ndarray,
    *,
    n_clusters: int,
    locality: float,
    random_state: int,
) -> np.ndarray:
    Rastermap = import_module("rastermap").Rastermap
    model = Rastermap(
        n_clusters=n_clusters,
        locality=locality,
        time_lag_window=0,
        normalize=False,
        mean_time=False,
        random_state=random_state,
        verbose=False,
    )
    model.fit(
        Usv=features,
        Vsv=np.asarray(V.T, dtype=np.float32),
        compute_X_embedding=False,
    )
    return np.asarray(model.isort, dtype=np.int64)


def aggregate_areas(
    U: np.ndarray,
    neurons: pd.DataFrame,
    raster_order: np.ndarray,
    *,
    groups_per_area: int,
    areas: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if groups_per_area < 1:
        raise ValueError("groups_per_area must be positive")

    ranks = pd.Series(np.arange(len(raster_order)), index=raster_order)
    group_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, int]] = []
    components: list[np.ndarray] = []

    for area in areas:
        area_ids = neurons.loc[
            neurons["area_group"].eq(area), "neuron_id"
        ].to_numpy(dtype=np.int64)
        if len(area_ids) < groups_per_area:
            raise ValueError(f"{area} has fewer than {groups_per_area} neurons")
        ordered = ranks.loc[area_ids].sort_values().index.to_numpy(dtype=np.int64)

        for area_group_id, neuron_ids in enumerate(np.array_split(ordered, groups_per_area)):
            superneuron_id = len(components)
            components.append(np.asarray(U[:, neuron_ids]).mean(axis=1))
            group_rows.append(
                {
                    "superneuron_id": superneuron_id,
                    "area_group": area,
                    "area_superneuron_id": area_group_id,
                    "n_neurons": len(neuron_ids),
                }
            )
            member_rows.extend(
                {
                    "neuron_id": int(neuron_id),
                    "superneuron_id": superneuron_id,
                    "raster_rank": int(ranks.loc[neuron_id]),
                }
                for neuron_id in neuron_ids
            )

    return (
        np.asarray(components).T,
        pd.DataFrame(group_rows),
        pd.DataFrame(member_rows),
    )


__all__ = ["DEFAULT_AREAS", "Superneurons", "rastermap_features"]
