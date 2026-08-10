import numpy as np
import pandas as pd

from superneurons import aggregate_areas, rastermap_features


def test_rastermap_features_match_normalized_activity_geometry() -> None:
    rng = np.random.default_rng(4)
    left, _ = np.linalg.qr(rng.normal(size=(24, 6)))
    temporal = rng.normal(size=(80, 6))
    temporal -= temporal.mean(axis=0)
    right, _ = np.linalg.qr(temporal)
    singular_values = np.linspace(9.0, 2.0, 6)
    U = left.T
    V = singular_values[:, None] * right.T

    activity = U.T @ V
    normalized = activity / activity.std(axis=1, keepdims=True)
    population_mean = normalized.mean(axis=0)
    population_mean /= np.linalg.norm(population_mean)
    normalized -= np.outer(normalized @ population_mean, population_mean)
    normalized /= np.sqrt(activity.shape[1])

    features = rastermap_features(U, V)
    np.testing.assert_allclose(features @ features.T, normalized @ normalized.T, atol=1e-6)


def test_area_aggregation_has_fixed_count_and_exact_mean_activity() -> None:
    rng = np.random.default_rng(8)
    U = rng.normal(size=(5, 28))
    V = rng.normal(size=(5, 31))
    neurons = pd.DataFrame(
        {
            "neuron_id": np.arange(28),
            "area_group": ["V1"] * 16 + ["mHV"] * 12,
        }
    )
    order = rng.permutation(28)

    components, groups, membership = aggregate_areas(
        U,
        neurons,
        order,
        groups_per_area=4,
        areas=("V1", "mHV"),
    )

    assert groups.groupby("area_group").size().to_dict() == {"V1": 4, "mHV": 4}
    assert groups.groupby("area_group").n_neurons.sum().to_dict() == {"V1": 16, "mHV": 12}

    first_ids = membership.loc[membership.superneuron_id.eq(0), "neuron_id"].to_numpy()
    expected = (U[:, first_ids].T @ V).mean(axis=0)
    np.testing.assert_allclose(components[:, 0] @ V, expected)
