from __future__ import annotations

from itertools import combinations, product
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .trials import _positive_integer


def exact_paired_sign_flip(values: ArrayLike) -> dict[str, float]:
    """Exact two-sided sign-flip test for finite paired differences."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("values must contain at least one finite paired difference")
    observed = float(np.mean(finite))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(finite))))
    null = np.mean(signs * finite, axis=1)
    return {
        "difference": observed,
        "pvalue": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
        "permutations": float(len(null)),
        "n_pairs": float(len(finite)),
    }


def fit_early_slope(
    midpoint: ArrayLike,
    dprime: ArrayLike,
    *,
    early_horizon: float,
) -> dict[str, float]:
    """Fit a per-trial linear rate within the predeclared early horizon."""

    x = np.asarray(midpoint, dtype=np.float64)
    y = np.asarray(dprime, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("midpoint and dprime must be aligned one-dimensional arrays")
    if not np.isfinite(float(early_horizon)):
        raise ValueError("early_horizon must be finite")
    keep = np.isfinite(x) & np.isfinite(y) & (x <= float(early_horizon))
    if np.count_nonzero(keep) < 2 or np.ptp(x[keep]) <= 0:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "n_blocks": float(np.count_nonzero(keep)),
        }
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_blocks": float(np.count_nonzero(keep)),
    }


def fit_saturation_curve(midpoint: ArrayLike, dprime: ArrayLike) -> dict[str, Any]:
    """Fit ``baseline + amplitude * (1 - exp(-k * trial))`` by grid search.

    The baseline and ``initial_rate`` are referenced to the first observed
    block midpoint, not trial zero.  The amplitude is not forced positive.
    ``plateau_observed`` requires at least 90% of the fitted amplitude to be
    attained within the data *and* a non-boundary grid optimum.  This keeps a
    half-completed rise or a grid-limited extrapolation from being described as
    an observed plateau.
    """

    x = np.asarray(midpoint, dtype=np.float64)
    y = np.asarray(dprime, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("midpoint and dprime must be aligned one-dimensional arrays")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4 or np.ptp(x) <= 0:
        return {
            "baseline": float("nan"), "amplitude": float("nan"),
            "plateau": float("nan"), "k": float("nan"),
            "initial_rate": float("nan"), "half_time": float("nan"),
            "r2": float("nan"), "plateau_observed": False,
            "fraction_of_amplitude_observed": float("nan"),
            "grid_boundary_hit": False, "reference_trial": float("nan"),
            "x_fitted": np.array([]), "y_fitted": np.array([]),
        }
    shifted = x - np.min(x)
    span = float(np.ptp(shifted))
    k_grid = np.geomspace(0.05 / span, 20.0 / span, 300)
    best = None
    for grid_index, k in enumerate(k_grid):
        basis = np.column_stack([np.ones(len(x)), 1.0 - np.exp(-k * shifted)])
        coefficient, *_ = np.linalg.lstsq(basis, y, rcond=None)
        predicted = basis @ coefficient
        sse = float(np.sum((y - predicted) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(k), coefficient, predicted, grid_index)
    assert best is not None
    sse, rate, coefficient, predicted, grid_index = best
    baseline, amplitude = map(float, coefficient)
    total = float(np.sum((y - np.mean(y)) ** 2))
    half_time = float(np.log(2.0) / rate)
    fraction_observed = float(1.0 - np.exp(-rate * span))
    grid_boundary_hit = bool(grid_index in (0, len(k_grid) - 1))
    order = np.argsort(x)
    return {
        "baseline": baseline,
        "amplitude": amplitude,
        "plateau": baseline + amplitude,
        "k": rate,
        "initial_rate": amplitude * rate,
        "half_time": half_time,
        "r2": float("nan") if total <= 0 else 1.0 - sse / total,
        "plateau_observed": bool(fraction_observed >= 0.90 and not grid_boundary_hit),
        "fraction_of_amplitude_observed": fraction_observed,
        "grid_boundary_hit": grid_boundary_hit,
        "reference_trial": float(np.min(x)),
        "x_fitted": x[order],
        "y_fitted": predicted[order],
    }


def _mouse_level_observations(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
) -> tuple[NDArray[np.float64], NDArray, NDArray]:
    """Collapse repeated sessions to one value per mouse and validate labels."""

    value = np.asarray(values, dtype=np.float64)
    group = np.asarray(groups, dtype=object)
    mouse = np.asarray(mouse_ids, dtype=object)
    if value.ndim != 1 or group.ndim != 1 or mouse.ndim != 1:
        raise ValueError("values, groups, and mouse_ids must be one-dimensional")
    if not (len(value) == len(group) == len(mouse)):
        raise ValueError("values, groups, and mouse_ids must align")
    if rewarded_label == unrewarded_label:
        raise ValueError("rewarded_label and unrewarded_label must be distinct")
    allowed = {rewarded_label, unrewarded_label}
    unexpected = set(group.tolist()) - allowed
    if unexpected:
        raise ValueError(f"unexpected group labels: {sorted(map(str, unexpected))}")
    if any(
        not isinstance(identifier, (str, np.str_)) or str(identifier).strip() == ""
        for identifier in mouse
    ):
        raise ValueError("every observation must have a non-empty string mouse_id")

    ordered_mice = list(dict.fromkeys(mouse.tolist()))
    collapsed_values: list[float] = []
    collapsed_groups: list[str] = []
    collapsed_mice: list[Any] = []
    for identifier in ordered_mice:
        selected = mouse == identifier
        labels = set(group[selected].tolist())
        if len(labels) != 1:
            raise ValueError(f"mouse {identifier!r} appears in more than one group")
        finite = value[selected & np.isfinite(value)]
        if len(finite) == 0:
            continue
        collapsed_values.append(float(np.mean(finite)))
        collapsed_groups.append(next(iter(labels)))
        collapsed_mice.append(identifier)
    observed_groups = set(collapsed_groups)
    if observed_groups != allowed:
        raise ValueError("both rewarded and unrewarded groups need a finite mouse-level value")
    return (
        np.asarray(collapsed_values, dtype=np.float64),
        np.asarray(collapsed_groups, dtype=object),
        np.asarray(collapsed_mice, dtype=object),
    )


def exact_group_permutation(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
    alternative: str = "greater",
) -> dict[str, Any]:
    """Exact label permutation after reducing repeated sessions to mice."""

    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    value, group, mouse = _mouse_level_observations(
        values,
        groups,
        mouse_ids=mouse_ids,
        rewarded_label=rewarded_label,
        unrewarded_label=unrewarded_label,
    )
    rewarded = group == rewarded_label
    n_rewarded = int(np.count_nonzero(rewarded))
    observed = float(np.mean(value[rewarded]) - np.mean(value[~rewarded]))
    null = []
    all_indices = np.arange(len(value))
    for chosen in combinations(all_indices, n_rewarded):
        mask = np.zeros(len(value), dtype=bool)
        mask[list(chosen)] = True
        null.append(float(np.mean(value[mask]) - np.mean(value[~mask])))
    null_values = np.asarray(null)
    if alternative == "greater":
        pvalue = float(np.mean(null_values >= observed - 1e-12))
    else:
        pvalue = float(np.mean(np.abs(null_values) >= abs(observed) - 1e-12))
    return {
        "difference": observed,
        "pvalue": pvalue,
        "permutations": float(len(null_values)),
        "n_mice": float(len(value)),
        "mouse_ids": mouse,
        "mouse_values": value,
        "mouse_groups": group,
    }


def bootstrap_group_difference(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a mean difference by resampling unique mice within group."""

    n_boot = _positive_integer("n_boot", n_boot)
    value, group, mouse = _mouse_level_observations(
        values,
        groups,
        mouse_ids=mouse_ids,
        rewarded_label=rewarded_label,
        unrewarded_label=unrewarded_label,
    )
    rewarded = value[group == rewarded_label]
    unrewarded = value[group == unrewarded_label]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = (
            np.mean(rng.choice(rewarded, size=len(rewarded), replace=True))
            - np.mean(rng.choice(unrewarded, size=len(unrewarded), replace=True))
        )
    return {
        "difference": float(np.mean(rewarded) - np.mean(unrewarded)),
        "ci": tuple(np.quantile(samples, [0.025, 0.975]).tolist()),
        "samples": samples,
        "n_mice": float(len(value)),
        "mouse_ids": mouse,
        "mouse_values": value,
        "mouse_groups": group,
    }


def simulate_mouse_curves(
    *,
    rewarded_mice: int = 4,
    unrewarded_mice: int = 9,
    rewarded_rate: float = 0.018,
    unrewarded_rate: float = 0.008,
    rewarded_plateau: float = 1.25,
    unrewarded_plateau: float = 1.05,
    baseline: float = 0.15,
    noise: float = 0.10,
    block_trials: int = 40,
    n_blocks: int = 8,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Create clearly synthetic mouse curves for the notebook design sandbox."""

    rewarded_mice = _positive_integer("rewarded_mice", rewarded_mice)
    unrewarded_mice = _positive_integer("unrewarded_mice", unrewarded_mice)
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    n_blocks = _positive_integer("n_blocks", n_blocks, minimum=2)
    if not np.isfinite(float(noise)) or float(noise) < 0:
        raise ValueError("noise must be finite and non-negative")
    rng = np.random.default_rng(seed)
    midpoint = (np.arange(n_blocks) + 0.5) * block_trials
    rows = []
    for group, count, rate, plateau in (
        ("rewarded", rewarded_mice, float(rewarded_rate), float(rewarded_plateau)),
        ("unrewarded", unrewarded_mice, float(unrewarded_rate), float(unrewarded_plateau)),
    ):
        amplitude = plateau - baseline
        k = rate / amplitude
        for mouse_index in range(count):
            mouse_baseline = baseline + rng.normal(0, noise * 0.35)
            mouse_amplitude = amplitude * rng.normal(1.0, 0.12)
            mouse_k = max(k * rng.normal(1.0, 0.15), 1e-6)
            dprime = mouse_baseline + mouse_amplitude * (1.0 - np.exp(-mouse_k * midpoint))
            dprime += rng.normal(0.0, noise, size=len(midpoint))
            rows.append(
                {
                    "mouse": f"{group[:1].upper()}{mouse_index + 1:02d}",
                    "group": group,
                    "midpoint": midpoint.astype(np.float64),
                    "dprime": dprime.astype(np.float64),
                }
            )
    return rows
