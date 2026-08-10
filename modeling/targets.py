"""Constraint-preserving transforms for d-prime distribution summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DistributionTargetTransform:
    """Map constrained summaries to an unconstrained regression space.

    The three quantiles are encoded as median, log lower gap, and log upper
    gap.  Leaf/circle fractions are encoded as additive log ratios against the
    non-selective fraction.  Standard deviation is log transformed.  Other
    metrics pass through unchanged.
    """

    metrics: tuple[str, ...]
    epsilon: float = 1e-6

    def __init__(self, metrics: Sequence[str], epsilon: float = 1e-6) -> None:
        object.__setattr__(self, "metrics", tuple(metrics))
        object.__setattr__(self, "epsilon", float(epsilon))
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be non-empty and unique")
        if not 0 < self.epsilon < 0.01:
            raise ValueError("epsilon must be between zero and 0.01")

    def _index(self, name: str) -> int:
        return self.metrics.index(name)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.metrics):
            raise ValueError("values must be rows x configured metrics")
        result = values.copy()

        quantiles = {"q05", "median", "q95"}
        if quantiles.issubset(self.metrics):
            low = values[:, self._index("q05")]
            center = values[:, self._index("median")]
            high = values[:, self._index("q95")]
            result[:, self._index("q05")] = center
            result[:, self._index("median")] = np.log(
                np.maximum(center - low, self.epsilon)
            )
            result[:, self._index("q95")] = np.log(
                np.maximum(high - center, self.epsilon)
            )

        if "sd_dprime" in self.metrics:
            index = self._index("sd_dprime")
            result[:, index] = np.log(np.maximum(values[:, index], self.epsilon))

        fraction_names = ("frac_leaf_selective", "frac_circle_selective")
        available = [name for name in fraction_names if name in self.metrics]
        if len(available) == 2:
            leaf_index, circle_index = map(self._index, fraction_names)
            leaf = np.clip(values[:, leaf_index], self.epsilon, 1.0)
            circle = np.clip(values[:, circle_index], self.epsilon, 1.0)
            total = leaf + circle
            excessive = total >= 1.0 - self.epsilon
            if excessive.any():
                factor = (1.0 - 2.0 * self.epsilon) / total[excessive]
                leaf[excessive] *= factor
                circle[excessive] *= factor
            neutral = np.maximum(1.0 - leaf - circle, self.epsilon)
            result[:, leaf_index] = np.log(leaf / neutral)
            result[:, circle_index] = np.log(circle / neutral)
        elif len(available) == 1:
            index = self._index(available[0])
            fraction = np.clip(values[:, index], self.epsilon, 1.0 - self.epsilon)
            result[:, index] = np.log(fraction / (1.0 - fraction))
        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.metrics):
            raise ValueError("values must be rows x configured metrics")
        result = values.copy()

        quantiles = {"q05", "median", "q95"}
        if quantiles.issubset(self.metrics):
            center = values[:, self._index("q05")]
            lower_gap = np.exp(np.clip(values[:, self._index("median")], -30, 30))
            upper_gap = np.exp(np.clip(values[:, self._index("q95")], -30, 30))
            result[:, self._index("q05")] = center - lower_gap
            result[:, self._index("median")] = center
            result[:, self._index("q95")] = center + upper_gap

        if "sd_dprime" in self.metrics:
            index = self._index("sd_dprime")
            result[:, index] = np.exp(np.clip(values[:, index], -30, 30))

        fraction_names = ("frac_leaf_selective", "frac_circle_selective")
        available = [name for name in fraction_names if name in self.metrics]
        if len(available) == 2:
            leaf_index, circle_index = map(self._index, fraction_names)
            logits = np.column_stack(
                [values[:, leaf_index], values[:, circle_index], np.zeros(len(values))]
            )
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            result[:, leaf_index] = probability[:, 0]
            result[:, circle_index] = probability[:, 1]
        elif len(available) == 1:
            index = self._index(available[0])
            result[:, index] = 1.0 / (1.0 + np.exp(-np.clip(values[:, index], -30, 30)))
        return result

    def validate(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("predictions contain non-finite values")
        if {"q05", "median", "q95"}.issubset(self.metrics):
            low = values[:, self._index("q05")]
            center = values[:, self._index("median")]
            high = values[:, self._index("q95")]
            if np.any(low > center) or np.any(center > high):
                raise ValueError("predicted quantiles are not ordered")
        if "sd_dprime" in self.metrics:
            if np.any(values[:, self._index("sd_dprime")] <= 0):
                raise ValueError("predicted standard deviations must be positive")
        fractions = [
            values[:, self._index(name)]
            for name in ("frac_leaf_selective", "frac_circle_selective")
            if name in self.metrics
        ]
        if fractions:
            stacked = np.column_stack(fractions)
            if np.any(stacked < 0) or np.any(stacked > 1):
                raise ValueError("predicted fractions must lie in [0, 1]")
            if stacked.shape[1] == 2 and np.any(stacked.sum(axis=1) > 1 + 1e-10):
                raise ValueError("leaf and circle fractions cannot sum above one")

