"""Typed configuration for the exposure-transfer analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_AREAS = ("V1", "mHV", "lHV", "aHV")
DEFAULT_METRICS = (
    "q05",
    "median",
    "q95",
    "sd_dprime",
    "frac_leaf_selective",
    "frac_circle_selective",
)
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class StudyConfig:
    """All choices that can alter model fitting or scoring.

    Keeping these choices in one immutable value makes sensitivity analyses
    explicit and ensures that reports contain the exact analysis contract.
    """

    areas: tuple[str, ...] = DEFAULT_AREAS
    metrics: tuple[str, ...] = DEFAULT_METRICS
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    training_cohort: str = "unsupervised"
    transfer_cohort: str = "supervised"
    bootstrap_repeats: int = 10_000
    random_seed: int = 2025

    def __post_init__(self) -> None:
        if not self.areas or len(set(self.areas)) != len(self.areas):
            raise ValueError("areas must be non-empty and unique")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be non-empty and unique")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive")
        if self.training_cohort == self.transfer_cohort:
            raise ValueError("training and transfer cohorts must differ")
        if self.bootstrap_repeats < 100:
            raise ValueError("bootstrap_repeats must be at least 100")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("areas", "metrics", "alphas"):
            result[name] = list(result[name])
        return result


PRIMARY_CONFIG = StudyConfig()

