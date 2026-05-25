"""Empirical-null inflation model result type."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmpiricalNullInflationModel:
    """Result of fitting the context-weighted empirical-null inflation."""

    method: str
    n_calibration: int
    baseline_empirical_inflation_factor: float
    effective_sample_size: float
    n_strict_null_calibration: int = 0
    n_stopped_or_null_calibration: int = 0
    context_center: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    context_bandwidth: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_contexts: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    sample_feature_families: tuple[str, ...] = ()
    sample_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_statistics: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_reference_scales: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    sample_degrees_of_freedom: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )


__all__ = ["EmpiricalNullInflationModel"]
