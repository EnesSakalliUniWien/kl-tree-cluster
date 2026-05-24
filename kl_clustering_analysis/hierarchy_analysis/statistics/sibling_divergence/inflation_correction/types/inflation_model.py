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
    context_center: float = 0.0
    context_bandwidth: float = 0.0
    sample_contexts: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_statistics: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_reference_scales: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    sample_degrees_of_freedom: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )


__all__ = ["EmpiricalNullInflationModel"]
