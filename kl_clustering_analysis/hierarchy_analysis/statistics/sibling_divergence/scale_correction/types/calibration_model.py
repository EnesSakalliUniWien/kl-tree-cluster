"""Empirical-null scale model result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class EmpiricalNullScaleModel:
    """Result of fitting the context-weighted empirical-null scale."""

    method: str
    n_calibration: int
    baseline_empirical_scale_factor: float
    max_observed_statistic_ratio: float = 1.0
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
    diagnostics: Dict = field(default_factory=dict)


__all__ = ["EmpiricalNullScaleModel"]
