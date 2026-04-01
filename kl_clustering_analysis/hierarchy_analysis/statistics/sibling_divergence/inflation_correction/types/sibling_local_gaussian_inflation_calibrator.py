"""Local Gaussian sibling-inflation calibrator type."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiblingLocalGaussianInflationCalibrator:
    """Fitted local Gaussian calibrator for sibling inflation deflation."""

    global_adjustment: float
    log_center: float
    center: float
    spread: float
    spread_status: str
    max_adjustment: float
    record_count: int
    sample_log_scales: np.ndarray = field(repr=False)
    sample_weights: np.ndarray = field(repr=False)
    sample_adjustments: np.ndarray = field(repr=False)


__all__ = ["SiblingLocalGaussianInflationCalibrator"]
