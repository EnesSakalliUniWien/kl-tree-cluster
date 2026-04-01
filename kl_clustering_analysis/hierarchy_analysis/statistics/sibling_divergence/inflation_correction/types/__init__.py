"""Shared type definitions for the inflation_correction package."""

from .calibration_model import CalibrationModel
from .sibling_local_gaussian_inflation_calibrator import (
    SiblingLocalGaussianInflationCalibrator,
)

__all__ = [
    "CalibrationModel",
    "SiblingLocalGaussianInflationCalibrator",
]
