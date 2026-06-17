"""Calibration-resolution helpers for adjusted sibling Wald annotation."""

from __future__ import annotations

from ..inflation_correction.conditional_deflation import predict_sibling_adjustment
from ..inflation_correction.types.sibling_local_gaussian_inflation_calibrator import (
    SiblingLocalGaussianInflationCalibrator,
)
from ..pair_testing.types.sibling_pair_record import SiblingPairRecord


def _resolve_calibration(
    sibling_test_record: SiblingPairRecord,
    calibrator: SiblingLocalGaussianInflationCalibrator,
) -> tuple[float, str]:
    """Return the sibling adjustment and label for one sibling test."""
    adjustment = predict_sibling_adjustment(
        calibrator,
        sibling_test_record.sibling_test_calibration_scale,
    )
    return adjustment, "local_gaussian_adjuster"


__all__ = ["_resolve_calibration"]
