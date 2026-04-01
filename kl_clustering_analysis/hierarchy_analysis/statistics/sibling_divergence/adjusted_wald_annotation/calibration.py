"""Calibration-resolution helpers for adjusted sibling Wald annotation."""

from __future__ import annotations

from importlib import import_module

from ..inflation_correction.types import SiblingLocalGaussianInflationCalibrator
from ..pair_testing.types import SiblingPairRecord


def _resolve_calibration(
    sibling_test_record: SiblingPairRecord,
    calibrator: SiblingLocalGaussianInflationCalibrator,
) -> tuple[float, str]:
    """Return the sibling adjustment and label for one sibling test."""
    adjusted_wald_annotation_module = import_module(__package__)
    adjustment = adjusted_wald_annotation_module.predict_sibling_adjustment(
        calibrator,
        sibling_test_record.sibling_test_calibration_scale,
    )
    return adjustment, "local_gaussian_adjuster"


__all__ = ["_resolve_calibration"]
