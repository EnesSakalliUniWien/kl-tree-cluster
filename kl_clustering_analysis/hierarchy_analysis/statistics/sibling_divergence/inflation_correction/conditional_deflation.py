"""Local Gaussian adjustment for the adjusted sibling Wald test.

Instead of shrinking the global inflation factor with a fitted slope,
this module estimates a node-specific correction directly from nearby
calibration pairs in log sibling-scale space:

    w_i(u) = w_i^null_prior * exp(-0.5 * ((log s_i - log s_u) / h)^2)
    a_u = weighted_mean(T_i / df_i, weights=w_i(u))

where ``s_i`` is the rough sibling scale carried by each record,
``w_i^null_prior`` is the sibling null prior (from Gate 2 edge p-values), and ``h``
is the null-prior-weighted spread of ``log(scale)`` over the calibration
sample. This makes the deflation local in the decomposition-derived geometry
rather than global in effective degrees-of-freedom space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..pair_testing.types import SiblingPairRecord
from .inflation_estimation import _positive_ratio_records
from .types import CalibrationModel

logger = logging.getLogger(__name__)


def _safe_weights(weights: np.ndarray) -> np.ndarray:
    """Return non-negative finite weights, defaulting to uniform weights."""
    clean = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    clean = np.where(np.isfinite(clean), clean, 0.0)
    if clean.sum() <= 0:
        return np.ones_like(clean, dtype=float)
    return clean


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a finite weighted mean using sanitized non-negative weights."""
    if len(values) == 0:
        return 0.0
    clean_weights = _safe_weights(weights)
    return float(np.average(np.asarray(values, dtype=float), weights=clean_weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted standard deviation."""
    if len(values) == 0:
        return 0.0
    clean_weights = _safe_weights(weights)
    values = np.asarray(values, dtype=float)
    mean_value = float(np.average(values, weights=clean_weights))
    variance = float(np.average((values - mean_value) ** 2, weights=clean_weights))
    return float(np.sqrt(max(variance, 0.0)))


def _record_sibling_scale(record: SiblingPairRecord) -> float:
    """Return the sibling-scale axis used for local calibration."""
    if np.isfinite(record.sibling_scale) and record.sibling_scale > 0:
        return float(record.sibling_scale)
    raise ValueError(
        f"Invalid sibling_scale={record.sibling_scale!r}; "
        "caller must supply records with sibling_scale > 0 "
        "(upstream filter on degrees_of_freedom > 0 should guarantee this)."
    )


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


def fit_sibling_inflation_calibrator(
    records: list[SiblingPairRecord],
    model: CalibrationModel,
) -> SiblingLocalGaussianInflationCalibrator:
    """Fit the local Gaussian sibling adjuster from valid sibling records."""
    valid = _positive_ratio_records(records)
    if not valid:
        return SiblingLocalGaussianInflationCalibrator(
            global_adjustment=model.global_inflation_factor,
            log_center=0.0,
            center=1.0,
            spread=0.0,
            spread_status="global_fallback_no_data",
            max_adjustment=max(1.0, float(model.max_observed_ratio)),
            record_count=0,
            sample_log_scales=np.array([], dtype=float),
            sample_weights=np.array([], dtype=float),
            sample_adjustments=np.array([], dtype=float),
        )

    sibling_scales = np.array(
        [_record_sibling_scale(record) for record in valid],
        dtype=float,
    )
    sample_log_scales = np.log(np.maximum(sibling_scales, 1.0))
    sibling_null_priors = np.array(
        [record.sibling_null_prior_from_edge_pvalue for record in valid], dtype=float
    )
    sample_adjustments = np.array(
        [record.stat / record.degrees_of_freedom for record in valid], dtype=float
    )
    positive_weight_mask = np.isfinite(sibling_null_priors) & (sibling_null_priors > 0)
    if not np.any(positive_weight_mask):
        return SiblingLocalGaussianInflationCalibrator(
            global_adjustment=model.global_inflation_factor,
            log_center=0.0,
            center=1.0,
            spread=0.0,
            spread_status="global_fallback_no_positive_weights",
            max_adjustment=max(
                1.0,
                float(np.max(sample_adjustments)),
                float(model.max_observed_ratio),
            ),
            record_count=0,
            sample_log_scales=np.array([], dtype=float),
            sample_weights=np.array([], dtype=float),
            sample_adjustments=np.array([], dtype=float),
        )

    sibling_scales = sibling_scales[positive_weight_mask]
    sample_log_scales = sample_log_scales[positive_weight_mask]
    sibling_null_priors = sibling_null_priors[positive_weight_mask]
    sample_adjustments = sample_adjustments[positive_weight_mask]

    log_center = _weighted_mean(sample_log_scales, sibling_null_priors)
    spread = _weighted_std(sample_log_scales, sibling_null_priors)
    if not np.isfinite(spread) or spread <= 1e-12:
        spread = 0.0
        spread_status = "global_fallback_zero_log_scale_spread"
    else:
        spread_status = "weighted_log_scale_std"

    calibrator = SiblingLocalGaussianInflationCalibrator(
        global_adjustment=model.global_inflation_factor,
        log_center=log_center,
        center=float(np.exp(log_center)),
        spread=spread,
        spread_status=spread_status,
        max_adjustment=max(
            1.0,
            float(np.max(sample_adjustments)),
            float(model.max_observed_ratio),
        ),
        record_count=len(sample_adjustments),
        sample_log_scales=sample_log_scales,
        sample_weights=_safe_weights(sibling_null_priors),
        sample_adjustments=sample_adjustments,
    )

    logger.debug(
        "Local sibling adjuster: center=%.2f, spread=%.4f, global_adjustment=%.4f, "
        "record_count=%d.",
        calibrator.center,
        calibrator.spread,
        calibrator.global_adjustment,
        calibrator.record_count,
    )
    return calibrator


def predict_sibling_adjustment(
    calibrator: SiblingLocalGaussianInflationCalibrator,
    sibling_scale: int | float,
) -> float:
    """Return the node-specific sibling adjustment from the local Gaussian fit.

    The target node is positioned at ``log(scale_u)``, where ``scale_u`` is the
    sibling scale inferred from the decomposition. The calibration sample is
    then reweighted with a Gaussian kernel in that one-dimensional log-scale
    space.
    """
    if calibrator.record_count == 0 or calibrator.spread <= 0:
        return float(np.clip(calibrator.global_adjustment, 1.0, calibrator.max_adjustment))

    log_target = float(np.log(max(float(sibling_scale), 1.0)))
    normalized_offsets = (
        calibrator.sample_log_scales - log_target
    ) / calibrator.spread
    kernel_weights = np.exp(-0.5 * normalized_offsets**2)
    local_weights = calibrator.sample_weights * kernel_weights

    if not np.isfinite(local_weights).any() or float(np.sum(local_weights)) <= 0:
        return float(np.clip(calibrator.global_adjustment, 1.0, calibrator.max_adjustment))

    clean_weights = _safe_weights(local_weights)
    local_adjustment = float(np.average(calibrator.sample_adjustments, weights=clean_weights))
    return float(np.clip(local_adjustment, 1.0, calibrator.max_adjustment))


__all__ = [
    "SiblingLocalGaussianInflationCalibrator",
    "fit_sibling_inflation_calibrator",
    "predict_sibling_adjustment",
]
