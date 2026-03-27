"""Local structural-dimension kernel deflation for the adjusted Wald test.

Instead of shrinking the global inflation factor with a fitted slope,
this module estimates a node-specific correction directly from nearby
calibration pairs in log-structural-dimension space:

    w_i(u) = w_i^null_prior * exp(-0.5 * ((log k_i - log k_u) / h)^2)
    c_u = weighted_mean(T_i / df_i, weights=w_i(u))

where ``k_i`` is the sibling structural dimension carried by each record,
``w_i^null_prior`` is the sibling null prior (from Gate 2 edge p-values), and ``h``
is the null-prior-weighted standard deviation of ``log(k)`` over the calibration
pool. This makes the deflation local in the decomposition-derived geometry
rather than global in effective-df space.
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


def _record_structural_dimension(record: SiblingPairRecord) -> float:
    """Return the structural-dimension axis used for local calibration."""
    if np.isfinite(record.structural_dimension) and record.structural_dimension > 0:
        return float(record.structural_dimension)
    raise ValueError(
        f"Invalid structural_dimension={record.structural_dimension!r}; "
        "caller must supply records with structural_dimension > 0 "
        "(upstream filter on degrees_of_freedom > 0 should guarantee this)."
    )


@dataclass(frozen=True)
class PoolStats:
    """Summary of the calibration pool used for local kernel deflation."""

    c_global: float
    mean_log_structural_dimension: float
    geometric_mean_structural_dimension: float
    bandwidth_log_structural_dimension: float
    bandwidth_status: str
    max_ratio: float
    n_records: int
    calibration_log_structural_dimensions: np.ndarray = field(repr=False)
    calibration_sibling_null_priors: np.ndarray = field(repr=False)
    calibration_stat_df_ratios: np.ndarray = field(repr=False)


def compute_pool_stats(
    records: list[SiblingPairRecord],
    model: CalibrationModel,
) -> PoolStats:
    """Compute the local-kernel calibration pool from valid sibling records."""
    valid = _positive_ratio_records(records)
    if not valid:
        return PoolStats(
            c_global=model.global_inflation_factor,
            mean_log_structural_dimension=0.0,
            geometric_mean_structural_dimension=1.0,
            bandwidth_log_structural_dimension=0.0,
            bandwidth_status="global_fallback_no_data",
            max_ratio=max(1.0, float(model.max_observed_ratio)),
            n_records=0,
            calibration_log_structural_dimensions=np.array([], dtype=float),
            calibration_sibling_null_priors=np.array([], dtype=float),
            calibration_stat_df_ratios=np.array([], dtype=float),
        )

    structural_dimensions = np.array(
        [_record_structural_dimension(record) for record in valid],
        dtype=float,
    )
    log_structural_dimensions = np.log(np.maximum(structural_dimensions, 1.0))
    sibling_null_priors = np.array(
        [record.sibling_null_prior_from_edge_pvalue for record in valid], dtype=float
    )
    stat_df_ratios = np.array(
        [record.stat / record.degrees_of_freedom for record in valid], dtype=float
    )
    positive_weight_mask = np.isfinite(sibling_null_priors) & (sibling_null_priors > 0)
    if not np.any(positive_weight_mask):
        return PoolStats(
            c_global=model.global_inflation_factor,
            mean_log_structural_dimension=0.0,
            geometric_mean_structural_dimension=1.0,
            bandwidth_log_structural_dimension=0.0,
            bandwidth_status="global_fallback_no_positive_weights",
            max_ratio=max(1.0, float(np.max(stat_df_ratios)), float(model.max_observed_ratio)),
            n_records=0,
            calibration_log_structural_dimensions=np.array([], dtype=float),
            calibration_sibling_null_priors=np.array([], dtype=float),
            calibration_stat_df_ratios=np.array([], dtype=float),
        )

    structural_dimensions = structural_dimensions[positive_weight_mask]
    log_structural_dimensions = log_structural_dimensions[positive_weight_mask]
    sibling_null_priors = sibling_null_priors[positive_weight_mask]
    stat_df_ratios = stat_df_ratios[positive_weight_mask]

    mean_log_structural_dimension = _weighted_mean(log_structural_dimensions, sibling_null_priors)
    bandwidth = _weighted_std(log_structural_dimensions, sibling_null_priors)
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        bandwidth = 0.0
        bandwidth_status = "global_fallback_zero_log_k_spread"
    else:
        bandwidth_status = "weighted_log_k_std"

    pool = PoolStats(
        c_global=model.global_inflation_factor,
        mean_log_structural_dimension=mean_log_structural_dimension,
        geometric_mean_structural_dimension=float(np.exp(mean_log_structural_dimension)),
        bandwidth_log_structural_dimension=bandwidth,
        bandwidth_status=bandwidth_status,
        max_ratio=max(1.0, float(np.max(stat_df_ratios)), float(model.max_observed_ratio)),
        n_records=len(stat_df_ratios),
        calibration_log_structural_dimensions=log_structural_dimensions,
        calibration_sibling_null_priors=_safe_weights(sibling_null_priors),
        calibration_stat_df_ratios=stat_df_ratios,
    )

    logger.debug(
        "Local structural-k kernel pool: center_k=%.2f, bandwidth=%.4f, "
        "c_global=%.4f, n_records=%d.",
        pool.geometric_mean_structural_dimension,
        pool.bandwidth_log_structural_dimension,
        pool.c_global,
        pool.n_records,
    )
    return pool


def predict_local_inflation_factor(
    pool: PoolStats,
    structural_dimension: int | float,
) -> float:
    """Return the node-specific inflation factor from the local kernel.

    The target node is positioned at ``log(k_u)``, where ``k_u`` is the
    sibling structural dimension inferred from the decomposition. The
    calibration pool is then reweighted with a Gaussian kernel in that
    one-dimensional log-k space.
    """
    if pool.n_records == 0 or pool.bandwidth_log_structural_dimension <= 0:
        return float(np.clip(pool.c_global, 1.0, pool.max_ratio))

    log_target = float(np.log(max(float(structural_dimension), 1.0)))
    normalized_offsets = (
        pool.calibration_log_structural_dimensions - log_target
    ) / pool.bandwidth_log_structural_dimension
    kernel_weights = np.exp(-0.5 * normalized_offsets**2)
    local_weights = pool.calibration_sibling_null_priors * kernel_weights

    if not np.isfinite(local_weights).any() or float(np.sum(local_weights)) <= 0:
        return float(np.clip(pool.c_global, 1.0, pool.max_ratio))

    clean_weights = _safe_weights(local_weights)
    local_inflation = float(np.average(pool.calibration_stat_df_ratios, weights=clean_weights))
    return float(np.clip(local_inflation, 1.0, pool.max_ratio))


__all__ = [
    "PoolStats",
    "compute_pool_stats",
    "predict_local_inflation_factor",
]
