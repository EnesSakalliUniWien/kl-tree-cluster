"""Post-selection inflation calibration for the adjusted Wald test.

Estimates the inflation factor c using continuous edge-weight calibration.
Each sibling pair is weighted by min(p_edge_left, p_edge_right), so
pairs where neither child is edge-significant dominate the estimate.
This avoids the circular dependency of the binary null-like oracle,
which fails on null data when the edge test passes everything.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..pair_testing.sibling_pair_collection import SiblingPairRecord
from .types import CalibrationModel

logger = logging.getLogger(__name__)


# =============================================================================
# Inflation estimation — continuous edge-weight calibration
# =============================================================================


def _compute_weighted_inflation(
    records: List[SiblingPairRecord],
) -> CalibrationModel:
    """Estimate global inflation c-hat as a weighted mean of T/k ratios.

    Each pair is weighted by its ``edge_weight`` (= min of the two children's
    BH-corrected edge p-values).  Pairs with high edge p-values (null-like)
    dominate; pairs with low edge p-values (signal) are down-weighted.

    The model is intercept-only: c-hat is constant across all pairs.
    """
    # Collect valid pairs: finite stat, positive df
    valid_records = [
        record for record in records if np.isfinite(record.stat) and record.degrees_of_freedom > 0
    ]
    if not valid_records:
        logger.warning(
            "Continuous-weight calibration: 0 valid pairs — " "using neutral c-hat = 1.0."
        )
        return CalibrationModel(
            method="weighted_mean",
            n_calibration=0,
            global_inflation_factor=1.0,
            max_observed_ratio=1.0,
            beta=np.zeros(3, dtype=np.float64),
            diagnostics={"fit_status": "neutral_no_data"},
        )

    stat_df_ratios = np.array([record.stat / record.degrees_of_freedom for record in valid_records])
    edge_weights = np.array([record.edge_weight for record in valid_records])
    # Filter out non-positive ratios (can happen with degenerate pairs)
    positive_ratio_mask = stat_df_ratios > 0
    stat_df_ratios = stat_df_ratios[positive_ratio_mask]
    edge_weights = edge_weights[positive_ratio_mask]

    if len(stat_df_ratios) == 0 or edge_weights.sum() == 0:
        logger.warning(
            "Continuous-weight calibration: no positive-ratio pairs — " "using neutral c-hat = 1.0."
        )
        return CalibrationModel(
            method="weighted_mean",
            n_calibration=0,
            global_inflation_factor=1.0,
            max_observed_ratio=1.0,
            beta=np.zeros(3, dtype=np.float64),
            diagnostics={"fit_status": "neutral_no_positive_ratios"},
        )

    max_observed_ratio = float(np.max(stat_df_ratios))

    inflation_factor = float(np.average(stat_df_ratios, weights=edge_weights))

    # Clamp: at least 1.0 (never inflate the statistic), at most max observed
    inflation_factor = max(inflation_factor, 1.0)
    inflation_factor = min(inflation_factor, max_observed_ratio)

    contributing_pair_count = int(np.sum(edge_weights > 0))

    effective_sample_size = (
        float(np.sum(edge_weights) ** 2 / np.sum(edge_weights**2))
        if np.sum(edge_weights**2) > 0
        else 0.0
    )

    diagnostics = {
        "fit_status": "weighted_mean",
        "n_valid_pairs": len(stat_df_ratios),
        "n_contributing": contributing_pair_count,
        "effective_n": effective_sample_size,
        "max_observed_ratio": max_observed_ratio,
        "median_ratio": float(np.median(stat_df_ratios)),
        "mean_weight": float(np.mean(edge_weights)),
        "max_weight": float(np.max(edge_weights)),
        "min_weight": float(np.min(edge_weights)),
    }

    logger.info(
        "Continuous-weight calibration: c-hat = %.4f from %d pairs "
        "(effective n = %.1f, max ratio = %.4f).",
        inflation_factor,
        len(stat_df_ratios),
        effective_sample_size,
        max_observed_ratio,
    )

    return CalibrationModel(
        method="weighted_mean",
        n_calibration=contributing_pair_count,
        global_inflation_factor=inflation_factor,
        max_observed_ratio=max_observed_ratio,
        beta=np.array([np.log(inflation_factor), 0.0, 0.0], dtype=np.float64),
        diagnostics=diagnostics,
    )


def fit_inflation_model(
    records: List[SiblingPairRecord],
) -> CalibrationModel:
    """Estimate the post-selection inflation factor using continuous edge weights.

    Uses all sibling pairs weighted by min(p_edge_left, p_edge_right).
    The model is intercept-only: a single global c-hat.
    """
    return _compute_weighted_inflation(records)


def predict_inflation_factor(
    model: CalibrationModel,
    branch_length_sum: float,
    n_reference: int,
) -> float:
    """Return the global inflation factor c-hat.

    The intercept-only model returns the same c-hat for every pair.
    branch_length_sum and n_reference are retained for API compatibility
    but are not used.
    """
    return model.global_inflation_factor


# =============================================================================
# Parametric inflation estimation — c(n) = α · n^(-β)
# =============================================================================


def fit_parametric_inflation_model(
    records: List[SiblingPairRecord],
) -> CalibrationModel:
    """Fit a power-law inflation model c(n) = α · n^(-β) from null-like pairs.

    Null-like pairs (neither child edge-significant) estimate pure
    post-selection inflation.  Their T/k ratios are regressed against
    n_parent via log-linear OLS::

        log(T/k) = log(α) - β · log(n_parent)

    The OLS fit is refined with ``scipy.optimize.curve_fit`` for robustness.
    Falls back to a global weighted-mean model when fewer than 3 null-like
    pairs are available.
    """
    from scipy.optimize import curve_fit as _curve_fit

    # Collect valid null-like pairs for calibration
    null_records = [
        r for r in records if r.is_null_like and np.isfinite(r.stat) and r.degrees_of_freedom > 0
    ]

    all_valid = [r for r in records if np.isfinite(r.stat) and r.degrees_of_freedom > 0]
    all_ratios = np.array([r.stat / r.degrees_of_freedom for r in all_valid])
    max_observed_ratio = float(np.max(all_ratios)) if len(all_ratios) > 0 else 1.0

    if len(null_records) < 3:
        logger.info(
            "Parametric Wald: only %d null-like pairs — " "falling back to global weighted mean.",
            len(null_records),
        )
        model = _compute_weighted_inflation(records)
        return CalibrationModel(
            method="parametric_fallback_global",
            n_calibration=model.n_calibration,
            global_inflation_factor=model.global_inflation_factor,
            max_observed_ratio=max_observed_ratio,
            beta=model.beta,
            diagnostics={**model.diagnostics, "fallback_reason": "insufficient_null_pairs"},
        )

    n_parents = np.array([float(r.n_parent) for r in null_records])
    c_values = np.array([r.stat / r.degrees_of_freedom for r in null_records])

    # Filter to positive values (required for log transform)
    mask = (n_parents > 0) & (c_values > 0)
    n_fit = n_parents[mask]
    c_fit = c_values[mask]

    if len(n_fit) < 2:
        logger.info(
            "Parametric Wald: only %d valid null-like pairs after filtering — "
            "falling back to global weighted mean.",
            len(n_fit),
        )
        model = _compute_weighted_inflation(records)
        return CalibrationModel(
            method="parametric_fallback_global",
            n_calibration=model.n_calibration,
            global_inflation_factor=model.global_inflation_factor,
            max_observed_ratio=max_observed_ratio,
            beta=model.beta,
            diagnostics={**model.diagnostics, "fallback_reason": "insufficient_positive_pairs"},
        )

    # Log-linear OLS: log(c) = log(α) - β · log(n)
    log_n = np.log(n_fit)
    log_c = np.log(c_fit)
    design = np.column_stack([np.ones(len(log_n)), log_n])
    params, _, _, _ = np.linalg.lstsq(design, log_c, rcond=None)
    log_alpha_ols, neg_beta_ols = params
    alpha = float(np.exp(log_alpha_ols))
    beta_val = float(-neg_beta_ols)

    # Refine with scipy curve_fit (non-linear least squares)
    try:
        popt, _ = _curve_fit(
            lambda n, a, b: a * np.power(n, -b),
            n_fit,
            c_fit,
            p0=[alpha, beta_val],
            maxfev=5000,
        )
        if popt[0] > 0 and popt[1] > 0:
            alpha, beta_val = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        pass

    # R-squared
    c_pred = alpha * n_fit ** (-beta_val)
    ss_res = float(np.sum((c_fit - c_pred) ** 2))
    ss_tot = float(np.sum((c_fit - np.mean(c_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Global fallback: weighted mean (used as floor in predict)
    global_model = _compute_weighted_inflation(records)

    logger.info(
        "Parametric Wald: α=%.4f, β=%.4f, R²=%.4f from %d null-like pairs " "(global ĉ=%.4f).",
        alpha,
        beta_val,
        r_squared,
        len(n_fit),
        global_model.global_inflation_factor,
    )

    return CalibrationModel(
        method="parametric_power_law",
        n_calibration=len(n_fit),
        global_inflation_factor=global_model.global_inflation_factor,
        max_observed_ratio=max_observed_ratio,
        beta=np.array([np.log(alpha), -beta_val, 0.0], dtype=np.float64),
        diagnostics={
            "fit_status": "parametric_power_law",
            "alpha": alpha,
            "beta": beta_val,
            "r_squared": r_squared,
            "n_null_like": len(n_fit),
            "n_range": [float(np.min(n_fit)), float(np.max(n_fit))],
            "c_range": [float(np.min(c_fit)), float(np.max(c_fit))],
            "global_inflation_factor": global_model.global_inflation_factor,
            "max_observed_ratio": max_observed_ratio,
        },
    )


def predict_parametric_inflation_factor(
    model: CalibrationModel,
    n_reference: int,
) -> float:
    """Predict per-node inflation c(n) = α · n^(-β).

    Clamped to [1.0, max_observed_ratio].  Falls back to
    ``global_inflation_factor`` when the model is not parametric.
    """
    if model.method != "parametric_power_law" or model.beta is None:
        return model.global_inflation_factor

    # Decode: beta[0] = log(α), beta[1] = -β
    log_alpha = model.beta[0]
    neg_beta = model.beta[1]
    c_pred = float(np.exp(log_alpha + neg_beta * np.log(max(n_reference, 1))))

    # Clamp: at least 1.0 (never inflate the statistic), at most max observed
    c_pred = max(c_pred, 1.0)
    c_pred = min(c_pred, model.max_observed_ratio)
    return c_pred


__all__ = [
    "CalibrationModel",
    "fit_inflation_model",
    "fit_parametric_inflation_model",
    "predict_inflation_factor",
    "predict_parametric_inflation_factor",
]
