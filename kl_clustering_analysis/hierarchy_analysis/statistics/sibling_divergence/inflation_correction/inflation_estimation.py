"""Post-selection inflation estimation for the adjusted Wald test.

Estimates the inflation factor ĉ using sibling null priors as weights.
Each sibling pair is weighted by its sibling_null_prior_from_edge_pvalue
(= min(p_edge_left, p_edge_right)), so pairs where neither child is
edge-significant dominate the estimate.  This avoids the circular
dependency of the binary null-like oracle, which fails on null data
when the edge test passes everything.
"""

from __future__ import annotations

import logging

import numpy as np

from ..pair_testing.types import SiblingPairRecord
from .types import CalibrationModel

logger = logging.getLogger(__name__)


# =============================================================================
# Inflation estimation — continuous edge-weight calibration
# =============================================================================


def _neutral_calibration_model(fit_status: str) -> CalibrationModel:
    """Return the neutral intercept-only calibration model."""
    return CalibrationModel(
        method="weighted_mean",
        n_calibration=0,
        global_inflation_factor=1.0,
        max_observed_ratio=1.0,
        diagnostics={"fit_status": fit_status},
    )


def _positive_ratio_records(records: list[SiblingPairRecord]) -> list[SiblingPairRecord]:
    """Return finite-stat, positive-df, positive-ratio records."""
    return [
        record
        for record in records
        if np.isfinite(record.stat)
        and record.degrees_of_freedom > 0
        and (record.stat / record.degrees_of_freedom) > 0
    ]


def fit_inflation_model(
    records: list[SiblingPairRecord],
) -> CalibrationModel:
    """Estimate the post-selection inflation factor ĉ as a weighted mean of T/k ratios.

    Uses all sibling pairs weighted by ``sibling_null_prior_from_edge_pvalue``
    (= min(p_edge_left, p_edge_right)).  Pairs with high null priors (null-like)
    dominate; pairs with low null priors (signal) are down-weighted.

    The model is intercept-only: a single global ĉ.
    """
    # Collect valid pairs: finite stat, positive df
    valid_records = [
        record for record in records if np.isfinite(record.stat) and record.degrees_of_freedom > 0
    ]
    if not valid_records:
        logger.warning(
            "Continuous-weight calibration: 0 valid pairs — " "using neutral c-hat = 1.0."
        )
        return _neutral_calibration_model("neutral_no_data")

    positive_ratio_records = _positive_ratio_records(valid_records)
    if not positive_ratio_records:
        logger.warning(
            "Continuous-weight calibration: no positive-ratio pairs — using neutral c-hat = 1.0."
        )
        return _neutral_calibration_model("neutral_no_positive_ratios")

    stat_df_ratios = np.array(
        [record.stat / record.degrees_of_freedom for record in positive_ratio_records]
    )
    sibling_null_priors = np.array(
        [record.sibling_null_prior_from_edge_pvalue for record in positive_ratio_records]
    )
    max_observed_ratio = max(float(np.max(stat_df_ratios)), 1.0)
    positive_weight_mask = np.isfinite(sibling_null_priors) & (sibling_null_priors > 0)
    if not np.any(positive_weight_mask):
        logger.warning(
            "Continuous-weight calibration: no positive-weight pairs — using neutral c-hat = 1.0."
        )
        return _neutral_calibration_model("neutral_no_positive_weights")

    stat_df_ratios = stat_df_ratios[positive_weight_mask]
    sibling_null_priors = sibling_null_priors[positive_weight_mask]

    inflation_factor = float(np.average(stat_df_ratios, weights=sibling_null_priors))

    # Clamp: at least 1.0 (never inflate the statistic), at most max observed
    inflation_factor = max(inflation_factor, 1.0)
    inflation_factor = min(inflation_factor, max_observed_ratio)

    contributing_pair_count = int(np.sum(sibling_null_priors > 0))

    effective_sample_size = (
        float(np.sum(sibling_null_priors) ** 2 / np.sum(sibling_null_priors**2))
        if np.sum(sibling_null_priors**2) > 0
        else 0.0
    )

    diagnostics = {
        "fit_status": "weighted_mean",
        "n_valid_pairs": len(positive_ratio_records),
        "n_contributing": contributing_pair_count,
        "effective_n": effective_sample_size,
        "max_observed_ratio": max_observed_ratio,
        "median_ratio": float(np.median(stat_df_ratios)),
        "mean_sibling_null_prior": float(np.mean(sibling_null_priors)),
        "max_sibling_null_prior": float(np.max(sibling_null_priors)),
        "min_sibling_null_prior": float(np.min(sibling_null_priors)),
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
        diagnostics=diagnostics,
    )


__all__ = [
    "CalibrationModel",
    "fit_inflation_model",
]
