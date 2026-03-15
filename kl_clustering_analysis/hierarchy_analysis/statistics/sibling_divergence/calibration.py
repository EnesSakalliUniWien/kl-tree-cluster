"""Post-selection inflation calibration for the adjusted Wald test.

Estimates the inflation factor c using continuous edge-weight calibration.
Each sibling pair is weighted by min(p_edge_left, p_edge_right), so
pairs where neither child is edge-significant dominate the estimate.
This avoids the circular dependency of the binary null-like oracle,
which fails on null data when the edge test passes everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .tree_traversal import SiblingPairRecord

logger = logging.getLogger(__name__)

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class CalibrationModel:
    """Result of fitting the post-selection inflation model.

    Stores the global inflation factor c-hat estimated via continuous
    edge-weight calibration.  The model is intercept-only: a single
    constant c-hat applied uniformly to all focal sibling pairs.
    """

    method: str
    n_calibration: int  # number of pairs that contributed (weight > 0)
    global_inflation_factor: float  # the estimated c-hat
    max_observed_ratio: float = 1.0  # max(r_i) — upper bound for c-hat
    beta: Optional[np.ndarray] = None  # kept for API compat; [log(c-hat), 0, 0]
    diagnostics: Dict = field(default_factory=dict)


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


__all__ = [
    "CalibrationModel",
    "fit_inflation_model",
    "predict_inflation_factor",
]
