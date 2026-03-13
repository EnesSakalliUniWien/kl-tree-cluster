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
    valid = [r for r in records if np.isfinite(r.stat) and r.degrees_of_freedom > 0]
    if not valid:
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

    ratios = np.array([r.stat / r.degrees_of_freedom for r in valid])
    weights = np.array([r.edge_weight for r in valid])

    # Filter out non-positive ratios (can happen with degenerate pairs)
    pos_mask = ratios > 0
    ratios = ratios[pos_mask]
    weights = weights[pos_mask]

    if len(ratios) == 0 or weights.sum() == 0:
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

    max_observed_ratio = float(np.max(ratios))
    c_hat = float(np.average(ratios, weights=weights))

    # Clamp: at least 1.0 (never inflate the statistic), at most max observed
    c_hat = max(c_hat, 1.0)
    c_hat = min(c_hat, max_observed_ratio)

    n_contributing = int(np.sum(weights > 0))
    effective_n = (
        float(np.sum(weights) ** 2 / np.sum(weights**2)) if np.sum(weights**2) > 0 else 0.0
    )

    diagnostics = {
        "fit_status": "weighted_mean",
        "n_valid_pairs": len(ratios),
        "n_contributing": n_contributing,
        "effective_n": effective_n,
        "max_observed_ratio": max_observed_ratio,
        "median_ratio": float(np.median(ratios)),
        "mean_weight": float(np.mean(weights)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
    }

    logger.info(
        "Continuous-weight calibration: c-hat = %.4f from %d pairs "
        "(effective n = %.1f, max ratio = %.4f).",
        c_hat,
        len(ratios),
        effective_n,
        max_observed_ratio,
    )

    return CalibrationModel(
        method="weighted_mean",
        n_calibration=n_contributing,
        global_inflation_factor=c_hat,
        max_observed_ratio=max_observed_ratio,
        # Store as [log(c_hat), 0, 0] for backward compat with predict_inflation_factor
        beta=np.array([np.log(c_hat), 0.0, 0.0], dtype=np.float64),
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
