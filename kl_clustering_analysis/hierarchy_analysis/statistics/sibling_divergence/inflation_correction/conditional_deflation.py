"""Conditional (per-node) deflation for the adjusted Wald test.

Instead of deflating every focal pair by the same global ĉ, this module
computes a per-node factor:

    ĉ_i = 1 + (ĉ_global − 1) · r_i

where r_i ∈ [0, 1] is a *trust weight* measuring how well node i's
degrees of freedom match the calibration pool that produced ĉ_global.

Bounded interpolation guarantees  1 ≤ ĉ_i ≤ ĉ_global  — the per-node
correction can only *reduce* over-deflation, never increase it.

The weight function (df-mismatch kernel):

    r_i = clip(1 − α · |log(k_i / k_pool_median)|, 0, 1)

where k_i is the node's sibling-test df and k_pool_median is the
edge-weight-weighted median df across the calibration pool.  α controls
sensitivity (default 0.3, validated on 101 benchmark cases).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from ..pair_testing.types import SiblingPairRecord
from .types import CalibrationModel

logger = logging.getLogger(__name__)


# ── Pool statistics ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoolStats:
    """Summary of the calibration pool, computed once per tree."""

    c_global: float
    median_log_df: float  # weighted median of log(df)
    median_df: float  # exp(median_log_df)
    n_records: int


def compute_pool_stats(
    records: List[SiblingPairRecord],
    model: CalibrationModel,
) -> PoolStats:
    """Compute calibration-pool summary from all valid sibling pair records.

    The weighted median uses ``edge_weight`` (= min of the two children's
    BH-corrected edge p-values), so null-like pairs dominate — matching the
    weighting scheme used to estimate ĉ_global itself.
    """
    valid = [r for r in records if np.isfinite(r.stat) and r.degrees_of_freedom > 0]
    if not valid:
        return PoolStats(
            c_global=model.global_inflation_factor,
            median_log_df=0.0,
            median_df=1.0,
            n_records=0,
        )

    log_dfs = np.array([np.log(max(r.degrees_of_freedom, 1)) for r in valid])
    weights = np.array([r.edge_weight for r in valid])

    # Weighted median via sorted cumulative weights
    sort_idx = np.argsort(log_dfs)
    sorted_log_dfs = log_dfs[sort_idx]
    sorted_weights = weights[sort_idx]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    if total_weight > 0:
        median_idx = np.searchsorted(cum_weights, total_weight / 2.0)
        median_idx = min(median_idx, len(sorted_log_dfs) - 1)
        median_log_df = float(sorted_log_dfs[median_idx])
    else:
        median_log_df = float(np.median(log_dfs))

    pool = PoolStats(
        c_global=model.global_inflation_factor,
        median_log_df=median_log_df,
        median_df=float(np.exp(median_log_df)),
        n_records=len(valid),
    )

    logger.debug(
        "Conditional deflation pool: median_df=%.2f, c_global=%.4f, " "n_records=%d.",
        pool.median_df,
        pool.c_global,
        pool.n_records,
    )
    return pool


# ── Per-node inflation factor ──────────────────────────────────────────────


def predict_conditional_inflation_factor(
    model: CalibrationModel,
    pool: PoolStats,
    degrees_of_freedom: int | float,
    *,
    alpha: float = 0.3,
) -> float:
    """Return the per-node inflation factor ĉ_i.

    Uses the df-mismatch kernel:

        r_i = clip(1 − alpha · |log(k_i / k_pool_median)|, 0, 1)
        ĉ_i = 1 + (ĉ_global − 1) · r_i

    Parameters
    ----------
    model : CalibrationModel
        Fitted global inflation model.
    pool : PoolStats
        Pre-computed calibration pool summary.
    degrees_of_freedom : int | float
        The sibling-test df (k) for the node being deflated.
    alpha : float
        Sensitivity of the df-mismatch kernel.  Larger values reduce
        deflation faster as df departs from the pool median.  Validated
        at 0.3 on 101 benchmark cases.

    Returns
    -------
    float
        Per-node inflation factor, guaranteed in [1.0, ĉ_global].
    """
    k_i = max(degrees_of_freedom, 1)
    k_median = max(pool.median_df, 1.0)
    df_mismatch = abs(np.log(k_i / k_median))
    r_i = float(np.clip(1.0 - alpha * df_mismatch, 0.0, 1.0))
    return 1.0 + (pool.c_global - 1.0) * r_i


__all__ = [
    "PoolStats",
    "compute_pool_stats",
    "predict_conditional_inflation_factor",
]
