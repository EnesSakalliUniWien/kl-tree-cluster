"""Cousin-adjusted Wald sibling divergence test.

Corrects the post-selection inflation of the Wald χ² statistic by
estimating the inflation factor c from *null-like calibration pairs* —
sibling pairs where neither child passed the edge (child-parent) test.

Architecture
------------
1. **Compute all raw Wald stats** T_i, k_i for every eligible sibling pair
   (same as the standard pipeline).
2. **Identify null-like pairs**: pairs where neither child is edge-significant.
   For these, r_i = T_i / k_i ≈ c(BL_i, n_i) is a pure estimate of
   the post-selection inflation at that location in the tree.
3. **Model c** as a function of (BL_sum, n_parent) via log-linear regression:
       log(r_i) = β₀ + β₁·log(BL_sum_i) + β₂·log(n_parent_i) + ε_i
4. **Deflate focal pairs**: T_adj = T / ĉ, p = χ²_sf(T_adj, k).

Fallback tiers
--------------
- ≥ 5 null-like pairs → log-linear regression on (BL_sum, n_parent)
- 3–4 null-like pairs → global median ĉ = median(r_i)
- < 3 null-like pairs → no calibration (raw Wald; flag warning)

Configuration
-------------
Toggle via ``config.SIBLING_TEST_METHOD = "cousin_adjusted_wald"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ..branch_length_utils import compute_mean_branch_length
from .cousin_pipeline_helpers import (
    SiblingPairRecord,
    apply_calibrated_results,
    collect_sibling_pair_records,
    count_null_focal_pairs,
    deflate_focal_pairs,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)

logger = logging.getLogger(__name__)

# Minimum null-like pairs for each calibration tier
_MIN_REGRESSION = 5
_MIN_MEDIAN = 3


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class CalibrationModel:
    """Result of fitting the post-selection inflation model.

    Stores the parameters needed to predict the inflation factor ĉ
    for a focal sibling pair, given its branch-length sum and parent
    sample size.  Used both during annotation (to deflate sibling
    test statistics) and during post-hoc merge (to ensure symmetric
    calibration).
    """

    method: str  # "regression", "median", "none"
    n_calibration: int  # number of null-like pairs used
    global_c_hat: float  # median(r_i) across null-like pairs (always computed)
    max_observed_ratio: float = 1.0  # max(r_i) — upper bound for ĉ predictions
    beta: Optional[np.ndarray] = None  # [β₀, β₁, β₂] for regression
    diagnostics: Dict = field(default_factory=dict)


# =============================================================================
# Inflation estimation
# =============================================================================


def _fit_inflation_model(
    records: List[SiblingPairRecord],
) -> CalibrationModel:
    """Estimate the post-selection inflation factor from null-like pairs.

    Uses log-linear regression on (BL_sum, n_parent) when ≥ 5 null-like pairs
    are available, otherwise falls back to global median, and finally to no
    calibration.
    """
    null_records = [r for r in records if r.is_null_like and np.isfinite(r.stat) and r.df > 0]

    if not null_records:
        logger.warning(
            "Cousin-adjusted Wald: 0 null-like pairs — "
            "no calibration possible; raw Wald stats will be used."
        )
        return CalibrationModel(
            method="none", n_calibration=0, global_c_hat=1.0, max_observed_ratio=1.0
        )

    # r_i = T_i / k_i  (expected value under H₀ + inflation is c)
    ratios = np.array([r.stat / r.df for r in null_records])
    bl_sums = np.array([r.bl_sum for r in null_records])
    n_parents = np.array([r.n_parent for r in null_records])

    # Guard: remove any zero/negative BL or n_parent (can't take log)
    valid = (bl_sums > 0) & (n_parents > 0) & np.isfinite(ratios) & (ratios > 0)
    ratios = ratios[valid]
    bl_sums = bl_sums[valid]
    n_parents = n_parents[valid]
    null_records = [r for r, v in zip(null_records, valid) if v]

    n_cal = len(ratios)
    global_c = float(np.median(ratios)) if n_cal > 0 else 1.0
    max_c = float(np.max(ratios)) if n_cal > 0 else 1.0

    if n_cal < _MIN_MEDIAN:
        logger.warning(
            "Cousin-adjusted Wald: only %d null-like pairs (need ≥%d) — "
            "raw Wald stats will be used (ĉ = 1.0).",
            n_cal,
            _MIN_MEDIAN,
        )
        return CalibrationModel(
            method="none", n_calibration=n_cal, global_c_hat=global_c, max_observed_ratio=max_c
        )

    if n_cal < _MIN_REGRESSION:
        logger.info(
            "Cousin-adjusted Wald: %d null-like pairs (need ≥%d for regression) — "
            "using global median ĉ = %.3f.",
            n_cal,
            _MIN_REGRESSION,
            global_c,
        )
        return CalibrationModel(
            method="median",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
        )

    # --- Log-linear regression ---
    # log(r) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)
    log_r = np.log(ratios)
    X = np.column_stack(
        [
            np.ones(n_cal),
            np.log(bl_sums),
            np.log(n_parents.astype(float)),
        ]
    )

    # OLS via pseudoinverse (robust to collinearity)
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, log_r, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning(
            "Cousin-adjusted Wald: regression failed — " "falling back to global median ĉ = %.3f.",
            global_c,
        )
        return CalibrationModel(
            method="median",
            n_calibration=n_cal,
            global_c_hat=global_c,
        )

    # Sanity: if R² is very low, regression is not informative → use median
    fitted = X @ beta
    ss_res = float(np.sum((log_r - fitted) ** 2))
    ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    diagnostics = {
        "r_squared": r_squared,
        "beta": beta.tolist(),
        "n_calibration": n_cal,
        "median_ratio": float(global_c),
        "max_observed_ratio": float(max_c),
    }

    if r_squared < 0.05:
        logger.info(
            "Cousin-adjusted Wald: regression R²=%.3f (< 0.05) — "
            "not informative; using global median ĉ = %.3f.",
            r_squared,
            global_c,
        )
        return CalibrationModel(
            method="median",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
            diagnostics=diagnostics,
        )

    logger.info(
        "Cousin-adjusted Wald: fitted regression on %d null-like pairs. "
        "β = [%.3f, %.3f, %.3f], R² = %.3f.",
        n_cal,
        beta[0],
        beta[1],
        beta[2],
        r_squared,
    )

    return CalibrationModel(
        method="regression",
        n_calibration=n_cal,
        global_c_hat=global_c,
        max_observed_ratio=max_c,
        beta=np.asarray(beta),
        diagnostics=diagnostics,
    )


def predict_inflation_factor(
    model: CalibrationModel,
    bl_sum: float,
    n_parent: int,
) -> float:
    """Predict inflation factor ĉ for a focal pair.

    The prediction is clamped to [1.0, max_observed_ratio] to prevent
    extrapolation beyond the calibration data.  Without the upper cap,
    the β₂·log(n_parent) term can over-estimate inflation at the root
    (largest n_parent), deflating T so aggressively that real signal is
    missed and the tree collapses to K=1.

    Parameters
    ----------
    model : CalibrationModel
        Fitted calibration model.
    bl_sum : float
        Sum of branch lengths from the two siblings to their parent.
    n_parent : int
        Number of leaves under the parent node.

    Returns
    -------
    float
        Predicted inflation factor, clamped to [1.0, max_observed_ratio].
    """
    if model.method == "none":
        return 1.0

    if model.method == "median":
        return max(model.global_c_hat, 1.0)  # ĉ ≥ 1 (inflation, never deflation)

    # Regression
    if model.beta is None:
        return max(model.global_c_hat, 1.0)

    if bl_sum <= 0 or n_parent <= 0:
        return max(model.global_c_hat, 1.0)

    log_c = model.beta[0] + model.beta[1] * np.log(bl_sum) + model.beta[2] * np.log(float(n_parent))
    c_hat = float(np.exp(log_c))
    # Clamp: never predict more inflation than actually observed in
    # null-like calibration pairs (prevents regression extrapolation),
    # and never deflate below 1 (that would inflate the test statistic).
    c_hat = min(c_hat, model.max_observed_ratio)
    return max(c_hat, 1.0)


# =============================================================================
# Pipeline: collect → test → calibrate → deflate
# =============================================================================


def _collect_all_pairs(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
    mean_bl: float | None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> Tuple[List[SiblingPairRecord], List[str]]:
    """Collect ALL binary-child parent nodes and compute raw Wald stats.

    Returns (records, non_binary_nodes).
    """
    return collect_sibling_pair_records(
        tree,
        nodes_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


def _deflate_and_test(
    records: List[SiblingPairRecord],
    model: CalibrationModel,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs and compute adjusted p-values.

    Returns:
        focal_parents: parent node IDs for focal (edge-significant) pairs
        focal_results: (T_adj, k, p_adj) per focal pair
        calibration_methods: method string per focal pair
    """

    def _resolve_calibration(rec: SiblingPairRecord) -> tuple[float, str]:
        c_hat = predict_inflation_factor(model, rec.bl_sum, rec.n_parent)
        return c_hat, f"adjusted_{model.method}"

    return deflate_focal_pairs(
        records,
        calibration_resolver=_resolve_calibration,
    )


def _apply_results_adjusted(
    df: pd.DataFrame,
    focal_parents: List[str],
    focal_results: List[Tuple[float, float, float]],
    calibration_methods: List[str],
    skipped_parents: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply deflated results with BH correction to DataFrame."""
    return apply_calibrated_results(
        df,
        focal_parents,
        focal_results,
        calibration_methods,
        skipped_parents,
        alpha,
        logger=logger,
        audit_label="Cousin-adjusted Wald",
    )


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_adjusted(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using cousin-adjusted Wald.

    Two-pass approach:
    1. Compute raw Wald χ² stats for ALL binary-child parent nodes.
    2. Identify *null-like* pairs (neither child edge-significant) as
       calibration data.  Fit log-linear model:
           log(T/k) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)
    3. For *focal* pairs (at least one child edge-significant), predict
       inflation ĉ and deflate:  T_adj = T / ĉ,  p = χ²_sf(T_adj, k).

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    nodes_statistics_dataframe : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' column.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with sibling divergence columns (same schema as standard test)
        plus ``Sibling_Test_Method`` column.
    """
    df = init_sibling_annotation_df(nodes_statistics_dataframe)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Pass 1: compute ALL raw Wald stats
    records, non_binary = _collect_all_pairs(
        tree,
        df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )

    # Mark non-binary/leaf nodes as skipped (never testable)
    mark_non_binary_as_skipped(df, non_binary, logger=logger)

    early_df = early_return_if_no_records(df, records)
    if early_df is not None:
        return early_df

    n_null, n_focal = count_null_focal_pairs(records)
    logger.info(
        "Cousin-adjusted Wald: %d total pairs (%d null-like, %d focal).",
        len(records),
        n_null,
        n_focal,
    )

    # Pass 2: fit inflation model from null-like pairs
    model = _fit_inflation_model(records)

    # Pass 3: deflate focals and compute p-values
    focal_parents, focal_results, cal_methods = _deflate_and_test(records, model)

    # Null-like parents are skipped (they are noise splits)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    df = _apply_results_adjusted(
        df,
        focal_parents,
        focal_results,
        cal_methods,
        skipped_parents,
        significance_level_alpha,
    )

    # Audit metadata
    df.attrs["sibling_divergence_audit"] = {
        "total_pairs": len(records),
        "null_like_pairs": n_null,
        "focal_pairs": n_focal,
        "calibration_method": model.method,
        "calibration_n": model.n_calibration,
        "global_c_hat": model.global_c_hat,
        "diagnostics": model.diagnostics,
        "test_method": "cousin_adjusted_wald",
    }

    # Store the fitted model object for downstream use (e.g., post-hoc merge
    # calibration).  This is stored separately from the human-readable audit
    # dict so that consumers can call predict_inflation_factor() directly.
    df.attrs["_calibration_model"] = model

    return df


__all__ = [
    "CalibrationModel",
    "annotate_sibling_divergence_adjusted",
    "predict_inflation_factor",
]
