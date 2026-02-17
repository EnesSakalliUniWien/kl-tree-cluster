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
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..branch_length_utils import compute_mean_branch_length
from ..multiple_testing import benjamini_hochberg_correction
from .sibling_divergence_test import (
    _either_child_significant,
    _get_binary_children,
    _get_sibling_data,
    sibling_divergence_test,
)

logger = logging.getLogger(__name__)

# Minimum null-like pairs for each calibration tier
_MIN_REGRESSION = 5
_MIN_MEDIAN = 3


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class _SiblingRecord:
    """Per–sibling-pair record for calibration pipeline."""

    parent: str
    left: str
    right: str
    stat: float  # raw Wald T
    df: int  # projection dimension k
    pval: float  # raw Wald p
    bl_sum: float  # branch-length sum (left + right)
    n_parent: int  # number of leaves under parent
    is_null_like: bool  # neither child edge-significant


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
    records: List[_SiblingRecord],
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
) -> List[_SiblingRecord]:
    """Collect ALL binary-child parent nodes and compute raw Wald stats."""
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. " "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()
    records: List[_SiblingRecord] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children
        left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(tree, parent, left, right)

        # Per-node spectral dimension and PCA projection
        _spectral_k: int | None = None
        _pca_proj: np.ndarray | None = None
        if spectral_dims is not None:
            _spectral_k = spectral_dims.get(parent)
        if pca_projections is not None:
            _pca_proj = pca_projections.get(parent)

        # Compute raw Wald stat
        stat, df, pval = sibling_divergence_test(
            left_dist,
            right_dist,
            float(n_l),
            float(n_r),
            branch_length_left=bl_l,
            branch_length_right=bl_r,
            mean_branch_length=mean_bl,
            test_id=f"sibling:{parent}",
            spectral_k=_spectral_k,
            pca_projection=_pca_proj,
        )

        # Branch-length sum
        bl_sum = 0.0
        if bl_l is not None and bl_r is not None:
            bl_sum = bl_l + bl_r
        elif bl_l is not None:
            bl_sum = bl_l
        elif bl_r is not None:
            bl_sum = bl_r
        # If no branch lengths at all, bl_sum stays 0.0

        n_parent = extract_node_sample_size(tree, parent)

        # Is this a null-like pair?  Neither child edge-significant.
        is_null = not _either_child_significant(left, right, sig_map)

        records.append(
            _SiblingRecord(
                parent=parent,
                left=left,
                right=right,
                stat=stat,
                df=int(df) if np.isfinite(df) else 0,
                pval=pval,
                bl_sum=bl_sum,
                n_parent=n_parent,
                is_null_like=is_null,
            )
        )

    return records


def _deflate_and_test(
    records: List[_SiblingRecord],
    model: CalibrationModel,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs and compute adjusted p-values.

    Returns:
        focal_parents: parent node IDs for focal (edge-significant) pairs
        focal_results: (T_adj, k, p_adj) per focal pair
        calibration_methods: method string per focal pair
    """
    focal_parents: List[str] = []
    focal_results: List[Tuple[float, float, float]] = []
    methods: List[str] = []

    for rec in records:
        if rec.is_null_like:
            continue  # null-like pairs are skipped (treated as no-split)

        if not np.isfinite(rec.stat) or rec.df <= 0:
            focal_parents.append(rec.parent)
            focal_results.append((np.nan, np.nan, np.nan))
            methods.append("invalid")
            continue

        c_hat = predict_inflation_factor(model, rec.bl_sum, rec.n_parent)
        t_adj = rec.stat / c_hat
        p_adj = float(chi2.sf(t_adj, df=rec.df))

        focal_parents.append(rec.parent)
        focal_results.append((t_adj, float(rec.df), p_adj))
        methods.append(f"adjusted_{model.method}")

    return focal_parents, focal_results, methods


def _apply_results_adjusted(
    df: pd.DataFrame,
    focal_parents: List[str],
    focal_results: List[Tuple[float, float, float]],
    calibration_methods: List[str],
    skipped_parents: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply deflated results with BH correction to DataFrame."""
    if skipped_parents:
        df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not focal_results:
        return df

    stats = np.array([r[0] for r in focal_results])
    dfs = np.array([r[1] for r in focal_results])
    pvals = np.array([r[2] for r in focal_results])

    invalid_mask = (~np.isfinite(stats)) | (~np.isfinite(dfs)) | (~np.isfinite(pvals))
    pvals_for_correction = np.where(np.isfinite(pvals), pvals, 1.0)

    reject, pvals_adj, _ = benjamini_hochberg_correction(pvals_for_correction, alpha=alpha)
    reject = np.where(invalid_mask, False, reject)

    n_invalid = int(np.sum(invalid_mask))
    if n_invalid:
        logger.warning(
            "Cousin-adjusted Wald audit: total_tests=%d, invalid_tests=%d. "
            "Conservative correction path applied (p=1.0, reject=False).",
            len(focal_results),
            n_invalid,
        )

    df.loc[focal_parents, "Sibling_Test_Statistic"] = stats
    df.loc[focal_parents, "Sibling_Degrees_of_Freedom"] = dfs
    df.loc[focal_parents, "Sibling_Divergence_P_Value"] = pvals
    df.loc[focal_parents, "Sibling_Divergence_P_Value_Corrected"] = pvals_adj
    df.loc[focal_parents, "Sibling_Divergence_Invalid"] = invalid_mask
    df.loc[focal_parents, "Sibling_BH_Different"] = reject
    df.loc[focal_parents, "Sibling_BH_Same"] = ~reject
    df.loc[focal_parents, "Sibling_Test_Method"] = calibration_methods

    return df


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
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    mean_bl = compute_mean_branch_length(tree)

    # Pass 1: compute ALL raw Wald stats
    records = _collect_all_pairs(
        tree,
        df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )

    if not records:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    n_null = sum(1 for r in records if r.is_null_like)
    n_focal = sum(1 for r in records if not r.is_null_like)
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
