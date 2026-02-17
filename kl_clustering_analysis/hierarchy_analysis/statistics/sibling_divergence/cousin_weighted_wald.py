"""Weighted cousin-adjusted Wald sibling divergence test.

Replaces the binary null-like/focal labeling of ``cousin_adjusted_wald``
with **continuous weights** derived from edge (child-parent) p-values.
This eliminates the arbitrary α=0.05 threshold that currently determines
which pairs are "null-like" calibration references.

Key difference from cousin_adjusted_wald
-----------------------------------------
- ``cousin_adjusted_wald``: Binary split — pairs with both edge p > α
  are "null-like" (weight=1), all others are "focal" (weight=0).
  Only null-like pairs contribute to the regression.
- ``cousin_weighted_wald``: ALL pairs contribute to the regression,
  weighted by w = min(p_edge_left, p_edge_right).  Pairs where both
  children have high edge p-values (truly null) dominate the fit;
  pairs with low edge p-values (likely signal) contribute minimally.

Why min(p_left, p_right)?
-------------------------
For a pair to be a good calibration reference, BOTH children must be
uninformative (no signal).  The bottleneck is the child more likely to
carry signal, so ``min()`` is the right aggregation — it equals zero
whenever either child is edge-significant, smoothly transitioning
from "useless for calibration" to "excellent calibration reference."

Architecture
------------
1. Compute all raw Wald stats T_i, k_i for every eligible sibling pair.
2. Retrieve edge p-values for each child (from Gate 2 results).
3. Compute weight w_i = min(p_edge_left, p_edge_right) for each pair.
4. Fit *weighted* intercept-only Gamma GLM:
       E[T_i/k_i] = exp(β₀)   with weights w_i.
   This gives ĉ = exp(β₀) = weighted mean of T/k, where null-like pairs
   (high edge p-values) dominate the estimate.
   Note: covariates log(BL_sum) and log(n_parent) were removed because they
   confound signal strength with post-selection inflation.
5. Deflate every focal pair: T_adj = T / ĉ, p = χ²_sf(T_adj, k).
6. Apply BH correction on adjusted p-values.

Focal / skip distinction
-------------------------
Even with continuous weights, we still need to decide which pairs to
*test* and which to *skip*.  A pair is skipped (treated as merge) when
NEITHER child passed the edge test — same as before.  The difference
is that the regression model is fit using all pairs weighted by
their null-likeness, rather than a binary subset.

Fallback tiers
--------------
- Weighted regression succeeds → per-pair ĉ prediction
- Weighted regression fails (< 3 valid pairs) → global weighted median
- < 3 pairs total → no calibration (raw Wald)

Configuration
-------------
Toggle via ``config.SIBLING_TEST_METHOD = "cousin_weighted_wald"``.
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

try:
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

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

# Minimum pairs for calibration tiers
_MIN_REGRESSION = 5
_MIN_MEDIAN = 3


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class _WeightedRecord:
    """Per-sibling-pair record with continuous weight."""

    parent: str
    left: str
    right: str
    stat: float  # raw Wald T
    df: int  # projection dimension k
    pval: float  # raw Wald p
    bl_sum: float  # branch-length sum (left + right)
    n_parent: int  # number of leaves under parent
    weight: float  # min(p_edge_left, p_edge_right), continuous [0, 1]
    is_null_like: bool  # neither child edge-significant (for skip/test decision)


@dataclass
class WeightedCalibrationModel:
    """Result of fitting the weighted inflation model.

    Public API — used by both annotation and post-hoc merge deflation.
    """

    method: str  # "gamma_glm", "weighted_regression", "weighted_median", "none"
    n_calibration: int  # number of pairs used (with weight > 0)
    global_c_hat: float  # weighted mean of ratios
    max_observed_ratio: float = 1.0  # upper bound for ĉ (from null-like pairs only)
    beta: Optional[np.ndarray] = None  # regression coefficients
    diagnostics: Dict = field(default_factory=dict)


# =============================================================================
# Edge p-value retrieval
# =============================================================================


def _get_edge_pvalue(
    node: str,
    pval_map: Dict[str, float],
) -> float:
    """Get the BH-corrected edge p-value for a node.

    Returns 1.0 if the node has no edge p-value (e.g. root, non-tested).
    A return of 1.0 means "maximally null-like" for weighting purposes.
    """
    val = pval_map.get(node, np.nan)
    if np.isfinite(val):
        return float(val)
    return 1.0


# =============================================================================
# Weighted inflation estimation
# =============================================================================


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of values with given weights."""
    if len(values) == 0:
        return 1.0
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative = np.cumsum(sorted_weights)
    half = cumulative[-1] / 2.0
    idx = np.searchsorted(cumulative, half)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def _fit_weighted_inflation_model(
    records: List[_WeightedRecord],
) -> WeightedCalibrationModel:
    """Estimate post-selection inflation using Gamma GLM on ALL pairs.

    Uses min(p_edge_left, p_edge_right) as frequency weights, so high-confidence
    null pairs dominate the fit while signal pairs contribute minimally.

    Under H₀, T ~ c·χ²(k), so r = T/k has E[r] = c and Var(r) = 2c²/k.
    The Gamma family with V(μ) = μ² matches this variance structure.
    A log link gives: log E[T_i/k_i] = β₀ + β₁·log(BL_i) + β₂·log(n_i).

    This replaces the previous log-normal WLS which had Jensen's bias ≈ -1/k.

    max_observed_ratio is computed from NULL-LIKE pairs only (is_null_like=True)
    to prevent focal/signal pairs from inflating the extrapolation clamp.
    """
    # Filter to valid records (finite stat, positive df and ratio)
    valid_records = [
        r
        for r in records
        if np.isfinite(r.stat) and r.df > 0 and r.stat > 0 and r.n_parent > 0 and r.weight > 0
    ]

    if not valid_records:
        logger.warning(
            "Weighted cousin Wald: 0 valid pairs — "
            "no calibration possible; raw Wald stats will be used."
        )
        return WeightedCalibrationModel(
            method="none",
            n_calibration=0,
            global_c_hat=1.0,
            max_observed_ratio=1.0,
        )

    ratios = np.array([r.stat / r.df for r in valid_records])
    weights = np.array([r.weight for r in valid_records])

    # max_observed_ratio from NULL-LIKE pairs only — prevents signal pairs
    # from inflating the extrapolation clamp ceiling.
    null_like_ratios = np.array([r.stat / r.df for r in valid_records if r.is_null_like])
    max_c = float(np.max(null_like_ratios)) if len(null_like_ratios) > 0 else float(np.max(ratios))

    n_cal = len(ratios)
    global_c = float(np.average(ratios, weights=weights))  # weighted arithmetic mean

    if n_cal < _MIN_MEDIAN:
        logger.warning(
            "Weighted cousin Wald: only %d valid pairs (need ≥%d) — "
            "raw Wald stats will be used (ĉ = 1.0).",
            n_cal,
            _MIN_MEDIAN,
        )
        return WeightedCalibrationModel(
            method="none",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
        )

    if n_cal < _MIN_REGRESSION:
        logger.info(
            "Weighted cousin Wald: %d valid pairs (need ≥%d for regression) — "
            "using weighted mean ĉ = %.3f.",
            n_cal,
            _MIN_REGRESSION,
            global_c,
        )
        return WeightedCalibrationModel(
            method="weighted_median",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
        )

    # --- Design matrix (intercept-only) ---
    # Covariates log(BL_sum) and log(n_parent) were removed because they
    # confound signal strength with post-selection inflation: under H₀
    # the inflation factor c is constant (independent of n and BL), but
    # a regression picks up the correlation between larger n → more power
    # → higher T/k from SIGNAL, not inflation.  Intercept-only Gamma GLM
    # estimates the weighted mean inflation factor.
    X = np.ones((n_cal, 1))

    # --- Try Gamma GLM with log link (preferred) ---
    beta = None
    r_squared = 0.0
    method = "weighted_median"  # fallback default
    glm_diagnostics: Dict = {}

    if _HAS_STATSMODELS:
        try:
            glm = sm.GLM(
                ratios,
                X,
                family=sm.families.Gamma(link=sm.families.links.Log()),
                freq_weights=weights,
            )
            result = glm.fit()
            beta = result.params

            # Deviance-based pseudo-R² (1 - deviance/null_deviance)
            if result.null_deviance > 0:
                r_squared = 1.0 - result.deviance / result.null_deviance
            else:
                r_squared = 0.0

            glm_diagnostics = {
                "deviance": float(result.deviance),
                "null_deviance": float(result.null_deviance),
                "aic": float(result.aic),
                "scale": float(result.scale),
                "converged": bool(result.converged),
            }
            method = "gamma_glm"

            logger.info(
                "Weighted cousin Wald: fitted Gamma GLM (intercept-only) on %d pairs "
                "(eff. n=%.1f). β₀ = %.3f → ĉ = %.3f, "
                "deviance = %.3f.",
                n_cal,
                float(np.sum(weights) ** 2 / np.sum(weights**2)),
                beta[0],
                float(np.exp(beta[0])),
                float(result.deviance),
            )
        except Exception as exc:
            logger.warning(
                "Weighted cousin Wald: Gamma GLM failed (%s) — " "falling back to WLS.",
                exc,
            )
            beta = None

    # --- Fallback: Weighted log-linear regression (WLS) ---
    if beta is None:
        log_r = np.log(ratios)
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = log_r * sqrt_w

        try:
            beta, _residuals, _rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning(
                "Weighted cousin Wald: WLS regression also failed — "
                "falling back to weighted mean ĉ = %.3f.",
                global_c,
            )
            return WeightedCalibrationModel(
                method="weighted_median",
                n_calibration=n_cal,
                global_c_hat=global_c,
                max_observed_ratio=max_c,
            )

        fitted = X @ beta
        ss_res = float(np.sum(weights * (log_r - fitted) ** 2))
        weighted_mean = float(np.average(log_r, weights=weights))
        ss_tot = float(np.sum(weights * (log_r - weighted_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        method = "weighted_regression"

        logger.info(
            "Weighted cousin Wald: fitted WLS regression (intercept-only) on %d pairs "
            "(effective n=%.1f). β₀ = %.3f → ĉ = %.3f, R² = %.3f.",
            n_cal,
            float(np.sum(weights) ** 2 / np.sum(weights**2)),
            beta[0],
            float(np.exp(beta[0])),
            r_squared,
        )

    diagnostics = {
        "r_squared": r_squared,
        "beta": beta.tolist() if beta is not None else None,
        "n_calibration": n_cal,
        "global_c_hat": float(global_c),
        "max_observed_ratio": float(max_c),
        "total_weight": float(np.sum(weights)),
        "effective_n": float(np.sum(weights) ** 2 / np.sum(weights**2)),
        "n_null_like": int(len(null_like_ratios)),
        **glm_diagnostics,
    }

    # Note: R² gate removed — with intercept-only design, R² is always 0
    # (no covariates to explain variance), which is expected, not a failure.

    return WeightedCalibrationModel(
        method=method,
        n_calibration=n_cal,
        global_c_hat=global_c,
        max_observed_ratio=max_c,
        beta=np.asarray(beta) if beta is not None else None,
        diagnostics=diagnostics,
    )


def predict_weighted_inflation_factor(
    model: WeightedCalibrationModel,
    bl_sum: float = 0.0,
    n_parent: int = 0,
) -> float:
    """Predict inflation factor ĉ for a pair.

    With intercept-only model, ĉ = exp(β₀) — a single global constant
    (the weighted mean of T/k).  bl_sum and n_parent are retained in
    the signature for API compatibility but are not used.

    Clamped to [1.0, max_observed_ratio] to prevent overestimation.
    """
    if model.method == "none":
        return 1.0

    if model.method == "weighted_median":
        return max(model.global_c_hat, 1.0)

    # Intercept-only GLM or WLS: ĉ = exp(β₀)
    if model.beta is None:
        return max(model.global_c_hat, 1.0)

    c_hat = float(np.exp(model.beta[0]))
    c_hat = min(c_hat, model.max_observed_ratio)
    return max(c_hat, 1.0)


# =============================================================================
# Pipeline: collect → weight → fit → deflate
# =============================================================================


def _collect_weighted_pairs(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
    mean_bl: float | None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
) -> List[_WeightedRecord]:
    """Collect ALL sibling pairs with continuous weights from edge p-values."""
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. " "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    # Build edge p-value map from the BH-corrected column
    pval_col = "Child_Parent_Divergence_P_Value_BH"
    if pval_col in nodes_df.columns:
        pval_map = nodes_df[pval_col].to_dict()
    else:
        # Fallback to raw p-values if BH not available
        pval_col_raw = "Child_Parent_Divergence_P_Value"
        pval_map = nodes_df[pval_col_raw].to_dict() if pval_col_raw in nodes_df.columns else {}

    records: List[_WeightedRecord] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children
        left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(tree, parent, left, right)

        # Look up spectral info for this parent node
        _spectral_k = spectral_dims.get(parent) if spectral_dims else None
        _pca_proj = pca_projections.get(parent) if pca_projections else None
        _pca_eig = pca_eigenvalues.get(parent) if pca_eigenvalues else None

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
            pca_eigenvalues=_pca_eig,
        )

        # Branch-length sum
        bl_sum = 0.0
        if bl_l is not None and bl_r is not None:
            bl_sum = bl_l + bl_r
        elif bl_l is not None:
            bl_sum = bl_l
        elif bl_r is not None:
            bl_sum = bl_r

        n_parent = extract_node_sample_size(tree, parent)

        # Continuous weight: min(p_edge_left, p_edge_right)
        p_left = _get_edge_pvalue(left, pval_map)
        p_right = _get_edge_pvalue(right, pval_map)
        weight = min(p_left, p_right)

        # Binary label still needed for skip/test decision
        is_null = not _either_child_significant(left, right, sig_map)

        records.append(
            _WeightedRecord(
                parent=parent,
                left=left,
                right=right,
                stat=stat,
                df=int(df) if np.isfinite(df) else 0,
                pval=pval,
                bl_sum=bl_sum,
                n_parent=n_parent,
                weight=weight,
                is_null_like=is_null,
            )
        )

    return records


def _deflate_and_test(
    records: List[_WeightedRecord],
    model: WeightedCalibrationModel,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs and compute adjusted p-values."""
    focal_parents: List[str] = []
    focal_results: List[Tuple[float, float, float]] = []
    methods: List[str] = []

    for rec in records:
        if rec.is_null_like:
            continue  # null-like pairs are skipped (merge)

        if not np.isfinite(rec.stat) or rec.df <= 0:
            focal_parents.append(rec.parent)
            focal_results.append((np.nan, np.nan, np.nan))
            methods.append("invalid")
            continue

        c_hat = predict_weighted_inflation_factor(model, rec.bl_sum, rec.n_parent)
        t_adj = rec.stat / c_hat
        p_adj = float(chi2.sf(t_adj, df=rec.df))

        focal_parents.append(rec.parent)
        focal_results.append((t_adj, float(rec.df), p_adj))
        methods.append(f"weighted_{model.method}")

    return focal_parents, focal_results, methods


def _apply_results(
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
            "Weighted cousin Wald audit: total_tests=%d, invalid_tests=%d. "
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


def annotate_sibling_divergence_weighted(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using weighted cousin-adjusted Wald.

    Instead of a binary null-like/focal split, uses edge p-values as
    continuous weights in the regression model.  All pairs contribute
    to the inflation estimate, weighted by min(p_edge_left, p_edge_right).

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    nodes_statistics_dataframe : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' and
        'Child_Parent_Divergence_P_Value_BH' columns.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with sibling divergence columns plus ``Sibling_Test_Method``.
    """
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Pass 1: compute ALL raw Wald stats with continuous weights
    records = _collect_weighted_pairs(
        tree, df, mean_bl, spectral_dims, pca_projections, pca_eigenvalues
    )

    if not records:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    n_null = sum(1 for r in records if r.is_null_like)
    n_focal = sum(1 for r in records if not r.is_null_like)
    total_weight = sum(r.weight for r in records)
    logger.info(
        "Weighted cousin Wald: %d total pairs (%d null-like, %d focal), " "total weight = %.2f.",
        len(records),
        n_null,
        n_focal,
        total_weight,
    )

    # Pass 2: fit weighted inflation model using ALL pairs
    model = _fit_weighted_inflation_model(records)

    # Pass 3: deflate focals and compute p-values
    focal_parents, focal_results, cal_methods = _deflate_and_test(records, model)

    # Null-like parents are skipped (merge)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    df = _apply_results(
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
        "test_method": "cousin_weighted_wald",
    }

    # Store the fitted model so downstream consumers (e.g. post-hoc merge)
    # can apply the same deflation — ensures symmetric calibration.
    df.attrs["_calibration_model"] = model

    return df


__all__ = [
    "WeightedCalibrationModel",
    "annotate_sibling_divergence_weighted",
    "predict_weighted_inflation_factor",
]
