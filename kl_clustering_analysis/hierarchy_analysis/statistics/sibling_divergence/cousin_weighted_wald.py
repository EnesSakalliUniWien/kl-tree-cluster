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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

from kl_clustering_analysis import config

from ..branch_length_utils import compute_mean_branch_length
from .cousin_pipeline_helpers import (
    apply_calibrated_results,
    count_null_focal_pairs,
    collect_sibling_pair_records,
    deflate_focal_pairs,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)

logger = logging.getLogger(__name__)

# Minimum pairs for calibration tiers
_MIN_REGRESSION = 5
_MIN_MEDIAN = 3
_WARNED_MISSING_STATSMODELS = False


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
    degrees_of_freedom: int  # projection dimension k
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


@dataclass(frozen=True)
class _WeightedCalibrationInputs:
    """Prepared numeric inputs for weighted calibration fitting."""

    ratio_values: np.ndarray
    weight_values: np.ndarray
    null_like_ratio_values: np.ndarray
    max_observed_ratio: float
    calibration_pair_count: int
    global_weighted_ratio: float
    effective_sample_size: float


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


def _resolve_statsmodels_availability() -> bool:
    """Return whether statsmodels-backed Gamma GLM can be used."""
    statsmodels_available = bool(_HAS_STATSMODELS and "sm" in globals())
    global _WARNED_MISSING_STATSMODELS
    if not statsmodels_available and not _WARNED_MISSING_STATSMODELS:
        logger.warning(
            "Weighted cousin Wald: statsmodels is unavailable; "
            "Gamma GLM calibration disabled. Falling back to weighted log-linear regression."
        )
        _WARNED_MISSING_STATSMODELS = True
    return statsmodels_available


def _filter_valid_weighted_records(records: List[_WeightedRecord]) -> List[_WeightedRecord]:
    """Keep records that are numerically valid for calibration fitting."""
    return [
        record
        for record in records
        if np.isfinite(record.stat)
        and record.degrees_of_freedom > 0
        and record.stat > 0
        and record.n_parent > 0
        and record.weight > 0
    ]


def _build_weighted_calibration_inputs(
    valid_records: List[_WeightedRecord],
) -> _WeightedCalibrationInputs:
    """Convert valid records into numeric vectors used by calibration models."""
    ratio_values = np.array([record.stat / record.degrees_of_freedom for record in valid_records])
    weight_values = np.array([record.weight for record in valid_records])
    null_like_ratio_values = np.array(
        [record.stat / record.degrees_of_freedom for record in valid_records if record.is_null_like]
    )

    if len(null_like_ratio_values) > 0:
        max_observed_ratio = float(np.max(null_like_ratio_values))
    else:
        max_observed_ratio = float(np.max(ratio_values))

    calibration_pair_count = len(ratio_values)
    global_weighted_ratio = float(np.average(ratio_values, weights=weight_values))

    total_weight = float(np.sum(weight_values))
    squared_weight_sum = float(np.sum(weight_values**2))
    if squared_weight_sum > 0.0:
        effective_sample_size = float((total_weight**2) / squared_weight_sum)
    else:
        effective_sample_size = 0.0

    return _WeightedCalibrationInputs(
        ratio_values=ratio_values,
        weight_values=weight_values,
        null_like_ratio_values=null_like_ratio_values,
        max_observed_ratio=max_observed_ratio,
        calibration_pair_count=calibration_pair_count,
        global_weighted_ratio=global_weighted_ratio,
        effective_sample_size=effective_sample_size,
    )


def _fit_intercept_only_gamma_glm(
    ratio_values: np.ndarray,
    weight_values: np.ndarray,
    calibration_pair_count: int,
    effective_sample_size: float,
) -> tuple[np.ndarray | None, float, Dict]:
    """Fit intercept-only Gamma GLM and return coefficients and diagnostics."""
    design_matrix = np.ones((calibration_pair_count, 1))

    try:
        gamma_glm = sm.GLM(
            ratio_values,
            design_matrix,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            freq_weights=weight_values,
        )
        gamma_glm_result = gamma_glm.fit()
        coefficient_vector = np.asarray(gamma_glm_result.params)

        if gamma_glm_result.null_deviance > 0:
            coefficient_of_determination = (
                1.0 - gamma_glm_result.deviance / gamma_glm_result.null_deviance
            )
        else:
            coefficient_of_determination = 0.0

        glm_diagnostics = {
            "deviance": float(gamma_glm_result.deviance),
            "null_deviance": float(gamma_glm_result.null_deviance),
            "aic": float(gamma_glm_result.aic),
            "scale": float(gamma_glm_result.scale),
            "converged": bool(gamma_glm_result.converged),
        }

        logger.info(
            "Weighted cousin Wald: fitted Gamma GLM (intercept-only) on %d pairs "
            "(eff. n=%.1f). β₀ = %.3f → ĉ = %.3f, "
            "deviance = %.3f.",
            calibration_pair_count,
            effective_sample_size,
            coefficient_vector[0],
            float(np.exp(coefficient_vector[0])),
            float(gamma_glm_result.deviance),
        )
        return coefficient_vector, coefficient_of_determination, glm_diagnostics
    except Exception as exception:  # pragma: no cover - exercised by tests via monkeypatch
        logger.warning(
            "Weighted cousin Wald: Gamma GLM failed (%s) — falling back to WLS.",
            exception,
        )
        return None, 0.0, {}


def _fit_intercept_only_weighted_log_regression(
    ratio_values: np.ndarray,
    weight_values: np.ndarray,
    calibration_pair_count: int,
    effective_sample_size: float,
    global_weighted_ratio: float,
) -> tuple[np.ndarray | None, float]:
    """Fit intercept-only weighted log-linear regression as fallback."""
    log_ratio_values = np.log(ratio_values)
    square_root_weights = np.sqrt(weight_values)

    design_matrix = np.ones((calibration_pair_count, 1))
    weighted_design_matrix = design_matrix * square_root_weights[:, np.newaxis]
    weighted_targets = log_ratio_values * square_root_weights

    try:
        coefficient_vector, _residuals, _rank, _singular_values = np.linalg.lstsq(
            weighted_design_matrix,
            weighted_targets,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        logger.warning(
            "Weighted cousin Wald: WLS regression also failed — "
            "falling back to weighted mean ĉ = %.3f.",
            global_weighted_ratio,
        )
        return None, 0.0

    fitted_log_ratios = design_matrix @ coefficient_vector
    sum_squared_residuals = float(np.sum(weight_values * (log_ratio_values - fitted_log_ratios) ** 2))
    weighted_log_ratio_mean = float(np.average(log_ratio_values, weights=weight_values))
    sum_squared_total = float(np.sum(weight_values * (log_ratio_values - weighted_log_ratio_mean) ** 2))
    coefficient_of_determination = (
        1.0 - sum_squared_residuals / sum_squared_total if sum_squared_total > 0 else 0.0
    )

    logger.info(
        "Weighted cousin Wald: fitted WLS regression (intercept-only) on %d pairs "
        "(effective n=%.1f). β₀ = %.3f → ĉ = %.3f, R² = %.3f.",
        calibration_pair_count,
        effective_sample_size,
        coefficient_vector[0],
        float(np.exp(coefficient_vector[0])),
        coefficient_of_determination,
    )

    return np.asarray(coefficient_vector), coefficient_of_determination


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
    statsmodels_available = _resolve_statsmodels_availability()
    valid_records = _filter_valid_weighted_records(records)

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
            diagnostics={"statsmodels_available": statsmodels_available},
        )

    calibration_inputs = _build_weighted_calibration_inputs(valid_records)

    if calibration_inputs.calibration_pair_count < _MIN_MEDIAN:
        logger.warning(
            "Weighted cousin Wald: only %d valid pairs (need ≥%d) — "
            "raw Wald stats will be used (ĉ = 1.0).",
            calibration_inputs.calibration_pair_count,
            _MIN_MEDIAN,
        )
        return WeightedCalibrationModel(
            method="none",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.global_weighted_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
            diagnostics={"statsmodels_available": statsmodels_available},
        )

    if calibration_inputs.calibration_pair_count < _MIN_REGRESSION:
        logger.info(
            "Weighted cousin Wald: %d valid pairs (need ≥%d for regression) — "
            "using weighted mean ĉ = %.3f.",
            calibration_inputs.calibration_pair_count,
            _MIN_REGRESSION,
            calibration_inputs.global_weighted_ratio,
        )
        return WeightedCalibrationModel(
            method="weighted_median",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.global_weighted_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
            diagnostics={"statsmodels_available": statsmodels_available},
        )

    coefficient_vector = None
    coefficient_of_determination = 0.0
    calibration_method = "weighted_median"
    gamma_glm_diagnostics: Dict = {}
    gamma_glm_attempted = False

    if statsmodels_available:
        gamma_glm_attempted = True
        (
            coefficient_vector,
            coefficient_of_determination,
            gamma_glm_diagnostics,
        ) = _fit_intercept_only_gamma_glm(
            calibration_inputs.ratio_values,
            calibration_inputs.weight_values,
            calibration_inputs.calibration_pair_count,
            calibration_inputs.effective_sample_size,
        )
        if coefficient_vector is not None:
            calibration_method = "gamma_glm"

    if coefficient_vector is None:
        coefficient_vector, coefficient_of_determination = _fit_intercept_only_weighted_log_regression(
            calibration_inputs.ratio_values,
            calibration_inputs.weight_values,
            calibration_inputs.calibration_pair_count,
            calibration_inputs.effective_sample_size,
            calibration_inputs.global_weighted_ratio,
        )
        if coefficient_vector is None:
            return WeightedCalibrationModel(
                method="weighted_median",
                n_calibration=calibration_inputs.calibration_pair_count,
                global_c_hat=calibration_inputs.global_weighted_ratio,
                max_observed_ratio=calibration_inputs.max_observed_ratio,
                diagnostics={"statsmodels_available": statsmodels_available},
            )
        calibration_method = "weighted_regression"

    diagnostics = {
        "r_squared": coefficient_of_determination,
        "beta": coefficient_vector.tolist() if coefficient_vector is not None else None,
        "n_calibration": calibration_inputs.calibration_pair_count,
        "global_c_hat": float(calibration_inputs.global_weighted_ratio),
        "max_observed_ratio": float(calibration_inputs.max_observed_ratio),
        "total_weight": float(np.sum(calibration_inputs.weight_values)),
        "effective_n": calibration_inputs.effective_sample_size,
        "n_null_like": int(len(calibration_inputs.null_like_ratio_values)),
        "statsmodels_available": statsmodels_available,
        "gamma_glm_attempted": gamma_glm_attempted,
        **gamma_glm_diagnostics,
    }

    # Note: R² gate removed — with intercept-only design, R² is always 0
    # (no covariates to explain variance), which is expected, not a failure.

    return WeightedCalibrationModel(
        method=calibration_method,
        n_calibration=calibration_inputs.calibration_pair_count,
        global_c_hat=calibration_inputs.global_weighted_ratio,
        max_observed_ratio=calibration_inputs.max_observed_ratio,
        beta=np.asarray(coefficient_vector) if coefficient_vector is not None else None,
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
    annotations_df: pd.DataFrame,
    mean_bl: float | None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
) -> Tuple[List[_WeightedRecord], List[str]]:
    """Collect ALL sibling pairs with continuous weights from edge p-values.

    Returns (records, non_binary_nodes).
    """
    base_records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )

    # Build edge p-value map from the BH-corrected column
    pval_col = "Child_Parent_Divergence_P_Value_BH"
    if pval_col in annotations_df.columns:
        pval_map = annotations_df[pval_col].to_dict()
    else:
        # Fallback to raw p-values if BH not available
        pval_col_raw = "Child_Parent_Divergence_P_Value"
        pval_map = (
            annotations_df[pval_col_raw].to_dict() if pval_col_raw in annotations_df.columns else {}
        )

    records: List[_WeightedRecord] = []
    for rec in base_records:
        # Continuous weight: min(p_edge_left, p_edge_right)
        p_left = _get_edge_pvalue(rec.left, pval_map)
        p_right = _get_edge_pvalue(rec.right, pval_map)
        weight = min(p_left, p_right)

        records.append(
            _WeightedRecord(
                parent=rec.parent,
                left=rec.left,
                right=rec.right,
                stat=rec.stat,
                degrees_of_freedom=rec.degrees_of_freedom,
                pval=rec.pval,
                bl_sum=rec.bl_sum,
                n_parent=rec.n_parent,
                weight=weight,
                is_null_like=rec.is_null_like,
            )
        )

    return records, non_binary


def _deflate_and_test(
    records: List[_WeightedRecord],
    model: WeightedCalibrationModel,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs and compute adjusted p-values."""
    def _resolve_calibration(rec: _WeightedRecord) -> tuple[float, str]:
        c_hat = predict_weighted_inflation_factor(model, rec.bl_sum, rec.n_parent)
        return c_hat, f"weighted_{model.method}"

    return deflate_focal_pairs(
        records,
        calibration_resolver=_resolve_calibration,
    )


def _apply_results(
    annotations_df: pd.DataFrame,
    focal_parents: List[str],
    focal_results: List[Tuple[float, float, float]],
    calibration_methods: List[str],
    skipped_parents: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply deflated results with BH correction to DataFrame."""
    return apply_calibrated_results(
        annotations_df,
        focal_parents,
        focal_results,
        calibration_methods,
        skipped_parents,
        alpha,
        logger=logger,
        audit_label="Weighted cousin Wald",
    )


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_weighted(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
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
    annotations_df : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' and
        'Child_Parent_Divergence_P_Value_BH' columns.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with sibling divergence columns plus ``Sibling_Test_Method``.
    """
    annotations_df = init_sibling_annotation_df(annotations_df)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Pass 1: compute ALL raw Wald stats with continuous weights
    records, non_binary = _collect_weighted_pairs(
        tree, annotations_df, mean_bl, spectral_dims, pca_projections, pca_eigenvalues
    )

    # Mark non-binary/leaf nodes as skipped (never testable)
    mark_non_binary_as_skipped(annotations_df, non_binary, logger=logger)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    n_null, n_focal = count_null_focal_pairs(records)
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

    annotations_df = _apply_results(
        annotations_df,
        focal_parents,
        focal_results,
        cal_methods,
        skipped_parents,
        significance_level_alpha,
    )

    # Audit metadata
    annotations_df.attrs["sibling_divergence_audit"] = {
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
    annotations_df.attrs["_calibration_model"] = model

    return annotations_df


__all__ = [
    "WeightedCalibrationModel",
    "annotate_sibling_divergence_weighted",
    "predict_weighted_inflation_factor",
]
