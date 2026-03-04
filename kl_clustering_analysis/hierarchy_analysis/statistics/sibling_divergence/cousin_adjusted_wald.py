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


@dataclass(frozen=True)
class _NullLikeCalibrationInputs:
    """Prepared numeric inputs for null-like calibration fitting."""

    ratio_values: np.ndarray
    branch_length_sum_values: np.ndarray
    parent_sample_size_values: np.ndarray
    calibration_pair_count: int
    median_ratio: float
    max_observed_ratio: float


# =============================================================================
# Inflation estimation
# =============================================================================


def _collect_null_like_records(records: List[SiblingPairRecord]) -> List[SiblingPairRecord]:
    """Return records that can be considered null-like calibration candidates."""
    return [
        record
        for record in records
        if record.is_null_like and np.isfinite(record.stat) and record.df > 0
    ]


def _build_null_like_calibration_inputs(
    null_like_records: List[SiblingPairRecord],
) -> _NullLikeCalibrationInputs:
    """Convert null-like records into validated numeric vectors for fitting."""
    ratio_values = np.array([record.stat / record.df for record in null_like_records])
    branch_length_sum_values = np.array([record.bl_sum for record in null_like_records])
    parent_sample_size_values = np.array([record.n_parent for record in null_like_records])

    valid_mask = (
        (branch_length_sum_values > 0)
        & (parent_sample_size_values > 0)
        & np.isfinite(ratio_values)
        & (ratio_values > 0)
    )
    ratio_values = ratio_values[valid_mask]
    branch_length_sum_values = branch_length_sum_values[valid_mask]
    parent_sample_size_values = parent_sample_size_values[valid_mask]

    calibration_pair_count = len(ratio_values)
    if calibration_pair_count > 0:
        median_ratio = float(np.median(ratio_values))
        max_observed_ratio = float(np.max(ratio_values))
    else:
        median_ratio = 1.0
        max_observed_ratio = 1.0

    return _NullLikeCalibrationInputs(
        ratio_values=ratio_values,
        branch_length_sum_values=branch_length_sum_values,
        parent_sample_size_values=parent_sample_size_values,
        calibration_pair_count=calibration_pair_count,
        median_ratio=median_ratio,
        max_observed_ratio=max_observed_ratio,
    )


def _fit_log_linear_inflation_regression(
    calibration_inputs: _NullLikeCalibrationInputs,
) -> tuple[np.ndarray | None, float]:
    """Fit log(ratio) regression against branch length and parent sample size."""
    log_ratio_values = np.log(calibration_inputs.ratio_values)
    design_matrix = np.column_stack(
        [
            np.ones(calibration_inputs.calibration_pair_count),
            np.log(calibration_inputs.branch_length_sum_values),
            np.log(calibration_inputs.parent_sample_size_values.astype(float)),
        ]
    )

    try:
        coefficient_vector, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design_matrix,
            log_ratio_values,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None, 0.0

    fitted_log_ratios = design_matrix @ coefficient_vector
    sum_squared_residuals = float(np.sum((log_ratio_values - fitted_log_ratios) ** 2))
    sum_squared_total = float(np.sum((log_ratio_values - np.mean(log_ratio_values)) ** 2))
    coefficient_of_determination = (
        1.0 - sum_squared_residuals / sum_squared_total if sum_squared_total > 0 else 0.0
    )
    return np.asarray(coefficient_vector), coefficient_of_determination


def _fit_inflation_model(
    records: List[SiblingPairRecord],
) -> CalibrationModel:
    """Estimate the post-selection inflation factor from null-like pairs.

    Uses log-linear regression on (BL_sum, n_parent) when ≥ 5 null-like pairs
    are available, otherwise falls back to global median, and finally to no
    calibration.
    """
    null_like_records = _collect_null_like_records(records)

    if not null_like_records:
        logger.warning(
            "Cousin-adjusted Wald: 0 null-like pairs — "
            "no calibration possible; raw Wald stats will be used."
        )
        return CalibrationModel(
            method="none", n_calibration=0, global_c_hat=1.0, max_observed_ratio=1.0
        )

    calibration_inputs = _build_null_like_calibration_inputs(null_like_records)

    if calibration_inputs.calibration_pair_count < _MIN_MEDIAN:
        logger.warning(
            "Cousin-adjusted Wald: only %d null-like pairs (need ≥%d) — "
            "raw Wald stats will be used (ĉ = 1.0).",
            calibration_inputs.calibration_pair_count,
            _MIN_MEDIAN,
        )
        return CalibrationModel(
            method="none",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.median_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
        )

    if calibration_inputs.calibration_pair_count < _MIN_REGRESSION:
        logger.info(
            "Cousin-adjusted Wald: %d null-like pairs (need ≥%d for regression) — "
            "using global median ĉ = %.3f.",
            calibration_inputs.calibration_pair_count,
            _MIN_REGRESSION,
            calibration_inputs.median_ratio,
        )
        return CalibrationModel(
            method="median",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.median_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
        )

    coefficient_vector, coefficient_of_determination = _fit_log_linear_inflation_regression(
        calibration_inputs
    )
    if coefficient_vector is None:
        logger.warning(
            "Cousin-adjusted Wald: regression failed — " "falling back to global median ĉ = %.3f.",
            calibration_inputs.median_ratio,
        )
        return CalibrationModel(
            method="median",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.median_ratio,
        )

    diagnostics = {
        "r_squared": coefficient_of_determination,
        "beta": coefficient_vector.tolist(),
        "n_calibration": calibration_inputs.calibration_pair_count,
        "median_ratio": float(calibration_inputs.median_ratio),
        "max_observed_ratio": float(calibration_inputs.max_observed_ratio),
    }

    if coefficient_of_determination < 0.05:
        logger.info(
            "Cousin-adjusted Wald: regression R²=%.3f (< 0.05) — "
            "not informative; using global median ĉ = %.3f.",
            coefficient_of_determination,
            calibration_inputs.median_ratio,
        )
        return CalibrationModel(
            method="median",
            n_calibration=calibration_inputs.calibration_pair_count,
            global_c_hat=calibration_inputs.median_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
            diagnostics=diagnostics,
        )

    logger.info(
        "Cousin-adjusted Wald: fitted regression on %d null-like pairs. "
        "β = [%.3f, %.3f, %.3f], R² = %.3f.",
        calibration_inputs.calibration_pair_count,
        coefficient_vector[0],
        coefficient_vector[1],
        coefficient_vector[2],
        coefficient_of_determination,
    )

    return CalibrationModel(
        method="regression",
        n_calibration=calibration_inputs.calibration_pair_count,
        global_c_hat=calibration_inputs.median_ratio,
        max_observed_ratio=calibration_inputs.max_observed_ratio,
        beta=np.asarray(coefficient_vector),
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
