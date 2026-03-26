"""Cousin-adjusted Wald sibling divergence annotation.

Corrects post-selection inflation in sibling Wald statistics by:
1. computing raw sibling statistics for all binary parent nodes,
2. fitting a global inflation factor from continuous edge-weighted T/df ratios,
3. localizing that global factor per node in structural-dimension space, and
4. deflating focal sibling pairs before BH correction.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ..branch_length_utils import compute_mean_branch_length
from ..projection.chi2_pvalue import WhiteningMode
from .fdr_annotation import (
    apply_sibling_bh_results,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .inflation_correction.conditional_deflation import (
    PoolStats,
    compute_pool_stats,
    predict_local_inflation_factor,
)
from .inflation_correction.inflation_estimation import CalibrationModel, fit_inflation_model
from .pair_testing.sibling_null_prior_interpolation import interpolate_sibling_null_priors
from .pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
    compute_adjusted_sibling_tests,
    count_null_focal_pairs,
)
from .pair_testing.types import SiblingPairRecord

logger = logging.getLogger(__name__)

# =============================================================================
# Pipeline: collect -> test -> calibrate -> deflate
# =============================================================================


def _resolve_calibration(
    sibling_test_record: SiblingPairRecord,
    model: CalibrationModel,
    pool: PoolStats | None,
) -> tuple[float, str]:
    """Return the inflation adjustment and label for one sibling test."""
    if pool is not None:
        inflation_factor = predict_local_inflation_factor(
            model,
            pool,
            sibling_test_record.structural_dimension,
        )
        return inflation_factor, "local_structural_k_kernel"
    # Global model: intercept-only, same c-hat for all pairs
    return model.global_inflation_factor, "global_weighted_mean"


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_adjusted(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
    pca_eigenvalues: Dict[str, np.ndarray] | None = None,
    child_pca_projections: Dict[str, list[np.ndarray]] | None = None,
    whitening: WhiteningMode = "per_component",
) -> pd.DataFrame:
    """Test sibling divergence using cousin-adjusted Wald.

    Runtime path:
    1. Compute raw Wald chi-squared stats for ALL binary-child parent nodes.
       All valid pairs contribute to calibration through continuous edge weights;
       focal pairs are the only ones tested after deflation.
    2. Estimate a global inflation factor using a weighted mean of T/df ratios,
       with weights ``min(p_edge_left, p_edge_right)``.
    3. Build a local kernel in log-structural-dimension space using the
       decomposition-derived sibling dimension ``k_struct`` and the
       edge-weighted log-k bandwidth of the calibration pool.
    4. For focal pairs, deflate ``T_adj = T / c_node`` and compute the
       adjusted sibling p-value from ``chi2.sf(T_adj, df_effective)``.

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    annotations_df : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' column.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with sibling divergence columns plus ``Sibling_Test_Method``.
    """
    annotations_df = init_sibling_annotation_df(annotations_df)

    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Pass 1: compute raw Wald stats for ALL pairs (needed for calibration)
    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=whitening,
    )

    # Mark non-binary/leaf nodes as skipped (never testable)
    mark_non_binary_as_skipped(annotations_df, non_binary, logger=logger)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    n_null, n_focal, n_blocked = count_null_focal_pairs(records)

    if n_blocked > 0:
        records = interpolate_sibling_null_priors(records, tree, annotations_df)

    # Pass 2: fit inflation model using continuous edge weights
    model = fit_inflation_model(records)

    # Pass 2b: compute pool stats for local per-node deflation
    pool = compute_pool_stats(records, model)

    # Pass 3: deflate focal pairs only and compute p-values
    tested_parent_ids, adjusted_test_summaries, adjustment_method_labels = (
        compute_adjusted_sibling_tests(
            records,
            resolve_inflation_adjustment=partial(_resolve_calibration, model=model, pool=pool),
        )
    )

    # Null-like parents are skipped (they are noise splits)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    # Apply BH correction to deflated sibling test results
    annotations_df = apply_sibling_bh_results(
        annotations_df,
        tested_parent_ids,
        adjusted_test_summaries,
        significance_level_alpha,
        logger=logger,
        audit_label="Cousin-adjusted Wald",
        method_labels=adjustment_method_labels,
        skipped_parents=skipped_parents,
    )

    # Audit metadata
    annotations_df.attrs["sibling_divergence_audit"] = {
        "total_pairs": len(records),
        "null_like_pairs": n_null,
        "focal_pairs": n_focal,
        "gate2_blocked_pairs": n_blocked,
        "calibration_method": model.method,
        "calibration_n": model.n_calibration,
        "global_inflation_factor": model.global_inflation_factor,
        "deflation_mode": "local_structural_k_kernel",
        "local_kernel_center_structural_dimension": pool.geometric_mean_structural_dimension,
        "local_kernel_bandwidth_log_structural_dimension": pool.bandwidth_log_structural_dimension,
        "local_kernel_bandwidth_status": pool.bandwidth_status,
        "single_feature_subtree_mode": config.SINGLE_FEATURE_SUBTREE_MODE,
        "diagnostics": model.diagnostics,
        "test_method": "cousin_adjusted_wald",
    }

    # Store the fitted model object for downstream use (e.g., post-hoc merge).
    annotations_df.attrs["_calibration_model"] = model

    return annotations_df


__all__ = ["annotate_sibling_divergence_adjusted"]
