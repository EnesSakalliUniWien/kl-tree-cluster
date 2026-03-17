"""Cousin-adjusted Wald sibling divergence annotation.

Corrects the post-selection inflation of the Wald chi-squared statistic
by estimating the inflation factor c from null-like calibration pairs,
then deflating focal pairs before BH correction.

The calibration model lives in ``calibration.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ..branch_length_utils import compute_mean_branch_length
from .bh_annotation import (
    apply_sibling_bh_results,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .inflation_correction.inflation_estimation import (
    CalibrationModel,
    fit_inflation_model,
    predict_inflation_factor,
)
from .pair_testing.sibling_pair_collection import (
    SiblingPairRecord,
    collect_sibling_pair_records,
    count_null_focal_pairs,
    deflate_focal_pairs,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline: collect -> test -> calibrate -> deflate
# =============================================================================


def _collect_all_pairs(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> Tuple[List[SiblingPairRecord], List[str]]:
    """Collect ALL binary-child parent nodes and compute raw Wald stats.

    All pairs are computed for calibration purposes; only focal pairs
    (at least one child edge-significant) are subsequently tested.

    Returns (records, non_binary_nodes).
    """
    return collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length,
        minimum_projection_dimension=minimum_projection_dimension,
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
        inflation_factor = predict_inflation_factor(
            model,
            rec.branch_length_sum,
            n_reference=rec.n_parent,
        )
        return inflation_factor, "adjusted_regression"

    return deflate_focal_pairs(
        records,
        calibration_resolver=_resolve_calibration,
    )


def _apply_results_adjusted(
    annotations_df: pd.DataFrame,
    focal_parents: List[str],
    focal_results: List[Tuple[float, float, float]],
    calibration_methods: List[str],
    skipped_parents: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply deflated results with BH correction to DataFrame."""
    return apply_sibling_bh_results(
        annotations_df,
        focal_parents,
        focal_results,
        alpha,
        logger=logger,
        audit_label="Cousin-adjusted Wald",
        method_labels=calibration_methods,
        skipped_parents=skipped_parents,
    )


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_adjusted(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
    pca_eigenvalues: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using cousin-adjusted Wald.

    Two-pass approach:
    1. Compute raw Wald chi-squared stats for ALL binary-child parent nodes.
       Null-like pairs provide calibration data; focal pairs are tested.
    2. Estimate inflation c-hat using continuous edge-weight calibration
       (weighted mean of T/k, where null-like pairs dominate via high weights).
    3. For *focal* pairs (at least one child edge-significant), deflate:
       T_adj = T / c-hat, p = chi-sq_sf(T_adj, k).

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

    _ = pca_eigenvalues  # not used by adjusted Wald; accepted for uniform dispatcher interface

    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Pass 1: compute raw Wald stats for ALL pairs (needed for calibration)
    records, non_binary = _collect_all_pairs(
        tree,
        annotations_df,
        mean_branch_length,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )

    # Mark non-binary/leaf nodes as skipped (never testable)
    mark_non_binary_as_skipped(annotations_df, non_binary, logger=logger)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    n_null, n_focal = count_null_focal_pairs(records)
    logger.info(
        "Cousin-adjusted Wald: %d total pairs (%d null-like, %d focal).",
        len(records),
        n_null,
        n_focal,
    )

    # Pass 2: fit inflation model using continuous edge weights
    model = fit_inflation_model(records)

    # Pass 3: deflate focal pairs only and compute p-values
    focal_parents, focal_results, cal_methods = _deflate_and_test(records, model)

    # Null-like parents are skipped (they are noise splits)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    annotations_df = _apply_results_adjusted(
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
        "global_inflation_factor": model.global_inflation_factor,
        "diagnostics": model.diagnostics,
        "test_method": "cousin_adjusted_wald",
    }

    # Store the fitted model object for downstream use (e.g., post-hoc merge).
    annotations_df.attrs["_calibration_model"] = model

    return annotations_df


__all__ = ["annotate_sibling_divergence_adjusted"]
