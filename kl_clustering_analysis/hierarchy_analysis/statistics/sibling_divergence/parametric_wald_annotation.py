"""Parametric Wald sibling divergence annotation.

Corrects post-selection inflation using a per-node power-law model:

    c(n) = α · n^(-β)

fitted from null-like calibration pairs (neither child edge-significant).
Each focal pair is deflated by its node-specific predicted c(n_parent),
then tested against χ²(k).

Falls back to global weighted-mean ĉ when fewer than 3 null-like pairs
are available for fitting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ..branch_length_utils import compute_mean_branch_length
from .fdr_annotation import (
    apply_sibling_bh_results,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .inflation_correction.inflation_estimation import (
    CalibrationModel,
    fit_parametric_inflation_model,
    predict_parametric_inflation_factor,
)
from .pair_testing.sibling_pair_collection import (
    SiblingPairRecord,
    collect_sibling_pair_records,
    count_null_focal_pairs,
    deflate_focal_pairs,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline: collect -> fit parametric -> deflate per-node -> apply
# =============================================================================


def _collect_all_pairs(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
    pca_eigenvalues: Dict[str, np.ndarray] | None = None,
    child_pca_projections: Dict[str, list[np.ndarray]] | None = None,
    whitening: str = "per_component",
) -> Tuple[List[SiblingPairRecord], List[str]]:
    """Collect ALL binary-child parent nodes and compute raw Wald stats."""
    return collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=whitening,
    )


def _deflate_and_test(
    records: List[SiblingPairRecord],
    model: CalibrationModel,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs using per-node c(n) and compute adjusted p-values."""

    def _resolve_calibration(rec: SiblingPairRecord) -> tuple[float, str]:
        c = predict_parametric_inflation_factor(model, n_reference=rec.n_parent)
        method_label = (
            "parametric_power_law"
            if model.method == "parametric_power_law"
            else "parametric_fallback"
        )
        return c, method_label

    return deflate_focal_pairs(
        records,
        calibration_resolver=_resolve_calibration,
    )


def _apply_results_parametric(
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
        audit_label="Parametric Wald",
        method_labels=calibration_methods,
        skipped_parents=skipped_parents,
    )


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_parametric(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
    pca_eigenvalues: Dict[str, np.ndarray] | None = None,
    child_pca_projections: Dict[str, list[np.ndarray]] | None = None,
    whitening: str = "per_component",
) -> pd.DataFrame:
    """Test sibling divergence using parametric Wald with c(n) = α · n^(-β).

    Three-pass approach:
    1. Compute raw Wald χ² stats for ALL binary-child parent nodes.
    2. Fit power-law inflation model c(n) = α · n^(-β) from null-like pairs.
    3. For *focal* pairs, deflate with per-node prediction:
       T_adj = T / c(n_parent), p = χ²_sf(T_adj, k).

    Falls back to global weighted-mean ĉ when fewer than 3 null-like pairs
    are available.

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

    # Pass 1: compute raw Wald stats for ALL pairs
    records, non_binary = _collect_all_pairs(
        tree,
        annotations_df,
        mean_branch_length,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=whitening,
    )

    mark_non_binary_as_skipped(annotations_df, non_binary, logger=logger)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    n_null, n_focal, n_blocked = count_null_focal_pairs(records)
    logger.info(
        "Parametric Wald: %d total pairs (%d null-like, %d focal, %d gate2-blocked).",
        len(records),
        n_null,
        n_focal,
        n_blocked,
    )

    # Pass 2: fit parametric inflation model c(n) = α · n^(-β)
    model = fit_parametric_inflation_model(records)

    # Pass 3: deflate focal pairs with per-node c(n) prediction
    focal_parents, focal_results, cal_methods = _deflate_and_test(records, model)

    # Null-like parents are skipped (they are noise splits)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    annotations_df = _apply_results_parametric(
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
        "gate2_blocked_pairs": n_blocked,
        "calibration_method": model.method,
        "calibration_n": model.n_calibration,
        "global_inflation_factor": model.global_inflation_factor,
        "diagnostics": model.diagnostics,
        "test_method": "parametric_wald",
    }

    annotations_df.attrs["_calibration_model"] = model

    return annotations_df


__all__ = ["annotate_sibling_divergence_parametric"]
