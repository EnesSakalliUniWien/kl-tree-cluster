"""Child-parent divergence annotation for Gate 2."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)

from .spectral_context import _compute_child_parent_spectral_context_with_audit
from .tree_testing import run_child_parent_tests_across_tree
from .child_parent_divergence_audit import (
    build_child_parent_divergence_audit,
    log_non_finite_child_parent_divergence_audit,
)
from .child_parent_divergence_tree_bh import (
    apply_child_parent_divergence_tree_bh_correction,
    attach_child_parent_stopping_edge_recovery_metadata,
)

logger = logging.getLogger(__name__)


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.EDGE_ALPHA,
    leaf_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Test child-parent divergence using the projected Wald pipeline.

    Uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR correction.
    This is the only supported multiple-testing correction method for Gate 2.
    """
    annotations_df = annotations_df.copy()
    edge_alpha = float(significance_level_alpha)

    tree_edges = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in tree_edges]
    child_ids = [child_id for _, child_id in tree_edges]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")

    child_leaf_counts = extract_leaf_counts(annotations_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(annotations_df, parent_ids)

    if leaf_data is None:
        node_spectral_dimensions = None
        node_pca_projections = None
        node_pca_eigenvalues = None
        single_feature_subtree_audit = None
    else:
        (
            node_spectral_dimensions,
            node_pca_projections,
            node_pca_eigenvalues,
            single_feature_subtree_audit,
        ) = _compute_child_parent_spectral_context_with_audit(
            tree,
            leaf_data,
        )

    annotations_df.attrs["_spectral_dims"] = node_spectral_dimensions
    annotations_df.attrs["_pca_projections"] = node_pca_projections
    annotations_df.attrs["_pca_eigenvalues"] = node_pca_eigenvalues

    if single_feature_subtree_audit is not None:
        annotations_df.attrs["_single_feature_subtree_audit"] = single_feature_subtree_audit

    (
        edge_test_statistics,
        edge_degrees_of_freedom,
        edge_p_values,
        invalid_test_flags,
    ) = run_child_parent_tests_across_tree(
        tree=tree,
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_leaf_counts=child_leaf_counts,
        parent_leaf_counts=parent_leaf_counts,
        spectral_dims=node_spectral_dimensions,
        pca_projections=node_pca_projections,
        pca_eigenvalues=node_pca_eigenvalues,
    )

    annotations_df.attrs["_edge_raw_test_data"] = {
        "child_ids": child_ids,
        "parent_ids": parent_ids,
        "test_stats": edge_test_statistics.copy(),
        "degrees_of_freedom": edge_degrees_of_freedom.copy(),
        "p_values": edge_p_values.copy(),
        "child_leaf_counts": child_leaf_counts.copy(),
        "parent_leaf_counts": parent_leaf_counts.copy(),
    }

    p_values_for_correction = np.where(np.isfinite(edge_p_values), edge_p_values, 1.0)
    non_finite_p_value_flags = ~np.isfinite(edge_p_values)
    invalid_test_count = int(np.sum(invalid_test_flags))
    non_finite_p_value_count = int(np.sum(non_finite_p_value_flags))

    log_non_finite_child_parent_divergence_audit(
        logger,
        child_ids=child_ids,
        edge_p_values=edge_p_values,
        invalid_test_count=invalid_test_count,
        non_finite_p_value_count=non_finite_p_value_count,
    )

    (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_edge_flags,
        tree_bh_result,
    ) = apply_child_parent_divergence_tree_bh_correction(
        tree=tree,
        p_values_for_correction=p_values_for_correction,
        child_ids=child_ids,
        edge_alpha=edge_alpha,
    )

    child_parent_edge_null_rejected_by_tree_bh = np.where(
        non_finite_p_value_flags,
        False,
        child_parent_edge_null_rejected_by_tree_bh,
    )

    annotations_df.attrs["child_parent_divergence_audit"] = build_child_parent_divergence_audit(
        total_tests=len(child_ids),
        invalid_test_count=invalid_test_count,
        non_finite_p_value_count=non_finite_p_value_count,
        child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_edge_flags=ancestor_blocked_edge_flags,
    )

    attach_child_parent_stopping_edge_recovery_metadata(
        annotations_df,
        tree=tree,
        child_ids=child_ids,
        child_parent_edge_null_rejected_by_tree_bh=child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh=child_parent_edge_corrected_p_values_by_tree_bh,
        tree_bh_result=tree_bh_result,
        ancestor_blocked_edge_flags=ancestor_blocked_edge_flags,
    )

    return assign_divergence_results(
        annotations_df=annotations_df,
        child_ids=child_ids,
        p_values=edge_p_values,
        p_values_corrected=child_parent_edge_corrected_p_values_by_tree_bh,
        reject_null=child_parent_edge_null_rejected_by_tree_bh,
        degrees_of_freedom=edge_degrees_of_freedom,
        invalid_test_flags=invalid_test_flags,
        tested_edge_flags=child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_edge_flags=ancestor_blocked_edge_flags,
    )


__all__ = ["annotate_child_parent_divergence"]
