"""Gate 2 orchestration for child-parent divergence testing."""

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

from ..multiple_testing.tree_bh import ChildParentEdgeTreeBHResult, apply_tree_bh_correction
from .child_parent_spectral_decomposition import (
    _compute_child_parent_spectral_context_with_audit,
)
from .child_parent_tree_testing import run_child_parent_tests_across_tree

logger = logging.getLogger(__name__)


def _apply_edge_multiple_testing_correction(
    tree: nx.DiGraph,
    p_values_for_correction: np.ndarray,
    child_ids: list[str],
    edge_alpha: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ChildParentEdgeTreeBHResult | None,
]:
    """Apply Tree-BH correction to child-parent edge p-values.

    Tree-BH (Tree-structured Benjamini-Hochberg) is the only supported
    Gate 2 multiple-testing correction method.
    """
    tree_bh_result = apply_tree_bh_correction(
        tree,
        p_values_for_correction,
        child_ids,
        alpha=edge_alpha,
    )
    child_parent_edge_null_rejected_by_tree_bh = (
        tree_bh_result.child_parent_edge_null_rejected_by_tree_bh
    )
    child_parent_edge_corrected_p_values_by_tree_bh = (
        tree_bh_result.child_parent_edge_corrected_p_values_by_tree_bh.copy()
    )
    child_parent_edge_tested_by_tree_bh = np.asarray(
        tree_bh_result.child_parent_edge_tested_by_tree_bh,
        dtype=bool,
    )
    ancestor_blocked_mask = ~child_parent_edge_tested_by_tree_bh
    child_parent_edge_corrected_p_values_by_tree_bh = np.where(
        child_parent_edge_tested_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        np.nan,
    )

    return (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_mask,
        tree_bh_result,
    )


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

    if invalid_test_count or non_finite_p_value_count:

        non_finite_p_value_indices = [
            edge_index
            for edge_index, p_value in enumerate(edge_p_values)
            if not np.isfinite(p_value)
        ]

        non_finite_p_value_node_ids = [
            child_ids[edge_index] for edge_index in non_finite_p_value_indices
        ]

        preview_node_ids = ", ".join(map(repr, non_finite_p_value_node_ids[:5]))

        logger.warning(
            "Child-parent divergence audit: total_tests=%d, invalid_tests=%d, "
            "non_finite_p_values=%d. Conservative correction path applied "
            "(p=1.0, reject=False) for nodes: %s",
            len(child_ids),
            invalid_test_count,
            non_finite_p_value_count,
            preview_node_ids,
        )

    (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_edge_flags,
        tree_bh_result,
    ) = _apply_edge_multiple_testing_correction(
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

    annotations_df.attrs["child_parent_divergence_audit"] = {
        "total_tests": int(len(child_ids)),
        "invalid_tests": invalid_test_count,
        "non_finite_p_values": non_finite_p_value_count,
        "conservative_path_tests": non_finite_p_value_count,
        "tested_edges": int(np.sum(child_parent_edge_tested_by_tree_bh)),
        "ancestor_blocked_edges": int(np.sum(ancestor_blocked_edge_flags)),
    }

    # Build stopping-edge recovery metadata for blocked edges
    if int(np.sum(ancestor_blocked_edge_flags)) > 0:
        from ..multiple_testing.stopping_edge_recovery import (
            recover_signal_neighbors,
            recover_stopping_edge_info,
        )
        from ..multiple_testing.stopping_edge_recovery.serialization import (
            STOPPING_EDGE_INFO_ATTR_KEY,
            build_stopping_edge_attrs,
        )

        assert tree_bh_result is not None
        stopping_edge_info_by_child = recover_stopping_edge_info(tree, tree_bh_result, child_ids)

        nearest_signal_neighbor_by_child = recover_signal_neighbors(
            tree,
            child_ids,
            child_parent_edge_null_rejected_by_tree_bh=child_parent_edge_null_rejected_by_tree_bh,
            child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
            child_parent_edge_corrected_p_values_by_tree_bh=child_parent_edge_corrected_p_values_by_tree_bh,
        )
        annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = build_stopping_edge_attrs(
            child_node_ids=child_ids,
            stopping_edge_info_by_child=stopping_edge_info_by_child,
            signal_neighbor_info_by_child=nearest_signal_neighbor_by_child,
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
