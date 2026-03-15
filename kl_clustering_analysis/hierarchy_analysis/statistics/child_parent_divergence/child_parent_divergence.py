"""Gate 2 orchestration for child-parent divergence testing."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

from ..multiple_testing import apply_multiple_testing_correction
from .child_parent_spectral_decomposition import compute_child_parent_spectral_context
from .child_parent_tree_testing import run_child_parent_tests_across_tree

logger = logging.getLogger(__name__)


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    significance_level_alpha: float = 0.05,
    fdr_method: str = "tree_bh",
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    minimum_projection_dimension: int | None = None,
) -> pd.DataFrame:
    """Test child-parent divergence using the projected Wald pipeline."""
    annotations_df = annotations_df.copy()
    edge_alpha = float(significance_level_alpha)

    tree_edges = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in tree_edges]
    child_ids = [child_id for _, child_id in tree_edges]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")

    child_leaf_counts = extract_leaf_counts(annotations_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(annotations_df, parent_ids)

    node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues = (
        compute_child_parent_spectral_context(
            tree,
            leaf_data,
            spectral_method,
        )
    )

    annotations_df.attrs["_spectral_dims"] = node_spectral_dimensions

    (
        edge_test_statistics,
        edge_degrees_of_freedom,
        edge_p_values,
        invalid_test_mask,
    ) = run_child_parent_tests_across_tree(
        tree=tree,
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_leaf_counts=child_leaf_counts,
        parent_leaf_counts=parent_leaf_counts,
        spectral_dims=node_spectral_dimensions,
        pca_projections=node_pca_projections,
        pca_eigenvalues=node_pca_eigenvalues,
        minimum_projection_dimension=minimum_projection_dimension,
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

    node_depths = compute_node_depths(tree)
    child_depths_for_correction = np.array([node_depths.get(cid, 0) for cid in child_ids])

    p_values_for_correction = np.where(np.isfinite(edge_p_values), edge_p_values, 1.0)
    non_finite_p_value_mask = ~np.isfinite(edge_p_values)
    invalid_test_count = int(np.sum(invalid_test_mask))
    non_finite_p_value_count = int(np.sum(non_finite_p_value_mask))

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

    reject_null_hypothesis, corrected_p_values = apply_multiple_testing_correction(
        p_values=p_values_for_correction,
        child_ids=child_ids,
        child_depths=child_depths_for_correction,
        alpha=edge_alpha,
        method=fdr_method,
        tree=tree,
    )

    reject_null_hypothesis = np.where(non_finite_p_value_mask, False, reject_null_hypothesis)

    annotations_df.attrs["child_parent_divergence_audit"] = {
        "total_tests": int(len(child_ids)),
        "invalid_tests": invalid_test_count,
        "non_finite_p_values": non_finite_p_value_count,
        "conservative_path_tests": non_finite_p_value_count,
    }

    return assign_divergence_results(
        annotations_df=annotations_df,
        child_ids=child_ids,
        p_values=edge_p_values,
        p_values_corrected=corrected_p_values,
        reject_null=reject_null_hypothesis,
        degrees_of_freedom=edge_degrees_of_freedom,
        invalid_mask=invalid_test_mask,
    )


__all__ = ["annotate_child_parent_divergence"]
