"""Child-parent divergence annotation for Gate 2."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from .child_parent_divergence_tree_bh import (
    apply_child_parent_divergence_tree_bh_correction,
)
from .spectral_context import SpectralContext, compute_child_parent_spectral_context
from .tree_testing import run_child_parent_tests_across_tree


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.EDGE_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    feature_space: FeatureSpace | None = None,
) -> pd.DataFrame:
    """Test child-parent divergence using the projected Wald pipeline.

    Uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR correction.
    This is the only supported multiple-testing correction method for Gate 2.
    """
    annotated_df, _spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        leaf_data=leaf_data,
        feature_space=feature_space,
    )
    return annotated_df


def annotate_child_parent_divergence_with_context(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.EDGE_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    feature_space: FeatureSpace | None = None,
) -> tuple[pd.DataFrame, SpectralContext]:
    """Test child-parent divergence and return typed Gate 2 spectral context."""
    annotations_df = annotations_df.copy()
    edge_alpha = float(significance_level_alpha)

    tree_edges = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in tree_edges]
    child_ids = [child_id for _, child_id in tree_edges]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")
    if leaf_data is None:
        raise ValueError(
            "Child-parent projected Wald tests require leaf_data so Gate 2 can provide "
            "spectral dimensions and PCA bases."
        )

    child_leaf_counts = extract_leaf_counts(annotations_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(annotations_df, parent_ids)

    (
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
    ) = compute_child_parent_spectral_context(
        tree,
        leaf_data,
        feature_space=feature_space,
    )

    spectral_context = SpectralContext(
        spectral_projection_dimensions_by_node=node_spectral_dimensions,
        principal_component_projections_by_node=node_pca_projections,
        principal_component_eigenvalues_by_node=node_pca_eigenvalues,
    )

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
        feature_space=feature_space,
    )

    if invalid_test_flags.any():
        invalid_child_ids = [
            child_ids[edge_index]
            for edge_index, invalid in enumerate(invalid_test_flags)
            if invalid
        ]
        raise ValueError(
            "Child-parent projected Wald tests returned invalid result flags for "
            f"child node(s): {invalid_child_ids[:5]!r}."
        )

    edge_result_arrays = {
        "test statistics": edge_test_statistics,
        "degrees of freedom": edge_degrees_of_freedom,
        "p-values": edge_p_values,
    }
    for result_name, result_values in edge_result_arrays.items():
        if not np.isfinite(result_values).all():
            bad_child_ids = [
                child_ids[edge_index]
                for edge_index, value in enumerate(result_values)
                if not np.isfinite(value)
            ]
            raise ValueError(
                f"Child-parent projected Wald {result_name} must be finite for every edge; "
                f"bad child node(s): {bad_child_ids[:5]!r}."
            )

    (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_edge_flags,
    ) = apply_child_parent_divergence_tree_bh_correction(
        tree=tree,
        p_values_for_correction=edge_p_values,
        child_ids=child_ids,
        edge_alpha=edge_alpha,
    )

    annotated_df = assign_divergence_results(
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
    return annotated_df, spectral_context


__all__ = [
    "annotate_child_parent_divergence",
    "annotate_child_parent_divergence_with_context",
]
