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
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

from ..multiple_testing import apply_multiple_testing_correction
from .child_parent_spectral_decomposition import compute_child_parent_spectral_context
from .child_parent_tree_testing import run_child_parent_tests_across_tree
from .edge_calibration import (
    deflate_edge_tests,
    fit_edge_inflation_model,
    _identify_null_like_edges,
)

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

    # --- Edge calibration: deflate post-selection inflation before BH ---
    if config.EDGE_CALIBRATION and node_spectral_dimensions is not None:
        null_like_mask = _identify_null_like_edges(
            parent_ids, node_spectral_dimensions, invalid_test_mask,
        )
        edge_cal_model = fit_edge_inflation_model(
            edge_test_statistics,
            edge_degrees_of_freedom,
            parent_leaf_counts,
            null_like_mask,
        )
        edge_test_statistics, edge_p_values = deflate_edge_tests(
            edge_test_statistics,
            edge_degrees_of_freedom,
            edge_p_values,
            parent_leaf_counts,
            edge_cal_model,
            invalid_test_mask,
        )
        annotations_df.attrs["edge_calibration_audit"] = {
            "n_calibration": edge_cal_model.n_calibration,
            "global_inflation_factor": edge_cal_model.global_inflation_factor,
            "max_observed_ratio": edge_cal_model.max_observed_ratio,
            "diagnostics": edge_cal_model.diagnostics,
        }

    node_depths = compute_node_depths(tree)
    child_depths_for_correction = np.array([node_depths.get(cid, 0) for cid in child_ids])

    p_values_for_correction = np.where(np.isfinite(edge_p_values), edge_p_values, 1.0)
    nonfinite_p_value_mask = ~np.isfinite(edge_p_values)
    invalid_test_count = int(np.sum(invalid_test_mask))
    nonfinite_p_value_count = int(np.sum(nonfinite_p_value_mask))
    if invalid_test_count or nonfinite_p_value_count:
        nonfinite_p_value_indices = [
            edge_index
            for edge_index, p_value in enumerate(edge_p_values)
            if not np.isfinite(p_value)
        ]
        nonfinite_p_value_node_ids = [
            child_ids[edge_index] for edge_index in nonfinite_p_value_indices
        ]
        preview_node_ids = ", ".join(map(repr, nonfinite_p_value_node_ids[:5]))
        logger.warning(
            "Child-parent divergence audit: total_tests=%d, invalid_tests=%d, "
            "nonfinite_p_values=%d. Conservative correction path applied "
            "(p=1.0, reject=False) for nodes: %s",
            len(child_ids),
            invalid_test_count,
            nonfinite_p_value_count,
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

    reject_null_hypothesis = np.where(nonfinite_p_value_mask, False, reject_null_hypothesis)

    annotations_df.attrs["child_parent_divergence_audit"] = {
        "total_tests": int(len(child_ids)),
        "invalid_tests": invalid_test_count,
        "nonfinite_p_values": nonfinite_p_value_count,
        "conservative_path_tests": nonfinite_p_value_count,
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
