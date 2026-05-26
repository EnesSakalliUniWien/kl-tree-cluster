"""Tree-wide execution helpers for Gate 2 child-parent annotation."""

from __future__ import annotations

import networkx as nx
import numpy as np

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ...branch_length_utils import (
    compute_mean_branch_length,
    extract_branch_length_observation,
)
from ..child_parent_projected_wald.child_parent_projected_wald_test import (
    run_child_parent_projected_wald_test,
)


def run_child_parent_tests_across_tree(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
    parent_leaf_counts: np.ndarray,
    spectral_dims: dict[str, int],
    pca_projections: dict[str, np.ndarray],
    pca_eigenvalues: dict[str, np.ndarray],
    feature_space: FeatureSpace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute projected Wald results for all child-parent edges in the tree."""
    n_edge_tests = len(child_ids)
    test_statistics = np.full(n_edge_tests, np.nan)
    degrees_of_freedom = np.full(n_edge_tests, np.nan)
    p_values = np.full(n_edge_tests, np.nan)
    invalid_test_flags = np.zeros(n_edge_tests, dtype=bool)

    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    for edge_index in range(n_edge_tests):
        child_dist = tree.nodes[child_ids[edge_index]]["distribution"]
        parent_dist = tree.nodes[parent_ids[edge_index]]["distribution"]

        branch_length: float | None = None
        if config.FELSENSTEIN_SCALING and tree.has_edge(
            parent_ids[edge_index], child_ids[edge_index]
        ):
            branch_length = extract_branch_length_observation(
                tree,
                parent_ids[edge_index],
                child_ids[edge_index],
            )

        node_spectral_dimension = spectral_dims[parent_ids[edge_index]]
        node_pca_projection = pca_projections[parent_ids[edge_index]]
        node_pca_eigenvalues = pca_eigenvalues[parent_ids[edge_index]]
        continuous_covariance_by_block = require_node_continuous_covariance_by_block(
            tree,
            parent_ids[edge_index],
            feature_space,
        )

        (
            edge_test_statistic,
            edge_degrees_of_freedom,
            edge_p_value,
            edge_test_invalid,
        ) = run_child_parent_projected_wald_test(
            np.asarray(child_dist, dtype=np.float64),
            np.asarray(parent_dist, dtype=np.float64),
            int(child_leaf_counts[edge_index]),
            int(parent_leaf_counts[edge_index]),
            branch_length,
            mean_branch_length,
            spectral_k=node_spectral_dimension,
            pca_projection=node_pca_projection,
            pca_eigenvalues=node_pca_eigenvalues,
            feature_space=feature_space,
            continuous_covariance_by_block=continuous_covariance_by_block,
        )

        test_statistics[edge_index], degrees_of_freedom[edge_index], p_values[edge_index] = (
            edge_test_statistic,
            edge_degrees_of_freedom,
            edge_p_value,
        )

        invalid_test_flags[edge_index] = bool(edge_test_invalid)

    return test_statistics, degrees_of_freedom, p_values, invalid_test_flags


__all__ = ["run_child_parent_tests_across_tree"]
