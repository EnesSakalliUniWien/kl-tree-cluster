"""Tree-wide execution for child-parent divergence tests."""

from __future__ import annotations

import networkx as nx
import numpy as np

from kl_clustering_analysis import config

from ...decomposition.backends.random_projection_backend import (
    derive_projection_seed_backend as derive_projection_seed,
)
from ..branch_length_utils import compute_mean_branch_length, sanitize_positive_branch_length
from .child_parent_projected_wald import run_child_parent_projected_wald_test


def run_child_parent_tests_across_tree(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
    parent_leaf_counts: np.ndarray,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
    minimum_projection_dimension: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute projected Wald results for all child-parent edges in the tree."""
    n_edge_tests = len(child_ids)
    test_statistics = np.full(n_edge_tests, np.nan)
    degrees_of_freedom = np.full(n_edge_tests, np.nan)
    p_values = np.full(n_edge_tests, np.nan)
    invalid_test_mask = np.zeros(n_edge_tests, dtype=bool)

    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    for edge_index in range(n_edge_tests):
        child_dist = tree.nodes[child_ids[edge_index]].get("distribution")
        parent_dist = tree.nodes[parent_ids[edge_index]].get("distribution")

        if child_dist is None or parent_dist is None or child_leaf_counts[edge_index] < 1:
            test_statistics[edge_index], degrees_of_freedom[edge_index], p_values[edge_index] = (
                0.0,
                0.0,
                1.0,
            )
            continue

        branch_length: float | None = None
        if tree.has_edge(parent_ids[edge_index], child_ids[edge_index]):
            branch_length = sanitize_positive_branch_length(
                tree.edges[parent_ids[edge_index], child_ids[edge_index]].get("branch_length")
            )

        test_seed = derive_projection_seed(
            config.PROJECTION_RANDOM_SEED,
            f"edge:{parent_ids[edge_index]}->{child_ids[edge_index]}",
        )

        node_spectral_dimension: int | None = None
        node_pca_projection: np.ndarray | None = None
        node_pca_eigenvalues: np.ndarray | None = None
        if spectral_dims is not None:
            node_spectral_dimension = spectral_dims.get(parent_ids[edge_index])
        if pca_projections is not None:
            node_pca_projection = pca_projections.get(parent_ids[edge_index])
        if pca_eigenvalues is not None:
            node_pca_eigenvalues = pca_eigenvalues.get(parent_ids[edge_index])

        projected_test_kwargs: dict[str, object] = {
            "spectral_k": node_spectral_dimension,
            "pca_projection": node_pca_projection,
            "pca_eigenvalues": node_pca_eigenvalues,
        }
        if minimum_projection_dimension is not None:
            projected_test_kwargs["minimum_projection_dimension"] = int(minimum_projection_dimension)

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
            test_seed,
            branch_length,
            mean_branch_length,
            **projected_test_kwargs,
        )
        test_statistics[edge_index], degrees_of_freedom[edge_index], p_values[edge_index] = (
            edge_test_statistic,
            edge_degrees_of_freedom,
            edge_p_value,
        )
        invalid_test_mask[edge_index] = bool(edge_test_invalid)

    return test_statistics, degrees_of_freedom, p_values, invalid_test_mask


__all__ = ["run_child_parent_tests_across_tree"]
