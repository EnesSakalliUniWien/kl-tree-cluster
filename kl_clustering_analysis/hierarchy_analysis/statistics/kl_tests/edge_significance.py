"""Edge significance testing for hierarchical clustering.

Tests whether each child's distribution significantly diverges from its parent
using a projected Wald test. Supports both binary (Bernoulli) and categorical
(multinomial) distributions.

Test statistic: T = ||R·z||² ~ χ²(k) where:
- z_i = (θ_child - θ_parent) / √(Var(θ̂)) is the standardized difference
- R is a k × d random projection matrix (k = O(log n))
- k is the projection dimension from the JL lemma

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
Bogomolov et al. (2021). "Hypotheses on a tree: new error rates and
    testing strategies". Biometrika, 108(3), 575-590.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import apply_multiple_testing_correction
from ..random_projection import compute_projection_dimension, generate_projection_matrix
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths


# =============================================================================
# Core Statistical Functions
# =============================================================================


def _compute_standardized_z(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
) -> np.ndarray:
    """Compute standardized z-scores for child vs parent.

    Supports both binary (1D) and categorical (2D) distributions.
    Under H₀ (child ~ parent), each z_i ~ N(0, 1).

    Parameters
    ----------
    child_dist
        Child node's distribution. Shape (d,) for binary or (d, K) for categorical.
    parent_dist
        Parent node's distribution. Same shape as child_dist.
    n_child
        Sample size (leaf count) for the child node.

    Returns
    -------
    np.ndarray
        Standardized z-scores, flattened to 1D.
    """
    # Variance under null: Var(θ̂) = θ(1-θ)/n (works for both binary and categorical)
    var = parent_dist * (1 - parent_dist) / n_child
    var = np.maximum(var, 1e-10)
    z = (child_dist - parent_dist) / np.sqrt(var)
    
    # Flatten if categorical (2D -> 1D)
    return z.ravel()


def _compute_projected_test(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    seed: int,
) -> tuple[float, int, float]:
    """Compute projected Wald test for one edge.

    Projects z-scores to k dimensions and computes T = ||R·z||² ~ χ²(k).

    Parameters
    ----------
    child_dist
        Child node's distribution.
    parent_dist
        Parent node's distribution.
    n_child
        Sample size for child node.
    seed
        Random seed for projection matrix.

    Returns
    -------
    tuple[float, int, float]
        (test_statistic, degrees_of_freedom, p_value)
    """
    z = _compute_standardized_z(child_dist, parent_dist, n_child)
    d = len(z)

    # Projection dimension from JL lemma: k = O(log n)
    k = compute_projection_dimension(n_child, d)

    # Project to k dimensions
    R = generate_projection_matrix(d, k, seed)
    projected = R @ z

    # Test statistic: sum of squared projected z-scores
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))

    return stat, k, pval


def _compute_p_values_via_projection(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute p-values for all edges via random projection.

    Parameters
    ----------
    tree
        Hierarchy with 'distribution' attribute on nodes.
    child_ids
        List of child node IDs.
    parent_ids
        List of parent node IDs (aligned with child_ids).
    child_leaf_counts
        Sample sizes for child nodes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (test_statistics, degrees_of_freedom, p_values)
    """
    n_edges = len(child_ids)
    stats = np.full(n_edges, np.nan)
    dfs = np.full(n_edges, np.nan)
    pvals = np.full(n_edges, np.nan)

    seed = config.PROJECTION_RANDOM_SEED

    for i in range(n_edges):
        child_dist = tree.nodes[child_ids[i]].get("distribution")
        parent_dist = tree.nodes[parent_ids[i]].get("distribution")

        if child_dist is None or parent_dist is None or child_leaf_counts[i] < 1:
            stats[i], dfs[i], pvals[i] = 0.0, 0, 1.0
            continue

        stats[i], dfs[i], pvals[i] = _compute_projected_test(
            np.asarray(child_dist, dtype=np.float64),
            np.asarray(parent_dist, dtype=np.float64),
            int(child_leaf_counts[i]),
            seed,
        )

    return stats, dfs, pvals


# =============================================================================
# Public API
# =============================================================================


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    significance_level_alpha: float = 0.05,
    fdr_method: str = "tree_bh",
) -> pd.DataFrame:
    """Test child-parent divergence using projected Wald test.

    Supports both binary (Bernoulli) and categorical (multinomial) distributions.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy. Nodes must have
        'distribution' attribute containing feature parameters.
    nodes_statistics_dataframe
        DataFrame indexed by node_id with 'leaf_count' column.
    significance_level_alpha
        FDR threshold for correction.
    fdr_method
        FDR correction method: "tree_bh", "flat", or "level_wise".

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with divergence test results.
    """
    nodes_dataframe = nodes_statistics_dataframe.copy()
    alpha = float(significance_level_alpha)

    edge_list = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in edge_list]
    child_ids = [child_id for _, child_id in edge_list]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")

    child_leaf_counts = extract_leaf_counts(nodes_dataframe, child_ids)

    test_stats, degrees_of_freedom, p_values = _compute_p_values_via_projection(
        tree, child_ids, parent_ids, child_leaf_counts
    )

    if not np.isfinite(p_values).all():
        bad_ids = [child_ids[i] for i, v in enumerate(p_values) if not np.isfinite(v)]
        preview = ", ".join(map(repr, bad_ids[:5]))
        raise ValueError(
            f"Non-finite child-parent divergence p-values for nodes: {preview}."
        )

    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    reject_null, p_values_corrected = apply_multiple_testing_correction(
        p_values=p_values,
        child_ids=child_ids,
        child_depths=child_depths,
        alpha=alpha,
        method=fdr_method,
        tree=tree,
    )

    return assign_divergence_results(
        nodes_dataframe=nodes_dataframe,
        child_ids=child_ids,
        p_values=p_values,
        p_values_corrected=p_values_corrected,
        reject_null=reject_null,
        degrees_of_freedom=degrees_of_freedom,
    )


__all__ = ["annotate_child_parent_divergence"]
