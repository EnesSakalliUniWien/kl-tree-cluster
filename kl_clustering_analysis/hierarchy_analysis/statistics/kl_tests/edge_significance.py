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


def _compute_mean_branch_length(tree: nx.DiGraph) -> float:
    """Compute mean branch length across all edges for normalization.

    Used to normalize branch lengths so that the Felsenstein adjustment
    doesn't shrink variance (which would cause over-splitting).

    Returns
    -------
    float
        Mean branch length, or 1.0 if no branch lengths found.
    """
    branch_lengths = [tree.edges[p, c].get("branch_length", 0) for p, c in tree.edges()]
    if not branch_lengths:
        return 1.0
    mean_bl = float(np.mean(branch_lengths))
    return mean_bl if mean_bl > 0 else 1.0


def _compute_standardized_z(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
) -> np.ndarray:
    """Compute standardized z-scores for child vs parent.

    Supports both binary (1D) and categorical (2D) distributions.
    Under H₀ (child ~ parent), each z_i ~ N(0, 1).

    Uses nested variance formula that accounts for child being within parent:
        Var(θ̂_child - θ̂_parent) = θ(1-θ) × (1/n_child - 1/n_parent)

    Applies normalized branch-length scaling following Felsenstein's (1985)
    Phylogenetic Independent Contrasts:
        Var_adjusted = Var × (1 + BL/mean_BL)

    The normalization ensures BL_norm ≥ 1, preventing variance shrinkage
    that would cause over-splitting.

    Parameters
    ----------
    child_dist
        Child node's distribution. Shape (d,) for binary or (d, K) for categorical.
    parent_dist
        Parent node's distribution. Same shape as child_dist.
    n_child
        Sample size (leaf count) for the child node.
    n_parent
        Sample size (leaf count) for the parent node.
    branch_length
        Distance from parent to child in the tree.
    mean_branch_length
        Mean branch length across tree for normalization.

    Returns
    -------
    np.ndarray
        Standardized z-scores, flattened to 1D.
    """
    # Nested variance formula: Var = θ(1-θ) × (1/n_child - 1/n_parent)
    # This accounts for the covariance between child and parent estimates
    # since child data is included in parent's calculation.
    nested_factor = 1.0 / n_child - 1.0 / n_parent

    # Edge case: if nested_factor <= 0 (shouldn't happen in valid tree),
    # fall back to naive formula
    if nested_factor <= 0:
        nested_factor = 1.0 / n_child

    var = parent_dist * (1 - parent_dist) * nested_factor

    # Felsenstein (1985) branch-length adjustment with normalization:
    # BL_norm = 1 + BL/mean_BL ensures multiplier ≥ 1
    # - Longer branches → larger variance → smaller z → harder to split
    # - Shorter branches → less extra variance → easier to split
    if (
        branch_length is not None
        and mean_branch_length is not None
        and mean_branch_length > 0
    ):
        bl_normalized = 1.0 + branch_length / mean_branch_length
        var = var * bl_normalized

    var = np.maximum(var, 1e-10)
    z = (child_dist - parent_dist) / np.sqrt(var)

    # [0.25, 0.75] -  [0.1, 0.9] / 2

    # Flatten if categorical (2D -> 1D)
    return z.ravel()


def _compute_projected_test(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    seed: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
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
    n_parent
        Sample size for parent node.
    seed
        Random seed for projection matrix.
    branch_length
        Distance from parent to child. Used for Felsenstein adjustment.
    mean_branch_length
        Mean branch length across tree for normalization.

    Returns
    -------
    tuple[float, int, float]
        (test_statistic, degrees_of_freedom, p_value)
    """
    z = _compute_standardized_z(
        child_dist, parent_dist, n_child, n_parent, branch_length, mean_branch_length
    )

    # Sanitize z-scores to prevent numerical instability (overflow in projection)
    z = np.nan_to_num(z, posinf=100.0, neginf=-100.0)
    z = np.clip(z, -100.0, 100.0)
    z = z.astype(np.float64)

    d = len(z)

    # Projection dimension from JL lemma: k = O(log n)
    k = compute_projection_dimension(n_child, d)

    # Project to k dimensions
    R = generate_projection_matrix(d, k, seed)

    try:
        if hasattr(R, "dot"):
            projected = R.dot(z)
        else:
            projected = R @ z
    except Exception as e:
        print(
            f"Projection failed (Edge): z.shape={z.shape}, R.shape={R.shape}, z_stats={np.min(z)}/{np.max(z)}"
        )
        raise e

    # Test statistic: sum of squared projected z-scores
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))

    return stat, k, pval


def _compute_p_values_via_projection(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
    parent_leaf_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute p-values for all edges via random projection.

    Extracts branch lengths from tree edges and applies Felsenstein's (1985)
    branch-length adjustment to variance. Uses corrected variance formula
    that accounts for child being nested within parent.

    Parameters
    ----------
    tree
        Hierarchy with 'distribution' attribute on nodes and 'branch_length'
        attribute on edges.
    child_ids
        List of child node IDs.
    parent_ids
        List of parent node IDs (aligned with child_ids).
    child_leaf_counts
        Sample sizes for child nodes.
    parent_leaf_counts
        Sample sizes for parent nodes.

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

    # Compute mean branch length for normalization
    mean_branch_length = _compute_mean_branch_length(tree)

    for i in range(n_edges):
        child_dist = tree.nodes[child_ids[i]].get("distribution")
        parent_dist = tree.nodes[parent_ids[i]].get("distribution")

        if child_dist is None or parent_dist is None or child_leaf_counts[i] < 1:
            stats[i], dfs[i], pvals[i] = 0.0, 0, 1.0
            continue

        # Extract branch length from tree edge (parent → child)
        branch_length = None
        if tree.has_edge(parent_ids[i], child_ids[i]):
            branch_length = tree.edges[parent_ids[i], child_ids[i]].get("branch_length")

        stats[i], dfs[i], pvals[i] = _compute_projected_test(
            np.asarray(child_dist, dtype=np.float64),
            np.asarray(parent_dist, dtype=np.float64),
            int(child_leaf_counts[i]),
            int(parent_leaf_counts[i]),
            seed,
            branch_length,
            mean_branch_length,
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
    parent_leaf_counts = extract_leaf_counts(nodes_dataframe, parent_ids)

    test_stats, degrees_of_freedom, p_values = _compute_p_values_via_projection(
        tree, child_ids, parent_ids, child_leaf_counts, parent_leaf_counts
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
