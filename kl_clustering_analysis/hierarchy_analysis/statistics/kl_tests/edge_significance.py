from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
from ..tree_bh_experimental import tree_bh_correction as _family_tree_bh
from .utils import get_local_kl_series


def _compute_node_depths(tree: nx.DiGraph) -> Dict[str, int]:
    """Compute depth of each node from the root.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy

    Returns
    -------
    Dict[str, int]
        Mapping from node_id to depth (root = 0)
    """
    # Find root (node with no parents)
    roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    if not roots:
        raise ValueError("Tree has no root node (all nodes have parents)")

    depths: Dict[str, int] = {}
    for root in roots:
        depths[root] = 0

    # BFS to compute depths
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        for child in tree.successors(node):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)

    return depths


def _level_wise_bh_correction(
    p_values: np.ndarray,
    child_depths: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply BH correction separately at each tree level (level-wise BH).

    This is a simpler variant that applies standard BH at each level
    independently, providing level-specific FDR control.

    Parameters
    ----------
    p_values
        Array of p-values for each edge
    child_depths
        Array of depths for each child node
    alpha
        Significance level for BH correction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input
    """
    n = len(p_values)
    reject_null = np.zeros(n, dtype=bool)
    adjusted_p = np.ones(n, dtype=float)

    if n == 0:
        return reject_null, adjusted_p

    # Group edges by depth level
    levels = sorted(set(child_depths))
    level_indices: Dict[int, List[int]] = defaultdict(list)
    for i, depth in enumerate(child_depths):
        level_indices[depth].append(i)

    for level in levels:
        indices = level_indices[level]
        if not indices:
            continue

        # Extract p-values for this level
        level_p_values = p_values[indices]

        # Apply standard BH at this level
        if len(level_p_values) > 0:
            level_reject, level_adjusted, _ = benjamini_hochberg_correction(
                level_p_values, alpha=alpha
            )

            # Store results
            for j, idx in enumerate(indices):
                reject_null[idx] = level_reject[j]
                adjusted_p[idx] = level_adjusted[j]

    return reject_null, adjusted_p


def _flat_bh_correction(
    p_values: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply standard flat BH correction across all p-values.

    Parameters
    ----------
    p_values
        Array of p-values for each edge
    alpha
        Significance level for BH correction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input
    """
    n = len(p_values)
    if n == 0:
        return np.zeros(0, dtype=bool), np.ones(0, dtype=float)

    reject_null, adjusted_p, _ = benjamini_hochberg_correction(p_values, alpha=alpha)
    return reject_null, adjusted_p


def _tree_bh_correction(
    p_values: np.ndarray,
    child_ids: List[str],
    child_depths: np.ndarray,
    alpha: float,
    method: str = "flat",
    tree: Optional[nx.DiGraph] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply BH correction with optional hierarchical structure awareness.

    Parameters
    ----------
    p_values
        Array of p-values for each edge
    child_ids
        List of child node identifiers
    child_depths
        Array of depths for each child node
    alpha
        Base significance level
    method
        Correction method:
        - "flat" (default): Standard BH across all edges
        - "level_wise": BH applied separately at each tree level
        - "tree_bh": Family-wise BH with ancestor-adjusted thresholds
                      (Bogomolov et al. 2021)
    tree
        Required for method="tree_bh". The tree structure.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input
    """
    if method == "tree_bh":
        if tree is None:
            raise ValueError("tree parameter required for method='tree_bh'")
        result = _family_tree_bh(tree, p_values, child_ids, alpha=alpha)
        return result.reject, result.adjusted_p
    elif method == "level_wise":
        return _level_wise_bh_correction(p_values, child_depths, alpha)
    else:
        # Default: flat BH across all edges
        return _flat_bh_correction(p_values, alpha)


def _add_default_significance_columns(_: pd.DataFrame) -> pd.DataFrame:
    """Prevent fallback defaults when edge-level statistics cannot be computed."""
    raise ValueError(
        "Child-parent divergence annotations cannot be computed for a leaf-only tree."
    )


def _extract_child_node_data(
    nodes_dataframe: pd.DataFrame, child_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract leaf counts and local KL values for child nodes.

    Parameters
    ----------
    nodes_dataframe
        DataFrame with node statistics
    child_ids
        List of child node identifiers

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (leaf_counts, local_kl_values) arrays aligned to child_ids
    """
    if "leaf_count" not in nodes_dataframe.columns:
        raise KeyError("Missing required column 'leaf_count' in nodes dataframe.")
    child_leaf_counts = nodes_dataframe["leaf_count"].reindex(child_ids).to_numpy()
    if np.isnan(child_leaf_counts).any():
        missing = [child_ids[i] for i, v in enumerate(child_leaf_counts) if np.isnan(v)]
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(f"Missing leaf_count values for child nodes: {preview}.")

    child_local_kl = get_local_kl_series(nodes_dataframe).reindex(child_ids).to_numpy()
    if np.isnan(child_local_kl).any():
        missing = [child_ids[i] for i, v in enumerate(child_local_kl) if np.isnan(v)]
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(
            f"Missing kl_divergence_local values for child nodes: {preview}."
        )

    return child_leaf_counts, child_local_kl


def _compute_variance_weighted_df(parent_distribution: np.ndarray) -> float:
    """Compute variance-weighted effective degrees of freedom.

    For Bernoulli features, the weight is 4*θ*(1-θ), which equals
    the variance of a Bernoulli normalized to [0,1] (max at θ=0.5).

    Parameters
    ----------
    parent_distribution
        Parent node's feature means (θ values), shape (n_features,)

    Returns
    -------
    float
        Effective degrees of freedom (sum of variance weights)
    """
    theta = np.asarray(parent_distribution, dtype=np.float64)
    variance_weights = 4.0 * theta * (1.0 - theta)
    return max(1.0, float(np.sum(variance_weights)))


def _compute_chi_square_p_values_per_edge(
    child_leaf_counts: np.ndarray,
    child_local_kl: np.ndarray,
    parent_distributions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute chi-square p-values for local KL divergence tests with per-edge df.

    Tests χ² = 2*N*KL ~ χ²(df_eff) where N is leaf count and df_eff is the
    variance-weighted effective degrees of freedom based on the parent distribution.

    Parameters
    ----------
    child_leaf_counts
        Leaf count for each child node
    child_local_kl
        Local KL(child||parent) for each child node
    parent_distributions
        List of parent distribution arrays (one per edge). Must not contain None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (p_values, degrees_of_freedom) arrays aligned to child nodes
    """
    n_edges = len(child_leaf_counts)
    p_values = np.full(n_edges, np.nan, dtype=float)
    degrees_of_freedom = np.full(n_edges, np.nan, dtype=float)

    valid_nodes = np.isfinite(child_local_kl) & (child_leaf_counts > 0)

    for i in range(n_edges):
        if not valid_nodes[i]:
            continue

        # Compute variance-weighted df from parent distribution
        if parent_distributions[i] is None:
            raise ValueError(
                f"Parent distribution is not available for edge {i}. "
                "Cannot compute variance-weighted degrees of freedom."
            )
        df = _compute_variance_weighted_df(parent_distributions[i])

        degrees_of_freedom[i] = df
        chi_square_statistic = 2.0 * child_leaf_counts[i] * child_local_kl[i]
        p_values[i] = float(chi2.sf(chi_square_statistic, df=df))

    return p_values, degrees_of_freedom


def _assign_test_results(
    nodes_dataframe: pd.DataFrame,
    child_ids: list[str],
    p_values: np.ndarray,
    p_values_corrected: np.ndarray,
    reject_null: np.ndarray,
) -> None:
    """Assign p-values and significance flags to tested nodes in-place.

    Parameters
    ----------
    nodes_dataframe
        DataFrame to update with results
    child_ids
        List of child node identifiers
    p_values
        Uncorrected p-values
    p_values_corrected
        BH-corrected p-values
    reject_null
        Boolean array indicating significance after BH correction
    """
    finite_p_mask = np.isfinite(p_values)
    tested_indices = np.flatnonzero(finite_p_mask)
    tested_node_ids = [child_ids[i] for i in tested_indices]

    if tested_node_ids:
        nodes_dataframe.loc[tested_node_ids, "Child_Parent_Divergence_P_Value"] = (
            p_values[tested_indices]
        )
        if p_values_corrected.size > 0:
            nodes_dataframe.loc[
                tested_node_ids, "Child_Parent_Divergence_P_Value_BH"
            ] = p_values_corrected
            nodes_dataframe.loc[
                tested_node_ids, "Child_Parent_Divergence_Significant"
            ] = reject_null


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    significance_level_alpha: float = 0.05,
    fdr_method: str = "tree_bh",
) -> pd.DataFrame:
    """Annotate each child node with local KL divergence significance test results.

    Tests whether each child's distribution significantly diverges from its parent
    using chi-square test on 2*N*KL(child||parent) ~ χ²(df_eff) where N is leaf count
    and df_eff is the variance-weighted effective degrees of freedom based on the
    parent's distribution: df_eff = Σ 4·θᵢ·(1-θᵢ).

    This weights features by their informativeness - features with θ near 0 or 1
    contribute less to the degrees of freedom than features with θ near 0.5.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy. Nodes should have
        'distribution' attribute containing feature means.
    nodes_statistics_dataframe
        DataFrame indexed by node_id with 'leaf_count' and KL divergence columns
    significance_level_alpha
        FDR threshold for correction
    fdr_method
        FDR correction method:
        - "tree_bh" (default): Family-wise BH with ancestor-adjusted thresholds
                     (Bogomolov et al. 2021). Provides proper hierarchical FDR control.
        - "flat": Standard Benjamini-Hochberg across all edges
        - "level_wise": BH applied separately at each tree level

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with columns:
        - Child_Parent_Divergence_P_Value: raw chi-square p-value
        - Child_Parent_Divergence_P_Value_BH: corrected p-value
        - Child_Parent_Divergence_Significant: boolean, True if rejected after correction
        - Child_Parent_Divergence_df: effective degrees of freedom used

    References
    ----------
    Bogomolov et al. (2021). "Hypotheses on a tree: new error rates and
    testing strategies". Biometrika, 108(3), 575-590.
    """
    nodes_dataframe = nodes_statistics_dataframe.copy()

    alpha = float(significance_level_alpha)

    # Extract all edges (parent -> child)
    edge_list = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in edge_list]
    child_ids = [child_id for _, child_id in edge_list]

    # Tree contains only leaves (no internal nodes) - cannot compute local divergence
    if not child_ids:
        return _add_default_significance_columns(nodes_dataframe)

    # Extract parent distributions for variance-weighted df
    parent_distributions = []
    for parent_id in parent_ids:
        node_data = tree.nodes.get(parent_id, {})
        dist = node_data.get("distribution")
        if dist is not None:
            parent_distributions.append(np.asarray(dist, dtype=np.float64))
        else:
            parent_distributions.append(None)

    # Extract child node data
    child_leaf_counts, child_local_kl = _extract_child_node_data(
        nodes_dataframe, child_ids
    )

    # Compute chi-square p-values with per-edge variance-weighted df
    p_values, degrees_of_freedom = _compute_chi_square_p_values_per_edge(
        child_leaf_counts, child_local_kl, parent_distributions
    )

    if not np.isfinite(p_values).all():
        bad_ids = [child_ids[i] for i, v in enumerate(p_values) if not np.isfinite(v)]
        preview = ", ".join(map(repr, bad_ids[:5]))
        raise ValueError(
            f"Non-finite child-parent divergence p-values for nodes: {preview}."
        )

    # Compute node depths for TreeBH hierarchical correction
    node_depths = _compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    # Apply FDR correction with chosen method
    reject_null, p_values_corrected = _tree_bh_correction(
        p_values, child_ids, child_depths, alpha=alpha, method=fdr_method, tree=tree
    )

    # Initialize columns with default values
    nodes_dataframe["Child_Parent_Divergence_P_Value"] = np.nan
    nodes_dataframe["Child_Parent_Divergence_P_Value_BH"] = np.nan
    nodes_dataframe["Child_Parent_Divergence_Significant"] = False
    nodes_dataframe["Child_Parent_Divergence_df"] = np.nan

    # Assign results to all child nodes (TreeBH returns full arrays)
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = (
        p_values_corrected
    )
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_df"] = degrees_of_freedom

    if nodes_dataframe["Child_Parent_Divergence_Significant"].isna().any():
        raise ValueError(
            "Child_Parent_Divergence_Significant contains missing values after "
            "annotation; aborting."
        )
    with pd.option_context("future.no_silent_downcasting", True):
        nodes_dataframe["Child_Parent_Divergence_Significant"] = nodes_dataframe[
            "Child_Parent_Divergence_Significant"
        ].astype(bool)

    return nodes_dataframe


__all__ = [
    "annotate_child_parent_divergence",
]
