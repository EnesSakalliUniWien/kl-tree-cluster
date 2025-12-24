from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from ..multiple_testing import apply_multiple_testing_correction
from .utils import get_local_kl_series
from .chi_square_test import kl_divergence_chi_square_test_batch
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
    extract_parent_distributions,
    raise_leaf_only_tree_error,
)
from .global_weighting import (
    compute_global_weight,
    compute_neutral_point,
    extract_global_weighting_config,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import compute_node_depths


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


def _extract_local_kl(
    nodes_dataframe: pd.DataFrame, child_ids: list[str]
) -> np.ndarray:
    """Extract local KL divergence values for child nodes.

    Parameters
    ----------
    nodes_dataframe
        DataFrame with node statistics including KL divergence columns
    child_ids
        List of child node identifiers

    Returns
    -------
    np.ndarray
        Local KL(child||parent) values aligned to child_ids

    Raises
    ------
    ValueError
        If any child nodes have missing local KL values
    """
    child_local_kl = get_local_kl_series(nodes_dataframe).reindex(child_ids).to_numpy()
    if np.isnan(child_local_kl).any():
        missing = [child_ids[i] for i, v in enumerate(child_local_kl) if np.isnan(v)]
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(
            f"Missing kl_divergence_local values for child nodes: {preview}."
        )

    return child_local_kl


def _compute_chi_square_p_values_per_edge(
    child_leaf_counts: np.ndarray,
    child_local_kl: np.ndarray,
    parent_distributions: list[np.ndarray],
    child_global_kl: Optional[np.ndarray] = None,
    global_weight_beta: float = 0.0,
    global_weight_method: str = "relative",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute chi-square p-values for local KL divergence tests with per-edge df.

    Tests χ² = 2*N*KL/w ~ χ²(df_eff) where N is leaf count, w is the global
    weight, and df_eff is the variance-weighted effective degrees of freedom
    based on the parent distribution.

    Global weighting (when enabled):
    - Adjusts test statistic by tree position: deeper nodes require stronger signal
    - Uses relative strength (KL_local/KL_global) to adaptively penalize noise
    - Preserves df (represents feature dimensionality, not depth)

    Parameters
    ----------
    child_leaf_counts
        Leaf count for each child node
    child_local_kl
        Local KL(child||parent) for each child node
    parent_distributions
        List of parent distribution arrays (one per edge). Must not contain None.
    child_global_kl
        Global KL(child||root) for each child node (optional, for global weighting)
    global_weight_beta
        Global weight strength parameter (0 = no weighting)
    global_weight_method
        Method for computing global weight ("fixed" or "relative")

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (p_values, degrees_of_freedom, global_weights) arrays aligned to child nodes
    """
    n_edges = len(child_leaf_counts)
    degrees_of_freedom = np.full(n_edges, np.nan, dtype=float)
    global_weights = np.ones(n_edges, dtype=float)

    valid_nodes = np.isfinite(child_local_kl) & (child_leaf_counts > 0)

    # Check if global weighting is enabled
    use_global_weighting = (
        global_weight_beta > 0
        and child_global_kl is not None
        and len(child_global_kl) == n_edges
    )

    # Compute data-driven neutral point for symmetric weighting
    neutral_point = 0.5  # default
    if use_global_weighting and global_weight_method == "relative":
        neutral_point = compute_neutral_point(
            child_local_kl=child_local_kl,
            child_global_kl=child_global_kl,
            valid_mask=valid_nodes,
        )

    # Compute degrees of freedom and global weights for each edge
    for i in range(n_edges):
        if not valid_nodes[i]:
            continue

        # Compute variance-weighted df from parent distribution
        if parent_distributions[i] is None:
            raise ValueError(
                f"Parent distribution is not available for edge {i}. "
                "Cannot compute variance-weighted degrees of freedom."
            )
        degrees_of_freedom[i] = _compute_variance_weighted_df(parent_distributions[i])

        # Compute global weight if enabled
        if use_global_weighting:
            global_weights[i] = compute_global_weight(
                child_local_kl=child_local_kl[i],
                child_global_kl=child_global_kl[i],
                beta=global_weight_beta,
                method=global_weight_method,
                neutral_point=neutral_point,
            )

    # Use vectorized chi-square test for all edges at once
    _, p_values = kl_divergence_chi_square_test_batch(
        kl_divergences=child_local_kl,
        sample_sizes=child_leaf_counts,
        degrees_of_freedom=degrees_of_freedom,
        weights=global_weights,
    )

    return p_values, degrees_of_freedom, global_weights


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
        return raise_leaf_only_tree_error(nodes_dataframe)

    # Extract parent distributions for variance-weighted df
    parent_distributions = extract_parent_distributions(tree, parent_ids)

    # Extract child node data
    child_leaf_counts = extract_leaf_counts(nodes_dataframe, child_ids)
    child_local_kl = _extract_local_kl(nodes_dataframe, child_ids)

    # Extract global KL if available and global weighting is enabled
    child_global_kl = None
    global_weight_beta = 0.0
    global_weight_method = config.GLOBAL_WEIGHT_METHOD

    if config.USE_GLOBAL_DIVERGENCE_WEIGHTING:
        child_global_kl, global_weight_beta, global_weight_method = (
            extract_global_weighting_config(
                nodes_dataframe=nodes_dataframe,
                child_ids=child_ids,
                child_local_kl=child_local_kl,
            )
        )

    # Compute chi-square p-values with per-edge variance-weighted df and global weights
    p_values, degrees_of_freedom, global_weights = (
        _compute_chi_square_p_values_per_edge(
            child_leaf_counts,
            child_local_kl,
            parent_distributions,
            child_global_kl=child_global_kl,
            global_weight_beta=global_weight_beta,
            global_weight_method=global_weight_method,
        )
    )

    if not np.isfinite(p_values).all():
        bad_ids = [child_ids[i] for i, v in enumerate(p_values) if not np.isfinite(v)]
        preview = ", ".join(map(repr, bad_ids[:5]))
        raise ValueError(
            f"Non-finite child-parent divergence p-values for nodes: {preview}."
        )

    # Compute node depths for TreeBH hierarchical correction
    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    # Apply FDR correction with chosen method
    reject_null, p_values_corrected = apply_multiple_testing_correction(
        p_values=p_values,
        child_ids=child_ids,
        child_depths=child_depths,
        alpha=alpha,
        method=fdr_method,
        tree=tree,
    )

    # Assign results to dataframe
    return assign_divergence_results(
        nodes_dataframe=nodes_dataframe,
        child_ids=child_ids,
        p_values=p_values,
        p_values_corrected=p_values_corrected,
        reject_null=reject_null,
        degrees_of_freedom=degrees_of_freedom,
        global_weights=global_weights,
    )


__all__ = [
    "annotate_child_parent_divergence",
]
