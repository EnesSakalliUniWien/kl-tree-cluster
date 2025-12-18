from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
from .utils import get_local_kl_series
from kl_clustering_analysis import config


def _add_default_significance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add local divergence significance columns with default values.

    Used when tree has no edges (leaf-only tree) and no statistical tests
    can be performed. All p-values set to NaN, significance flags to False.
    """
    df["Local_P_Value_Uncorrected"] = np.nan
    df["Local_P_Value_Corrected"] = np.nan
    df["Local_BH_Significant"] = False
    return df


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
    child_leaf_counts = (
        nodes_dataframe.get(
            "leaf_count", pd.Series(index=nodes_dataframe.index, dtype=float)
        )
        .reindex(child_ids)
        .to_numpy()
    )

    child_local_kl = get_local_kl_series(nodes_dataframe).reindex(child_ids).to_numpy()

    return child_leaf_counts, child_local_kl


def _compute_chi_square_p_values(
    child_leaf_counts: np.ndarray,
    child_local_kl: np.ndarray,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Compute chi-square p-values for local KL divergence tests.

    Tests χ² = 2*N*KL ~ χ²(F) where N is leaf count and F is degrees of freedom.
    Only computes for nodes with finite KL and positive leaf count.

    Parameters
    ----------
    child_leaf_counts
        Leaf count for each child node
    child_local_kl
        Local KL(child||parent) for each child node
    degrees_of_freedom
        Number of features (chi-square test df)

    Returns
    -------
    np.ndarray
        P-values array (NaN for invalid nodes)
    """
    p_values = np.full(len(child_leaf_counts), np.nan, dtype=float)
    valid_nodes = np.isfinite(child_local_kl) & (child_leaf_counts > 0)

    if np.any(valid_nodes):
        chi_square_statistics = (
            2.0 * child_leaf_counts[valid_nodes] * child_local_kl[valid_nodes]
        )
        p_values[valid_nodes] = chi2.sf(chi_square_statistics, df=degrees_of_freedom)

    return p_values


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
        nodes_dataframe.loc[tested_node_ids, "Local_P_Value_Uncorrected"] = p_values[
            tested_indices
        ]
        if p_values_corrected.size > 0:
            nodes_dataframe.loc[tested_node_ids, "Local_P_Value_Corrected"] = (
                p_values_corrected
            )
            nodes_dataframe.loc[tested_node_ids, "Local_BH_Significant"] = reject_null


def _compute_effective_df_for_nodes(
    nodes_dataframe: pd.DataFrame, total_number_of_features: int
) -> pd.Series:
    """Pre-compute variance-weighted effective df for all nodes.

    Parameters
    ----------
    nodes_dataframe
        DataFrame with 'distribution' column
    total_number_of_features
        Fallback df value

    Returns
    -------
    pd.Series
        Effective df for each node, indexed by node_id
    """

    def calc_effective_df(distribution):
        if distribution is None or not hasattr(distribution, "__len__"):
            return float(total_number_of_features)
        parent_theta = np.asarray(distribution, dtype=np.float64)
        variance_weights = 4.0 * parent_theta * (1.0 - parent_theta)
        return float(np.sum(variance_weights))

    return nodes_dataframe["distribution"].apply(calc_effective_df)


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = 0.05,
) -> pd.DataFrame:
    """Annotate each child node with local KL divergence significance test results.

    Tests whether each child's distribution significantly diverges from its parent
    using chi-square test on 2*N*KL(child||parent) ~ χ²(F_eff) where N is leaf count
    and F_eff is variance-weighted effective degrees of freedom. Applies Benjamini-Hochberg
    correction for multiple testing.

    Uses variance-weighted df by default: features are weighted by their Bernoulli variance
    in the parent distribution (w_i = 4·θ_i·(1-θ_i)). This automatically down-weights
    uninformative features (θ near 0 or 1).

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy
    nodes_statistics_dataframe
        DataFrame indexed by node_id with 'leaf_count', 'distribution', and KL divergence columns
    total_number_of_features
        Number of features (used as fallback if parent distribution unavailable)
    significance_level_alpha
        FDR threshold for Benjamini-Hochberg correction

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with columns:
        - Local_P_Value_Uncorrected: raw chi-square p-value
        - Local_P_Value_Corrected: BH-corrected p-value
        - Local_BH_Significant: boolean, True if rejected after BH correction
    """
    nodes_dataframe = nodes_statistics_dataframe.copy()
    alpha = float(significance_level_alpha)

    # Extract all edges
    edge_list = list(tree.edges())

    # Tree contains only leaves (no internal nodes) - cannot compute local divergence
    if not edge_list:
        return _add_default_significance_columns(nodes_dataframe)

    # Pre-compute effective df for all nodes (vectorized, fast)
    effective_df_series = _compute_effective_df_for_nodes(
        nodes_dataframe, total_number_of_features
    )

    # Build arrays aligned to edges
    parent_ids = [parent_id for parent_id, _ in edge_list]
    child_ids = [child_id for _, child_id in edge_list]

    # Extract child data (vectorized)
    child_leaf_counts, child_local_kl = _extract_child_node_data(
        nodes_dataframe, child_ids
    )

    # Get parent df for each edge (vectorized lookup)
    parent_df_values = effective_df_series.reindex(parent_ids).to_numpy()

    # Compute chi-square statistics (vectorized)
    valid_nodes = np.isfinite(child_local_kl) & (child_leaf_counts > 0)
    p_values = np.full(len(child_ids), np.nan, dtype=float)

    if np.any(valid_nodes):
        chi_square_statistics = (
            2.0 * child_leaf_counts[valid_nodes] * child_local_kl[valid_nodes]
        )
        valid_df = parent_df_values[valid_nodes]

        # Vectorized p-value computation with per-edge df
        # Note: chi2.sf can handle array of x values with scalar or array df
        p_values[valid_nodes] = chi2.sf(chi_square_statistics, df=valid_df)

    # Apply Benjamini-Hochberg FDR correction with adaptive α if enabled
    finite_p_mask = np.isfinite(p_values)

    if config.USE_ADAPTIVE_ALPHA and np.any(valid_nodes):
        # Compute adaptive α based on median sample size and df
        median_n = np.median(child_leaf_counts[valid_nodes])
        median_df = np.median(valid_df)
        effective_alpha = config.compute_adaptive_alpha(alpha, median_n, median_df)
    else:
        effective_alpha = alpha

    reject_null, p_values_corrected, _ = benjamini_hochberg_correction(
        p_values[finite_p_mask], alpha=effective_alpha
    )

    # Initialize columns with default values
    nodes_dataframe["Local_P_Value_Uncorrected"] = np.nan
    nodes_dataframe["Local_P_Value_Corrected"] = np.nan
    nodes_dataframe["Local_BH_Significant"] = False

    # Assign results to tested nodes
    _assign_test_results(
        nodes_dataframe, child_ids, p_values, p_values_corrected, reject_null
    )

    # Ensure boolean dtype
    with pd.option_context("future.no_silent_downcasting", True):
        nodes_dataframe["Local_BH_Significant"] = (
            nodes_dataframe["Local_BH_Significant"].fillna(False).astype(bool)
        )

    return nodes_dataframe


__all__ = [
    "annotate_child_parent_divergence",
]
