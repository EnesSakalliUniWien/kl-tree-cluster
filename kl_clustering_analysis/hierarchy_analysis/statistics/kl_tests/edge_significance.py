from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
from .utils import get_local_kl_series


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


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = 0.05,
) -> pd.DataFrame:
    """Annotate each child node with local KL divergence significance test results.

    Tests whether each child's distribution significantly diverges from its parent
    using chi-square test on 2*N*KL(child||parent) ~ χ²(F) where N is leaf count
    and F is number of features. Applies Benjamini-Hochberg correction for multiple testing.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy
    nodes_statistics_dataframe
        DataFrame indexed by node_id with 'leaf_count' and KL divergence columns
    total_number_of_features
        Degrees of freedom for chi-square test
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

    degrees_of_freedom = int(total_number_of_features)
    alpha = float(significance_level_alpha)

    # Extract all child nodes from tree edges
    edge_list = list(tree.edges())
    child_ids = [child_id for _, child_id in edge_list]

    # Tree contains only leaves (no internal nodes) - cannot compute local divergence
    if not child_ids:
        return _add_default_significance_columns(nodes_dataframe)

    # Extract child node data
    child_leaf_counts, child_local_kl = _extract_child_node_data(
        nodes_dataframe, child_ids
    )

    # Compute chi-square p-values
    p_values = _compute_chi_square_p_values(
        child_leaf_counts, child_local_kl, degrees_of_freedom
    )

    # Apply Benjamini-Hochberg FDR correction
    finite_p_mask = np.isfinite(p_values)
    reject_null, p_values_corrected, _ = benjamini_hochberg_correction(
        p_values[finite_p_mask], alpha=alpha
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
