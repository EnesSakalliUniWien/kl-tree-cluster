"""
KL divergence and mutual information analysis for hierarchical clustering.

Functions for computing mutual information matrices between feature patterns
across nodes in hierarchical tree structures.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import mutual_info_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tree.poset_tree import PosetTree


def calculate_kl_divergence_mutual_information_matrix(
    tree: PosetTree,
    node_kl_stats_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate mutual information matrix between binary feature patterns across nodes.

    This function computes the mutual information between nodes based on their
    binary feature patterns (thresholded at 0.5 from node distributions).

    Mutual information measures how much knowing one node's binary pattern tells you
    about another node's binary pattern. High MI means nodes have interdependent
    binary feature patterns.

    Process:
    1. Extract binary distributions from internal tree nodes in stats DataFrame.
    2. Threshold distributions to binary (0/1) at 0.5.
    3. Compute pairwise mutual information between all node pairs.

    Args:
        tree: PosetTree instance representing the hierarchy.
        node_kl_stats_df: DataFrame from `calculate_hierarchy_kl_divergence()`.
            Must contain 'distribution' and 'is_leaf' columns.

    Returns:
        Tuple of (mi_matrix, binary_patterns_df):
            - mi_matrix: DataFrame with pairwise mutual information between nodes.
            - binary_patterns_df: DataFrame with binary patterns for each node.
    """
    # Filter for internal nodes from the DataFrame
    internal_nodes_df = node_kl_stats_df[~node_kl_stats_df["is_leaf"]]
    internal_node_names = internal_nodes_df.index.tolist()

    if len(internal_node_names) == 0:
        # Return empty DataFrames if no internal nodes
        empty_mi = pd.DataFrame()
        empty_patterns = pd.DataFrame()
        return empty_mi, empty_patterns

    # Extract binary patterns from the stats DataFrame
    binary_patterns = {}
    for node_name in internal_node_names:
        if node_name in node_kl_stats_df.index:
            distribution = node_kl_stats_df.loc[node_name, "distribution"]
            if distribution is not None and hasattr(distribution, "__len__"):
                # Convert to numpy array if it's a list
                if isinstance(distribution, list):
                    distribution = np.array(distribution)
                binary_pattern = (distribution >= 0.5).astype(int)
                binary_patterns[node_name] = binary_pattern

    if len(binary_patterns) == 0:
        # Return empty DataFrames if no valid binary patterns
        empty_mi = pd.DataFrame(index=internal_node_names, columns=internal_node_names)
        empty_patterns = pd.DataFrame(index=internal_node_names)
        return empty_mi, empty_patterns

    # Create DataFrame where each row is a node and columns are binary features
    binary_patterns_df = pd.DataFrame(binary_patterns).T

    # Calculate mutual information matrix between nodes
    nodes = binary_patterns_df.index.tolist()
    mi_matrix = pd.DataFrame(
        np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes
    )

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                # Self-MI is 1.0 (perfect dependence with itself)
                mi_matrix.loc[node_i, node_j] = 1.0
            else:
                # Calculate MI between the two binary patterns
                mi = mutual_info_score(
                    binary_patterns_df.loc[node_i].values,
                    binary_patterns_df.loc[node_j].values,
                )
                mi_matrix.loc[node_i, node_j] = mi

    # Normalize MI matrix to [0, 1] for interpretability (excluding diagonal)
    # Use the maximum non-diagonal value for normalization
    non_diag_values = mi_matrix.values[~np.eye(len(nodes), dtype=bool)]
    if len(non_diag_values) > 0 and non_diag_values.max() > 0:
        max_mi = non_diag_values.max()
        mi_matrix = mi_matrix / max_mi  # Normalize to [0, 1]
        # Ensure diagonal is exactly 1.0
        for node in nodes:
            mi_matrix.loc[node, node] = 1.0
    else:
        # All MI values are zero or no valid comparisons
        for node in nodes:
            mi_matrix.loc[node, node] = 1.0

    return mi_matrix, binary_patterns_df


def get_highly_similar_nodes(
    mi_matrix: pd.DataFrame,
    threshold: float = 0.8,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Find pairs of nodes with high mutual information (MI).

    Args:
        mi_matrix: MI matrix from calculate_kl_divergence_mutual_information_matrix
        threshold: Minimum MI to include (default: 0.8, normalized scale 0-1)
        exclude_self: Whether to exclude self-MI (default: True)

    Returns:
        DataFrame with columns ['Node_1', 'Node_2', 'Mutual_Information']
        Sorted by MI in descending order

    Example:
        >>> high_mi = get_highly_similar_nodes(mi_matrix, threshold=0.9)
        >>> print(high_mi)
          Node_1 Node_2  Mutual_Information
        0     N3     N4                0.95
        1     N2     N5                0.92
    """
    pairs = []
    nodes = mi_matrix.index.tolist()

    for i, node_1 in enumerate(nodes):
        for j, node_2 in enumerate(nodes):
            # Skip if excluding self-correlation
            if exclude_self and i == j:
                continue

            # Skip duplicate pairs (only keep upper triangle)
            if i >= j:
                continue

            mi_value = mi_matrix.loc[node_1, node_2]

            if mi_value >= threshold:
                pairs.append(
                    {"Node_1": node_1, "Node_2": node_2, "Mutual_Information": mi_value}
                )

    result_df = pd.DataFrame(pairs)

    if len(result_df) > 0:
        result_df = result_df.sort_values("Mutual_Information", ascending=False)

    return result_df


def get_node_mutual_information_summary(
    mi_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate summary statistics for each node's mutual information with other nodes.

    Args:
        mi_matrix: MI matrix from calculate_kl_divergence_mutual_information_matrix

    Returns:
        DataFrame with summary statistics for each node:
            - 'Node': Node name
            - 'Mean_MI': Mean MI with all other nodes
            - 'Max_MI': Maximum MI with any other node
            - 'Min_MI': Minimum MI with any other node
            - 'Std_MI': Standard deviation of MI values

    Example:
        >>> summary = get_node_mutual_information_summary(mi_matrix)
        >>> print(summary)
          Node  Mean_MI  Max_MI  Min_MI   Std_MI
        0   N3     0.75    0.95    0.45     0.18
        1   N4     0.82    0.95    0.60     0.12
    """
    summary_data = []

    for node in mi_matrix.index:
        # Get MI with all other nodes (exclude self)
        other_mi = mi_matrix.loc[node, mi_matrix.columns != node]

        summary_data.append(
            {
                "Node": node,
                "Mean_MI": other_mi.mean(),
                "Max_MI": other_mi.max(),
                "Min_MI": other_mi.min(),
                "Std_MI": other_mi.std(),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Mean_MI", ascending=False)

    return summary_df


# Backward-compatible wrappers (naming used elsewhere/tests)
def calculate_kl_divergence_correlation_matrix(
    tree: PosetTree, node_kl_stats_df: pd.DataFrame
):
    """
    Backward-compatible alias. Returns (mi_matrix, kl_divergence_df).
    """
    return calculate_kl_divergence_mutual_information_matrix(tree, node_kl_stats_df)


def get_highly_correlated_nodes(
    mi_matrix: pd.DataFrame, threshold: float = 0.8, exclude_self: bool = True
) -> pd.DataFrame:
    """
    Backward-compatible alias for get_highly_similar_nodes.
    """
    return get_highly_similar_nodes(
        mi_matrix, threshold=threshold, exclude_self=exclude_self
    )


def get_node_correlation_summary(mi_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias for get_node_mutual_information_summary.
    """
    return get_node_mutual_information_summary(mi_matrix)
