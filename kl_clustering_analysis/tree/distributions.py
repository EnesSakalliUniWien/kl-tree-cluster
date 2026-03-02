"""
Distribution population for tree nodes.

This module handles bottom-up propagation of probability distributions
from leaf nodes to internal nodes, optionally using branch lengths
for harmonic weighting.
"""

from typing import Any, Dict

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd


def _calculate_leaf_distribution(
    tree: nx.DiGraph,
    node_id: str,
    leaf_matrix: npt.NDArray[np.float64],
    label_to_row_idx: Dict[Any, int],
) -> None:
    """Set distribution and leaf count for a leaf node."""
    label = tree.nodes[node_id].get("label", node_id)
    try:
        row_idx = label_to_row_idx[label]
    except KeyError as exc:
        raise KeyError(
            f"Leaf label {label!r} was not found in leaf_data index."
        ) from exc

    feature_probabilities = np.asarray(leaf_matrix[row_idx], dtype=np.float64).reshape(-1)
    tree.nodes[node_id]["distribution"] = feature_probabilities
    tree.nodes[node_id]["leaf_count"] = 1


def _calculate_hierarchy_node_distribution(
    tree: nx.DiGraph,
    node_id: str,
) -> None:
    """Weighted mean of children distributions."""
    children = list(tree.successors(node_id))
    weighted_distribution_sum = 0.0
    total_weight = 0.0
    total_descendant_leaves = 0

    for child_id in children:
        child_leaves = int(tree.nodes[child_id]["leaf_count"])
        child_distribution = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        total_descendant_leaves += child_leaves

        # Original behavior: weight = leaf_count
        weight = child_leaves

        weighted_distribution_sum += child_distribution * weight
        total_weight += weight

    tree.nodes[node_id]["leaf_count"] = total_descendant_leaves
    tree.nodes[node_id]["distribution"] = weighted_distribution_sum / total_weight


def populate_distributions(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
) -> None:
    """
    Populate 'distribution' and 'leaf_count' for all nodes bottom-up.

    Traverses in postorder so children are processed before parents.

    Parameters
    ----------
    tree
        A directed tree (e.g., PosetTree) with 'is_leaf' node attributes.
    leaf_data
        DataFrame where index matches leaf labels and columns are features.
    """
    # Find root
    root = tree.graph.get("root") or next(n for n, d in tree.in_degree() if d == 0)

    # Vectorized extraction of leaf values; avoids per-row Series allocation from iterrows().
    leaf_feature_matrix = leaf_data.to_numpy(dtype=np.float64, copy=False)
    label_to_row_idx = {label: i for i, label in enumerate(leaf_data.index)}

    # Process nodes bottom-up (leaves first, then parents)
    for node_id in nx.dfs_postorder_nodes(tree, source=root):
        is_leaf = tree.nodes[node_id].get("is_leaf", False)

        if is_leaf:
            _calculate_leaf_distribution(tree, node_id, leaf_feature_matrix, label_to_row_idx)
        else:
            _calculate_hierarchy_node_distribution(tree, node_id)
