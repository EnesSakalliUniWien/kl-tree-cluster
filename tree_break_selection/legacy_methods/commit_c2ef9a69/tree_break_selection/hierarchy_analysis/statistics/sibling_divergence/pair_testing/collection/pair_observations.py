"""Sibling pair observation extraction from the tree structure."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
)


def identify_binary_sibling_children(
    tree: nx.DiGraph,
    parent_node_id: str,
) -> tuple[str, str] | None:
    """Return the sibling-child pair when a parent node is binary."""
    children = list(tree.successors(parent_node_id))
    if len(children) != 2:
        return None
    return children[0], children[1]


def extract_sibling_pair_observations(
    tree: nx.DiGraph,
    parent_node_id: str,
    left_child_id: str,
    right_child_id: str,
) -> tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]:
    """Extract sibling distributions, sample sizes, and branch lengths."""
    left_branch = tree.edges[parent_node_id, left_child_id].get("branch_length")
    right_branch = tree.edges[parent_node_id, right_child_id].get("branch_length")

    return (
        extract_node_distribution(tree, left_child_id),
        extract_node_distribution(tree, right_child_id),
        extract_node_sample_size(tree, left_child_id),
        extract_node_sample_size(tree, right_child_id),
        left_branch,
        right_branch,
    )


def compute_sibling_branch_length_sum(
    branch_length_left: float | None,
    branch_length_right: float | None,
) -> float:
    """Return the total sibling branch-length contribution."""
    if branch_length_left is not None and branch_length_right is not None:
        return float(branch_length_left + branch_length_right)
    if branch_length_left is None and branch_length_right is None:
        return 0.0
    raise ValueError(
        f"Inconsistent branch lengths: left={branch_length_left!r}, right={branch_length_right!r}. "
        "Supply either both branch lengths or neither."
    )


__all__ = [
    "compute_sibling_branch_length_sum",
    "extract_sibling_pair_observations",
    "identify_binary_sibling_children",
]
