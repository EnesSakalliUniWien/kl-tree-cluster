"""Sibling pair observation extraction from the tree structure."""

from __future__ import annotations

import networkx as nx
import numpy as np

from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_sibling_branch_length_sum,
    extract_branch_length_observation,
)

SiblingPairObservations = tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    float | None,
    float | None,
]


def identify_binary_sibling_children(
    tree: nx.DiGraph,
    parent_node_id: object,
) -> tuple[object, object] | None:
    """Return the sibling-child pair when a parent node is binary, else ``None``."""
    children = list(tree.successors(parent_node_id))
    if len(children) != 2:
        return None
    return children[0], children[1]


def extract_sibling_pair_observations(
    tree: nx.DiGraph,
    parent_node_id: object,
    left_child_id: object,
    right_child_id: object,
) -> SiblingPairObservations:
    """Extract sibling distributions, sample sizes, and optional branch lengths."""
    left_branch = extract_branch_length_observation(
        tree,
        parent_node_id,
        left_child_id,
    )
    right_branch = extract_branch_length_observation(
        tree,
        parent_node_id,
        right_child_id,
    )

    return (
        extract_node_distribution(tree, left_child_id),
        extract_node_distribution(tree, right_child_id),
        extract_node_sample_size(tree, left_child_id),
        extract_node_sample_size(tree, right_child_id),
        left_branch,
        right_branch,
    )

__all__ = [
    "compute_sibling_branch_length_sum",
    "extract_sibling_pair_observations",
    "identify_binary_sibling_children",
    "SiblingPairObservations",
]
