"""
Distribution population for tree nodes.

This module handles bottom-up propagation of typed raw feature coordinates
from leaf nodes to internal nodes.
"""

from typing import Any, Dict

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from kl_clustering_analysis.core_utils.tree_utils import bottom_up_nodes
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    resolve_feature_space,
    validate_feature_matrix,
    validate_feature_vector,
)

CONTINUOUS_COVARIANCE_BY_BLOCK = "continuous_covariance_by_block"
_CONTINUOUS_SCATTER_BY_BLOCK = "_continuous_scatter_by_block"


def _calculate_leaf_distribution(
    tree: nx.DiGraph,
    node_id: object,
    leaf_matrix: npt.NDArray[np.float64],
    label_to_row_idx: Dict[Any, int],
    feature_space: FeatureSpace,
) -> None:
    """Set a leaf's distribution to its row in ``leaf_matrix``.

    For Bernoulli data the stored vector is

        distribution = leaf_matrix[row_idx, :]

    The stored distribution is always a flat raw-coordinate vector. The
    feature-space contract supplies the block structure used later by Wald and
    spectral covariance code.

    Args:
        tree: Directed tree containing the leaf node.
        node_id: Leaf node whose distribution should be populated.
        leaf_matrix: Matrix whose rows correspond to leaf labels.
        label_to_row_idx: Mapping from leaf label to row index.

    Raises:
        KeyError: If the leaf label is not present in ``label_to_row_idx``.
    """
    label = tree.nodes[node_id]["label"]
    try:
        row_idx = label_to_row_idx[label]
    except KeyError as exc:
        raise KeyError(f"Leaf label {label!r} was not found in leaf_data index.") from exc

    feature_probabilities = validate_feature_vector(
        leaf_matrix[row_idx],
        feature_space,
    )
    tree.nodes[node_id]["distribution"] = feature_probabilities
    tree.nodes[node_id]["leaf_count"] = 1
    if feature_space.has_continuous_blocks:
        tree.nodes[node_id][_CONTINUOUS_SCATTER_BY_BLOCK] = {
            block.name: np.zeros(
                (block.raw_dimension, block.raw_dimension),
                dtype=np.float64,
            )
            for block in feature_space.continuous_blocks
        }


def _calculate_hierarchy_node_distribution(
    tree: nx.DiGraph,
    node_id: object,
    feature_space: FeatureSpace,
) -> None:
    """Compute an internal node's distribution from its immediate children.

    If each child already stores ``distribution`` and ``leaf_count``, then

        parent_distribution =
            sum(child_leaf_count_i * child_distribution_i) /
            sum(child_leaf_count_i)

    and

        parent_leaf_count = sum(child_leaf_count_i)

    If child entries satisfy the active feature-space constraints, parent
    entries do too because internal distributions are convex combinations of
    children.

    Args:
        tree: Directed tree whose child nodes already have ``distribution`` and
            ``leaf_count`` attributes.
        node_id: Internal node whose distribution should be computed.

    Raises:
        ValueError: If the node has no children or if the combined child weight
            is not positive.
    """
    children = list(tree.successors(node_id))
    if not children:
        raise ValueError(f"Internal node {node_id!r} has no children.")

    weighted_distribution_sum = 0.0
    total_weight = 0.0
    total_descendant_leaves = 0

    for child_id in children:
        child_leaf_count = int(tree.nodes[child_id]["leaf_count"])
        child_distribution = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        total_descendant_leaves += child_leaf_count

        weighted_distribution_sum += child_distribution * child_leaf_count
        total_weight += child_leaf_count

    if total_weight <= 0:
        raise ValueError(
            f"Internal node {node_id!r} has no descendant leaves; malformed tree. "
            f"Children: {children!r}."
        )

    tree.nodes[node_id]["leaf_count"] = total_descendant_leaves
    parent_distribution = weighted_distribution_sum / total_weight
    tree.nodes[node_id]["distribution"] = parent_distribution

    if feature_space.has_continuous_blocks:
        tree.nodes[node_id][_CONTINUOUS_SCATTER_BY_BLOCK] = (
            _calculate_hierarchy_continuous_scatter(
                tree,
                children,
                parent_distribution,
                feature_space,
            )
        )


def _calculate_hierarchy_continuous_scatter(
    tree: nx.DiGraph,
    children: list[object],
    parent_distribution: npt.NDArray[np.float64],
    feature_space: FeatureSpace,
) -> dict[str, npt.NDArray[np.float64]]:
    """Combine child scatter matrices around the parent mean."""
    scatter_by_block: dict[str, npt.NDArray[np.float64]] = {}
    for block in feature_space.continuous_blocks:
        column_indices = list(block.column_indices)
        parent_block_mean = parent_distribution[column_indices]
        block_scatter = np.zeros(
            (block.raw_dimension, block.raw_dimension),
            dtype=np.float64,
        )
        for child_id in children:
            child_leaf_count = int(tree.nodes[child_id]["leaf_count"])
            child_distribution = np.asarray(
                tree.nodes[child_id]["distribution"],
                dtype=np.float64,
            )
            child_block_mean = child_distribution[column_indices]
            child_scatter = tree.nodes[child_id][_CONTINUOUS_SCATTER_BY_BLOCK][block.name]
            mean_delta = child_block_mean - parent_block_mean
            block_scatter += child_scatter + child_leaf_count * np.outer(
                mean_delta,
                mean_delta,
            )
        scatter_by_block[block.name] = block_scatter
    return scatter_by_block


def _finalize_continuous_covariances(
    tree: nx.DiGraph,
    feature_space: FeatureSpace,
) -> None:
    """Write unbiased empirical covariance blocks and remove scatter work state."""
    for node_id in tree.nodes:
        node_attrs = tree.nodes[node_id]
        node_attrs.pop(CONTINUOUS_COVARIANCE_BY_BLOCK, None)
        if not feature_space.has_continuous_blocks:
            node_attrs.pop(_CONTINUOUS_SCATTER_BY_BLOCK, None)
            continue

        node_leaf_count = int(node_attrs["leaf_count"])
        scatter_by_block = node_attrs[_CONTINUOUS_SCATTER_BY_BLOCK]
        covariance_denominator = max(node_leaf_count - 1, 1)
        node_attrs[CONTINUOUS_COVARIANCE_BY_BLOCK] = {
            block.name: np.asarray(scatter_by_block[block.name], dtype=np.float64)
            / float(covariance_denominator)
            for block in feature_space.continuous_blocks
        }
        node_attrs.pop(_CONTINUOUS_SCATTER_BY_BLOCK, None)


def populate_distributions(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    feature_space: FeatureSpace | None = None,
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
    active_feature_space = resolve_feature_space(tuple(leaf_data.columns), feature_space)

    # Vectorized extraction of leaf values; avoids per-row Series allocation from iterrows().
    leaf_feature_matrix = validate_feature_matrix(
        leaf_data.to_numpy(dtype=np.float64, copy=False),
        active_feature_space,
        value_name="leaf_data",
    )
    label_to_row_idx = {label: i for i, label in enumerate(leaf_data.index)}

    # Process nodes bottom-up (leaves first, then parents)
    for node_id in bottom_up_nodes(tree):
        is_leaf = tree.nodes[node_id]["is_leaf"]

        if is_leaf:
            _calculate_leaf_distribution(
                tree,
                node_id,
                leaf_feature_matrix,
                label_to_row_idx,
                active_feature_space,
            )
        else:
            _calculate_hierarchy_node_distribution(tree, node_id, active_feature_space)

    _finalize_continuous_covariances(tree, active_feature_space)


def require_node_continuous_covariance_by_block(
    tree: nx.DiGraph,
    node_id: object,
    feature_space: FeatureSpace | None,
) -> dict[str, npt.NDArray[np.float64]] | None:
    """Return node covariance blocks when continuous features are part of the contract."""
    if feature_space is None or not feature_space.has_continuous_blocks:
        return None
    covariance_by_block = tree.nodes[node_id][CONTINUOUS_COVARIANCE_BY_BLOCK]
    expected_block_names = {block.name for block in feature_space.continuous_blocks}
    actual_block_names = set(covariance_by_block)
    if actual_block_names != expected_block_names:
        raise ValueError(
            f"Node {node_id!r} continuous covariance blocks do not match feature_space. "
            f"expected={sorted(expected_block_names)!r}, actual={sorted(actual_block_names)!r}."
        )
    return covariance_by_block


__all__ = [
    "CONTINUOUS_COVARIANCE_BY_BLOCK",
    "populate_distributions",
    "require_node_continuous_covariance_by_block",
]
