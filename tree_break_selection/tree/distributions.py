"""
Distribution population for tree nodes.

This module handles bottom-up propagation of typed raw feature coordinates
from leaf nodes to internal nodes. Internal node distributions are empirical
subtree barycenters: leaf-count-weighted means of their child distributions.
"""

import os
from typing import Any, Dict, Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    resolve_feature_space,
    validate_feature_matrix,
)

CONTINUOUS_COVARIANCE_BY_BLOCK = "continuous_covariance_by_block"
ContinuousCovariancePolicy = Literal["parent_total", "guarded_within_child"]
DEFAULT_CONTINUOUS_COVARIANCE_POLICY: ContinuousCovariancePolicy = "parent_total"
GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY: ContinuousCovariancePolicy = (
    "guarded_within_child"
)
DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT = 8
_CONTINUOUS_SCATTER_BY_BLOCK = "_continuous_scatter_by_block"
MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION = 4096
MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_BYTES = 512 * 1024 * 1024
MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION_ENV = (
    "TBS_MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION"
)
MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV = "TBS_MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB"


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return int(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.")
    return value


def _continuous_covariance_limits() -> tuple[int, int]:
    """Return active exact dense continuous covariance limits."""
    max_block_dimension = _positive_int_from_env(
        MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION_ENV,
        MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION,
    )
    max_work_mib = _positive_int_from_env(
        MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV,
        MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_BYTES // (1024 * 1024),
    )
    return max_block_dimension, max_work_mib * 1024 * 1024


def validate_continuous_covariance_policy(
    policy: str,
) -> ContinuousCovariancePolicy:
    """Validate and normalize the continuous covariance policy name."""
    value = str(policy)
    if value not in {
        DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
        GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY,
    }:
        raise ValueError(
            "continuous_covariance_policy must be 'parent_total' or "
            f"'guarded_within_child'; got {policy!r}."
        )
    return value  # type: ignore[return-value]


def validate_continuous_covariance_min_child_leaf_count(value: int) -> int:
    """Validate the child-mass guard used by guarded continuous covariance."""
    count = int(value)
    if count <= 0:
        raise ValueError(
            f"continuous_covariance_min_child_leaf_count must be positive; got {value!r}."
        )
    return count


def _use_dense_continuous_covariance(
    tree: nx.DiGraph,
    feature_space: FeatureSpace,
) -> bool:
    """Return whether full dense continuous covariance storage is feasible."""
    if not feature_space.has_continuous_blocks:
        return False

    node_count = int(tree.number_of_nodes())
    max_block_dimension, max_work_bytes = _continuous_covariance_limits()
    for block in feature_space.continuous_blocks:
        block_dimension = int(block.raw_dimension)
        if block_dimension > max_block_dimension:
            return False

        estimated_work_bytes = node_count * block_dimension * block_dimension * 8
        if estimated_work_bytes > max_work_bytes:
            return False
    return True


def _calculate_leaf_distribution(
    tree: nx.DiGraph,
    node_id: object,
    leaf_matrix: npt.NDArray[np.float64],
    label_to_row_idx: Dict[Any, int],
    feature_space: FeatureSpace,
    use_dense_continuous_covariance: bool,
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

    feature_probabilities = leaf_matrix[row_idx]
    tree.nodes[node_id]["distribution"] = feature_probabilities
    tree.nodes[node_id]["leaf_count"] = 1
    if feature_space.has_continuous_blocks:
        tree.nodes[node_id][_CONTINUOUS_SCATTER_BY_BLOCK] = {
            block.name: (
                np.zeros((block.raw_dimension, block.raw_dimension), dtype=np.float64)
                if use_dense_continuous_covariance
                else np.zeros(block.raw_dimension, dtype=np.float64)
            )
            for block in feature_space.continuous_blocks
        }


def _calculate_hierarchy_node_distribution(
    tree: nx.DiGraph,
    node_id: object,
    feature_space: FeatureSpace,
    use_dense_continuous_covariance: bool,
) -> None:
    """Compute an internal node's distribution from its immediate children.

    If each child already stores ``distribution`` and ``leaf_count``, then the
    parent distribution is the empirical subtree barycenter

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
        tree.nodes[node_id][_CONTINUOUS_SCATTER_BY_BLOCK] = _calculate_hierarchy_continuous_scatter(
            tree,
            children,
            parent_distribution,
            feature_space,
            use_dense_continuous_covariance,
        )


def _calculate_hierarchy_continuous_scatter(
    tree: nx.DiGraph,
    children: list[object],
    parent_distribution: npt.NDArray[np.float64],
    feature_space: FeatureSpace,
    use_dense_continuous_covariance: bool,
) -> dict[str, npt.NDArray[np.float64]]:
    """Combine child scatter state around the parent mean."""
    scatter_by_block: dict[str, npt.NDArray[np.float64]] = {}
    for block in feature_space.continuous_blocks:
        column_indices = list(block.column_indices)
        parent_block_mean = parent_distribution[column_indices]
        block_scatter = (
            np.zeros((block.raw_dimension, block.raw_dimension), dtype=np.float64)
            if use_dense_continuous_covariance
            else np.zeros(block.raw_dimension, dtype=np.float64)
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
            if use_dense_continuous_covariance:
                block_scatter += child_scatter + child_leaf_count * np.outer(
                    mean_delta,
                    mean_delta,
                )
            else:
                block_scatter += child_scatter + child_leaf_count * (mean_delta * mean_delta)
        scatter_by_block[block.name] = block_scatter
    return scatter_by_block


def _finalize_continuous_covariances(
    tree: nx.DiGraph,
    feature_space: FeatureSpace,
) -> None:
    """Write empirical covariance blocks and remove scatter work state.

    Dense-feasible regimes store full covariance matrices. High-dimensional
    regimes store diagonal variance vectors, which keeps the continuous Gaussian
    model usable without allocating one dense ``p x p`` matrix per tree node.
    """
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
    use_dense_continuous_covariance = _use_dense_continuous_covariance(
        tree,
        active_feature_space,
    )

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
                use_dense_continuous_covariance,
            )
        else:
            _calculate_hierarchy_node_distribution(
                tree,
                node_id,
                active_feature_space,
                use_dense_continuous_covariance,
            )

    _finalize_continuous_covariances(
        tree,
        active_feature_space,
    )


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


def resolve_node_continuous_covariance_by_block(
    tree: nx.DiGraph,
    node_id: object,
    feature_space: FeatureSpace | None,
    *,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
) -> dict[str, npt.NDArray[np.float64]] | None:
    """Resolve the continuous covariance blocks used by node-level Wald tests.

    ``parent_total`` preserves the historical empirical subtree covariance. The
    guarded within-child candidate uses the immediate children's residual
    covariance only when both child subtrees have enough leaves; otherwise it
    falls back to parent-total covariance.
    """
    if feature_space is None or not feature_space.has_continuous_blocks:
        return None

    policy = validate_continuous_covariance_policy(continuous_covariance_policy)
    min_child_leaf_count = validate_continuous_covariance_min_child_leaf_count(
        continuous_covariance_min_child_leaf_count
    )
    parent_covariance_by_block = require_node_continuous_covariance_by_block(
        tree,
        node_id,
        feature_space,
    )
    if policy == DEFAULT_CONTINUOUS_COVARIANCE_POLICY:
        return parent_covariance_by_block

    children = list(tree.successors(node_id))
    if len(children) != 2:
        return parent_covariance_by_block

    child_leaf_counts = [int(tree.nodes[child_id]["leaf_count"]) for child_id in children]
    if any(count < min_child_leaf_count for count in child_leaf_counts):
        return parent_covariance_by_block

    residual_denominator = sum(count - 1 for count in child_leaf_counts)
    if residual_denominator <= 0:
        return parent_covariance_by_block

    child_covariances_by_block = [
        require_node_continuous_covariance_by_block(
            tree,
            child_id,
            feature_space,
        )
        for child_id in children
    ]

    within_covariance_by_block: dict[str, npt.NDArray[np.float64]] = {}
    for block in feature_space.continuous_blocks:
        weighted_covariance_sum = np.zeros_like(
            np.asarray(parent_covariance_by_block[block.name], dtype=np.float64),
            dtype=np.float64,
        )
        for child_leaf_count, child_covariance_by_block in zip(
            child_leaf_counts,
            child_covariances_by_block,
            strict=True,
        ):
            weighted_covariance_sum += float(child_leaf_count - 1) * np.asarray(
                child_covariance_by_block[block.name], dtype=np.float64
            )
        within_covariance_by_block[block.name] = weighted_covariance_sum / float(
            residual_denominator
        )

    return within_covariance_by_block


__all__ = [
    "CONTINUOUS_COVARIANCE_BY_BLOCK",
    "DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT",
    "DEFAULT_CONTINUOUS_COVARIANCE_POLICY",
    "GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY",
    "MAX_EXACT_CONTINUOUS_COVARIANCE_BLOCK_DIMENSION_ENV",
    "MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV",
    "populate_distributions",
    "require_node_continuous_covariance_by_block",
    "resolve_node_continuous_covariance_by_block",
    "validate_continuous_covariance_min_child_leaf_count",
    "validate_continuous_covariance_policy",
]
