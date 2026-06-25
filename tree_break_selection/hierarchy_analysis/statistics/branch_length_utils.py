"""Shared utilities for branch-length validation and aggregation.

These helpers define a single, consistent policy:
- missing branch-length attributes are optional observations
- present branch-length attributes must be finite and non-negative
- only strictly positive observations contribute to the tree mean
- positive branch time is used as a dimensionless variance multiplier
"""

from __future__ import annotations

import networkx as nx
import numpy as np

EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE = "none"
EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED = "normalized_branch_length"
EDGE_BRANCH_LENGTH_VARIANCE_POLICIES = (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
)


def validate_edge_branch_length_variance_policy(value: str) -> str:
    """Return the configured edge branch-length variance policy."""
    policy = str(value)
    if policy not in EDGE_BRANCH_LENGTH_VARIANCE_POLICIES:
        raise ValueError(
            "edge_branch_length_variance_policy must be 'none' or "
            f"'normalized_branch_length'; got {value!r}."
        )
    return policy


def validate_branch_length_observation(
    value: object,
    *,
    value_name: str = "branch_length",
) -> float:
    """Return a finite non-negative branch length or raise a clear error."""
    try:
        branch_length = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{value_name} must be a finite non-negative branch length; got {value!r}."
        ) from exc
    if not np.isfinite(branch_length) or branch_length < 0.0:
        raise ValueError(
            f"{value_name} must be a finite non-negative branch length; got {value!r}."
        )
    return branch_length


def extract_branch_length_observation(
    tree: nx.DiGraph,
    parent_id: object,
    child_id: object,
) -> float | None:
    """Return an observed branch length for an edge, or ``None`` when absent."""
    edge_attributes = tree.edges[parent_id, child_id]
    if "branch_length" not in edge_attributes:
        return None
    return validate_branch_length_observation(
        edge_attributes["branch_length"],
        value_name=f"branch_length for edge {parent_id!r}->{child_id!r}",
    )


def compute_mean_branch_length(tree: nx.DiGraph) -> float | None:
    """Compute mean branch length across valid edges.

    Uses only strictly positive observed branch lengths. Zero-length edges are
    valid observations but do not define the positive normalization scale.
    Returns ``None`` when the tree has no positive branch-length observations.
    """
    branch_lengths: list[float] = []
    for parent, child in tree.edges():
        branch_length = extract_branch_length_observation(tree, parent, child)
        if branch_length is not None and branch_length > 0.0:
            branch_lengths.append(branch_length)
    if not branch_lengths:
        return None
    return float(np.mean(branch_lengths))


def compute_sibling_branch_length_sum(
    branch_length_left: float | None,
    branch_length_right: float | None,
) -> float:
    """Return the sibling branch-length sum for complete optional observations."""
    left_branch_length = (
        None
        if branch_length_left is None
        else validate_branch_length_observation(
            branch_length_left,
            value_name="branch_length_left",
        )
    )
    right_branch_length = (
        None
        if branch_length_right is None
        else validate_branch_length_observation(
            branch_length_right,
            value_name="branch_length_right",
        )
    )

    if left_branch_length is None and right_branch_length is None:
        return 0.0
    if left_branch_length is not None and right_branch_length is not None:
        return left_branch_length + right_branch_length
    raise ValueError(
        "Inconsistent branch lengths: "
        f"left={branch_length_left!r}, right={branch_length_right!r}. "
        "Supply either both sibling branch lengths or neither."
    )


__all__ = [
    "EDGE_BRANCH_LENGTH_VARIANCE_POLICIES",
    "EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE",
    "EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED",
    "compute_mean_branch_length",
    "compute_sibling_branch_length_sum",
    "extract_branch_length_observation",
    "validate_edge_branch_length_variance_policy",
    "validate_branch_length_observation",
]
