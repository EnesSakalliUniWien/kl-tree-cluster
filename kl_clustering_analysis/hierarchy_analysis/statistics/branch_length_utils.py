"""Shared utilities for branch-length validation and aggregation.

These helpers define a single, consistent policy:
- missing branch-length attributes are optional observations
- present branch-length attributes must be finite and non-negative
- only strictly positive observations contribute to the tree mean
"""

from __future__ import annotations

import networkx as nx
import numpy as np


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


def felsenstein_sibling_multiplier(
    branch_length_sum: float,
    mean_branch_length: float | None,
) -> float:
    """Compute the Felsenstein (1985) variance multiplier for a sibling pair.

    For a sibling contrast with total branch length b_L + b_R:

        multiplier = 1 + (b_L + b_R) / (2 · b̄)

    The factor of 2 accounts for summing two branches; the +1 ensures the
    multiplier is always ≥ 1 (no variance shrinkage).

    Longer total branch length → larger multiplier → larger variance
    → smaller z-scores → harder to declare siblings different.

    Parameters
    ----------
    branch_length_sum : float
        Sum of the two sibling branch lengths (b_L + b_R). Must be > 0.
    mean_branch_length : float | None
        Mean branch length across the tree. Required and must be > 0.

    Returns
    -------
    float
        Variance multiplier ≥ 1.

    Raises
    ------
    ValueError
        If ``branch_length_sum`` or ``mean_branch_length`` is not finite and
        positive, or if ``mean_branch_length`` is ``None``.
    """
    try:
        branch_length_sum_value = float(branch_length_sum)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "branch_length_sum must be finite and positive when computing "
            f"Felsenstein scaling; got {branch_length_sum!r}."
        ) from exc
    if not np.isfinite(branch_length_sum_value) or branch_length_sum_value <= 0:
        raise ValueError(
            "branch_length_sum must be finite and positive when computing "
            f"Felsenstein scaling; got {branch_length_sum!r}."
        )

    try:
        mean_branch_length_value = (
            None if mean_branch_length is None else float(mean_branch_length)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "mean_branch_length is required when branch_length_sum is provided. "
            f"Got mean_branch_length={mean_branch_length!r}, "
            f"branch_length_sum={branch_length_sum!r}. "
            "Ensure the tree has valid branch lengths before running "
            "the Felsenstein-adjusted sibling divergence test."
        ) from exc

    if (
        mean_branch_length_value is None
        or not np.isfinite(mean_branch_length_value)
        or mean_branch_length_value <= 0
    ):
        raise ValueError(
            "mean_branch_length is required when branch_length_sum is provided. "
            f"Got mean_branch_length={mean_branch_length!r}, "
            f"branch_length_sum={branch_length_sum!r}. "
            "Ensure the tree has valid branch lengths before running "
            "the Felsenstein-adjusted sibling divergence test."
        )
    return 1.0 + branch_length_sum_value / (2.0 * mean_branch_length_value)


__all__ = [
    "compute_mean_branch_length",
    "compute_sibling_branch_length_sum",
    "extract_branch_length_observation",
    "felsenstein_sibling_multiplier",
    "validate_branch_length_observation",
]
