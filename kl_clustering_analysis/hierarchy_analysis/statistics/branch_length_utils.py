"""Shared utilities for branch-length sanitization and aggregation.

These helpers define a single, consistent policy:
- only finite, strictly positive branch lengths are valid for Felsenstein scaling
- missing/non-finite/non-positive values are treated as unavailable
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def sanitize_positive_branch_length(value: object) -> float | None:
    """Return a finite positive branch length, else ``None``."""
    if value is None:
        return None
    try:
        branch_length = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(branch_length) or branch_length <= 0:
        return None
    return branch_length


def compute_mean_branch_length(tree: nx.DiGraph) -> float | None:
    """Compute mean branch length across valid edges.

    Uses only edges that carry a ``branch_length`` attribute and whose values are
    finite and strictly positive. Returns ``None`` when no such values exist.
    """
    branch_lengths: list[float] = []
    for parent, child in tree.edges():
        if "branch_length" not in tree.edges[parent, child]:
            continue
        branch_length = sanitize_positive_branch_length(tree.edges[parent, child]["branch_length"])
        if branch_length is not None:
            branch_lengths.append(branch_length)
    if not branch_lengths:
        return None
    return float(np.mean(branch_lengths))


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
        If ``mean_branch_length`` is ``None`` or ≤ 0.
    """
    if mean_branch_length is None or mean_branch_length <= 0:
        raise ValueError(
            "mean_branch_length is required when branch_length_sum is provided. "
            f"Got mean_branch_length={mean_branch_length!r}, "
            f"branch_length_sum={branch_length_sum!r}. "
            "Ensure the tree has valid branch lengths before running "
            "the Felsenstein-adjusted sibling divergence test."
        )
    return 1.0 + branch_length_sum / (2.0 * mean_branch_length)


__all__ = [
    "sanitize_positive_branch_length",
    "compute_mean_branch_length",
    "felsenstein_sibling_multiplier",
]
