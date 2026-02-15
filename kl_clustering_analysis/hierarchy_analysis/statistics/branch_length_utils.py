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
        branch_length = sanitize_positive_branch_length(
            tree.edges[parent, child]["branch_length"]
        )
        if branch_length is not None:
            branch_lengths.append(branch_length)
    if not branch_lengths:
        return None
    return float(np.mean(branch_lengths))


__all__ = ["sanitize_positive_branch_length", "compute_mean_branch_length"]

