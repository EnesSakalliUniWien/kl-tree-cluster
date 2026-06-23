"""Branch-length computation for merge-based tree construction.

Provides the ultrametric subtraction used when converting a linkage matrix
(or sklearn ``AgglomerativeClustering`` output) into per-edge branch lengths,
as well as a shared ``node_id`` naming helper.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def node_id(idx: int, n_leaves: int) -> str:
    """Map a flat merge index to a PosetTree node-ID string.

    Indices ``0 … n_leaves-1`` become ``L0 … L{n-1}`` (leaves);
    indices ``≥ n_leaves`` become ``N{idx}`` (internal nodes).
    """
    return f"L{idx}" if idx < n_leaves else f"N{idx}"


def compute_ultrametric_branch_lengths(
    n_leaves: int,
    children: np.ndarray,
    distances: Optional[np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """Compute per-edge branch lengths from merge distances.

    For each merge step *k* that joins children *a*, *b* at height
    ``distances[k]``:

        branch_length(parent → child) = distances[k] − merge_height(child)

    Leaf merge-heights are 0.  When *distances* is ``None`` every edge
    receives a default branch length of ``1.0``.

    Parameters
    ----------
    n_leaves
        Number of original leaf nodes.
    children
        ``(n_leaves - 1, 2)`` array of child-index pairs (from scipy/sklearn).
    distances
        Optional ``(n_leaves - 1,)`` array of merge distances.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Mapping ``(parent_id, child_id) → branch_length``.
    """
    edge_lengths: Dict[Tuple[str, str], float] = {}

    if distances is None:
        for k, (a, b) in enumerate(children):
            parent = node_id(n_leaves + k, n_leaves)
            edge_lengths[(parent, node_id(int(a), n_leaves))] = 1.0
            edge_lengths[(parent, node_id(int(b), n_leaves))] = 1.0
        return edge_lengths

    # Ultrametric subtraction: track merge heights as we go.
    merge_heights: Dict[str, float] = {node_id(i, n_leaves): 0.0 for i in range(n_leaves)}

    for k, (a, b) in enumerate(children):
        parent = node_id(n_leaves + k, n_leaves)
        height = float(distances[k])
        merge_heights[parent] = height

        left_id = node_id(int(a), n_leaves)
        right_id = node_id(int(b), n_leaves)
        edge_lengths[(parent, left_id)] = height - merge_heights[left_id]
        edge_lengths[(parent, right_id)] = height - merge_heights[right_id]

    return edge_lengths


__all__ = ["node_id", "compute_ultrametric_branch_lengths"]
