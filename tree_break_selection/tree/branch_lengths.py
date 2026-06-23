"""Branch-length computation for merge-based tree construction.

Provides the normalized ultrametric subtraction used when converting a linkage
matrix (or sklearn ``AgglomerativeClustering`` output) into per-edge branch
lengths, as well as a shared ``node_id`` naming helper.
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
    """Compute normalized per-edge branch lengths from merge distances.

    For each merge step *k* that joins children *a*, *b* at height
    ``distances[k]``:

        raw_length(parent -> child) = distances[k] - merge_height(child)

        branch_length(parent -> child) = raw_length(parent -> child) / root_height

    Leaf merge-heights are 0. ``root_height`` is the final merge height. When
    the root height is zero, all normalized branch lengths are zero. With a
    positive root height, every root-to-leaf path sums to 1, so linkage-derived
    branch lengths represent relative change over tree time rather than raw
    distance units.

    Parameters
    ----------
    n_leaves
        Number of original leaf nodes.
    children
        ``(n_leaves - 1, 2)`` array of child-index pairs (from scipy/sklearn).
    distances
        ``(n_leaves - 1,)`` array of merge distances.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Mapping ``(parent_id, child_id) -> normalized_branch_length``.
    """
    edge_lengths: Dict[Tuple[str, str], float] = {}

    if distances is None:
        raise ValueError("merge distances are required to compute branch lengths.")
    distances_array = np.asarray(distances, dtype=float)
    if distances_array.shape != (len(children),):
        raise ValueError(
            "merge distances must have one entry per merge step; "
            f"got distances.shape={distances_array.shape}, children.shape={children.shape}."
        )
    if distances_array.size == 0:
        return edge_lengths
    if not np.all(np.isfinite(distances_array)):
        raise ValueError("merge distances must be finite to compute branch lengths.")
    if np.any(distances_array < 0.0):
        raise ValueError("merge distances must be non-negative to compute branch lengths.")
    if np.any(np.diff(distances_array) < -float(np.finfo(np.float64).eps)):
        raise ValueError("merge distances must be nondecreasing to compute branch lengths.")

    root_height = float(distances_array[-1])
    normalization = root_height if root_height > 0.0 else 1.0

    # Ultrametric subtraction: track merge heights as we go.
    merge_heights: Dict[str, float] = {node_id(i, n_leaves): 0.0 for i in range(n_leaves)}

    for k, (a, b) in enumerate(children):
        parent = node_id(n_leaves + k, n_leaves)
        height = float(distances_array[k])
        merge_heights[parent] = height

        left_id = node_id(int(a), n_leaves)
        right_id = node_id(int(b), n_leaves)
        left_length = height - merge_heights[left_id]
        right_length = height - merge_heights[right_id]
        if left_length < -float(np.finfo(np.float64).eps) or right_length < -float(
            np.finfo(np.float64).eps
        ):
            raise ValueError("merge distances imply a negative branch length.")
        edge_lengths[(parent, left_id)] = max(left_length, 0.0) / normalization
        edge_lengths[(parent, right_id)] = max(right_length, 0.0) / normalization

    return edge_lengths


__all__ = ["node_id", "compute_ultrametric_branch_lengths"]
