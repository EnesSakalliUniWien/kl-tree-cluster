"""Tree traversal helpers for spectral dimension estimation.

Utilities for identifying leaves, computing bottom-up descendant indices,
and building per-node data matrices from a hierarchical tree and leaf data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np


def is_leaf(tree: nx.DiGraph, node_id: str) -> bool:
    """Check whether *node_id* is a leaf in *tree*."""
    is_leaf_attr = tree.nodes[node_id].get("is_leaf")
    if is_leaf_attr is not None:
        return bool(is_leaf_attr)
    return tree.out_degree(node_id) == 0


def precompute_descendants(
    tree: nx.DiGraph,
    label_to_idx: Dict[str, int],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Bottom-up precomputation of descendant leaf indices and internal nodes.

    Runs in O(N) total by propagating index lists from leaves to root,
    avoiding O(N²) ``nx.descendants()`` calls.

    Parameters
    ----------
    tree
        Directed hierarchy.
    label_to_idx
        Mapping from leaf labels to row indices in the data matrix.

    Returns
    -------
    desc_indices
        node_id → list of leaf row indices in the data matrix.
    desc_internal
        node_id → list of internal descendant node IDs.
    """
    desc_indices: Dict[str, list] = {}
    desc_internal: Dict[str, list] = {}

    for node_id in reversed(list(nx.topological_sort(tree))):
        if is_leaf(tree, node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_indices[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
            desc_internal[node_id] = []
        else:
            indices: list[int] = []
            internals: list[str] = []
            for child in tree.successors(node_id):
                indices.extend(desc_indices.get(child, []))
                if not is_leaf(tree, child):
                    internals.append(child)
                internals.extend(desc_internal.get(child, []))
            desc_indices[node_id] = indices
            desc_internal[node_id] = internals

    return desc_indices, desc_internal


def build_subtree_data(
    tree: nx.DiGraph,
    X: np.ndarray,
    desc_indices: Dict[str, list],
    desc_internal: Dict[str, list],
    node_id: str,
    d: int,
    include_internal: bool,
) -> Optional[np.ndarray]:
    """Build the data matrix for *node_id*'s subtree.

    Collects descendant leaf rows and optionally appends internal node
    distribution vectors.

    Parameters
    ----------
    tree
        Directed hierarchy.
    X
        Full data matrix (all leaves, n × d).
    desc_indices
        Pre-computed descendant leaf indices (from ``precompute_descendants``).
    desc_internal
        Pre-computed internal descendant node IDs.
    node_id
        The node whose subtree data to build.
    d
        Number of features (columns in X).
    include_internal
        If True, append internal node distribution vectors to the data.

    Returns
    -------
    np.ndarray or None
        Data matrix for this subtree, or *None* when there are fewer than
        2 descendant leaves.
    """
    row_indices = desc_indices[node_id]
    if len(row_indices) < 2:
        return None

    leaf_rows = X[row_indices, :]

    if include_internal:
        internal_rows = []
        for inode in desc_internal[node_id]:
            dist = tree.nodes[inode].get("distribution")
            if dist is not None:
                dist_arr = np.asarray(dist, dtype=np.float64)
                if dist_arr.shape == (d,):
                    internal_rows.append(dist_arr)
        if internal_rows:
            return np.vstack([leaf_rows, np.array(internal_rows)])
    return leaf_rows


__all__ = [
    "is_leaf",
    "precompute_descendants",
    "build_subtree_data",
]
