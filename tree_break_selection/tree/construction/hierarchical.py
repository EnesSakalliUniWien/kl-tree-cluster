"""Construct :class:`PosetTree` instances from hierarchical merge output."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from tree_break_selection.tree.branch_lengths import compute_ultrametric_branch_lengths, node_id
from tree_break_selection.tree.poset_tree import PosetTree

# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------


def _tree_from_merges(
    n_leaves: int,
    leaf_names: List[str],
    children: np.ndarray,
    distances: Optional[np.ndarray],
) -> "PosetTree":
    """Populate a :class:`PosetTree` from merge arrays and optional distances.

    Parameters
    ----------
    n_leaves
        Number of original samples.
    leaf_names
        Labels for the leaf nodes (length ``n_leaves``).
    children
        ``(n_leaves - 1, 2)`` array of child index pairs produced by scipy or
        sklearn.
    distances
        ``(n_leaves - 1,)`` array of merge distances. Branch lengths are computed
        via ultrametric subtraction (see
        :func:`~tree_break_selection.tree.branch_lengths.compute_ultrametric_branch_lengths`).
    Returns
    -------
    PosetTree
    """
    tree = PosetTree()

    edge_lengths = compute_ultrametric_branch_lengths(n_leaves, children, distances)

    # Add leaf nodes.
    for i, name in enumerate(leaf_names):
        tree.add_node(node_id(i, n_leaves), is_leaf=True, label=name)

    # Add internal merges.
    for k, (a, b) in enumerate(children):
        nid = node_id(n_leaves + k, n_leaves)
        left_id = node_id(int(a), n_leaves)
        right_id = node_id(int(b), n_leaves)

        tree.add_node(nid, is_leaf=False, label=nid)
        left_length = edge_lengths[(nid, left_id)]
        right_length = edge_lengths[(nid, right_id)]
        tree.add_edge(nid, left_id, branch_length=left_length)
        tree.add_edge(nid, right_id, branch_length=right_length)

    # Discover & cache root.
    roots = [u for u, degree in tree.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, got {roots}")
    tree.graph["root"] = roots[0]

    return tree


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def tree_from_linkage(
    linkage_matrix: np.ndarray,
    leaf_names: Optional[List[str]] = None,
) -> "PosetTree":
    """Build a :class:`PosetTree` from a SciPy linkage matrix.

    Parameters
    ----------
    linkage_matrix
        A ``(n-1, 4)`` NumPy array from :func:`scipy.cluster.hierarchy.linkage`.
    leaf_names
        Optional list of leaf labels; defaults to ``leaf_0 … leaf_{n-1}``.

    Returns
    -------
    PosetTree
    """
    n_leaves = linkage_matrix.shape[0] + 1
    if leaf_names is None:
        leaf_names = [f"leaf_{i}" for i in range(n_leaves)]

    children = linkage_matrix[:, :2].astype(int)
    distances = linkage_matrix[:, 2]

    return _tree_from_merges(n_leaves, leaf_names, children, distances)
