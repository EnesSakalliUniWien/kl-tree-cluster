"""I/O helpers for constructing :class:`PosetTree` from external representations.

Each public function accepts raw clustering output (linkage matrix, sklearn model,
undirected edge list) and returns a fully-initialised :class:`PosetTree`.

A shared ``_build_tree_from_merges`` helper either computes ultrametric branch
lengths from merge heights or assigns explicit placeholder lengths for
topology-only trees that will be refit by a downstream optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from tree_break_selection.tree.branch_lengths import compute_ultrametric_branch_lengths, node_id

if TYPE_CHECKING:
    from tree_break_selection.tree.poset_tree import PosetTree


def _get_poset_tree_cls() -> type["PosetTree"]:
    """Lazy import to avoid circular dependency with poset_tree.py."""
    from tree_break_selection.tree.poset_tree import PosetTree

    return PosetTree


# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------


def _build_tree_from_merges(
    cls: type["PosetTree"],
    n_leaves: int,
    leaf_names: List[str],
    children: np.ndarray,
    distances: Optional[np.ndarray],
    fallback_branch_length: float | None = None,
) -> "PosetTree":
    """Populate a :class:`PosetTree` from merge arrays and optional distances.

    Parameters
    ----------
    cls
        The concrete ``PosetTree`` class (or subclass) to instantiate.
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
    fallback_branch_length
        Optional non-negative branch length assigned to every edge when only
        merge topology is desired and calibrated branch lengths will be fitted
        later.

    Returns
    -------
    PosetTree
    """
    G = cls()

    # Compute all edge branch lengths up-front unless this is a topology-only
    # construction for a downstream branch-length optimizer.
    if fallback_branch_length is None:
        edge_lengths = compute_ultrametric_branch_lengths(n_leaves, children, distances)
    else:
        fallback_branch_length = float(fallback_branch_length)
        if not np.isfinite(fallback_branch_length) or fallback_branch_length < 0.0:
            raise ValueError("fallback_branch_length must be finite and non-negative.")
        edge_lengths = {}

    # Add leaf nodes.
    for i, name in enumerate(leaf_names):
        G.add_node(node_id(i, n_leaves), is_leaf=True, label=name)

    # Add internal merges.
    for k, (a, b) in enumerate(children):
        nid = node_id(n_leaves + k, n_leaves)
        left_id = node_id(int(a), n_leaves)
        right_id = node_id(int(b), n_leaves)

        G.add_node(nid, is_leaf=False, label=nid)
        left_length = (
            fallback_branch_length
            if fallback_branch_length is not None
            else edge_lengths[(nid, left_id)]
        )
        right_length = (
            fallback_branch_length
            if fallback_branch_length is not None
            else edge_lengths[(nid, right_id)]
        )
        G.add_edge(nid, left_id, branch_length=left_length)
        G.add_edge(nid, right_id, branch_length=right_length)

    # Discover & cache root.
    roots = [u for u, d in G.in_degree() if d == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, got {roots}")
    G.graph["root"] = roots[0]

    return G


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
    cls = _get_poset_tree_cls()
    n_leaves = linkage_matrix.shape[0] + 1
    if leaf_names is None:
        leaf_names = [f"leaf_{i}" for i in range(n_leaves)]

    children = linkage_matrix[:, :2].astype(int)
    distances = linkage_matrix[:, 2]

    return _build_tree_from_merges(cls, n_leaves, leaf_names, children, distances)


def tree_from_linkage_topology(
    linkage_matrix: np.ndarray,
    leaf_names: Optional[List[str]] = None,
    *,
    fallback_branch_length: float = 1.0,
) -> "PosetTree":
    """Build a tree from linkage topology with placeholder branch lengths.

    This constructor is intended for linkage methods such as centroid and
    median that may emit nonmonotone merge heights. The placeholder lengths
    are not calibrated branch times; callers should replace them with a
    fixed-topology branch-length optimizer before using branch-time statistics.
    """
    cls = _get_poset_tree_cls()
    n_leaves = linkage_matrix.shape[0] + 1
    if leaf_names is None:
        leaf_names = [f"leaf_{i}" for i in range(n_leaves)]

    children = linkage_matrix[:, :2].astype(int)

    return _build_tree_from_merges(
        cls,
        n_leaves,
        leaf_names,
        children,
        distances=None,
        fallback_branch_length=fallback_branch_length,
    )


def tree_from_agglomerative(
    X: np.ndarray,
    leaf_names: Optional[List[str]] = None,
    linkage: str = "average",
    metric: str = "euclidean",
) -> "PosetTree":
    """Build a :class:`PosetTree` from an :class:`AgglomerativeClustering` fit.

    Parameters
    ----------
    X
        Feature matrix of shape ``(n_samples, n_features)``.
    leaf_names
        Optional list of labels; defaults to ``leaf_0 … leaf_{n-1}``.
    linkage, metric
        Passed through to :class:`AgglomerativeClustering`.

    Returns
    -------
    PosetTree
    """
    cls = _get_poset_tree_cls()
    n = int(X.shape[0])
    if leaf_names is None:
        leaf_names = [f"leaf_{i}" for i in range(n)]

    model = AgglomerativeClustering(
        n_clusters=1,
        linkage=linkage,
        metric=metric,
        compute_distances=True,
    )
    model.fit(X)

    children = model.children_
    distances = model.distances_

    return _build_tree_from_merges(cls, n, leaf_names, children, distances)


def tree_from_undirected_edges(
    edges: Iterable[Tuple],
) -> "PosetTree":
    """Orient an undirected weighted tree and promote it to a :class:`PosetTree`.

    Parameters
    ----------
    edges
        Iterable of ``(u, v, weight)`` tuples.

    Returns
    -------
    PosetTree
    """
    cls = _get_poset_tree_cls()
    U = nx.Graph()
    U.add_weighted_edges_from(edges)
    if U.number_of_nodes() == 0 or not nx.is_tree(U):
        raise ValueError("from_undirected_edges requires a non-empty undirected tree.")

    # Pick a leaf as root (deterministic choice).
    leaves = [n for n, d in U.degree() if d == 1]
    root = leaves[0]

    G = cls()
    for n in U.nodes():
        G.add_node(n)

    visited = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v, attr in U[u].items():
            if v not in visited:
                visited.add(v)
                G.add_edge(u, v, weight=float(attr["weight"]))
                queue.append(v)

    # Annotate leaves.
    for n in G.nodes:
        is_leaf_node = G.out_degree(n) == 0
        G.nodes[n]["is_leaf"] = is_leaf_node
        G.nodes[n]["label"] = n
    G.graph["root"] = root

    return G
