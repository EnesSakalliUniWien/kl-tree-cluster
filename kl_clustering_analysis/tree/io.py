"""I/O helpers for constructing :class:`PosetTree` from external representations.

Each public function accepts raw clustering output (linkage matrix, sklearn model,
undirected edge list) and returns a fully-initialised :class:`PosetTree`.

A shared ``_build_tree_from_merges`` helper delegates branch-length computation
to :func:`~kl_clustering_analysis.tree.branch_lengths.compute_ultrametric_branch_lengths`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from kl_clustering_analysis.tree.branch_lengths import compute_ultrametric_branch_lengths, node_id

if TYPE_CHECKING:
    from kl_clustering_analysis.tree.poset_tree import PosetTree


def _get_poset_tree_cls() -> type["PosetTree"]:
    """Lazy import to avoid circular dependency with poset_tree.py."""
    from kl_clustering_analysis.tree.poset_tree import PosetTree

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
        Optional ``(n_leaves - 1,)`` array of merge distances.  When supplied,
        branch lengths are computed via ultrametric subtraction (see
        :func:`~kl_clustering_analysis.tree.branch_lengths.compute_ultrametric_branch_lengths`).
        When ``None`` every edge receives a default ``branch_length`` of ``1.0``.

    Returns
    -------
    PosetTree
    """
    G = cls()

    # Compute all edge branch lengths up-front.
    edge_lengths = compute_ultrametric_branch_lengths(n_leaves, children, distances)

    # Add leaf nodes.
    for i, name in enumerate(leaf_names):
        G.add_node(node_id(i, n_leaves), is_leaf=True, label=str(name))

    # Add internal merges.
    for k, (a, b) in enumerate(children):
        nid = node_id(n_leaves + k, n_leaves)
        left_id = node_id(int(a), n_leaves)
        right_id = node_id(int(b), n_leaves)

        G.add_node(nid, is_leaf=False)
        G.add_edge(nid, left_id, branch_length=edge_lengths[(nid, left_id)])
        G.add_edge(nid, right_id, branch_length=edge_lengths[(nid, right_id)])

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


def tree_from_agglomerative(
    X: np.ndarray,
    leaf_names: Optional[List[str]] = None,
    linkage: str = "average",
    metric: str = "euclidean",
    compute_distances: bool = True,
) -> "PosetTree":
    """Build a :class:`PosetTree` from an :class:`AgglomerativeClustering` fit.

    Parameters
    ----------
    X
        Feature matrix of shape ``(n_samples, n_features)``.
    leaf_names
        Optional list of labels; defaults to ``leaf_0 … leaf_{n-1}``.
    linkage, metric, compute_distances
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
        compute_distances=compute_distances,
    )
    model.fit(X)

    children = model.children_
    distances = getattr(model, "distances_", None)

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

    # Pick a leaf as root (deterministic choice).
    leaves = [n for n, d in U.degree() if d == 1]
    root = leaves[0] if leaves else next(iter(U.nodes))

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
                G.add_edge(u, v, weight=float(attr.get("weight", 1.0)))
                queue.append(v)

    # Annotate leaves.
    for n in G.nodes:
        G.nodes[n]["is_leaf"] = G.out_degree(n) == 0
    G.graph["root"] = root

    return G
