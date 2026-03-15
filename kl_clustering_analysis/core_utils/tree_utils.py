"""Tree utility functions for hierarchical clustering.

Low-level tree operations that don't depend on hierarchy_analysis
modules, avoiding circular import issues.
"""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx


def bottom_up_nodes(tree: nx.DiGraph) -> Iterator[str]:
    """Yield nodes in bottom-up order (leaves first, root last).

    This is the single canonical traversal for all bottom-up aggregation
    passes across the codebase — descendant sets, distribution propagation,
    split-flag precomputation, and spectral index collection should all
    call this instead of inlining their own topological sort.

    Equivalent to ``reversed(list(nx.topological_sort(tree)))`` and to
    ``nx.dfs_postorder_nodes(tree, source=root)`` on a rooted tree, but
    expressed as a single source of truth.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy.

    Yields
    ------
    str
        Node identifiers, leaves first, root last.
    """
    return reversed(list(nx.topological_sort(tree)))


def compute_node_depths(tree: nx.DiGraph) -> dict[str, int]:
    """Compute depth of each node from the root via BFS.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy.

    Returns
    -------
    Dict[str, int]
        Mapping from node_id to depth (root = 0).

    Raises
    ------
    ValueError
        If the tree has no root node (all nodes have parents).
    """
    roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    if not roots:
        raise ValueError("Tree has no root node (all nodes have parents)")

    depths: dict[str, int] = {}
    for root in roots:
        depths[root] = 0

    queue = list(roots)
    while queue:
        node = queue.pop(0)
        for child in tree.successors(node):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)

    return depths


__all__ = [
    "bottom_up_nodes",
    "compute_node_depths",
]
