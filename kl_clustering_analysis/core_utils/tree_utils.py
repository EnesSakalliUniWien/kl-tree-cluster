"""Tree utility functions for hierarchical clustering.

Low-level tree operations that don't depend on hierarchy_analysis
modules, avoiding circular import issues.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx


def compute_node_depths(tree: nx.DiGraph) -> Dict[str, int]:
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

    depths: Dict[str, int] = {}
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
    "compute_node_depths",
]
