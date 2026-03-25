from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import networkx as nx


def get_unique_parent_id(tree: nx.DiGraph, node_id: str) -> str | None:
    """Return a node's unique parent or raise if the graph is not tree-shaped."""
    predecessors = [str(parent_id) for parent_id in tree.predecessors(node_id)]
    if not predecessors:
        return None
    if len(predecessors) > 1:
        raise ValueError(
            "stopping-edge recovery expects a rooted tree with at most one parent per node; "
            f"node {node_id!r} has parents {predecessors!r}."
        )
    return predecessors[0]


def build_tree_distance_resolver(tree: nx.DiGraph) -> Callable[[str, str], float]:
    """Return a cached shortest-path distance function on the undirected tree view."""
    tree_undirected = tree.to_undirected(as_view=True)

    @lru_cache(maxsize=None)
    def tree_distance(node_a: str, node_b: str) -> float:
        if node_a == node_b:
            return 0.0
        try:
            return float(nx.shortest_path_length(tree_undirected, node_a, node_b))
        except nx.NetworkXNoPath:
            return float("inf")

    return tree_distance
