"""Strict topology helpers for rooted ``PosetTree`` instances."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes, compute_node_depths


def is_leaf(tree: nx.DiGraph, node_id: object) -> bool:
    """Return the explicit leaf flag for a node."""
    return bool(tree.nodes[node_id]["is_leaf"])


def get_leaf_nodes(tree: nx.DiGraph, node: object | None = None) -> list[object]:
    """Collect leaf node ids globally or inside one subtree."""
    if node is None:
        candidates = list(tree.nodes)
    elif is_leaf(tree, node):
        candidates = [node]
    else:
        candidates = list(nx.descendants(tree, node))
    return [candidate for candidate in candidates if is_leaf(tree, candidate)]


def get_leaf_values(
    tree: nx.DiGraph,
    *,
    node: object | None = None,
    use_labels: bool = True,
    sort: bool = True,
) -> list[object]:
    """Collect leaf labels or ids globally or inside one subtree."""
    leaf_nodes = get_leaf_nodes(tree, node)
    values = [
        tree.nodes[leaf_node]["label"] if use_labels else leaf_node
        for leaf_node in leaf_nodes
    ]
    return sorted(values) if sort else values


def compute_descendant_leaf_sets(
    tree: nx.DiGraph,
    *,
    use_labels: bool = True,
) -> dict[object, frozenset]:
    """Map each node to the set of descendant leaf labels or ids."""
    descendant_sets: dict[object, frozenset] = {}
    for node in bottom_up_nodes(tree):
        if is_leaf(tree, node):
            leaf_value = tree.nodes[node]["label"] if use_labels else node
            descendant_sets[node] = frozenset([leaf_value])
            continue

        child_sets = [descendant_sets[child] for child in tree.successors(node)]
        descendant_sets[node] = frozenset.union(*child_sets)
    return descendant_sets


def _single_parent(tree: nx.DiGraph, node_id: object) -> object:
    parents = list(tree.predecessors(node_id))
    if len(parents) != 1:
        raise ValueError(f"Expected node {node_id!r} to have exactly one parent.")
    return parents[0]


def lowest_common_ancestor(
    tree: nx.DiGraph,
    node_a: object,
    node_b: object,
    *,
    depths: dict[object, int] | None = None,
) -> object:
    """Find the lowest common ancestor of two nodes in a rooted tree."""
    if node_a == node_b:
        return node_a

    node_depths = depths if depths is not None else compute_node_depths(tree)
    try:
        depth_a = node_depths[node_a]
        depth_b = node_depths[node_b]
    except KeyError as exc:
        raise ValueError(
            f"Cannot compute LCA for nodes outside the rooted tree: {node_a!r}, {node_b!r}."
        ) from exc

    current_a, current_b = node_a, node_b
    for _ in range(max(depth_a - depth_b, 0)):
        current_a = _single_parent(tree, current_a)
    for _ in range(max(depth_b - depth_a, 0)):
        current_b = _single_parent(tree, current_b)

    while current_a != current_b:
        current_a = _single_parent(tree, current_a)
        current_b = _single_parent(tree, current_b)

    return current_a


def lowest_common_ancestor_for_set(
    tree: nx.DiGraph,
    nodes: Iterable[object],
) -> object:
    """Find the lowest common ancestor for a non-empty node collection."""
    node_iterator = iter(nodes)
    try:
        lca = next(node_iterator)
    except StopIteration as exc:
        raise ValueError("Cannot find LCA for an empty node set.") from exc

    root = tree.graph["root"]
    depths = compute_node_depths(tree)
    for node in node_iterator:
        lca = lowest_common_ancestor(tree, lca, node, depths=depths)
        if lca == root:
            return root
    return lca


__all__ = [
    "compute_descendant_leaf_sets",
    "get_leaf_nodes",
    "get_leaf_values",
    "is_leaf",
    "lowest_common_ancestor",
    "lowest_common_ancestor_for_set",
]
