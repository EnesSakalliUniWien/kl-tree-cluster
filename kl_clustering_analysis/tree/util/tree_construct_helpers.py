from __future__ import annotations
from typing import Any, Dict
import networkx as nx


def count_cluster_leaves(node: Any) -> int:
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if left is None and right is None:
        return 1
    return count_cluster_leaves(left) + count_cluster_leaves(right)


def add_cluster_node_recursive(G: nx.DiGraph, node: Any, leaf_names: list[str]) -> str:
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    is_leaf = left is None and right is None
    if is_leaf:
        leaf_index = int(getattr(node, "id"))
        node_id = f"L{leaf_index}"
        G.add_node(node_id, label=leaf_names[leaf_index], is_leaf=True)
        return node_id
    node_id = f"N{int(getattr(node, 'id'))}"
    left_id = add_cluster_node_recursive(G, left, leaf_names)
    right_id = add_cluster_node_recursive(G, right, leaf_names)
    G.add_node(node_id, is_leaf=False)
    G.add_edge(node_id, left_id)
    G.add_edge(node_id, right_id)
    return node_id


def add_nested_tuple_recursive(
    G: nx.DiGraph, node: Any, counters: Dict[str, int]
) -> str:
    if not isinstance(node, tuple):
        node_id = f"L{counters['leaf']}"
        counters["leaf"] += 1
        G.add_node(node_id, label=node, is_leaf=True)
        return node_id

    if len(node) not in (2, 3):
        raise ValueError("nested tuple nodes must have 2 or 3 items")

    left = node[0]
    right = node[1]

    left_id = add_nested_tuple_recursive(G, left, counters)
    right_id = add_nested_tuple_recursive(G, right, counters)

    node_id = f"N{counters['internal']}"
    counters["internal"] += 1
    attrs: Dict[str, Any] = {"is_leaf": False}
    G.add_node(node_id, **attrs)
    G.add_edge(node_id, left_id)
    G.add_edge(node_id, right_id)
    return node_id
