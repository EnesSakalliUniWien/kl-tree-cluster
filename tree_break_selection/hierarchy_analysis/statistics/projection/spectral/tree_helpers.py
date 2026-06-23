"""Tree traversal helpers for spectral dimension estimation."""

from __future__ import annotations

from typing import Dict

import networkx as nx

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes


def is_leaf(tree: nx.DiGraph, node_id: str) -> bool:
    """Check whether *node_id* is a leaf in *tree*."""
    return tree.out_degree(node_id) == 0


def precompute_descendants(
    tree: nx.DiGraph,
    label_to_idx: Dict[str, int],
) -> Dict[str, list[int]]:
    """Bottom-up precomputation of descendant leaf indices.

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
    """
    desc_indices: Dict[str, list] = {}

    for node_id in bottom_up_nodes(tree):
        if is_leaf(tree, node_id):
            lbl = tree.nodes[node_id]["label"]
            if lbl not in label_to_idx:
                raise ValueError(
                    f"Leaf label {lbl!r} for node {node_id!r} is missing from leaf_data."
                )
            desc_indices[node_id] = [label_to_idx[lbl]]
        else:
            indices: list[int] = []
            for child in tree.successors(node_id):
                indices.extend(desc_indices[child])
            desc_indices[node_id] = indices
    return desc_indices


def precompute_descendant_internal_nodes(tree: nx.DiGraph) -> Dict[str, list[object]]:
    """Bottom-up precomputation of descendant internal node IDs.

    Internal barycenter rows are a diagnostic reconstruction of the older
    spectral path. They are deterministic descendants, so this helper records
    their node IDs without treating them as extra independent leaves.
    """
    desc_internal: Dict[str, list[object]] = {}

    for node_id in bottom_up_nodes(tree):
        if is_leaf(tree, node_id):
            desc_internal[node_id] = []
            continue
        internal_nodes: list[object] = []
        for child in tree.successors(node_id):
            if not is_leaf(tree, child):
                internal_nodes.append(child)
            internal_nodes.extend(desc_internal[child])
        desc_internal[node_id] = internal_nodes
    return desc_internal


__all__ = [
    "is_leaf",
    "precompute_descendants",
    "precompute_descendant_internal_nodes",
]
