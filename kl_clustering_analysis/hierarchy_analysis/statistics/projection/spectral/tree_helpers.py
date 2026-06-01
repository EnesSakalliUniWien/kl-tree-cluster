"""Tree traversal helpers for spectral dimension estimation."""

from __future__ import annotations

from typing import Dict

import networkx as nx

from kl_clustering_analysis.core_utils.tree_utils import bottom_up_nodes


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


__all__ = [
    "is_leaf",
    "precompute_descendants",
]
