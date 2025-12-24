"""Post-hoc merge logic for reducing over-splitting in hierarchical clustering.

This module provides tree-respecting post-hoc merging that combines clusters
whose distributions are not significantly different after FDR correction.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from .statistics.multiple_testing import benjamini_hochberg_correction


def _get_leaf_clusters_under_node(
    node: str,
    cluster_roots: Set[str],
    tree: nx.DiGraph,
) -> List[str]:
    """Get all cluster root nodes that are descendants of the given node.

    Parameters
    ----------
    node
        The tree node to search under.
    cluster_roots
        Set of current cluster root node IDs.
    tree
        The hierarchy tree.

    Returns
    -------
    List[str]
        List of cluster root nodes that are descendants of `node`.
    """
    if node in cluster_roots:
        return [node]

    descendants = nx.descendants(tree, node)
    return [n for n in descendants if n in cluster_roots]


def _find_lca_pair(node_a: str, node_b: str, tree: nx.DiGraph, root: str) -> str:
    """Find the lowest common ancestor of two nodes.

    Parameters
    ----------
    node_a, node_b
        The two nodes to find the LCA for.
    tree
        The hierarchy tree.
    root
        The root node of the tree.

    Returns
    -------
    str
        The lowest common ancestor node ID.
    """
    ancestors_a = set(nx.ancestors(tree, node_a))
    ancestors_a.add(node_a)

    current = node_b
    while current not in ancestors_a:
        parents = list(tree.predecessors(current))
        if not parents:
            return root
        current = parents[0]

    return current


def apply_posthoc_merge(
    cluster_roots: Set[str],
    alpha: float,
    tree: nx.DiGraph,
    children: Dict[str, List[str]],
    root: str,
    test_divergence: Callable[[str, str, str], Tuple[float, float]],
) -> Set[str]:
    """Apply tree-respecting post-hoc merging to reduce over-splitting.

    Collects all sibling-boundary cluster pairs, applies FDR correction once,
    then greedily merges non-overlapping pairs by p-value (highest first).

    Only compares clusters across sibling boundaries in the tree, preserving
    hierarchical structure.

    Parameters
    ----------
    cluster_roots
        Set of current cluster root node IDs.
    alpha
        Significance level for the merge test.
    tree
        The hierarchy tree (nx.DiGraph).
    children
        Pre-computed mapping from node ID to list of child node IDs.
    root
        The root node ID of the tree.
    test_divergence
        Callable that takes (cluster_a, cluster_b, common_ancestor) and returns
        (test_statistic, p_value).

    Returns
    -------
    Set[str]
        Updated set of cluster root nodes after merging.
    """
    cluster_roots = set(cluster_roots)

    # Collect all sibling-boundary pairs
    pairs: List[Tuple[str, str, str, float]] = []

    for node in tree.nodes:
        node_children = children[node]
        if len(node_children) != 2:
            continue

        left_child, right_child = node_children

        left_clusters = _get_leaf_clusters_under_node(left_child, cluster_roots, tree)
        right_clusters = _get_leaf_clusters_under_node(right_child, cluster_roots, tree)

        if not left_clusters or not right_clusters:
            continue

        for lc in left_clusters:
            for rc in right_clusters:
                _, p_value = test_divergence(lc, rc, node)
                lca = _find_lca_pair(lc, rc, tree, root)
                pairs.append((lc, rc, lca, p_value))

    if not pairs:
        return cluster_roots

    # Single FDR correction on all pairs
    p_values = np.array([p[3] for p in pairs])
    reject, _, _ = benjamini_hochberg_correction(p_values, alpha=alpha)

    # Get mergeable pairs (failed to reject H0 = clusters are similar)
    # Sort by p-value descending (most similar first)
    mergeable = [(i, pairs[i]) for i, r in enumerate(reject) if not r]
    mergeable.sort(key=lambda x: -p_values[x[0]])

    # Greedily merge non-overlapping pairs
    merged: Set[str] = set()
    for _, (lc, rc, lca, _) in mergeable:
        if lc in merged or rc in merged:
            continue

        cluster_roots.discard(lc)
        cluster_roots.discard(rc)
        cluster_roots -= nx.descendants(tree, lca)
        cluster_roots.add(lca)

        merged.add(lc)
        merged.add(rc)

    return cluster_roots
