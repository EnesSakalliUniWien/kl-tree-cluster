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


def apply_posthoc_merge(
    cluster_roots: Set[str],
    alpha: float,
    tree: nx.DiGraph,
    children: Dict[str, List[str]],
    root: str,
    test_divergence: Callable[[str, str, str], Tuple[float, float, float]],
) -> Tuple[Set[str], List[Dict]]:
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
        (test_statistic, degrees_of_freedom, p_value).

    Returns
    -------
    Tuple[Set[str], List[Dict]]
        - Updated set of cluster root nodes after merging.
        - Audit trail list of all candidate merges and their outcomes.
    """
    cluster_roots = set(cluster_roots)

    # Collect all sibling-boundary pairs
    pairs: List[Dict] = []

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
                # test_divergence returns (test_statistic, df, p_value)
                test_stat, df, p_value = test_divergence(lc, rc, node)
                # For any lc under left_child and rc under right_child in a tree,
                # the lowest common ancestor is the current boundary node.
                pairs.append(
                    {
                        "left_cluster": lc,
                        "right_cluster": rc,
                        "lca": node,
                        "p_value": float(p_value),
                        "test_stat": float(test_stat),
                        "df": float(df),
                    }
                )

    if not pairs:
        return cluster_roots, []

    # Single FDR correction on all pairs
    p_values = np.array([p["p_value"] for p in pairs])
    reject, _, _ = benjamini_hochberg_correction(p_values, alpha=alpha)

    # Update pairs with significance status
    for i, is_rejected in enumerate(reject):
        pairs[i]["is_significant"] = bool(is_rejected)
        pairs[i]["was_merged"] = False  # Initialize

    # Get mergeable pairs (failed to reject H0 = clusters are similar).
    # We no longer block ALL merges under an LCA just because one pair
    # under it is significant — that was overly conservative and prevented
    # obviously similar pairs from merging (e.g., C≈D blocked because A≠B
    # under the same LCA).  BH correction already handles multiplicity;
    # each pair's reject/fail-to-reject decision stands on its own.
    #
    # Sort by p-value descending (most similar first) for greedy merge.
    mergeable_indices = [i for i, r in enumerate(reject) if not r]
    mergeable_indices.sort(key=lambda i: -p_values[i])

    # Greedily merge non-overlapping pairs.
    # Use targeted removal: only discard the two merged cluster roots,
    # then add the LCA.  The old code used `cluster_roots -= nx.descendants(tree, lca)`
    # which would silently absorb a third unrelated cluster root that happened to
    # be under the same LCA.
    #
    # Antichain guard: before merging, verify that no OTHER cluster root is a
    # descendant of the proposed LCA.  If it is, the merge would create a
    # non-partition (ancestor + descendant both in the root set).  Skip the
    # merge rather than silently absorbing uninvolved clusters.
    merged_roots_count = 0
    for idx in mergeable_indices:
        lc = pairs[idx]["left_cluster"]
        rc = pairs[idx]["right_cluster"]
        lca = pairs[idx]["lca"]

        # Only merge "live" cluster roots.
        if lc not in cluster_roots or rc not in cluster_roots:
            continue

        # Antichain check: would other cluster roots end up under the LCA?
        lca_descendants = nx.descendants(tree, lca)
        other_roots_under_lca = {
            r for r in cluster_roots
            if r in lca_descendants and r != lc and r != rc
        }
        if other_roots_under_lca:
            # Merging here would violate the antichain invariant.
            # Skip this pair.
            continue

        cluster_roots.discard(lc)
        cluster_roots.discard(rc)
        cluster_roots.add(lca)

        pairs[idx]["was_merged"] = True
        merged_roots_count += 1

    return cluster_roots, pairs
