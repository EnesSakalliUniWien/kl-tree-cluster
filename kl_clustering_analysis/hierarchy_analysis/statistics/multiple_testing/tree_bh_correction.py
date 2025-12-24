"""TreeBH hierarchical FDR correction implementation.

This module implements the TreeBH procedure from Bogomolov et al. (2021),
which provides hierarchical FDR control for tree-structured hypotheses.

The key insight is that TreeBH applies BH correction within each *family*
(children of the same parent), not across all nodes at a level. The threshold
for each family is adjusted based on the rejection history in ancestor families.

References
----------
Bogomolov, M., Peterson, C. B., Benjamini, Y., & Sabatti, C. (2021).
"Hypotheses on a tree: new error rates and testing strategies".
Biometrika, 108(3), 575-590.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx
from statsmodels.stats.multitest import multipletests

from kl_clustering_analysis.tree.poset_tree import compute_node_depths


@dataclass
class TreeBHResult:
    """Results from TreeBH correction.

    Attributes
    ----------
    reject : np.ndarray
        Boolean array of rejections (True = reject null hypothesis)
    adjusted_p : np.ndarray
        Adjusted p-values
    level_thresholds : Dict[int, float]
        Threshold used at each level
    family_results : Dict[str, Dict]
        Per-family results for debugging
    """

    reject: np.ndarray
    adjusted_p: np.ndarray
    level_thresholds: Dict[int, float]
    family_results: Dict[str, Dict]


def _get_root_nodes(tree: nx.DiGraph) -> List[str]:
    """Find root nodes (nodes with no parents)."""
    return [n for n in tree.nodes() if tree.in_degree(n) == 0]


def _get_families_by_parent(
    tree: nx.DiGraph, child_ids: List[str]
) -> Dict[str, List[int]]:
    """Group child indices by their parent node.

    Returns a dict mapping parent_id -> list of indices into child_ids
    for children of that parent.
    """
    families: Dict[str, List[int]] = defaultdict(list)

    for i, child_id in enumerate(child_ids):
        # Find parent of this child
        predecessors = list(tree.predecessors(child_id))
        if predecessors:
            parent_id = predecessors[0]  # Assume single parent (tree)
            families[parent_id].append(i)

    return dict(families)


def _compute_family_threshold(
    tree: nx.DiGraph,
    parent_id: str,
    family_rejection_counts: Dict[str, Tuple[int, int]],
    alpha: float,
    roots: Set[str],
) -> float:
    """Compute the adjusted threshold for a family.

    The threshold is α × product of (rejections/tests) for all ancestor families.

    Parameters
    ----------
    tree
        The tree structure
    parent_id
        Parent node whose children form this family
    family_rejection_counts
        Dict of (n_rejections, n_tests) for each parent already processed
    alpha
        Base significance level
    roots
        Set of root node IDs

    Returns
    -------
    float
        Adjusted threshold for this family
    """
    adjusted_alpha = alpha

    # Walk up the tree from parent to root, multiplying rejection proportions
    current = parent_id
    while current not in roots:
        # Find the grandparent (parent of current)
        predecessors = list(tree.predecessors(current))
        if not predecessors:
            break

        grandparent = predecessors[0]

        # Get rejection proportion from grandparent's family
        if grandparent in family_rejection_counts:
            n_rej, n_tests = family_rejection_counts[grandparent]
            if n_tests > 0:
                proportion = n_rej / n_tests
                adjusted_alpha *= proportion

        current = grandparent

    return adjusted_alpha


def tree_bh_correction(
    tree: nx.DiGraph,
    p_values: np.ndarray,
    child_ids: List[str],
    alpha: float = 0.05,
    verbose: bool = False,
) -> TreeBHResult:
    """Apply TreeBH hierarchical FDR correction.

    The TreeBH procedure:
    1. Start from the root and work down level by level
    2. At each level, for each family (children of same parent):
       - Apply BH with threshold = α × product of (rejections/tests) in ancestor families
    3. Only test families whose parent hypothesis was rejected

    This provides FDR control at each level while respecting the tree structure.

    Parameters
    ----------
    tree
        Directed graph representing the hierarchy (parent -> child edges)
    p_values
        Array of p-values, one per edge (child node)
    child_ids
        List of child node IDs corresponding to p_values
    alpha
        Base significance level
    verbose
        If True, print debugging information

    Returns
    -------
    TreeBHResult
        Object containing rejection decisions, adjusted p-values, and diagnostics

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> tree = nx.DiGraph()
    >>> tree.add_edges_from([("root", "A"), ("root", "B"), ("A", "C")])
    >>> p_values = np.array([0.01, 0.02, 0.03])
    >>> child_ids = ["A", "B", "C"]
    >>> result = tree_bh_correction(tree, p_values, child_ids)
    >>> result.reject
    array([ True,  True,  True])
    """
    n = len(p_values)
    reject = np.zeros(n, dtype=bool)
    adjusted_p = np.ones(n, dtype=float)

    if n == 0:
        return TreeBHResult(
            reject=reject,
            adjusted_p=adjusted_p,
            level_thresholds={},
            family_results={},
        )

    # Compute node depths
    node_depths = compute_node_depths(tree)

    # Map child_ids to their depths
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    # Group children by parent (family)
    families = _get_families_by_parent(tree, child_ids)

    # Track which parents have been "rejected" (their edge was significant)
    # Initially, root is always considered "rejected" (we always test level 1)
    roots = set(_get_root_nodes(tree))
    rejected_parents: Set[str] = roots.copy()

    # Track rejection proportions for threshold adjustment
    # Key: parent_id, Value: (n_rejections, n_tests)
    family_rejection_counts: Dict[str, Tuple[int, int]] = {}

    # Process levels in order
    levels = sorted(set(child_depths))
    level_thresholds: Dict[int, float] = {}
    family_results: Dict[str, Dict] = {}

    if verbose:
        print(f"TreeBH: {n} edges, {len(levels)} levels, {len(families)} families")
        print(f"Roots: {roots}")

    for level in levels:
        if verbose:
            print(f"\n--- Level {level} ---")

        # Find families to test at this level
        # A family is tested if its parent was rejected
        families_at_level = []
        for parent_id, child_indices in families.items():
            # Check if any child is at this level
            level_children = [i for i in child_indices if child_depths[i] == level]
            if level_children and parent_id in rejected_parents:
                families_at_level.append((parent_id, level_children))

        if not families_at_level:
            if verbose:
                print(f"  No families to test at level {level}")
            continue

        # Process each family at this level
        for parent_id, child_indices in families_at_level:
            # Compute adjusted threshold for this family
            adjusted_alpha = _compute_family_threshold(
                tree, parent_id, family_rejection_counts, alpha, roots
            )

            if verbose:
                print(
                    f"  Family under {parent_id}: {len(child_indices)} children, "
                    f"α_adj = {adjusted_alpha:.6f}"
                )

            # Extract p-values for this family
            family_p_values = p_values[child_indices]

            # Apply BH within this family
            if len(family_p_values) > 0 and adjusted_alpha > 0:
                family_reject, family_adjusted, _, _ = multipletests(
                    family_p_values,
                    alpha=adjusted_alpha,
                    method="fdr_bh",
                    is_sorted=False,
                    returnsorted=False,
                )

                # Store results
                for j, idx in enumerate(child_indices):
                    reject[idx] = family_reject[j]
                    adjusted_p[idx] = family_adjusted[j]

                    # If rejected, add this child as a potential parent for next level
                    if family_reject[j]:
                        rejected_parents.add(child_ids[idx])

                # Record rejection counts for this family
                n_rejections = int(np.sum(family_reject))
                n_tests = len(family_p_values)
                family_rejection_counts[parent_id] = (n_rejections, n_tests)

                family_results[parent_id] = {
                    "level": level,
                    "n_tests": n_tests,
                    "n_rejections": n_rejections,
                    "adjusted_alpha": adjusted_alpha,
                    "child_ids": [child_ids[i] for i in child_indices],
                    "p_values": family_p_values.tolist(),
                    "rejected": family_reject.tolist(),
                }

                if verbose:
                    print(
                        f"    Rejected {n_rejections}/{n_tests} "
                        f"(proportion: {n_rejections / n_tests:.2%})"
                    )

        # Store the base threshold for this level (just α for reference)
        level_thresholds[level] = alpha

    return TreeBHResult(
        reject=reject,
        adjusted_p=adjusted_p,
        level_thresholds=level_thresholds,
        family_results=family_results,
    )


__all__ = ["tree_bh_correction", "TreeBHResult"]
