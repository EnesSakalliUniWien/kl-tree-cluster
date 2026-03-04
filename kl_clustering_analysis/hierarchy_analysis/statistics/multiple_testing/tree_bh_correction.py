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

from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths


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


def _initialize_tree_bh_outputs(total_hypothesis_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize rejection mask and adjusted p-value array."""
    rejection_mask = np.zeros(total_hypothesis_count, dtype=bool)
    adjusted_p_values = np.ones(total_hypothesis_count, dtype=float)
    return rejection_mask, adjusted_p_values


def _compute_child_depths(tree: nx.DiGraph, child_ids: List[str]) -> np.ndarray:
    """Map each child id to its depth in the tree."""
    node_depths = compute_node_depths(tree)
    return np.array([node_depths.get(child_id, 0) for child_id in child_ids])


def _collect_testable_families_at_level(
    families_by_parent: Dict[str, List[int]],
    child_depths: np.ndarray,
    level: int,
    rejected_parent_nodes: Set[str],
) -> List[Tuple[str, List[int]]]:
    """Collect families at a level whose parent node was rejected."""
    families_to_test: List[Tuple[str, List[int]]] = []
    for parent_id, child_indices in families_by_parent.items():
        level_child_indices = [index for index in child_indices if child_depths[index] == level]
        if level_child_indices and parent_id in rejected_parent_nodes:
            families_to_test.append((parent_id, level_child_indices))
    return families_to_test


def _apply_bh_within_family(
    family_p_values: np.ndarray,
    adjusted_alpha: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply BH correction within one family."""
    if len(family_p_values) == 0 or adjusted_alpha <= 0:
        return None

    family_reject, family_adjusted, _, _ = multipletests(
        family_p_values,
        alpha=adjusted_alpha,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )
    return family_reject, family_adjusted


def _record_family_bh_outcome(
    *,
    parent_id: str,
    level: int,
    family_child_indices: List[int],
    family_p_values: np.ndarray,
    family_reject_mask: np.ndarray,
    family_adjusted_p_values: np.ndarray,
    child_ids: List[str],
    rejection_mask: np.ndarray,
    adjusted_p_values: np.ndarray,
    rejected_parent_nodes: Set[str],
    family_rejection_counts: Dict[str, Tuple[int, int]],
    family_results: Dict[str, Dict],
    adjusted_alpha: float,
) -> tuple[int, int]:
    """Store one family's BH outcome in all TreeBH tracking structures."""
    for within_family_index, global_index in enumerate(family_child_indices):
        rejection_mask[global_index] = family_reject_mask[within_family_index]
        adjusted_p_values[global_index] = family_adjusted_p_values[within_family_index]
        if family_reject_mask[within_family_index]:
            rejected_parent_nodes.add(child_ids[global_index])

    rejection_count = int(np.sum(family_reject_mask))
    test_count = len(family_p_values)
    family_rejection_counts[parent_id] = (rejection_count, test_count)
    family_results[parent_id] = {
        "level": level,
        "n_tests": test_count,
        "n_rejections": rejection_count,
        "adjusted_alpha": adjusted_alpha,
        "child_ids": [child_ids[index] for index in family_child_indices],
        "p_values": family_p_values.tolist(),
        "rejected": family_reject_mask.tolist(),
    }
    return rejection_count, test_count


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
    total_hypothesis_count = len(p_values)
    rejection_mask, adjusted_p_values = _initialize_tree_bh_outputs(total_hypothesis_count)

    if total_hypothesis_count == 0:
        return TreeBHResult(
            reject=rejection_mask,
            adjusted_p=adjusted_p_values,
            level_thresholds={},
            family_results={},
        )

    child_depths = _compute_child_depths(tree, child_ids)
    families_by_parent = _get_families_by_parent(tree, child_ids)
    root_nodes = set(_get_root_nodes(tree))
    rejected_parent_nodes: Set[str] = root_nodes.copy()
    family_rejection_counts: Dict[str, Tuple[int, int]] = {}
    levels = sorted(set(child_depths))
    level_thresholds: Dict[int, float] = {}
    family_results: Dict[str, Dict] = {}

    if verbose:
        print(
            f"TreeBH: {total_hypothesis_count} edges, {len(levels)} levels, "
            f"{len(families_by_parent)} families"
        )
        print(f"Roots: {root_nodes}")

    for level in levels:
        if verbose:
            print(f"\n--- Level {level} ---")

        families_at_level = _collect_testable_families_at_level(
            families_by_parent,
            child_depths,
            level,
            rejected_parent_nodes,
        )

        if not families_at_level:
            if verbose:
                print(f"  No families to test at level {level}")
            continue

        for parent_id, family_child_indices in families_at_level:
            adjusted_alpha = _compute_family_threshold(
                tree,
                parent_id,
                family_rejection_counts,
                alpha,
                root_nodes,
            )

            if verbose:
                print(
                    f"  Family under {parent_id}: {len(family_child_indices)} children, "
                    f"α_adj = {adjusted_alpha:.6f}"
                )

            family_p_values = p_values[family_child_indices]
            family_bh_result = _apply_bh_within_family(family_p_values, adjusted_alpha)
            if family_bh_result is None:
                continue

            family_reject_mask, family_adjusted_p_values = family_bh_result
            rejection_count, test_count = _record_family_bh_outcome(
                parent_id=parent_id,
                level=level,
                family_child_indices=family_child_indices,
                family_p_values=family_p_values,
                family_reject_mask=family_reject_mask,
                family_adjusted_p_values=family_adjusted_p_values,
                child_ids=child_ids,
                rejection_mask=rejection_mask,
                adjusted_p_values=adjusted_p_values,
                rejected_parent_nodes=rejected_parent_nodes,
                family_rejection_counts=family_rejection_counts,
                family_results=family_results,
                adjusted_alpha=adjusted_alpha,
            )

            if verbose:
                print(
                    f"    Rejected {rejection_count}/{test_count} "
                    f"(proportion: {rejection_count / test_count:.2%})"
                )

        level_thresholds[level] = alpha

    return TreeBHResult(
        reject=rejection_mask,
        adjusted_p=adjusted_p_values,
        level_thresholds=level_thresholds,
        family_results=family_results,
    )


__all__ = ["tree_bh_correction", "TreeBHResult"]
