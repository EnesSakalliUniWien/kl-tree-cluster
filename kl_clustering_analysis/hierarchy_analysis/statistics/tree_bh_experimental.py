"""Experimental TreeBH hierarchical FDR correction implementation.

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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from statsmodels.stats.multitest import multipletests


@dataclass
class TreeBHResult:
    """Results from TreeBH correction."""

    reject: np.ndarray  # Boolean array of rejections
    adjusted_p: np.ndarray  # Adjusted p-values
    level_thresholds: Dict[int, float]  # Threshold used at each level
    family_results: Dict[str, Dict]  # Per-family results for debugging


def _get_root_nodes(tree: nx.DiGraph) -> List[str]:
    """Find root nodes (nodes with no parents)."""
    return [n for n in tree.nodes() if tree.in_degree(n) == 0]


def _compute_node_depths(tree: nx.DiGraph) -> Dict[str, int]:
    """Compute depth of each node from root via BFS."""
    roots = _get_root_nodes(tree)
    if not roots:
        raise ValueError("Tree has no root node")

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
    node_depths = _compute_node_depths(tree)

    # Map child_ids to their indices and depths
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
            # α_family = α × product of (rejections/tests) along path from root
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


# =============================================================================
# Test functions
# =============================================================================


def _create_test_tree_simple() -> Tuple[nx.DiGraph, List[str], np.ndarray]:
    """Create a simple test tree with known structure.

    Tree structure::

          root
         /    \\
       L1a    L1b
       / \\    / \\
     L2a L2b L2c L2d

    Returns (tree, child_ids for edges, p_values)
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "L1a"),
            ("root", "L1b"),
            ("L1a", "L2a"),
            ("L1a", "L2b"),
            ("L1b", "L2c"),
            ("L1b", "L2d"),
        ]
    )

    # Edges are: root->L1a, root->L1b, L1a->L2a, L1a->L2b, L1b->L2c, L1b->L2d
    child_ids = ["L1a", "L1b", "L2a", "L2b", "L2c", "L2d"]

    # p-values: L1a and L1b significant, L2a significant, others not
    p_values = np.array([0.01, 0.02, 0.01, 0.3, 0.4, 0.5])

    return tree, child_ids, p_values


def _create_test_tree_asymmetric() -> Tuple[nx.DiGraph, List[str], np.ndarray]:
    """Create an asymmetric tree for testing.

    Tree structure::

          root
         /    \\
       L1a    L1b
       / \\      \\
     L2a L2b    L2c
     /
   L3a

    Returns (tree, child_ids for edges, p_values)
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "L1a"),
            ("root", "L1b"),
            ("L1a", "L2a"),
            ("L1a", "L2b"),
            ("L1b", "L2c"),
            ("L2a", "L3a"),
        ]
    )

    child_ids = ["L1a", "L1b", "L2a", "L2b", "L2c", "L3a"]
    # All significant
    p_values = np.array([0.01, 0.02, 0.01, 0.01, 0.01, 0.01])

    return tree, child_ids, p_values


def test_tree_bh_simple():
    """Test TreeBH on a simple balanced tree."""
    tree, child_ids, p_values = _create_test_tree_simple()

    result = tree_bh_correction(tree, p_values, child_ids, alpha=0.05, verbose=True)

    print("\n=== Simple Tree Test Results ===")
    print(f"Rejections: {result.reject}")
    print(f"Adjusted p-values: {result.adjusted_p}")
    print(f"Level thresholds: {result.level_thresholds}")

    # Verify expectations
    # L1a and L1b should be rejected (p=0.01, 0.02 < 0.05)
    assert result.reject[0], "L1a should be rejected"
    assert result.reject[1], "L1b should be rejected"

    # L2a should be rejected (p=0.01)
    # But threshold is adjusted: α × (2/2) = 0.05 (both L1 rejected)
    assert result.reject[2], "L2a should be rejected"

    # L2b, L2c, L2d have high p-values, should not be rejected
    assert not result.reject[3], "L2b should not be rejected"
    assert not result.reject[4], "L2c should not be rejected"
    assert not result.reject[5], "L2d should not be rejected"

    print("✓ Simple tree test passed!")
    return result


def test_tree_bh_asymmetric():
    """Test TreeBH on an asymmetric tree."""
    tree, child_ids, p_values = _create_test_tree_asymmetric()

    result = tree_bh_correction(tree, p_values, child_ids, alpha=0.05, verbose=True)

    print("\n=== Asymmetric Tree Test Results ===")
    print(f"Child IDs: {child_ids}")
    print(f"Rejections: {result.reject}")

    # All should be rejected since all p-values are 0.01
    assert all(result.reject), "All hypotheses should be rejected"

    print("✓ Asymmetric tree test passed!")
    return result


def test_tree_bh_threshold_adjustment():
    """Test that thresholds are properly adjusted based on ancestor rejections."""
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "L1a"),
            ("root", "L1b"),
            ("L1a", "L2a"),
            ("L1a", "L2b"),
        ]
    )

    child_ids = ["L1a", "L1b", "L2a", "L2b"]
    # Only L1a significant, L1b not
    p_values = np.array([0.01, 0.2, 0.03, 0.03])

    result = tree_bh_correction(tree, p_values, child_ids, alpha=0.05, verbose=True)

    print("\n=== Threshold Adjustment Test Results ===")
    print(f"Family results: {result.family_results}")

    # L1a rejected, L1b not
    assert result.reject[0], "L1a should be rejected"
    assert not result.reject[1], "L1b should not be rejected"

    # For L2a, L2b: parent is L1a which was rejected
    # Threshold = 0.05 × (1/2) = 0.025 (only 1 of 2 level-1 rejected)
    # p=0.03 > 0.025, so should NOT be rejected with adjusted threshold

    # Check the family threshold
    if "L1a" in result.family_results:
        family_info = result.family_results["L1a"]
        print(f"L1a family adjusted_alpha: {family_info['adjusted_alpha']:.4f}")
        # Should be 0.05 × 0.5 = 0.025
        assert abs(family_info["adjusted_alpha"] - 0.025) < 0.001

    print("✓ Threshold adjustment test passed!")
    return result


def test_tree_bh_vs_flat_bh():
    """Compare TreeBH with flat BH on the same data."""
    tree, child_ids, p_values = _create_test_tree_simple()

    # TreeBH
    tree_result = tree_bh_correction(tree, p_values, child_ids, alpha=0.05)

    # Flat BH
    flat_reject, flat_adjusted, _, _ = multipletests(
        p_values, alpha=0.05, method="fdr_bh"
    )

    print("\n=== TreeBH vs Flat BH Comparison ===")
    print(f"p-values:        {p_values}")
    print(f"TreeBH reject:   {tree_result.reject}")
    print(f"Flat BH reject:  {flat_reject}")
    print(f"TreeBH adj_p:    {tree_result.adjusted_p}")
    print(f"Flat BH adj_p:   {flat_adjusted}")

    # They may differ - TreeBH can be more or less conservative depending on structure
    print(f"\nTreeBH rejections: {sum(tree_result.reject)}")
    print(f"Flat BH rejections: {sum(flat_reject)}")

    print("✓ Comparison complete!")


def run_all_tests():
    """Run all TreeBH tests."""
    print("=" * 60)
    print("TESTING TreeBH IMPLEMENTATION")
    print("=" * 60)

    test_tree_bh_simple()
    test_tree_bh_asymmetric()
    test_tree_bh_threshold_adjustment()
    test_tree_bh_vs_flat_bh()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
