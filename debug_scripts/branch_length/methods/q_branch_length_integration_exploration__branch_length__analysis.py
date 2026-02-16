"""
Purpose: Exploring How to Leverage Branch Length Information.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_integration_exploration__branch_length__analysis.py
"""

#!/usr/bin/env python3
"""
Exploring How to Leverage Branch Length Information
====================================================

The literature (TreeCluster, PhytClust) suggests three model-free approaches:

1. SUM-LENGTH: Sum of branch lengths in subtree < threshold
2. MAX-DIAMETER: Max pairwise distance in cluster < threshold
3. RELATIVE RATIO: Divergence / Branch Length ratio

This script explores how each would work with your current framework.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
import networkx as nx


# =============================================================================
# PART 1: Understanding Current Tree Structure
# =============================================================================


def get_children(tree: PosetTree, node_id: str) -> list:
    """Get children of a node in PosetTree."""
    return list(tree.successors(node_id))


def is_leaf(tree: PosetTree, node_id: str) -> bool:
    """Check if node is a leaf."""
    return tree.out_degree(node_id) == 0


def get_height(tree: PosetTree, node_id: str) -> float:
    """Get height attribute of a node."""
    return tree.nodes[node_id].get("height", 0)


def get_all_internal_nodes(tree: PosetTree) -> list:
    """Get all internal (non-leaf) nodes."""
    return [n for n in tree.nodes if not is_leaf(tree, n)]


def get_leaves_under(tree: PosetTree, node_id: str) -> list:
    """Get all leaves under a node."""
    if is_leaf(tree, node_id):
        return [node_id]
    return [n for n in nx.descendants(tree, node_id) if is_leaf(tree, n)]


def analyze_tree_branch_structure(tree: PosetTree) -> pd.DataFrame:
    """Extract branch length structure from tree."""
    records = []

    for node_id in get_all_internal_nodes(tree):
        # Get node's height (merge distance)
        node_height = get_height(tree, node_id)

        # Get children info
        children = get_children(tree, node_id)
        if len(children) != 2:
            continue

        left_id, right_id = children
        left_height = get_height(tree, left_id)
        right_height = get_height(tree, right_id)

        # Branch lengths to children
        bl_left = node_height - left_height
        bl_right = node_height - right_height

        # Subtree sizes
        left_leaves = get_leaves_under(tree, left_id)
        right_leaves = get_leaves_under(tree, right_id)

        records.append(
            {
                "node_id": node_id,
                "node_height": node_height,
                "left_height": left_height,
                "right_height": right_height,
                "branch_left": bl_left,
                "branch_right": bl_right,
                "total_branch": bl_left + bl_right,
                "n_left": len(left_leaves),
                "n_right": len(right_leaves),
                "n_total": len(left_leaves) + len(right_leaves),
            }
        )

    return pd.DataFrame(records)


def compute_subtree_branch_sum(tree: PosetTree, node_id: str) -> float:
    """Compute sum of all branch lengths in subtree (TreeCluster Sum-length)."""
    if is_leaf(tree, node_id):
        return 0

    total = 0
    node_height = get_height(tree, node_id)

    for child_id in get_children(tree, node_id):
        child_height = get_height(tree, child_id)
        branch_to_child = node_height - child_height
        total += branch_to_child + compute_subtree_branch_sum(tree, child_id)

    return total


def compute_max_diameter(tree: PosetTree, node_id: str) -> float:
    """Compute max pairwise distance in subtree (TreeCluster Max-diameter).

    For efficiency, uses the fact that in a tree, max diameter =
    height of node * 2 (approximately, for ultrametric trees).
    For non-ultrametric: need to track max distance from node to any leaf.
    """
    leaves = get_leaves_under(tree, node_id)
    if len(leaves) <= 1:
        return 0

    # Use 2 * node_height as approximation
    return 2 * get_height(tree, node_id)


# =============================================================================
# PART 2: Simulate and Compare Approaches
# =============================================================================


def generate_hierarchical_clusters(
    n_clusters: int = 4,
    n_per_cluster: int = 50,
    n_features: int = 100,
    between_cluster_div: float = 0.3,
    within_cluster_div: float = 0.05,
    K: int = 4,
    seed: int = 42,
):
    """Generate data with known hierarchical structure."""
    rng = np.random.RandomState(seed)

    # Create cluster ancestors with hierarchical structure
    # Level 0: root ancestor
    root_ancestor = rng.randint(0, K, size=n_features)

    # Level 1: two super-clusters
    def mutate(seq, div, rng):
        n_mut = int(div * len(seq))
        mut_seq = seq.copy()
        sites = rng.choice(len(seq), n_mut, replace=False)
        for s in sites:
            options = [c for c in range(K) if c != seq[s]]
            mut_seq[s] = rng.choice(options)
        return mut_seq

    super_A = mutate(root_ancestor, between_cluster_div, rng)
    super_B = mutate(root_ancestor, between_cluster_div, rng)

    # Level 2: individual clusters
    cluster_ancestors = []
    labels = []

    for i in range(n_clusters // 2):
        cluster_ancestors.append(mutate(super_A, between_cluster_div / 2, rng))
    for i in range(n_clusters // 2, n_clusters):
        cluster_ancestors.append(mutate(super_B, between_cluster_div / 2, rng))

    # Generate samples
    all_samples = []
    for c_idx, ancestor in enumerate(cluster_ancestors):
        for _ in range(n_per_cluster):
            sample = mutate(ancestor, within_cluster_div, rng)
            all_samples.append(sample)
            labels.append(c_idx)

    data = np.array(all_samples)
    labels = np.array(labels)

    return data, labels


def build_tree_with_heights(data: np.ndarray) -> PosetTree:
    """Build PosetTree from data with height information."""
    # Compute pairwise Hamming distances
    distances = pdist(data, metric="hamming")

    # Build linkage
    Z = linkage(distances, method="weighted")

    # Create tree
    tree = PosetTree.from_linkage(Z)

    return tree


def compare_split_decisions(
    tree: PosetTree,
    data: np.ndarray,
    true_labels: np.ndarray,
    sum_threshold: float,
    diameter_threshold: float,
):
    """Compare different split decision criteria."""

    # Build leaf position map (leaf node_id -> data index)
    all_leaves = tree.get_leaves(return_labels=False)
    leaf_positions = {}
    for leaf_id in all_leaves:
        # The label attribute contains the original index
        label = tree.nodes[leaf_id].get("label", leaf_id)
        try:
            leaf_positions[leaf_id] = int(label)
        except (ValueError, TypeError):
            # If label is not an int, try to parse it
            leaf_positions[leaf_id] = all_leaves.index(leaf_id)

    results = []

    for node_id in get_all_internal_nodes(tree):
        children = get_children(tree, node_id)
        if len(children) != 2:
            continue

        # Get node properties
        node_height = get_height(tree, node_id)
        subtree_sum = compute_subtree_branch_sum(tree, node_id)
        max_diam = compute_max_diameter(tree, node_id)

        # Get true cluster composition
        leaves = get_leaves_under(tree, node_id)
        leaf_labels = []
        for leaf_id in leaves:
            if leaf_id in leaf_positions:
                idx = leaf_positions[leaf_id]
                if idx < len(true_labels):
                    leaf_labels.append(true_labels[idx])

        n_unique_clusters = len(set(leaf_labels)) if leaf_labels else 0

        # Is this a "real" split? (subtree contains multiple true clusters)
        should_split = n_unique_clusters > 1

        # TreeCluster Sum-length decision
        sum_decision = subtree_sum > sum_threshold

        # TreeCluster Max-diameter decision
        diam_decision = max_diam > diameter_threshold

        # Branch length between siblings
        left_id, right_id = children
        left_h = get_height(tree, left_id)
        right_h = get_height(tree, right_id)
        sibling_branch = (node_height - left_h) + (node_height - right_h)

        results.append(
            {
                "node_id": node_id,
                "height": node_height,
                "subtree_sum": subtree_sum,
                "max_diameter": max_diam,
                "sibling_branch": sibling_branch,
                "n_leaves": len(leaves),
                "n_true_clusters": n_unique_clusters,
                "should_split": should_split,
                "sum_decision": sum_decision,
                "diam_decision": diam_decision,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# PART 3: Key Integration Patterns
# =============================================================================


def demonstrate_integration_patterns():
    """Show how branch length can integrate with existing framework."""

    print("=" * 80)
    print("HOW TO LEVERAGE BRANCH LENGTH INFORMATION")
    print("=" * 80)

    print("""
CURRENT APPROACH:
-----------------
For each node, decide split using:
  1. Binary structure check
  2. Child-parent divergence test (statistical)
  3. Sibling divergence test (statistical)

INTEGRATION OPTIONS:
--------------------

OPTION A: REPLACE Statistical Tests with Branch Length
-------------------------------------------------------
Use TreeCluster's Sum-length criterion:
  
  def should_split(node):
      subtree_branch_sum = compute_subtree_branch_sum(node)
      return subtree_branch_sum > THRESHOLD

Pros: Model-free, O(n), proven effective
Cons: Need to choose threshold, loses statistical rigor

OPTION B: ADD Branch Length as Pre-Filter
-----------------------------------------
Only run expensive statistical tests if branch length warrants:

  def should_split(node):
      sibling_branch = get_sibling_branch_length(node)
      
      # Skip if branches too short (no signal possible)
      if sibling_branch < MIN_BRANCH:
          return False
      
      # Skip if branches too long (definitely different)
      if sibling_branch > MAX_BRANCH:
          return True
          
      # Middle ground: use statistical tests
      return run_statistical_tests(node)

Pros: Faster, avoids unreliable tests at extremes
Cons: Need to tune MIN/MAX thresholds

OPTION C: USE Branch Length to Calibrate Tests
----------------------------------------------
What we explored earlier - adjust expected divergence based on branch length:

  def should_split(node):
      D_obs = compute_sibling_divergence(node)
      b = get_sibling_branch_length(node)
      D_expected = expected_divergence(b)  # From calibration
      
      return D_obs > D_expected + k * SE(D)

Pros: Principled, accounts for evolutionary drift
Cons: Model-dependent (calibration specific to data type)

OPTION D: USE Branch Length as Validity Index (PhytClust)
---------------------------------------------------------
Don't fix number of clusters - find optimal k by minimizing intra-cluster
branch length:

  def find_optimal_clustering(tree):
      for k in range(2, max_k):
          clusters = cut_tree_into_k_clusters(tree, k)
          score = sum(intra_cluster_branch_sum(c) for c in clusters)
          validity = compute_validity_index(k, score)
      return best_k

Pros: Threshold-free, finds natural cluster number
Cons: More complex, may not match your decomposition framework

RECOMMENDATION:
---------------
Start with OPTION B (branch length as pre-filter):

1. It's model-free
2. It integrates naturally with your existing gates
3. It provides computational speedup
4. It avoids unreliable statistical tests at extremes

The key insight from the literature:
- Short branches → siblings are similar → DON'T SPLIT (save compute)
- Long branches → siblings definitely differ → SPLIT (skip tests)
- Medium branches → use statistical tests (current approach)
""")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("EXPLORING BRANCH LENGTH INTEGRATION")
    print("=" * 80)

    # Generate test data
    print("\n--- Generating hierarchical cluster data ---")
    data, labels = generate_hierarchical_clusters(
        n_clusters=4,
        n_per_cluster=50,
        n_features=100,
        between_cluster_div=0.3,
        within_cluster_div=0.05,
        seed=42,
    )
    print(f"Data shape: {data.shape}")
    print(f"True clusters: {len(np.unique(labels))}")

    # Build tree
    print("\n--- Building tree ---")
    tree = build_tree_with_heights(data)

    # Analyze branch structure
    print("\n--- Branch Structure Analysis ---")
    branch_df = analyze_tree_branch_structure(tree)
    print(f"\nBranch length statistics:")
    print(
        branch_df[
            ["node_height", "branch_left", "branch_right", "total_branch"]
        ].describe()
    )

    # Compare split decisions
    print("\n--- Comparing Split Decision Criteria ---")

    # Use median as threshold (automatic)
    sum_threshold = branch_df["total_branch"].median() * 2
    diam_threshold = branch_df["node_height"].median() * 2

    print(f"Sum threshold (auto): {sum_threshold:.4f}")
    print(f"Diameter threshold (auto): {diam_threshold:.4f}")

    comparison_df = compare_split_decisions(
        tree,
        data,
        labels,
        sum_threshold=sum_threshold,
        diameter_threshold=diam_threshold,
    )

    # Evaluate accuracy
    print("\n--- Decision Accuracy ---")

    for criterion in ["sum_decision", "diam_decision"]:
        correct = (comparison_df[criterion] == comparison_df["should_split"]).mean()
        tp = (
            (comparison_df[criterion] == True) & (comparison_df["should_split"] == True)
        ).sum()
        fp = (
            (comparison_df[criterion] == True)
            & (comparison_df["should_split"] == False)
        ).sum()
        fn = (
            (comparison_df[criterion] == False)
            & (comparison_df["should_split"] == True)
        ).sum()
        tn = (
            (comparison_df[criterion] == False)
            & (comparison_df["should_split"] == False)
        ).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\n{criterion}:")
        print(f"  Accuracy:  {correct:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Show relationship between branch length and true cluster boundaries
    print("\n--- Branch Length at True Cluster Boundaries ---")

    true_splits = comparison_df[comparison_df["should_split"] == True]
    false_splits = comparison_df[comparison_df["should_split"] == False]

    print(f"\nNodes that SHOULD split (n={len(true_splits)}):")
    print(f"  Mean sibling_branch: {true_splits['sibling_branch'].mean():.4f}")
    print(f"  Mean height:         {true_splits['height'].mean():.4f}")

    print(f"\nNodes that should NOT split (n={len(false_splits)}):")
    print(f"  Mean sibling_branch: {false_splits['sibling_branch'].mean():.4f}")
    print(f"  Mean height:         {false_splits['height'].mean():.4f}")

    # Compute separation
    if len(true_splits) > 0 and len(false_splits) > 0:
        sep = (
            true_splits["sibling_branch"].mean() - false_splits["sibling_branch"].mean()
        ) / (
            true_splits["sibling_branch"].std()
            + false_splits["sibling_branch"].std()
            + 1e-10
        )
        print(f"\nSeparation (Cohen's d): {sep:.2f}")

    # Show integration patterns
    demonstrate_integration_patterns()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY")
    print("=" * 80)
    print("""
Branch length provides a MODEL-FREE signal for cluster boundaries:

1. Nodes with LONG sibling branches → likely true cluster boundaries
2. Nodes with SHORT sibling branches → likely within-cluster variation

You can leverage this by:
- Using branch length as a PRE-FILTER before statistical tests
- Using it to SET EXPECTATIONS for divergence tests
- Using it as a VALIDITY INDEX to find optimal cluster count

The TreeCluster paper shows this works across:
- Microbiome OTU clustering
- HIV transmission clustering  
- Multiple sequence alignment

All without any model assumptions!
""")


if __name__ == "__main__":
    main()
