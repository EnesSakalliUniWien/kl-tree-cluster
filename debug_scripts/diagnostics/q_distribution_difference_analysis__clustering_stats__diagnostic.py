"""
Purpose: Analyze distribution differences between parent and child nodes.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/diagnostics/q_distribution_difference_analysis__clustering_stats__diagnostic.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def analyze_differences(tree, stats_df, entropy_label):
    """Analyze distribution differences for edges in the tree."""
    print(f"\n{'=' * 60}")
    print(f"Distribution Difference Analysis: entropy={entropy_label}")
    print("=" * 60)

    diff_l1_norms = []  # Sum of |θ_child - θ_parent|
    diff_l2_norms = []  # ||θ_child - θ_parent||₂
    max_diffs = []  # Max |θ_child - θ_parent| per edge

    for parent, child in tree.edges():
        child_dist = tree.nodes[child].get("distribution")
        parent_dist = tree.nodes[parent].get("distribution")

        if child_dist is None or parent_dist is None:
            continue

        diff = np.abs(np.asarray(child_dist) - np.asarray(parent_dist))
        diff_l1_norms.append(np.sum(diff))
        diff_l2_norms.append(np.sqrt(np.sum(diff**2)))
        max_diffs.append(np.max(diff))

    diff_l1 = np.array(diff_l1_norms)
    diff_l2 = np.array(diff_l2_norms)
    max_d = np.array(max_diffs)

    print(f"\nL1 norm of (θ_child - θ_parent):")
    print(f"  Min:    {diff_l1.min():.4f}")
    print(f"  Median: {np.median(diff_l1):.4f}")
    print(f"  Mean:   {diff_l1.mean():.4f}")
    print(f"  Max:    {diff_l1.max():.4f}")

    print(f"\nL2 norm of (θ_child - θ_parent):")
    print(f"  Min:    {diff_l2.min():.4f}")
    print(f"  Median: {np.median(diff_l2):.4f}")
    print(f"  Mean:   {diff_l2.mean():.4f}")
    print(f"  Max:    {diff_l2.max():.4f}")

    print(f"\nMax feature difference per edge:")
    print(f"  Min:    {max_d.min():.4f}")
    print(f"  Median: {np.median(max_d):.4f}")
    print(f"  Mean:   {max_d.mean():.4f}")
    print(f"  Max:    {max_d.max():.4f}")


def analyze_cluster_purity(tree, stats_df, labels, entropy_label):
    """Analyze how pure the subtrees are with respect to true labels."""
    print(f"\nCluster Purity Analysis:")

    # Get leaf labels
    leaves = [n for n in tree.nodes() if tree.nodes[n].get("is_leaf", False)]
    leaf_labels = {
        tree.nodes[n].get("label"): labels.get(tree.nodes[n].get("label"))
        for n in leaves
    }

    # For each internal node, compute entropy of true labels in its subtree
    def get_subtree_leaves(node):
        if tree.nodes[node].get("is_leaf", False):
            return [tree.nodes[node].get("label")]
        children = list(tree.successors(node))
        result = []
        for c in children:
            result.extend(get_subtree_leaves(c))
        return result

    purities = []
    for node in tree.nodes():
        if tree.nodes[node].get("is_leaf", False):
            continue
        subtree_leaves = get_subtree_leaves(node)
        if len(subtree_leaves) < 2:
            continue

        # Get true labels for subtree
        true_labels = [leaf_labels.get(l) for l in subtree_leaves]
        true_labels = [l for l in true_labels if l is not None]

        if not true_labels:
            continue

        # Compute purity = fraction of most common label
        from collections import Counter

        counts = Counter(true_labels)
        purity = counts.most_common(1)[0][1] / len(true_labels)
        purities.append(purity)

    purities = np.array(purities)
    print(f"  Subtree purity (fraction of dominant label):")
    print(f"    Min:    {purities.min():.3f}")
    print(f"    Median: {np.median(purities):.3f}")
    print(f"    Mean:   {purities.mean():.3f}")
    print(f"    Pure (>0.95): {(purities > 0.95).sum()} / {len(purities)}")
    print(f"    Mixed (<0.7): {(purities < 0.7).sum()} / {len(purities)}")


def main():
    np.random.seed(42)

    for entropy in [0.1, 0.2, 0.3]:
        print(f"\n{'#' * 70}")
        print(f"# ENTROPY = {entropy}")
        print("#" * 70)

        data_dict, labels = generate_random_feature_matrix(
            n_rows=200, n_cols=50, n_clusters=3, entropy_param=entropy, random_seed=42
        )
        df = pd.DataFrame.from_dict(data_dict, orient="index")

        # Build tree
        Z = linkage(pdist(df.values, metric="hamming"), method="average")
        tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())

        # Populate distributions
        tree.populate_node_divergences(df)

        analyze_differences(tree, tree.stats_df, entropy)
        analyze_cluster_purity(tree, tree.stats_df, labels, entropy)


if __name__ == "__main__":
    main()
