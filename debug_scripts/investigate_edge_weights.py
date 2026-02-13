#!/usr/bin/env python3
"""
Explore Edge Weights and MRCA Distances
========================================

This script investigates using actual edge weights (distances) in the tree:
1. Edge weight from parent to each child
2. Distance from MRCA (parent) to each child
3. Sum of sibling edge weights
4. Difference in sibling distances (asymmetry)
5. Path lengths to leaves

The goal is to find if edge weights can predict true cluster boundaries.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from benchmarks.shared.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def compute_edge_based_metrics(tree, stats_df):
    """
    Compute edge-based metrics for each internal node.

    For each internal node (MRCA of its children):
    - left_edge_weight: weight of edge to left child
    - right_edge_weight: weight of edge to right child
    - total_edge_weight: sum of both edge weights
    - edge_weight_diff: |left - right| (asymmetry)
    - edge_weight_ratio: max/min ratio
    - path_to_leaves_left: total path length to all leaves under left child
    - path_to_leaves_right: total path length to all leaves under right child
    """

    def get_path_length_to_leaves(tree, node_id, accumulated=0):
        """Recursively compute sum of path lengths to all descendant leaves."""
        children = list(tree.successors(node_id))
        if len(children) == 0:  # leaf
            return accumulated

        total = 0
        for child_id in children:
            # Get edge weight
            edge_data = tree.get_edge_data(node_id, child_id)
            edge_weight = edge_data.get("weight", 0) if edge_data else 0

            # For leaves, the "edge weight" is the parent height (since leaf height = 0)
            if tree.nodes[child_id].get("is_leaf", False):
                child_height = 0
            else:
                child_height = tree.nodes[child_id].get("height", 0)

            node_height = tree.nodes[node_id].get("height", 0)
            actual_edge_length = node_height - child_height

            total += get_path_length_to_leaves(
                tree, child_id, accumulated + actual_edge_length
            )

        return total

    def count_leaves(tree, node_id):
        """Count leaves under a node."""
        if tree.nodes[node_id].get("is_leaf", False):
            return 1
        return sum(count_leaves(tree, c) for c in tree.successors(node_id))

    edge_metrics = []

    for node_id in stats_df.index:
        if stats_df.loc[node_id, "is_leaf"]:
            edge_metrics.append(
                {
                    "node_id": node_id,
                    "left_edge_weight": np.nan,
                    "right_edge_weight": np.nan,
                    "total_edge_weight": np.nan,
                    "edge_weight_diff": np.nan,
                    "edge_weight_ratio": np.nan,
                    "left_path_to_leaves": np.nan,
                    "right_path_to_leaves": np.nan,
                    "path_asymmetry": np.nan,
                    "avg_edge_to_child": np.nan,
                    "left_n_leaves": np.nan,
                    "right_n_leaves": np.nan,
                }
            )
            continue

        children = list(tree.successors(node_id))
        if len(children) != 2:
            edge_metrics.append(
                {
                    "node_id": node_id,
                    "left_edge_weight": np.nan,
                    "right_edge_weight": np.nan,
                    "total_edge_weight": np.nan,
                    "edge_weight_diff": np.nan,
                    "edge_weight_ratio": np.nan,
                    "left_path_to_leaves": np.nan,
                    "right_path_to_leaves": np.nan,
                    "path_asymmetry": np.nan,
                    "avg_edge_to_child": np.nan,
                    "left_n_leaves": np.nan,
                    "right_n_leaves": np.nan,
                }
            )
            continue

        left_id, right_id = children
        node_height = tree.nodes[node_id].get("height", 0)

        # Get actual edge weights (distance from MRCA to each child)
        left_height = (
            0
            if tree.nodes[left_id].get("is_leaf", False)
            else tree.nodes[left_id].get("height", 0)
        )
        right_height = (
            0
            if tree.nodes[right_id].get("is_leaf", False)
            else tree.nodes[right_id].get("height", 0)
        )

        left_edge = node_height - left_height
        right_edge = node_height - right_height

        # Path lengths to leaves
        left_path = get_path_length_to_leaves(tree, left_id, 0)
        right_path = get_path_length_to_leaves(tree, right_id, 0)

        # Number of leaves
        left_n = count_leaves(tree, left_id)
        right_n = count_leaves(tree, right_id)

        edge_metrics.append(
            {
                "node_id": node_id,
                "left_edge_weight": left_edge,
                "right_edge_weight": right_edge,
                "total_edge_weight": left_edge + right_edge,
                "edge_weight_diff": abs(left_edge - right_edge),
                "edge_weight_ratio": max(left_edge, right_edge)
                / (min(left_edge, right_edge) + 1e-10),
                "left_path_to_leaves": left_path,
                "right_path_to_leaves": right_path,
                "path_asymmetry": abs(left_path - right_path),
                "avg_edge_to_child": (left_edge + right_edge) / 2,
                "left_n_leaves": left_n,
                "right_n_leaves": right_n,
            }
        )

    edge_df = pd.DataFrame(edge_metrics).set_index("node_id")
    return stats_df.join(edge_df)


def generate_and_analyze(
    n_clusters=4, n_per_cluster=50, n_features=100, divergence=0.3, seed=42
):
    """Generate data and analyze edge-based metrics."""

    # Generate phylogenetic data
    sample_dict, cluster_assignments, distributions, metadata = (
        generate_phylogenetic_data(
            n_taxa=n_clusters,
            n_features=n_features,
            n_categories=4,
            samples_per_taxon=n_per_cluster,
            mutation_rate=divergence,
            random_seed=seed,
        )
    )

    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])

    # Build linkage and tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Convert to probability format
    K = int(data.max()) + 1
    n_samples, n_feats = data.shape
    prob_data = np.zeros((n_samples, n_feats * K))
    for i in range(n_samples):
        for j in range(n_feats):
            prob_data[i, j * K + data[i, j]] = 1.0
    prob_df = pd.DataFrame(prob_data, index=sample_names)

    # Run decomposition
    tree.decompose(leaf_data=prob_df)
    stats_df = tree.stats_df.copy()

    # Add edge-based metrics
    stats_df = compute_edge_based_metrics(tree, stats_df)

    # Add true cluster information
    leaf_labels = {name: lbl for name, lbl in zip(sample_names, labels)}
    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf"):
            sample_name = tree.nodes[node_id].get("label")
            if sample_name and sample_name in leaf_labels:
                leaf_labels[node_id] = leaf_labels[sample_name]

    def get_leaves_under(tree, node_id):
        if tree.out_degree(node_id) == 0:
            return [node_id]
        leaves = []
        for child in tree.successors(node_id):
            leaves.extend(get_leaves_under(tree, child))
        return leaves

    true_info = []
    for node_id in stats_df.index:
        if stats_df.loc[node_id, "is_leaf"]:
            n_true = 1
            should_split = False
        else:
            leaves = get_leaves_under(tree, node_id)
            leaf_lbls = [leaf_labels.get(l) for l in leaves if l in leaf_labels]
            n_true = len(set(leaf_lbls)) if leaf_lbls else 0
            should_split = n_true > 1
        true_info.append(
            {
                "node_id": node_id,
                "n_true_clusters": n_true,
                "should_split": should_split,
            }
        )

    true_df = pd.DataFrame(true_info).set_index("node_id")
    stats_df = stats_df.join(true_df)

    return stats_df, tree


def analyze_edge_metrics(stats_df):
    """Analyze predictive power of edge-based metrics."""

    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split"])

    print("=" * 90)
    print("EDGE WEIGHT / MRCA DISTANCE ANALYSIS")
    print("=" * 90)
    print(
        f"\nInternal nodes: {len(internal)}, Should split: {internal['should_split'].sum()}"
    )

    metrics = [
        ("height", "Height (merge distance)"),
        ("left_edge_weight", "Left edge weight (MRCA → left child)"),
        ("right_edge_weight", "Right edge weight (MRCA → right child)"),
        ("total_edge_weight", "Total edge weight (sum of both)"),
        ("avg_edge_to_child", "Avg edge to child"),
        ("edge_weight_diff", "Edge weight difference (|L-R|)"),
        ("edge_weight_ratio", "Edge weight ratio (max/min)"),
        ("left_path_to_leaves", "Path length to left leaves"),
        ("right_path_to_leaves", "Path length to right leaves"),
        ("path_asymmetry", "Path asymmetry (|L-R|)"),
        ("sibling_branch_sum", "Sibling branch sum (existing)"),
        ("branch_length", "Branch length to parent"),
    ]

    print("\n" + "-" * 90)
    print(
        f"{'Metric':<40} {'AUC':>7} {'Corr':>7} {'d':>7} {'Mean(T)':>10} {'Mean(F)':>10}"
    )
    print("-" * 90)

    results = []

    for col, name in metrics:
        if col not in internal.columns:
            continue

        valid = internal.dropna(subset=[col])
        valid = valid[np.isfinite(valid[col])]

        if len(valid) < 10:
            continue

        true_split = valid[valid["should_split"] == True][col]
        no_split = valid[valid["should_split"] == False][col]

        if len(true_split) < 3 or len(no_split) < 3:
            continue

        try:
            auc = roc_auc_score(valid["should_split"], valid[col])
            corr, _ = stats.pointbiserialr(
                valid["should_split"].astype(int), valid[col]
            )
        except:
            auc, corr = np.nan, np.nan

        pooled_std = np.sqrt((true_split.std() ** 2 + no_split.std() ** 2) / 2)
        d = (true_split.mean() - no_split.mean()) / (pooled_std + 1e-10)

        print(
            f"{name:<40} {auc:>7.3f} {corr:>7.3f} {d:>7.3f} {true_split.mean():>10.4f} {no_split.mean():>10.4f}"
        )

        results.append(
            {
                "metric": col,
                "name": name,
                "auc": auc,
                "corr": corr,
                "d": d,
                "mean_true": true_split.mean(),
                "mean_false": no_split.mean(),
            }
        )

    return pd.DataFrame(results)


def explore_derived_metrics(stats_df):
    """Explore derived metrics that might be more useful."""

    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split", "height", "total_edge_weight"])

    print("\n" + "=" * 90)
    print("DERIVED METRICS (combinations)")
    print("=" * 90)

    # Derived metrics
    # 1. Height normalized by total edge weight to children
    internal["height_over_total_edge"] = internal["height"] / (
        internal["total_edge_weight"] + 1e-10
    )

    # 2. "Jump" = height minus avg child edge (how much this merge stands out)
    internal["height_jump"] = internal["height"] - internal["avg_edge_to_child"]

    # 3. Relative edge: total edge / height
    internal["edge_fraction"] = internal["total_edge_weight"] / (
        internal["height"] + 1e-10
    )

    # 4. Normalized asymmetry: edge diff / total edge
    internal["norm_asymmetry"] = internal["edge_weight_diff"] / (
        internal["total_edge_weight"] + 1e-10
    )

    # 5. Path per leaf (normalized by tree size)
    internal["left_path_per_leaf"] = internal["left_path_to_leaves"] / (
        internal["left_n_leaves"] + 1e-10
    )
    internal["right_path_per_leaf"] = internal["right_path_to_leaves"] / (
        internal["right_n_leaves"] + 1e-10
    )
    internal["avg_path_per_leaf"] = (
        internal["left_path_per_leaf"] + internal["right_path_per_leaf"]
    ) / 2

    derived = [
        ("height_over_total_edge", "Height / total_edge_weight"),
        ("height_jump", "Height - avg_edge_to_child"),
        ("edge_fraction", "Total edge / height"),
        ("norm_asymmetry", "Normalized asymmetry"),
        ("avg_path_per_leaf", "Avg path length per leaf"),
    ]

    print(f"\n{'Derived Metric':<35} {'AUC':>7} {'Corr':>7}")
    print("-" * 55)

    for col, name in derived:
        valid = internal.dropna(subset=[col])
        valid = valid[np.isfinite(valid[col])]

        if len(valid) < 10:
            continue

        try:
            auc = roc_auc_score(valid["should_split"], valid[col])
            corr, _ = stats.pointbiserialr(
                valid["should_split"].astype(int), valid[col]
            )
            print(f"{name:<35} {auc:>7.3f} {corr:>7.3f}")
        except Exception as e:
            print(f"{name:<35} Error: {e}")


def show_example_nodes(stats_df):
    """Show example nodes with edge metrics."""

    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split", "total_edge_weight"])
    internal = internal.sort_values("height", ascending=False)

    print("\n" + "=" * 90)
    print("EXAMPLE NODES (high height)")
    print("=" * 90)

    cols = [
        "should_split",
        "height",
        "left_edge_weight",
        "right_edge_weight",
        "total_edge_weight",
        "edge_weight_diff",
        "left_n_leaves",
        "right_n_leaves",
    ]

    print("\nTop 10 highest nodes:")
    print(internal[cols].head(10).to_string())

    print("\n\nNodes that SHOULD split (highest 5):")
    should = internal[internal["should_split"] == True]
    print(should[cols].head(5).to_string())

    print("\nNodes that should NOT split (highest 5):")
    shouldnt = internal[internal["should_split"] == False]
    print(shouldnt[cols].head(5).to_string())


def main():
    print("=" * 90)
    print("EDGE WEIGHTS AND MRCA DISTANCE INVESTIGATION")
    print("=" * 90)
    print("""
This script analyzes actual edge weights (distances) in the dendrogram:
- Edge weight = distance from parent (MRCA) to child
- For a node at height h with children at heights h_L and h_R:
  - Left edge weight = h - h_L
  - Right edge weight = h - h_R
  - (For leaf children, child height = 0)
""")

    scenarios = [
        (4, 50, 100, 0.3, "Base case"),
        (8, 25, 100, 0.3, "8 clusters"),
        (4, 50, 100, 0.1, "Low divergence"),
    ]

    all_results = []

    for n_clusters, n_per, n_feat, div, desc in scenarios:
        print(f"\n\n{'#' * 90}")
        print(f"# SCENARIO: {desc}")
        print(f"# Clusters={n_clusters}, Samples/cluster={n_per}, Divergence={div}")
        print(f"{'#' * 90}")

        stats_df, tree = generate_and_analyze(
            n_clusters=n_clusters,
            n_per_cluster=n_per,
            n_features=n_feat,
            divergence=div,
            seed=42,
        )

        results = analyze_edge_metrics(stats_df)
        results["scenario"] = desc
        all_results.append(results)

        if desc == "Base case":
            explore_derived_metrics(stats_df)
            show_example_nodes(stats_df)

    # Summary
    all_df = pd.concat(all_results, ignore_index=True)

    print("\n\n" + "=" * 90)
    print("SUMMARY: MEAN AUC ACROSS SCENARIOS")
    print("=" * 90)

    summary = (
        all_df.groupby("metric")["auc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
EDGE WEIGHT METRICS:

1. LEFT/RIGHT EDGE WEIGHT = distance from MRCA (parent) to each child
   - This is: parent_height - child_height
   - For leaf children: parent_height (since leaf height = 0)

2. TOTAL EDGE WEIGHT = left_edge + right_edge
   - This equals: 2*parent_height - left_height - right_height
   - For siblings where both are leaves: 2 * parent_height

3. EDGE WEIGHT DIFF = |left_edge - right_edge|
   - Measures asymmetry of the merge
   
KEY INSIGHT:
- High height = late merge = likely cluster boundary ✓
- Edge weights are DERIVED from heights
- The predictive power comes from HEIGHT, not edge weights per se
""")


if __name__ == "__main__":
    main()
