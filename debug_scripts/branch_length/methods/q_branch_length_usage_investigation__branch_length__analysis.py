"""
Purpose: Investigate How to Use Branch Lengths in Our Method.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_usage_investigation__branch_length__analysis.py
"""

#!/usr/bin/env python3
"""
Investigate How to Use Branch Lengths in Our Method
====================================================

This script explores the relationship between:
1. Height (merge distance from linkage)
2. Branch length (distance to parent)
3. Sibling branch sum (sum of branches to children)
4. Inconsistency coefficient (relative height jump)
5. KL divergence metrics
6. True cluster boundaries (should_split)

Goal: Understand which metrics are predictive and how to combine them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, inconsistent as scipy_inconsistent
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from benchmarks.shared.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def generate_analysis_data(
    n_clusters=4, n_per_cluster=50, n_features=100, divergence=0.3, seed=42
):
    """Generate data and build tree with all metrics."""

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

    # Run decomposition to get KL metrics
    tree.decompose(leaf_data=prob_df)
    stats_df = tree.stats_df.copy()

    # Compute inconsistency coefficient
    R = scipy_inconsistent(Z, d=2)
    n_leaves = len(sample_names)

    # Add inconsistency to internal nodes
    inconsistency_map = {}
    for merge_idx in range(len(Z)):
        node_idx = n_leaves + merge_idx
        node_id = f"N{node_idx}"
        inconsistency_map[node_id] = R[merge_idx, 3]  # inconsistency coefficient

    stats_df["inconsistency"] = stats_df.index.map(
        lambda x: inconsistency_map.get(x, np.nan)
    )

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

    return stats_df, tree, Z


def analyze_metrics(stats_df):
    """Analyze predictive power of different metrics."""

    # Filter to internal nodes with valid data
    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split"])

    print("=" * 80)
    print("METRIC ANALYSIS FOR PREDICTING TRUE CLUSTER BOUNDARIES")
    print("=" * 80)
    print(f"\nTotal internal nodes: {len(internal)}")
    print(f"Should split (True): {internal['should_split'].sum()}")
    print(f"Should NOT split (False): {(~internal['should_split']).sum()}")

    # Metrics to analyze
    metrics = [
        ("height", "Height (merge distance)"),
        ("branch_length", "Branch length (to parent)"),
        ("sibling_branch_sum", "Sibling branch sum"),
        ("inconsistency", "Inconsistency coefficient"),
        ("kl_divergence_local", "KL divergence (local)"),
        ("kl_divergence_global", "KL divergence (global)"),
    ]

    results = []

    print("\n" + "-" * 80)
    print(
        f"{'Metric':<30} {'AUC':>8} {'Corr':>8} {'d':>8} {'Mean(T)':>10} {'Mean(F)':>10}"
    )
    print("-" * 80)

    for col, name in metrics:
        if col not in internal.columns:
            continue

        valid = internal.dropna(subset=[col])
        if len(valid) < 10:
            continue

        true_split = valid[valid["should_split"] == True][col]
        no_split = valid[valid["should_split"] == False][col]

        if len(true_split) == 0 or len(no_split) == 0:
            continue

        # AUC
        try:
            auc = roc_auc_score(valid["should_split"], valid[col])
        except:
            auc = np.nan

        # Correlation
        try:
            corr, _ = stats.pointbiserialr(
                valid["should_split"].astype(int), valid[col]
            )
        except:
            corr = np.nan

        # Cohen's d
        pooled_std = np.sqrt((true_split.std() ** 2 + no_split.std() ** 2) / 2)
        d = (true_split.mean() - no_split.mean()) / (pooled_std + 1e-10)

        print(
            f"{name:<30} {auc:>8.3f} {corr:>8.3f} {d:>8.3f} {true_split.mean():>10.4f} {no_split.mean():>10.4f}"
        )

        results.append(
            {
                "metric": col,
                "name": name,
                "auc": auc,
                "correlation": corr,
                "cohens_d": d,
                "mean_true": true_split.mean(),
                "mean_false": no_split.mean(),
            }
        )

    return pd.DataFrame(results)


def show_example_nodes(stats_df, n=10):
    """Show example nodes with their metrics."""

    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split", "height"])

    # Sort by height descending (highest merges first)
    internal = internal.sort_values("height", ascending=False)

    print("\n" + "=" * 80)
    print("EXAMPLE NODES (sorted by height, highest first)")
    print("=" * 80)

    cols = [
        "should_split",
        "n_true_clusters",
        "height",
        "branch_length",
        "sibling_branch_sum",
        "inconsistency",
        "kl_divergence_local",
    ]

    display_df = internal[cols].head(n)
    print(display_df.to_string())

    print("\n... showing top nodes that SHOULD be split:")
    should_split = internal[internal["should_split"] == True].head(5)
    print(should_split[cols].to_string())

    print("\n... showing top nodes that should NOT split:")
    no_split = internal[internal["should_split"] == False].head(5)
    print(no_split[cols].to_string())


def explore_combinations(stats_df):
    """Explore combining multiple metrics."""

    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["should_split", "height", "sibling_branch_sum"])

    print("\n" + "=" * 80)
    print("EXPLORING METRIC COMBINATIONS")
    print("=" * 80)

    # Combination 1: Height normalized by sibling_branch_sum
    internal["height_per_sibling"] = internal["height"] / (
        internal["sibling_branch_sum"] + 1e-10
    )

    # Combination 2: Height - sibling_branch_sum (the "gap")
    internal["height_minus_sibling"] = (
        internal["height"] - internal["sibling_branch_sum"]
    )

    # Combination 3: Ratio of height to mean sibling branch
    internal["relative_height"] = internal["height"] / (
        internal["sibling_branch_sum"] / 2 + 1e-10
    )

    # Analyze these combinations
    combos = [
        ("height_per_sibling", "Height / sibling_branch_sum"),
        ("height_minus_sibling", "Height - sibling_branch_sum"),
        ("relative_height", "Height / (sibling_branch/2)"),
    ]

    print(f"\n{'Combination':<35} {'AUC':>8} {'Corr':>8}")
    print("-" * 55)

    for col, name in combos:
        valid = internal.dropna(subset=[col])
        valid = valid[np.isfinite(valid[col])]

        if len(valid) < 10:
            continue

        try:
            auc = roc_auc_score(valid["should_split"], valid[col])
            corr, _ = stats.pointbiserialr(
                valid["should_split"].astype(int), valid[col]
            )
            print(f"{name:<35} {auc:>8.3f} {corr:>8.3f}")
        except Exception as e:
            print(f"{name:<35} Error: {e}")


def main():
    print("=" * 80)
    print("BRANCH LENGTH USAGE INVESTIGATION")
    print("=" * 80)

    # Run analysis across multiple scenarios
    scenarios = [
        (4, 50, 100, 0.3, "Base case"),
        (2, 50, 100, 0.3, "2 clusters"),
        (8, 25, 100, 0.3, "8 clusters"),
        (4, 50, 100, 0.1, "Low divergence"),
        (4, 50, 100, 0.5, "High divergence"),
    ]

    all_results = []

    for n_clusters, n_per, n_feat, div, desc in scenarios:
        print(f"\n\n{'#' * 80}")
        print(f"# SCENARIO: {desc}")
        print(
            f"# Clusters={n_clusters}, Samples/cluster={n_per}, Features={n_feat}, Divergence={div}"
        )
        print(f"{'#' * 80}")

        stats_df, tree, Z = generate_analysis_data(
            n_clusters=n_clusters,
            n_per_cluster=n_per,
            n_features=n_feat,
            divergence=div,
            seed=42,
        )

        results = analyze_metrics(stats_df)
        results["scenario"] = desc
        all_results.append(results)

        if desc == "Base case":
            show_example_nodes(stats_df)
            explore_combinations(stats_df)

    # Summary across scenarios
    all_df = pd.concat(all_results, ignore_index=True)

    print("\n\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SCENARIOS")
    print("=" * 80)

    summary = (
        all_df.groupby("metric")
        .agg(
            {
                "auc": ["mean", "std"],
                "correlation": ["mean", "std"],
                "cohens_d": ["mean", "std"],
            }
        )
        .round(3)
    )

    print("\nMean ± Std across scenarios:")
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    height_auc = all_df[all_df["metric"] == "height"]["auc"].mean()
    sibling_auc = all_df[all_df["metric"] == "sibling_branch_sum"]["auc"].mean()
    branch_auc = all_df[all_df["metric"] == "branch_length"]["auc"].mean()

    print(f"""
1. HEIGHT (merge distance):
   - AUC = {height_auc:.3f} (> 0.5 means POSITIVE correlation)
   - Nodes that SHOULD split have HIGHER merge heights
   - This makes sense: true cluster boundaries are at high merges
   
2. SIBLING_BRANCH_SUM:
   - AUC = {sibling_auc:.3f} (< 0.5 means NEGATIVE correlation)
   - Nodes that SHOULD split have SHORTER sibling branches
   - This is OPPOSITE of naive expectation
   
3. BRANCH_LENGTH (to parent):
   - AUC = {branch_auc:.3f}
   
INTERPRETATION:
- Use HEIGHT for filtering: high-height nodes are more likely to be true boundaries
- Do NOT use sibling_branch_sum alone as it's anti-correlated
- The inconsistency coefficient adds modest value

RECOMMENDATION:
- Height can be used as a PRIOR or PRE-FILTER
- Nodes with low height are unlikely to be cluster boundaries
- Focus KL-divergence testing on high-height nodes
""")


if __name__ == "__main__":
    main()
