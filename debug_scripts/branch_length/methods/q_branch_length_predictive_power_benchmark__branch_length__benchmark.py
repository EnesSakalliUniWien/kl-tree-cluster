"""
Purpose: Branch Length Predictive Power Analysis.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_predictive_power_benchmark__branch_length__benchmark.py
"""

#!/usr/bin/env python3
"""
Branch Length Predictive Power Analysis
========================================

This script runs benchmarks to empirically track the relationship between
branch length metrics and split decisions across different data scenarios.

The goal is to identify:
1. When branch length is predictive of true cluster boundaries
2. When branch length correlates with statistical test outcomes
3. How branch length relates to clustering accuracy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy import stats
from datetime import datetime

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from benchmarks.shared.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_test_data(
    n_clusters: int,
    n_per_cluster: int,
    n_features: int,
    divergence: float,
    seed: int = 42,
):
    """Generate hierarchical data using the phylogenetic generator."""
    # Use the phylogenetic generator
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

    # Convert to arrays
    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])

    return data, labels, sample_names


def build_and_decompose(
    data: np.ndarray, labels: np.ndarray, sample_names: list
) -> dict:
    """Build tree and run decomposition, return comprehensive results."""
    # Build tree - pass leaf_names to match the probability data index
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Convert data to probability format (one-hot to frequency)
    K = int(data.max()) + 1
    n_samples, n_features = data.shape
    prob_data = np.zeros((n_samples, n_features * K))
    for i in range(n_samples):
        for j in range(n_features):
            prob_data[i, j * K + data[i, j]] = 1.0

    prob_df = pd.DataFrame(prob_data, index=sample_names)

    # Run decomposition
    try:
        decomp_results = tree.decompose(leaf_data=prob_df)
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None

    # Get stats_df with branch length info
    stats_df = tree.stats_df.copy()

    # Add true cluster information
    # Map sample_name -> true label
    leaf_labels = {}
    for i, (name, lbl) in enumerate(zip(sample_names, labels)):
        leaf_labels[name] = lbl

    # Also need to map tree's internal node IDs (L0, L1, etc.) to labels via the node 'label' attribute
    # The tree stores leaves as L0, L1, ... with attribute label=sample_name
    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf"):
            sample_name = tree.nodes[node_id].get("label")
            if sample_name and sample_name in leaf_labels:
                leaf_labels[node_id] = leaf_labels[sample_name]

    # For each internal node, compute true cluster composition
    def get_leaves_under(tree, node_id):
        if tree.out_degree(node_id) == 0:
            return [node_id]
        leaves = []
        for child in tree.successors(node_id):
            leaves.extend(get_leaves_under(tree, child))
        return leaves

    true_cluster_info = []
    for node_id in stats_df.index:
        if stats_df.loc[node_id, "is_leaf"]:
            n_true_clusters = 1
            should_split = False
        else:
            leaves = get_leaves_under(tree, node_id)
            leaf_lbls = [leaf_labels.get(l) for l in leaves if l in leaf_labels]
            n_true_clusters = len(set(leaf_lbls)) if leaf_lbls else 0
            should_split = n_true_clusters > 1

        true_cluster_info.append(
            {
                "node_id": node_id,
                "n_true_clusters": n_true_clusters,
                "should_split": should_split,
            }
        )

    true_df = pd.DataFrame(true_cluster_info).set_index("node_id")
    stats_df = stats_df.join(true_df)

    return {
        "tree": tree,
        "stats_df": stats_df,
        "decomp_results": decomp_results,
        "labels": labels,
        "sample_names": sample_names,
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def compute_predictive_metrics(stats_df: pd.DataFrame) -> dict:
    """Compute how well branch length predicts split decisions."""
    # Filter to internal nodes with valid data
    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["sibling_branch_sum", "should_split"])

    if len(internal) < 10:
        return {"n_nodes": len(internal), "error": "Too few nodes"}

    # Correlation between branch length and should_split
    corr_sibling, p_sibling = stats.pointbiserialr(
        internal["should_split"].astype(int), internal["sibling_branch_sum"]
    )

    corr_height, p_height = stats.pointbiserialr(
        internal["should_split"].astype(int), internal["height"]
    )

    # Effect size (Cohen's d)
    true_split = internal[internal["should_split"] == True]["sibling_branch_sum"]
    no_split = internal[internal["should_split"] == False]["sibling_branch_sum"]

    if len(true_split) > 0 and len(no_split) > 0:
        pooled_std = np.sqrt((true_split.std() ** 2 + no_split.std() ** 2) / 2)
        cohens_d = (true_split.mean() - no_split.mean()) / (pooled_std + 1e-10)
    else:
        cohens_d = np.nan

    # AUC-ROC for branch length as classifier
    from sklearn.metrics import roc_auc_score

    try:
        auc_sibling = roc_auc_score(
            internal["should_split"], internal["sibling_branch_sum"]
        )
    except:
        auc_sibling = np.nan

    try:
        auc_height = roc_auc_score(internal["should_split"], internal["height"])
    except:
        auc_height = np.nan

    # Correlation with statistical test outcomes
    corr_cp = np.nan
    corr_sib = np.nan

    if "Child_Parent_Divergence_Significant" in internal.columns:
        valid = internal.dropna(subset=["Child_Parent_Divergence_Significant"])
        if len(valid) > 5:
            corr_cp, _ = stats.pointbiserialr(
                valid["Child_Parent_Divergence_Significant"].astype(int),
                valid["sibling_branch_sum"],
            )

    if "Sibling_BH_Different" in internal.columns:
        valid = internal.dropna(subset=["Sibling_BH_Different"])
        if len(valid) > 5:
            corr_sib, _ = stats.pointbiserialr(
                valid["Sibling_BH_Different"].astype(int), valid["sibling_branch_sum"]
            )

    return {
        "n_internal_nodes": len(internal),
        "n_should_split": internal["should_split"].sum(),
        "corr_sibling_should_split": corr_sibling,
        "p_sibling": p_sibling,
        "corr_height_should_split": corr_height,
        "p_height": p_height,
        "cohens_d": cohens_d,
        "auc_sibling": auc_sibling,
        "auc_height": auc_height,
        "corr_sibling_cp_test": corr_cp,
        "corr_sibling_sibling_test": corr_sib,
        "mean_branch_true_split": true_split.mean() if len(true_split) > 0 else np.nan,
        "mean_branch_no_split": no_split.mean() if len(no_split) > 0 else np.nan,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def run_benchmark():
    """Run comprehensive benchmark across scenarios."""
    print("=" * 80)
    print("BRANCH LENGTH PREDICTIVE POWER BENCHMARK")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define scenarios
    scenarios = [
        # (n_clusters, n_per_cluster, n_features, divergence)
        (2, 50, 100, 0.1),  # Easy: 2 clusters, low divergence
        (2, 50, 100, 0.3),  # Medium: 2 clusters, medium divergence
        (2, 50, 100, 0.5),  # Hard: 2 clusters, high divergence
        (4, 50, 100, 0.1),  # 4 clusters, low
        (4, 50, 100, 0.3),  # 4 clusters, medium
        (4, 50, 100, 0.5),  # 4 clusters, high
        (8, 25, 100, 0.3),  # 8 clusters, medium
        (4, 100, 50, 0.3),  # More samples, fewer features
        (4, 25, 200, 0.3),  # Fewer samples, more features
    ]

    n_replicates = 5
    all_results = []

    for scenario in scenarios:
        n_clusters, n_per_cluster, n_features, divergence = scenario
        scenario_name = f"C{n_clusters}_N{n_per_cluster}_D{n_features}_div{divergence}"

        print(f"\n--- Scenario: {scenario_name} ---")

        for rep in range(n_replicates):
            seed = 42 + rep * 1000 + hash(scenario_name) % 1000

            try:
                # Generate data
                data, labels, sample_names = generate_test_data(
                    n_clusters=n_clusters,
                    n_per_cluster=n_per_cluster,
                    n_features=n_features,
                    divergence=divergence,
                    seed=seed,
                )

                # Build and decompose
                result = build_and_decompose(data, labels, sample_names)

                if result is None:
                    continue

                # Compute predictive metrics
                metrics = compute_predictive_metrics(result["stats_df"])

                metrics["scenario"] = scenario_name
                metrics["n_clusters"] = n_clusters
                metrics["n_per_cluster"] = n_per_cluster
                metrics["n_features"] = n_features
                metrics["divergence"] = divergence
                metrics["replicate"] = rep

                all_results.append(metrics)

            except Exception as e:
                print(f"  Replicate {rep} failed: {e}")
                continue

        # Print summary for this scenario
        scenario_results = [
            r for r in all_results if r.get("scenario") == scenario_name
        ]
        if scenario_results:
            df = pd.DataFrame(scenario_results)
            print(
                f"  AUC (sibling_branch): {df['auc_sibling'].mean():.3f} ± {df['auc_sibling'].std():.3f}"
            )
            print(
                f"  Cohen's d:            {df['cohens_d'].mean():.3f} ± {df['cohens_d'].std():.3f}"
            )
            print(
                f"  Corr (should_split):  {df['corr_sibling_should_split'].mean():.3f}"
            )

    # Save full results
    results_df = pd.DataFrame(all_results)
    output_path = repo_root / "results" / "branch_length_predictive_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    # Group by divergence level
    print("\n--- By Divergence Level ---")
    for div in sorted(results_df["divergence"].unique()):
        subset = results_df[results_df["divergence"] == div]
        print(f"\nDivergence = {div}:")
        print(f"  Mean AUC:      {subset['auc_sibling'].mean():.3f}")
        print(f"  Mean Cohen's d: {subset['cohens_d'].mean():.3f}")
        print(f"  Mean Corr:      {subset['corr_sibling_should_split'].mean():.3f}")

    # Group by number of clusters
    print("\n--- By Number of Clusters ---")
    for nc in sorted(results_df["n_clusters"].unique()):
        subset = results_df[results_df["n_clusters"] == nc]
        print(f"\nn_clusters = {nc}:")
        print(f"  Mean AUC:      {subset['auc_sibling'].mean():.3f}")
        print(f"  Mean Cohen's d: {subset['cohens_d'].mean():.3f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # When is branch length predictive?
    good_predictive = results_df[results_df["auc_sibling"] > 0.6]
    poor_predictive = results_df[results_df["auc_sibling"] < 0.5]

    if len(good_predictive) > 0:
        print("\nScenarios where branch length IS predictive (AUC > 0.6):")
        summary = (
            good_predictive.groupby("scenario")
            .agg(
                {
                    "auc_sibling": "mean",
                    "cohens_d": "mean",
                }
            )
            .sort_values("auc_sibling", ascending=False)
        )
        print(summary.head(10))

    if len(poor_predictive) > 0:
        print("\nScenarios where branch length is NOT predictive (AUC < 0.5):")
        summary = (
            poor_predictive.groupby("scenario")
            .agg(
                {
                    "auc_sibling": "mean",
                    "cohens_d": "mean",
                }
            )
            .sort_values("auc_sibling")
        )
        print(summary.head(10))

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    overall_auc = results_df["auc_sibling"].mean()
    overall_d = results_df["cohens_d"].mean()

    if overall_auc > 0.6:
        print(f"\nBranch length IS predictive overall (AUC={overall_auc:.3f})")
        print("Consider using it as a pre-filter or supplementary criterion.")
    elif overall_auc > 0.5:
        print(f"\nBranch length has WEAK predictive power (AUC={overall_auc:.3f})")
        print("Use cautiously, only in specific scenarios.")
    else:
        print(f"\nBranch length is NOT predictive (AUC={overall_auc:.3f})")
        print("Do not use for split decisions. Keep as diagnostic only.")

    return results_df


if __name__ == "__main__":
    results = run_benchmark()
