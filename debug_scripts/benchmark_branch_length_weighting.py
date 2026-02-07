#!/usr/bin/env python3
"""
Benchmark: Compare clustering with and without branch length weighting.

This script compares:
1. Current method: distributions weighted by leaf_count only
2. Harmonic method: distributions weighted by leaf_count / branch_length
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datetime import datetime

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics import (
    compute_node_divergences,
)
from kl_clustering_analysis.benchmarking.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def generate_test_data(
    n_clusters: int,
    n_per_cluster: int,
    n_features: int,
    mutation_rate: float,
    seed: int = 42,
):
    """Generate hierarchical data using the phylogenetic generator."""
    sample_dict, cluster_assignments, distributions, metadata = (
        generate_phylogenetic_data(
            n_taxa=n_clusters,
            n_features=n_features,
            n_categories=4,
            samples_per_taxon=n_per_cluster,
            mutation_rate=mutation_rate,
            random_seed=seed,
        )
    )

    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])

    return data, labels, sample_names


def data_to_probability_df(data: np.ndarray, sample_names: list) -> pd.DataFrame:
    """Convert categorical data to one-hot probability format."""
    K = int(data.max()) + 1
    n_samples, n_features = data.shape
    prob_data = np.zeros((n_samples, n_features * K))
    for i in range(n_samples):
        for j in range(n_features):
            prob_data[i, j * K + data[i, j]] = 1.0
    return pd.DataFrame(prob_data, index=sample_names)


def get_cluster_assignments(tree: PosetTree, stats_df: pd.DataFrame) -> dict:
    """Extract cluster assignments from decomposed tree."""
    # Find nodes where we should NOT split (these are cluster roots)
    # A node is a cluster root if it's significant but its children are not

    def get_leaves_under(node_id):
        if tree.nodes[node_id].get("is_leaf", False):
            return [node_id]
        leaves = []
        for child in tree.successors(node_id):
            leaves.extend(get_leaves_under(child))
        return leaves

    # Get significant split decisions from stats_df
    root = tree.graph.get("root") or next(n for n, d in tree.in_degree() if d == 0)

    # Simple approach: use the 'Sibling_BH_Different' column if available
    # Otherwise, just use final clusters from decomposition
    if hasattr(tree, "final_clusters_"):
        clusters = tree.final_clusters_
    else:
        # Fallback: all leaves in one cluster
        clusters = {0: get_leaves_under(root)}

    return clusters


def run_single_comparison(
    n_clusters: int,
    n_per_cluster: int,
    n_features: int,
    mutation_rate: float,
    seed: int,
):
    """Run a single comparison between scoring methods."""
    # Generate data
    data, true_labels, sample_names = generate_test_data(
        n_clusters, n_per_cluster, n_features, mutation_rate, seed
    )

    # Convert to probability format
    prob_df = data_to_probability_df(data, sample_names)

    # Build tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="ward")

    # Map sample names to true labels
    name_to_label = dict(zip(sample_names, true_labels))

    results = {}

    # We only run the "current" method (leaf-count weighting) for distributions,
    # but we will evaluate different SCORING metrics (KL, KL/BL, KL*BL)

    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Compute divergences (standard leaf-count weighting)
    # The compute_node_divergences signature has changed - just call it simply
    # It now calculates composite scores internally with default lambda=0.2
    stats_df = compute_node_divergences(tree, prob_df)

    # Get KL divergence statistics for internal nodes
    internal_nodes = stats_df[~stats_df["is_leaf"]]

    # For each internal node, check if its children span multiple true clusters
    def get_leaves_under(node_id):
        if tree.nodes[node_id].get("is_leaf", False):
            return [tree.nodes[node_id].get("label", node_id)]
        leaves = []
        for child in tree.successors(node_id):
            leaves.extend(get_leaves_under(child))
        return leaves

    # For each internal node, compute if children have different true labels
    node_analysis = []
    for node_id in internal_nodes.index:
        children = list(tree.successors(node_id))
        if len(children) == 2:
            left_leaves = get_leaves_under(children[0])
            right_leaves = get_leaves_under(children[1])

            left_labels = set(name_to_label.get(l, -1) for l in left_leaves)
            right_labels = set(name_to_label.get(l, -1) for l in right_leaves)

            # True boundary: children have non-overlapping label sets
            is_true_boundary = (
                len(left_labels & right_labels) == 0
                and len(left_labels) == 1
                and len(right_labels) == 1
            )

            # Get local KL and branch length for this node's children
            item = {
                "node_id": node_id,
                "is_true_boundary": is_true_boundary,
                # Scores will be summed over children
            }

            kl_vals = []
            bl_vals = []

            for child in children:
                if child in stats_df.index:
                    kl = stats_df.loc[child, "kl_divergence_local"]
                    bl = tree.edges[node_id, child].get("branch_length", 1.0)
                    kl_vals.append(kl)
                    bl_vals.append(bl)

            if len(kl_vals) == 2:
                # Metric 1: Sum of KL (Standard)
                item["score_standard"] = sum(kl_vals)

                # Metric 4: Integrated Score (calculated inside compute_node_divergences)
                # We need to sum up the pre-calculated composite scores from children
                comp_scores = []
                for child in children:
                    comp_scores.append(
                        tree.nodes[child].get("kl_divergence_local_composite", 0.0)
                    )
                item["score_composite"] = sum(comp_scores)

                node_analysis.append(item)

    if node_analysis:
        analysis_df = pd.DataFrame(node_analysis)
        try:
            from sklearn.metrics import roc_auc_score

            # Calculate AUC for each scoring method
            metrics = {"standard": "score_standard", "composite": "score_composite"}

            for name, col in metrics.items():
                if col in analysis_df.columns:
                    auc = roc_auc_score(
                        analysis_df["is_true_boundary"], analysis_df[col]
                    )

                    # Store results
                    results[name] = {
                        "mean_kl_local": internal_nodes["kl_divergence_local"].mean(),
                        "mean_kl_global": internal_nodes["kl_divergence_global"].mean(),
                        "auc_boundary_detection": auc,
                        # dummy vals
                        "std_kl_local": 0,
                        "std_kl_global": 0,
                    }

        except:
            pass

    return results


def run_benchmark():
    """Run benchmark across multiple scenarios."""
    print("=" * 80)
    print("BRANCH LENGTH WEIGHTING BENCHMARK")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    scenarios = [
        # (n_clusters, n_per_cluster, n_features, mutation_rate, name)
        (2, 30, 50, 0.05, "Easy-2"),
        (2, 30, 50, 0.20, "Medium-2"),
        (2, 30, 50, 0.40, "Hard-2"),
        (4, 20, 50, 0.05, "Easy-4"),
        (4, 20, 50, 0.20, "Medium-4"),
        (4, 20, 50, 0.40, "Hard-4"),
        (8, 15, 50, 0.10, "Medium-8"),
    ]

    all_results = []

    for n_clusters, n_per_cluster, n_features, mutation_rate, name in scenarios:
        print(f"\n--- Scenario: {name} ---")
        print(
            f"  Clusters={n_clusters}, Samples/cluster={n_per_cluster}, "
            f"Features={n_features}, MutationRate={mutation_rate}"
        )

        # Run multiple seeds for stability
        scenario_results = {"standard": [], "composite": []}

        for seed in range(5):
            try:
                result = run_single_comparison(
                    n_clusters, n_per_cluster, n_features, mutation_rate, seed
                )
                for method in ["standard", "composite"]:
                    if method in result:
                        scenario_results[method].append(result[method])
            except Exception as e:
                print(f"    Seed {seed} failed: {e}")
                import traceback

                traceback.print_exc()

        # Aggregate results
        for method in ["standard", "composite"]:
            if method in scenario_results and scenario_results[method]:
                agg = {
                    "scenario": name,
                    "method": method,
                    "n_clusters": n_clusters,
                    "mutation_rate": mutation_rate,
                }
                for key in [
                    "mean_kl_local",
                    "std_kl_local",
                    "mean_kl_global",
                    "std_kl_global",
                    "auc_boundary_detection",
                ]:
                    values = [
                        r[key] for r in scenario_results[method] if not np.isnan(r[key])
                    ]
                    agg[f"avg_{key}"] = np.mean(values) if values else np.nan
                all_results.append(agg)

    # Create results DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Pivot for comparison
    for metric in [
        "avg_mean_kl_local",
        "avg_mean_kl_global",
        "avg_auc_boundary_detection",
    ]:
        print(f"\n{metric}:")
        pivot = df.pivot(index="scenario", columns="method", values=metric)
        # Calculate diffs relative to "standard"
        if "standard" in pivot.columns:
            for col in pivot.columns:
                if col != "standard":
                    pivot[f"diff_{col}"] = pivot[col] - pivot["standard"]
                    pivot[f"pct_{col}"] = (
                        pivot[f"diff_{col}"] / pivot["standard"]
                    ) * 100
        print(pivot.to_string())

    return df


def quick_visual_test():
    """Quick visual test showing distribution differences."""
    print("\n" + "=" * 80)
    print("QUICK VISUAL TEST")
    print("=" * 80)
    print(
        "Skipping visual test as we are now testing scoring metrics on the same distributions."
    )


if __name__ == "__main__":
    quick_visual_test()
    print("\n")
    df = run_benchmark()

    # Save results
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(
        results_dir / f"branch_length_weighting_benchmark_{timestamp}.csv", index=False
    )
    print(f"\nResults saved to {results_dir}")
