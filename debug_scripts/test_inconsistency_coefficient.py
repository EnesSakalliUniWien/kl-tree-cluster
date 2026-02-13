#!/usr/bin/env python3
"""
Test Inconsistency Coefficient as Predictive Feature
=====================================================

The literature (Jain & Dubes 1988, Zahn 1971) shows that the INCONSISTENCY
COEFFICIENT (relative height jump) is the correct metric for detecting
cluster boundaries, NOT absolute branch length.

Formula: I(k) = (h_k - mean_local) / std_local

Where local is computed over d levels below node k.

This explains our finding: absolute branch length has NEGATIVE correlation
with true splits because true boundaries occur at HEIGHT JUMPS (inconsistencies),
not at absolute heights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, inconsistent
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from benchmarks.shared.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def compute_inconsistency_for_tree(Z, depth=2):
    """Compute inconsistency coefficient for each merge in linkage matrix."""
    R = inconsistent(Z, d=depth)
    # R columns: [mean, std, n_links, inconsistency_coef]
    return R


def run_test():
    """Test inconsistency coefficient vs absolute branch length."""
    print("=" * 80)
    print("INCONSISTENCY COEFFICIENT ANALYSIS")
    print("=" * 80)
    print("Reference: Jain & Dubes (1988), Zahn (1971)")
    print()

    scenarios = [
        # (n_clusters, n_per_cluster, n_features, divergence)
        (4, 50, 100, 0.3),
        (8, 25, 100, 0.3),
        (4, 50, 100, 0.1),
        (4, 50, 100, 0.5),
    ]

    all_results = []

    for scenario in scenarios:
        n_clusters, n_per_cluster, n_features, divergence = scenario
        scenario_name = f"C{n_clusters}_N{n_per_cluster}_D{n_features}_div{divergence}"

        print(f"\n--- Scenario: {scenario_name} ---")

        for rep in range(3):
            seed = 42 + rep * 1000

            # Generate data
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

            # Build linkage
            distances = pdist(data, metric="hamming")
            Z = linkage(distances, method="weighted")

            # Compute inconsistency coefficients
            R = compute_inconsistency_for_tree(Z, depth=2)

            # Build tree to get ground truth
            tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

            # Map leaf labels
            leaf_labels = {name: lbl for name, lbl in zip(sample_names, labels)}
            for node_id in tree.nodes():
                if tree.nodes[node_id].get("is_leaf"):
                    sample_name = tree.nodes[node_id].get("label")
                    if sample_name and sample_name in leaf_labels:
                        leaf_labels[node_id] = leaf_labels[sample_name]

            # For each internal node, determine should_split
            def get_leaves_under(tree, node_id):
                if tree.out_degree(node_id) == 0:
                    return [node_id]
                leaves = []
                for child in tree.successors(node_id):
                    leaves.extend(get_leaves_under(tree, child))
                return leaves

            # Match linkage rows to tree internal nodes
            n_leaves = len(sample_names)
            merge_data = []

            for merge_idx in range(len(Z)):
                node_idx = n_leaves + merge_idx
                node_id = f"N{node_idx}"

                # Get true cluster composition
                leaves = get_leaves_under(tree, node_id)
                leaf_lbls = [leaf_labels.get(l) for l in leaves if l in leaf_labels]
                n_true_clusters = len(set(leaf_lbls)) if leaf_lbls else 0
                should_split = n_true_clusters > 1

                # Get inconsistency and height
                height = Z[merge_idx, 2]
                inconsistency = R[merge_idx, 3]
                mean_local = R[merge_idx, 0]
                std_local = R[merge_idx, 1]

                merge_data.append(
                    {
                        "merge_idx": merge_idx,
                        "height": height,
                        "inconsistency": inconsistency,
                        "mean_local": mean_local,
                        "std_local": std_local,
                        "should_split": should_split,
                        "n_true_clusters": n_true_clusters,
                    }
                )

            df = pd.DataFrame(merge_data)

            # Filter to valid data
            valid = df[(df["should_split"].notna()) & (df["inconsistency"].notna())]
            valid = valid[valid["inconsistency"] > 0]  # Exclude zero inconsistency

            if len(valid) < 10:
                continue

            # Compute predictive metrics
            try:
                # Inconsistency as predictor
                auc_incons = roc_auc_score(
                    valid["should_split"], valid["inconsistency"]
                )
                corr_incons, _ = stats.pointbiserialr(
                    valid["should_split"].astype(int), valid["inconsistency"]
                )
            except:
                auc_incons = np.nan
                corr_incons = np.nan

            try:
                # Height as predictor
                auc_height = roc_auc_score(valid["should_split"], valid["height"])
                corr_height, _ = stats.pointbiserialr(
                    valid["should_split"].astype(int), valid["height"]
                )
            except:
                auc_height = np.nan
                corr_height = np.nan

            # Effect sizes
            true_split = valid[valid["should_split"] == True]
            no_split = valid[valid["should_split"] == False]

            if len(true_split) > 0 and len(no_split) > 0:
                # Cohen's d for inconsistency
                pooled_std_i = np.sqrt(
                    (
                        true_split["inconsistency"].std() ** 2
                        + no_split["inconsistency"].std() ** 2
                    )
                    / 2
                )
                d_incons = (
                    true_split["inconsistency"].mean()
                    - no_split["inconsistency"].mean()
                ) / (pooled_std_i + 1e-10)

                # Cohen's d for height
                pooled_std_h = np.sqrt(
                    (true_split["height"].std() ** 2 + no_split["height"].std() ** 2)
                    / 2
                )
                d_height = (true_split["height"].mean() - no_split["height"].mean()) / (
                    pooled_std_h + 1e-10
                )
            else:
                d_incons = np.nan
                d_height = np.nan

            result = {
                "scenario": scenario_name,
                "replicate": rep,
                "auc_inconsistency": auc_incons,
                "auc_height": auc_height,
                "corr_inconsistency": corr_incons,
                "corr_height": corr_height,
                "d_inconsistency": d_incons,
                "d_height": d_height,
                "mean_incons_true": true_split["inconsistency"].mean(),
                "mean_incons_false": no_split["inconsistency"].mean(),
                "mean_height_true": true_split["height"].mean(),
                "mean_height_false": no_split["height"].mean(),
            }
            all_results.append(result)

        # Print scenario summary
        scenario_df = pd.DataFrame(
            [r for r in all_results if r["scenario"] == scenario_name]
        )
        if len(scenario_df) > 0:
            print(
                f"  Inconsistency: AUC={scenario_df['auc_inconsistency'].mean():.3f}, "
                f"Corr={scenario_df['corr_inconsistency'].mean():.3f}, "
                f"d={scenario_df['d_inconsistency'].mean():.3f}"
            )
            print(
                f"  Height:        AUC={scenario_df['auc_height'].mean():.3f}, "
                f"Corr={scenario_df['corr_height'].mean():.3f}, "
                f"d={scenario_df['d_height'].mean():.3f}"
            )

    # Overall summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    print("\n                      AUC        Correlation    Cohen's d")
    print("-" * 60)
    print(
        f"Inconsistency:      {results_df['auc_inconsistency'].mean():.3f}          "
        f"{results_df['corr_inconsistency'].mean():.3f}          "
        f"{results_df['d_inconsistency'].mean():.3f}"
    )
    print(
        f"Height (absolute):  {results_df['auc_height'].mean():.3f}         "
        f"{results_df['corr_height'].mean():.3f}         "
        f"{results_df['d_height'].mean():.3f}"
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    inc_better = (
        results_df["auc_inconsistency"].mean() > results_df["auc_height"].mean()
    )

    if inc_better:
        improvement = (
            results_df["auc_inconsistency"].mean() - results_df["auc_height"].mean()
        )
        print(f"\n✅ INCONSISTENCY COEFFICIENT is BETTER by {improvement:.3f} AUC")
        print(
            "   The relative height jump (how much this merge differs from local neighbors)"
        )
        print("   is more predictive than absolute height.")
    else:
        print("\n❌ Inconsistency coefficient is NOT better than absolute height")

    print("\n" + "=" * 80)
    print("HOW TO USE THIS IN OUR METHOD")
    print("=" * 80)
    print("""
The inconsistency coefficient can complement our KL-divergence test:

1. CURRENT: KL-divergence test determines if child distributions differ from parent
2. PROPOSED: Weight or combine with inconsistency coefficient

Options:
A) Pre-filter: Only apply expensive KL-test to nodes with high inconsistency
B) Post-filter: Require BOTH significant KL and high inconsistency
C) Combined score: Aggregate KL p-value with inconsistency

Since inconsistency measures tree STRUCTURE (height jumps) and KL measures
DISTRIBUTION differences, they capture complementary information.
""")

    # Save results
    output_path = repo_root / "results" / "inconsistency_vs_height_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    results = run_test()
