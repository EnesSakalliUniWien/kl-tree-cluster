"""
Purpose: Run a quick end-to-end sanity benchmark of the current clustering pipeline.
Inputs: Synthetic datasets generated in-script.
Outputs: Console benchmark summary and clustering quality metrics.
Expected runtime: ~10-60 seconds.
How to run: python debug_scripts/smoke/q_pipeline_sanity_benchmark__pipeline__smoke.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def extract_clusters(decomp):
    """Extract cluster assignments from decomposition results."""
    cluster_assignments = decomp.get("cluster_assignments", {})
    leaf_to_cluster = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            leaf_to_cluster[leaf] = cluster_id
    return leaf_to_cluster


def run_benchmark():
    np.random.seed(42)
    print("=" * 70)
    print("NESTED VARIANCE + NORMALIZED BL BENCHMARK")
    print("=" * 70)
    print("\nFormula: Var = θ(1-θ) × (1/n_child - 1/n_parent) × (1 + BL/mean_BL)")
    print()
    print("K | entropy | Pred K | ARI   | CP_sig | Sib_diff | Status")
    print("-" * 70)

    results = []

    # Test with different entropy levels
    for true_k in [2, 3, 4]:
        for entropy in [0.1, 0.2, 0.3]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")

            # Build tree
            Z = linkage(pdist(df.values, metric="hamming"), method="average")
            tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())

            # Decompose
            decomp = tree.decompose(leaf_data=df, alpha_local=0.01, sibling_alpha=0.01)

            # Get test stats
            stats_df = tree.stats_df
            cp_sig = stats_df.get(
                "Child_Parent_Divergence_Significant", pd.Series()
            ).sum()
            sib_diff = stats_df.get("Sibling_BH_Different", pd.Series()).sum()

            # Extract predicted clusters
            leaf_to_cluster = extract_clusters(decomp)

            pred_labels = [leaf_to_cluster.get(name, 0) for name in df.index]
            true_labels = [labels[name] for name in df.index]

            pred_k = len(set(pred_labels))
            ari = adjusted_rand_score(true_labels, pred_labels)

            diff = pred_k - true_k
            if diff == 0:
                status = "✓ EXACT"
            elif diff > 0:
                status = f"+{diff} OVER"
            else:
                status = f"{diff} UNDER"

            print(
                f"{true_k} | {entropy:.1f}     | {pred_k:^6} | {ari:.3f} | {cp_sig:^6} | {sib_diff:^8} | {status}"
            )

            results.append(
                {
                    "true_k": true_k,
                    "entropy": entropy,
                    "pred_k": pred_k,
                    "ari": ari,
                    "diff": diff,
                }
            )

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    exact = sum(1 for r in results if r["diff"] == 0)
    over = sum(1 for r in results if r["diff"] > 0)
    under = sum(1 for r in results if r["diff"] < 0)
    mean_ari = np.mean([r["ari"] for r in results])

    print(f"  Exact: {exact}/{len(results)}")
    print(f"  Over-split: {over}/{len(results)}")
    print(f"  Under-split: {under}/{len(results)}")
    print(f"  Mean ARI: {mean_ari:.3f}")


if __name__ == "__main__":
    run_benchmark()
