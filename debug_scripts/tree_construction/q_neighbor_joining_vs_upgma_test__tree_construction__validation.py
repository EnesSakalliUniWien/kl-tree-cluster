"""
Purpose: Test Neighbor-Joining tree construction vs UPGMA (average linkage).
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/tree_construction/q_neighbor_joining_vs_upgma_test__tree_construction__validation.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings("ignore")

from skbio import DistanceMatrix
from skbio.tree import nj

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def nj_to_clusters(nj_tree, k):
    """Extract k clusters from a neighbor-joining tree.

    Strategy: Use the tree's internal structure to cut at k clusters.
    Since NJ is unrooted, we use longest-path midpoint rooting.
    """
    # Get all tip names
    tip_names = [tip.name for tip in nj_tree.tips()]
    n = len(tip_names)

    # For simplicity, we'll use a different approach:
    # Extract the cophenetic distance matrix from NJ tree
    # Then use hierarchical clustering on that

    # Build cophenetic distance matrix from NJ tree
    cophenetic = np.zeros((n, n))
    tip_to_idx = {name: i for i, name in enumerate(tip_names)}

    for i, tip1 in enumerate(nj_tree.tips()):
        for j, tip2 in enumerate(nj_tree.tips()):
            if i < j:
                # Distance in tree between two tips
                dist = tip1.distance(tip2)
                cophenetic[i, j] = dist
                cophenetic[j, i] = dist

    # Now do hierarchical clustering on cophenetic distances
    coph_condensed = squareform(cophenetic)
    Z = linkage(coph_condensed, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")

    # Map back to original names
    return {tip_names[i]: labels[i] for i in range(n)}


def test_neighbor_joining():
    """Compare UPGMA vs Neighbor-Joining."""
    print("=" * 80)
    print("NEIGHBOR-JOINING vs UPGMA (Average Linkage)")
    print("=" * 80)
    print("\nNJ doesn't assume molecular clock - may handle noisy data better")
    print("-" * 80)
    print("K | Ent | UPGMA | NJ    | Δ     | Winner")
    print("-" * 80)

    results = []

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
            true_labels = [labels[name] for name in df.index]

            # UPGMA (current default)
            D = pdist(df.values, metric="hamming")
            Z_upgma = linkage(D, method="average")
            pred_upgma = fcluster(Z_upgma, t=true_k, criterion="maxclust")
            ari_upgma = adjusted_rand_score(true_labels, pred_upgma)

            # Neighbor-Joining
            D_square = squareform(D)
            dm = DistanceMatrix(D_square, ids=df.index.tolist())

            try:
                nj_tree = nj(dm)
                cluster_map = nj_to_clusters(nj_tree, true_k)
                pred_nj = [cluster_map[name] for name in df.index]
                ari_nj = adjusted_rand_score(true_labels, pred_nj)
            except Exception as e:
                ari_nj = -1
                print(f"  NJ failed: {e}")

            delta = ari_nj - ari_upgma
            if delta > 0.05:
                winner = "NJ ✓"
            elif delta < -0.05:
                winner = "UPGMA ✓"
            else:
                winner = "≈ Tie"

            print(
                f"{true_k} | {entropy:.1f} | {ari_upgma:.3f} | {ari_nj:.3f} | {delta:+.3f} | {winner}"
            )

            results.append(
                {
                    "k": true_k,
                    "entropy": entropy,
                    "upgma": ari_upgma,
                    "nj": ari_nj,
                    "delta": delta,
                }
            )

    print("-" * 80)

    # Summary
    df_results = pd.DataFrame(results)
    print(f"\nSummary:")
    print(f"  Mean UPGMA ARI: {df_results['upgma'].mean():.3f}")
    print(f"  Mean NJ ARI:    {df_results['nj'].mean():.3f}")
    print(f"  Mean Δ:         {df_results['delta'].mean():+.3f}")

    # By entropy level
    print(f"\nBy entropy level:")
    for ent in [0.1, 0.2, 0.3]:
        subset = df_results[df_results["entropy"] == ent]
        print(
            f"  ent={ent}: UPGMA={subset['upgma'].mean():.3f}, NJ={subset['nj'].mean():.3f}, Δ={subset['delta'].mean():+.3f}"
        )


if __name__ == "__main__":
    np.random.seed(42)
    test_neighbor_joining()
