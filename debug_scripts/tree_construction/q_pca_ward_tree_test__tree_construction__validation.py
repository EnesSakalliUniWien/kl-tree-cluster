"""
Purpose: Test the best tree construction method: PCA + Ward.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/tree_construction/q_pca_ward_tree_test__tree_construction__validation.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def main():
    np.random.seed(42)

    print("=" * 80)
    print("COMPARISON: Current Default vs PCA+Ward")
    print("=" * 80)
    print("\nFormat: Default ARI → PCA+Ward ARI (improvement)")
    print("-" * 80)
    print("K | Entropy | Default | PCA+Ward | Δ ARI  | Winner")
    print("-" * 80)

    default_wins = 0
    pca_ward_wins = 0
    ties = 0

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

            # Current default: Hamming + Average
            Z_default = linkage(pdist(df.values, metric="hamming"), method="average")
            pred_default = fcluster(Z_default, t=true_k, criterion="maxclust")
            ari_default = adjusted_rand_score(true_labels, pred_default)

            # New: PCA(10) + Ward
            pca = PCA(n_components=min(10, df.shape[1]))
            X_pca = pca.fit_transform(df.values)
            Z_pca = linkage(X_pca, method="ward")
            pred_pca = fcluster(Z_pca, t=true_k, criterion="maxclust")
            ari_pca = adjusted_rand_score(true_labels, pred_pca)

            delta = ari_pca - ari_default

            if delta > 0.05:
                winner = "PCA+Ward ✓"
                pca_ward_wins += 1
            elif delta < -0.05:
                winner = "Default ✓"
                default_wins += 1
            else:
                winner = "≈ Tie"
                ties += 1

            print(
                f"{true_k} | {entropy:.1f}     | {ari_default:.3f}   | {ari_pca:.3f}    | {delta:+.3f}  | {winner}"
            )

    print("-" * 80)
    print(
        f"\nSummary: Default wins: {default_wins}, PCA+Ward wins: {pca_ward_wins}, Ties: {ties}"
    )

    # Now test varying PCA components
    print("\n" + "=" * 80)
    print("TUNING: PCA Components")
    print("=" * 80)
    print("\nARI for entropy=0.2, K=3 with different PCA components:")

    entropy, true_k = 0.2, 3
    data_dict, labels = generate_random_feature_matrix(
        n_rows=200, n_cols=50, n_clusters=true_k, entropy_param=entropy, random_seed=42
    )
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    true_labels = [labels[name] for name in df.index]

    for n_comp in [2, 3, 5, 7, 10, 15, 20, 30]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(df.values)
        var_explained = pca.explained_variance_ratio_.sum()

        Z = linkage(X_pca, method="ward")
        pred = fcluster(Z, t=true_k, criterion="maxclust")
        ari = adjusted_rand_score(true_labels, pred)

        print(f"  PCA({n_comp:2d}): ARI={ari:.3f}, Var explained={var_explained:.1%}")


if __name__ == "__main__":
    main()
