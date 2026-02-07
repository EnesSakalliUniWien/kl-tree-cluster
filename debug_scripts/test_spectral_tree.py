"""Test Spectral Embedding + Hierarchical Clustering.

Spectral clustering itself doesn't produce a tree.
But we can:
1. Compute spectral embedding (Laplacian eigenvectors)
2. Run hierarchical clustering (Ward) on embedded coordinates
3. This gives a tree in spectral space

This is similar to PCA+Ward but uses graph Laplacian instead of linear PCA.
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import warnings

warnings.filterwarnings("ignore")

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)


def test_spectral_embedding_tree():
    """Compare different embedding methods + Ward linkage."""
    print("=" * 80)
    print("EMBEDDING + WARD LINKAGE COMPARISON")
    print("=" * 80)
    print("\nAll methods produce a tree by running Ward on embedded coordinates")
    print("-" * 80)
    print("K | Ent | Raw   | PCA   | Spectral | Best")
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
            X = df.values

            # Method 1: Raw Hamming + Average (current default)
            D = pdist(X, metric="hamming")
            Z_raw = linkage(D, method="average")
            pred_raw = fcluster(Z_raw, t=true_k, criterion="maxclust")
            ari_raw = adjusted_rand_score(true_labels, pred_raw)

            # Method 2: PCA + Ward
            n_components = min(10, X.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            Z_pca = linkage(X_pca, method="ward")
            pred_pca = fcluster(Z_pca, t=true_k, criterion="maxclust")
            ari_pca = adjusted_rand_score(true_labels, pred_pca)

            # Method 3: Spectral Embedding + Ward
            # Use k-NN graph for affinity (works better with binary data)
            try:
                se = SpectralEmbedding(
                    n_components=n_components,
                    affinity="nearest_neighbors",
                    n_neighbors=10,
                    random_state=42,
                )
                X_spectral = se.fit_transform(X)
                Z_spectral = linkage(X_spectral, method="ward")
                pred_spectral = fcluster(Z_spectral, t=true_k, criterion="maxclust")
                ari_spectral = adjusted_rand_score(true_labels, pred_spectral)
            except Exception as e:
                ari_spectral = -1

            # Find best
            aris = {"Raw": ari_raw, "PCA": ari_pca, "Spectral": ari_spectral}
            best = max(aris, key=aris.get)

            print(
                f"{true_k} | {entropy:.1f} | {ari_raw:.3f} | {ari_pca:.3f} | {ari_spectral:.3f}    | {best}"
            )

            results.append(
                {
                    "k": true_k,
                    "entropy": entropy,
                    "raw": ari_raw,
                    "pca": ari_pca,
                    "spectral": ari_spectral,
                }
            )

    print("-" * 80)

    # Summary
    df_res = pd.DataFrame(results)
    print(f"\nMean ARI:")
    print(f"  Raw (Hamming+Avg): {df_res['raw'].mean():.3f}")
    print(f"  PCA + Ward:        {df_res['pca'].mean():.3f}")
    print(f"  Spectral + Ward:   {df_res['spectral'].mean():.3f}")

    print(f"\nBy entropy level:")
    for ent in [0.1, 0.2, 0.3]:
        subset = df_res[df_res["entropy"] == ent]
        print(
            f"  ent={ent}: Raw={subset['raw'].mean():.3f}, PCA={subset['pca'].mean():.3f}, Spectral={subset['spectral'].mean():.3f}"
        )

    # Count wins
    print(f"\nWins (best ARI per test case):")
    wins = {"Raw": 0, "PCA": 0, "Spectral": 0}
    for _, row in df_res.iterrows():
        aris = {"Raw": row["raw"], "PCA": row["pca"], "Spectral": row["spectral"]}
        best = max(aris, key=aris.get)
        wins[best] += 1
    for method, count in wins.items():
        print(f"  {method}: {count}/{len(df_res)}")


def test_spectral_parameters():
    """Tune spectral embedding parameters."""
    print("\n" + "=" * 80)
    print("SPECTRAL EMBEDDING PARAMETER TUNING")
    print("=" * 80)
    print("\nTesting different n_neighbors and n_components on entropy=0.2, K=3")
    print("-" * 80)

    entropy, true_k = 0.2, 3
    data_dict, labels = generate_random_feature_matrix(
        n_rows=200, n_cols=50, n_clusters=true_k, entropy_param=entropy, random_seed=42
    )
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    true_labels = [labels[name] for name in df.index]
    X = df.values

    # Baseline
    D = pdist(X, metric="hamming")
    Z_raw = linkage(D, method="average")
    pred_raw = fcluster(Z_raw, t=true_k, criterion="maxclust")
    ari_raw = adjusted_rand_score(true_labels, pred_raw)
    print(f"Baseline (Raw Hamming+Avg): {ari_raw:.3f}")
    print()

    print("n_neighbors \\ n_components |   3   |   5   |  10   |  20")
    print("-" * 60)

    for n_neighbors in [5, 10, 15, 20, 30]:
        row = [f"      {n_neighbors:2d}                   |"]
        for n_components in [3, 5, 10, 20]:
            try:
                se = SpectralEmbedding(
                    n_components=n_components,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
                X_emb = se.fit_transform(X)
                Z = linkage(X_emb, method="ward")
                pred = fcluster(Z, t=true_k, criterion="maxclust")
                ari = adjusted_rand_score(true_labels, pred)
                row.append(f"{ari:.3f} |")
            except:
                row.append("  -   |")
        print(" ".join(row))


if __name__ == "__main__":
    np.random.seed(42)
    test_spectral_embedding_tree()
    test_spectral_parameters()
