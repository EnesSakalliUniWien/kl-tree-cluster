"""Benchmark different tree construction methods for robustness to noise.

Compare:
1. Different distance metrics (hamming, jaccard, cosine, correlation)
2. Different linkage methods (average, complete, ward, single)
3. Dimensionality reduction before clustering (PCA, UMAP)
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def test_linkage_distance_combinations():
    """Test various linkage + distance metric combinations."""
    print("=" * 80)
    print("PART 1: Linkage Method × Distance Metric Comparison")
    print("=" * 80)

    # Distance metrics that work with binary data
    distance_metrics = [
        "hamming",
        "jaccard",
        "dice",
        "rogerstanimoto",
        "cosine",
        "correlation",
    ]
    # Linkage methods (ward requires euclidean)
    linkage_methods = ["average", "complete", "single", "weighted"]

    results = []

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [3]:  # Focus on K=3 for clarity
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            true_labels = [labels[name] for name in df.index]

            for dist in distance_metrics:
                for link in linkage_methods:
                    try:
                        D = pdist(df.values, metric=dist)
                        # Handle NaN/inf in distance matrix
                        D = np.nan_to_num(D, nan=0.5, posinf=1.0, neginf=0.0)
                        Z = linkage(D, method=link)
                        pred = fcluster(Z, t=true_k, criterion="maxclust")
                        ari = adjusted_rand_score(true_labels, pred)
                    except Exception as e:
                        ari = -1  # Failed

                    results.append(
                        {
                            "entropy": entropy,
                            "distance": dist,
                            "linkage": link,
                            "ari": ari,
                        }
                    )

    # Summarize
    df_results = pd.DataFrame(results)

    # Pivot: average ARI across entropies
    print("\nAverage ARI across entropy levels (K=3):")
    print("-" * 80)
    pivot = df_results.pivot_table(
        values="ari", index="distance", columns="linkage", aggfunc="mean"
    )
    print(pivot.round(3).to_string())

    # Best combo per entropy
    print("\n\nBest combination per entropy level:")
    print("-" * 80)
    for ent in [0.1, 0.2, 0.3]:
        subset = df_results[df_results["entropy"] == ent]
        best = subset.loc[subset["ari"].idxmax()]
        print(
            f"  entropy={ent}: {best['distance']} + {best['linkage']} → ARI={best['ari']:.3f}"
        )

    return df_results


def test_ward_with_euclidean():
    """Test Ward linkage (requires Euclidean distance)."""
    print("\n" + "=" * 80)
    print("PART 2: Ward Linkage (Euclidean distance)")
    print("=" * 80)

    print("\nWard linkage requires Euclidean distance. Testing on binary data:")
    print("-" * 80)

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [2, 3, 4]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            true_labels = [labels[name] for name in df.index]

            # Ward with Euclidean
            Z = linkage(df.values, method="ward", metric="euclidean")
            pred = fcluster(Z, t=true_k, criterion="maxclust")
            ari_ward = adjusted_rand_score(true_labels, pred)

            # Compare to current default (hamming + average)
            Z_default = linkage(pdist(df.values, metric="hamming"), method="average")
            pred_default = fcluster(Z_default, t=true_k, criterion="maxclust")
            ari_default = adjusted_rand_score(true_labels, pred_default)

            diff = ari_ward - ari_default
            marker = "↑" if diff > 0.05 else ("↓" if diff < -0.05 else "≈")

            print(
                f"  ent={entropy}, K={true_k}: Ward={ari_ward:.3f}, Hamming+Avg={ari_default:.3f} {marker}"
            )


def test_pca_preprocessing():
    """Test PCA dimensionality reduction before clustering."""
    print("\n" + "=" * 80)
    print("PART 3: PCA Preprocessing")
    print("=" * 80)

    print("\nPCA to denoise before hierarchical clustering:")
    print("-" * 80)

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [3]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            true_labels = [labels[name] for name in df.index]

            results_line = [f"ent={entropy}:"]

            # No PCA (baseline)
            Z = linkage(pdist(df.values, metric="hamming"), method="average")
            pred = fcluster(Z, t=true_k, criterion="maxclust")
            ari_base = adjusted_rand_score(true_labels, pred)
            results_line.append(f"No PCA={ari_base:.3f}")

            # Different PCA components
            for n_comp in [5, 10, 20]:
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(df.values)
                Z = linkage(X_pca, method="ward")
                pred = fcluster(Z, t=true_k, criterion="maxclust")
                ari = adjusted_rand_score(true_labels, pred)
                results_line.append(f"PCA({n_comp})={ari:.3f}")

            print("  " + ", ".join(results_line))


def test_umap_preprocessing():
    """Test UMAP for non-linear dimensionality reduction."""
    print("\n" + "=" * 80)
    print("PART 4: UMAP Preprocessing")
    print("=" * 80)

    try:
        import umap
    except ImportError:
        print("  UMAP not installed. Skipping.")
        return

    print("\nUMAP to find non-linear structure:")
    print("-" * 80)

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [3]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            true_labels = [labels[name] for name in df.index]

            results_line = [f"ent={entropy}:"]

            # Baseline
            Z = linkage(pdist(df.values, metric="hamming"), method="average")
            pred = fcluster(Z, t=true_k, criterion="maxclust")
            ari_base = adjusted_rand_score(true_labels, pred)
            results_line.append(f"Base={ari_base:.3f}")

            # UMAP with different n_neighbors
            for n_neighbors in [5, 15, 30]:
                reducer = umap.UMAP(
                    n_components=10,
                    n_neighbors=n_neighbors,
                    min_dist=0.0,
                    metric="hamming",
                    random_state=42,
                )
                X_umap = reducer.fit_transform(df.values)
                Z = linkage(X_umap, method="ward")
                pred = fcluster(Z, t=true_k, criterion="maxclust")
                ari = adjusted_rand_score(true_labels, pred)
                results_line.append(f"UMAP(nn={n_neighbors})={ari:.3f}")

            print("  " + ", ".join(results_line))


def test_feature_selection():
    """Test removing low-variance features before clustering."""
    print("\n" + "=" * 80)
    print("PART 5: Feature Selection (Remove Low-Variance Features)")
    print("=" * 80)

    print("\nKeep only high-variance features:")
    print("-" * 80)

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [3]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            true_labels = [labels[name] for name in df.index]

            results_line = [f"ent={entropy}:"]

            # Baseline (all features)
            Z = linkage(pdist(df.values, metric="hamming"), method="average")
            pred = fcluster(Z, t=true_k, criterion="maxclust")
            ari_base = adjusted_rand_score(true_labels, pred)
            results_line.append(f"All={ari_base:.3f}")

            # Keep top K% variance features
            variances = df.var(axis=0)
            for keep_pct in [0.75, 0.50, 0.25]:
                threshold = variances.quantile(1 - keep_pct)
                high_var_cols = variances[variances >= threshold].index
                df_filtered = df[high_var_cols]

                Z = linkage(
                    pdist(df_filtered.values, metric="hamming"), method="average"
                )
                pred = fcluster(Z, t=true_k, criterion="maxclust")
                ari = adjusted_rand_score(true_labels, pred)
                results_line.append(f"Top{int(keep_pct * 100)}%={ari:.3f}")

            print("  " + ", ".join(results_line))


def main():
    np.random.seed(42)

    df_results = test_linkage_distance_combinations()
    test_ward_with_euclidean()
    test_pca_preprocessing()
    test_umap_preprocessing()
    test_feature_selection()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings will show which tree construction method is most robust
to noisy data. Look for methods that maintain high ARI even at entropy=0.2-0.3.
""")


if __name__ == "__main__":
    main()
