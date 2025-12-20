"""Grid Search Evaluation for Clustering Performance.

This script evaluates the clustering algorithm across a grid of parameters:
- Number of features
- Number of clusters
- Number of samples
- Noise levels (entropy_param)

Generates 3D visualizations of ARI scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs

from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark" / "grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# Parameter grids - realistic ranges for biological/real-world data
# Samples: small cohort (100) → large study (1000)
# Features: small panel (100) → full transcriptome (5000)
# Clusters: subtypes (3) → cell types (10)
# Noise: clean data (2%) → noisy data (20%)

N_SAMPLES_GRID = [100, 200, 300, 500, 750, 1000]  # 6 levels - realistic study sizes
N_FEATURES_GRID = [
    100,
    250,
    500,
    1000,
    2000,
    5000,
]  # 6 levels - gene panel to transcriptome
N_CLUSTERS_GRID = [3, 4, 5, 6, 8, 10]  # 6 levels - realistic cluster counts
NOISE_LEVELS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20]  # 6 levels - clean to noisy

N_SEEDS = 3  # Average over 3 seeds for stability
BASE_SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def run_clustering(data_df):
    """Run the KL-TE clustering algorithm."""
    try:
        Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomposition = tree.decompose(leaf_data=data_df)

        cluster_report = decomposition.get("cluster_assignments", {})

        # Build label map
        label_map = {}
        cluster_id = 0
        for cl_key, info in cluster_report.items():
            for leaf in info["leaves"]:
                label_map[leaf] = cluster_id
            cluster_id += 1

        # Get labels - unassigned go to cluster 0
        labels = []
        for name in data_df.index:
            labels.append(label_map.get(name, 0))

        n_found = len(cluster_report) if cluster_report else 1
        return np.array(labels), n_found

    except Exception as e:
        print(f"    Clustering error: {e}")
        return np.zeros(len(data_df), dtype=int), 1


def evaluate_single(n_samples, n_features, n_clusters, noise, seed):
    """Evaluate a single configuration."""
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=n_samples,
        n_cols=n_features,
        n_clusters=n_clusters,
        entropy_param=noise,
        balanced_clusters=True,
        random_seed=seed,
    )

    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    pred_labels, n_found = run_clustering(data_df)
    ari = adjusted_rand_score(true_labels, pred_labels)

    return ari, n_found


def run_grid_search(noise_level):
    """Run grid search for one noise level."""
    results = []
    total = len(N_SAMPLES_GRID) * len(N_FEATURES_GRID) * len(N_CLUSTERS_GRID)
    count = 0

    for n_samples in N_SAMPLES_GRID:
        for n_features in N_FEATURES_GRID:
            for n_clusters in N_CLUSTERS_GRID:
                count += 1

                ari_list = []
                k_list = []

                for i in range(N_SEEDS):
                    seed = BASE_SEED + i * 100
                    ari, k = evaluate_single(
                        n_samples, n_features, n_clusters, noise_level, seed
                    )
                    ari_list.append(ari)
                    k_list.append(k)

                results.append(
                    {
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_clusters": n_clusters,
                        "noise": noise_level,
                        "ari": np.mean(ari_list),
                        "k_found": np.mean(k_list),
                    }
                )

                if count % 5 == 0:
                    print(f"    {count}/{total} done")

    return pd.DataFrame(results)


def plot_3d_scatter(df, noise, filepath):
    """Create 3D scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = df["n_features"].values
    y = df["n_clusters"].values
    z = df["n_samples"].values
    c = df["ari"].values

    # Size based on ARI
    sizes = 50 + 150 * np.clip(c, 0, 1)

    scatter = ax.scatter(
        x,
        y,
        z,
        c=c,
        cmap="RdYlGn",
        s=sizes,
        alpha=0.8,
        edgecolors="k",
        linewidth=0.5,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Features")
    ax.set_ylabel("Clusters")
    ax.set_zlabel("Samples")
    ax.set_title(f"Clustering ARI - Noise: {noise:.0%}")

    cbar = fig.colorbar(scatter, shrink=0.6, pad=0.1)
    cbar.set_label("ARI")

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_comparison(all_df, filepath):
    """Plot all noise levels side by side."""
    noise_vals = sorted(all_df["noise"].unique())
    n = len(noise_vals)

    fig = plt.figure(figsize=(5 * n, 5))

    for i, noise in enumerate(noise_vals):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        df = all_df[all_df["noise"] == noise]

        x = df["n_features"].values
        y = df["n_clusters"].values
        z = df["n_samples"].values
        c = df["ari"].values

        sizes = 30 + 100 * np.clip(c, 0, 1)
        ax.scatter(x, y, z, c=c, cmap="RdYlGn", s=sizes, alpha=0.8, vmin=0, vmax=1)

        ax.set_xlabel("Features")
        ax.set_ylabel("Clusters")
        ax.set_zlabel("Samples")
        ax.set_title(f"Noise: {noise:.0%}")

    plt.suptitle("Clustering Performance vs Noise", fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_ari_trends(all_df, filepath):
    """Plot ARI trends vs noise."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overall
    ax = axes[0, 0]
    grouped = all_df.groupby("noise")["ari"].agg(["mean", "std"])
    ax.errorbar(
        grouped.index * 100,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=5,
        linewidth=2,
    )
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("ARI")
    ax.set_title("Overall ARI vs Noise")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # 2. By clusters
    ax = axes[0, 1]
    for k in N_CLUSTERS_GRID:
        df_k = all_df[all_df["n_clusters"] == k]
        means = df_k.groupby("noise")["ari"].mean()
        ax.plot(means.index * 100, means.values, marker="o", label=f"k={k}")
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("ARI")
    ax.set_title("ARI by Cluster Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # 3. By samples
    ax = axes[1, 0]
    for n in N_SAMPLES_GRID:
        df_n = all_df[all_df["n_samples"] == n]
        means = df_n.groupby("noise")["ari"].mean()
        ax.plot(means.index * 100, means.values, marker="s", label=f"n={n}")
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("ARI")
    ax.set_title("ARI by Sample Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # 4. By features
    ax = axes[1, 1]
    for d in N_FEATURES_GRID:
        df_d = all_df[all_df["n_features"] == d]
        means = df_d.groupby("noise")["ari"].mean()
        ax.plot(means.index * 100, means.values, marker="^", label=f"d={d}")
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("ARI")
    ax.set_title("ARI by Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.suptitle("Clustering Performance Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRID SEARCH EVALUATION")
    print("=" * 60)

    print(f"\nParameters:")
    print(f"  Samples:  {N_SAMPLES_GRID}")
    print(f"  Features: {N_FEATURES_GRID}")
    print(f"  Clusters: {N_CLUSTERS_GRID}")
    print(f"  Noise:    {NOISE_LEVELS}")
    print(f"  Seeds:    {N_SEEDS}")

    all_results = []

    # Run for each noise level
    for noise in NOISE_LEVELS:
        print(f"\n--- Noise: {noise:.0%} ---")
        df = run_grid_search(noise)
        all_results.append(df)

        print(f"  Mean ARI: {df['ari'].mean():.3f}")
        print(f"  Best ARI: {df['ari'].max():.3f}")

        # Save individual 3D plot
        plot_3d_scatter(
            df, noise, RESULTS_DIR / f"grid_3d_noise_{int(noise * 100):02d}.png"
        )

    # Combine results
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'grid_search_results.csv'}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_df, RESULTS_DIR / "grid_3d_comparison.png")
    plot_ari_trends(all_df, RESULTS_DIR / "grid_ari_trends.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = all_df.groupby("noise")["ari"].agg(["mean", "std", "min", "max"])
    print(summary.round(3))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
