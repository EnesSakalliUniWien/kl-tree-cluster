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

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# Parameter grids
N_SAMPLES_GRID = [50, 100, 200]
N_FEATURES_GRID = [30, 60, 120]
N_CLUSTERS_GRID = [2, 3, 4, 5]
# Note: entropy_param controls noise. 0.0 = pure templates (identical samples per cluster)
# We use small values for "low noise" since 0.0 creates degenerate cases
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]  # entropy_param values

# Number of random seeds to average over
N_SEEDS = 2
BASE_SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def run_clustering(data_df):
    """Run the KL-TE clustering algorithm and return cluster assignments."""
    try:
        Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomposition = tree.decompose(leaf_data=data_df)

        cluster_report = decomposition.get("cluster_report", {})

        # Build label map
        label_map = {}
        for cl_id, info in cluster_report.items():
            for leaf in info["members"]:
                label_map[leaf] = cl_id

        # Get labels in order - assign unassigned to cluster -1
        labels = []
        for name in data_df.index:
            if name in label_map:
                labels.append(label_map[name])
            else:
                # Sample not in any cluster - assign to cluster 0 as fallback
                labels.append(0)

        n_clusters_found = len(cluster_report) if cluster_report else 1
        return np.array(labels), n_clusters_found
    except Exception as e:
        print(f"  Error in clustering: {e}")
        return None, 0


def evaluate_configuration(n_samples, n_features, n_clusters, noise, seed):
    """Generate data and evaluate clustering for a single configuration."""
    # Generate binary data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=n_samples,
        n_cols=n_features,
        n_clusters=n_clusters,
        entropy_param=noise,
        balanced_clusters=True,
        random_seed=seed,
    )

    # Convert to DataFrame
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    # Run clustering
    result = run_clustering(data_df)

    if result[0] is None:
        return np.nan, 0

    pred_labels, n_clusters_found = result

    # Calculate ARI
    ari = adjusted_rand_score(true_labels, pred_labels)

    return ari, n_clusters_found


def run_grid_search(noise_level, verbose=True):
    """Run grid search for a single noise level."""
    results = []

    total = len(N_SAMPLES_GRID) * len(N_FEATURES_GRID) * len(N_CLUSTERS_GRID)
    count = 0

    for n_samples in N_SAMPLES_GRID:
        for n_features in N_FEATURES_GRID:
            for n_clusters in N_CLUSTERS_GRID:
                count += 1

                # Average over multiple seeds
                ari_scores = []
                k_found_list = []

                for seed_idx in range(N_SEEDS):
                    seed = BASE_SEED + seed_idx * 100
                    ari, k_found = evaluate_configuration(
                        n_samples, n_features, n_clusters, noise_level, seed
                    )
                    if not np.isnan(ari):
                        ari_scores.append(ari)
                        k_found_list.append(k_found)

                avg_ari = np.mean(ari_scores) if ari_scores else np.nan
                avg_k = np.mean(k_found_list) if k_found_list else np.nan

                results.append(
                    {
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_clusters": n_clusters,
                        "noise": noise_level,
                        "ari": avg_ari,
                        "k_found": avg_k,
                        "k_error": avg_k - n_clusters
                        if not np.isnan(avg_k)
                        else np.nan,
                    }
                )

                if verbose and count % 10 == 0:
                    print(f"  Progress: {count}/{total} configurations")

    return pd.DataFrame(results)


def plot_3d_results(df, noise_level, ax=None, save_path=None):
    """Create 3D scatter plot of ARI scores."""
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

    # Map to coordinates
    x = df["n_features"].values
    y = df["n_clusters"].values
    z = df["n_samples"].values
    colors = df["ari"].values

    # Handle NaN values
    valid_mask = ~np.isnan(colors)
    x, y, z, colors = x[valid_mask], y[valid_mask], z[valid_mask], colors[valid_mask]

    # Size proportional to ARI (larger = better)
    sizes = 50 + 150 * np.clip(colors, 0, 1)

    # Create scatter plot
    scatter = ax.scatter(
        x,
        y,
        z,
        c=colors,
        cmap="RdYlGn",
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Number of Features", fontsize=10)
    ax.set_ylabel("Number of Clusters", fontsize=10)
    ax.set_zlabel("Number of Samples", fontsize=10)
    ax.set_title(
        f"Clustering Performance (ARI)\nNoise Level: {noise_level:.0%}", fontsize=12
    )

    # Add colorbar
    if ax.figure is not None:
        cbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Adjusted Rand Index (ARI)", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return ax


def plot_noise_comparison(all_results, save_path=None):
    """Create multi-panel plot comparing different noise levels."""
    n_noise = len(NOISE_LEVELS)
    fig = plt.figure(figsize=(6 * min(n_noise, 3), 5 * ((n_noise + 2) // 3)))

    for idx, noise in enumerate(NOISE_LEVELS):
        ax = fig.add_subplot(
            (n_noise + 2) // 3, min(n_noise, 3), idx + 1, projection="3d"
        )
        df_noise = all_results[all_results["noise"] == noise]

        x = df_noise["n_features"].values
        y = df_noise["n_clusters"].values
        z = df_noise["n_samples"].values
        colors = df_noise["ari"].values

        valid_mask = ~np.isnan(colors)
        if valid_mask.sum() > 0:
            x, y, z, colors = (
                x[valid_mask],
                y[valid_mask],
                z[valid_mask],
                colors[valid_mask],
            )
            sizes = 30 + 100 * np.clip(colors, 0, 1)

            scatter = ax.scatter(
                x, y, z, c=colors, cmap="RdYlGn", s=sizes, alpha=0.7, vmin=0, vmax=1
            )

        ax.set_xlabel("Features", fontsize=8)
        ax.set_ylabel("Clusters", fontsize=8)
        ax.set_zlabel("Samples", fontsize=8)
        ax.set_title(f"Noise: {noise:.0%}", fontsize=10)

    plt.suptitle("Clustering Performance Across Noise Levels", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_ari_vs_noise(all_results, save_path=None):
    """Plot ARI degradation as noise increases."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overall ARI vs Noise
    ax1 = axes[0, 0]
    noise_summary = all_results.groupby("noise")["ari"].agg(
        ["mean", "std", "min", "max"]
    )
    ax1.errorbar(
        noise_summary.index * 100,
        noise_summary["mean"],
        yerr=noise_summary["std"],
        marker="o",
        capsize=5,
        linewidth=2,
    )
    ax1.fill_between(
        noise_summary.index * 100, noise_summary["min"], noise_summary["max"], alpha=0.2
    )
    ax1.set_xlabel("Noise Level (%)")
    ax1.set_ylabel("ARI")
    ax1.set_title("Overall ARI vs Noise Level")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # 2. ARI by n_clusters
    ax2 = axes[0, 1]
    for k in N_CLUSTERS_GRID:
        df_k = all_results[all_results["n_clusters"] == k]
        means = df_k.groupby("noise")["ari"].mean()
        ax2.plot(means.index * 100, means.values, marker="o", label=f"k={k}")
    ax2.set_xlabel("Noise Level (%)")
    ax2.set_ylabel("ARI")
    ax2.set_title("ARI by Number of Clusters")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # 3. ARI by n_samples
    ax3 = axes[1, 0]
    for n in N_SAMPLES_GRID:
        df_n = all_results[all_results["n_samples"] == n]
        means = df_n.groupby("noise")["ari"].mean()
        ax3.plot(means.index * 100, means.values, marker="s", label=f"n={n}")
    ax3.set_xlabel("Noise Level (%)")
    ax3.set_ylabel("ARI")
    ax3.set_title("ARI by Sample Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)

    # 4. ARI by n_features
    ax4 = axes[1, 1]
    for d in N_FEATURES_GRID:
        df_d = all_results[all_results["n_features"] == d]
        means = df_d.groupby("noise")["ari"].mean()
        ax4.plot(means.index * 100, means.values, marker="^", label=f"d={d}")
    ax4.set_xlabel("Noise Level (%)")
    ax4.set_ylabel("ARI")
    ax4.set_title("ARI by Number of Features")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)

    plt.suptitle("Clustering Performance Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def create_heatmap(all_results, noise_level, save_path=None):
    """Create heatmap of ARI for fixed noise level."""
    df = all_results[all_results["noise"] == noise_level]

    # Average over n_clusters for a 2D heatmap (samples x features)
    pivot = df.pivot_table(
        values="ari", index="n_samples", columns="n_features", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Mean ARI (averaged over k)\nNoise Level: {noise_level:.0%}")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10
            )

    plt.colorbar(im, label="ARI")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GRID SEARCH: Clustering Performance Evaluation")
    print("=" * 70)

    print(f"\nParameter Grid:")
    print(f"  Samples:  {N_SAMPLES_GRID}")
    print(f"  Features: {N_FEATURES_GRID}")
    print(f"  Clusters: {N_CLUSTERS_GRID}")
    print(f"  Noise:    {NOISE_LEVELS}")
    print(f"  Seeds:    {N_SEEDS}")

    total_configs = (
        len(N_SAMPLES_GRID)
        * len(N_FEATURES_GRID)
        * len(N_CLUSTERS_GRID)
        * len(NOISE_LEVELS)
        * N_SEEDS
    )
    print(f"\nTotal evaluations: {total_configs}")

    # Collect all results
    all_results = []

    # ========== SIMULATION 1: No Noise ==========
    print("\n" + "=" * 70)
    print("SIMULATION 1: No Noise (entropy=0.0)")
    print("=" * 70)

    df_no_noise = run_grid_search(noise_level=0.0)
    all_results.append(df_no_noise)

    # Summary
    print(f"\nResults Summary (No Noise):")
    print(f"  Mean ARI: {df_no_noise['ari'].mean():.3f}")
    print(f"  Min ARI:  {df_no_noise['ari'].min():.3f}")
    print(f"  Max ARI:  {df_no_noise['ari'].max():.3f}")
    print(f"  Perfect (ARI=1): {(df_no_noise['ari'] == 1.0).sum()}/{len(df_no_noise)}")

    # Generate 3D plot for no noise
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    plot_3d_results(
        df_no_noise,
        0.0,
        ax=ax,
        save_path=RESULTS_DIR / "grid_search_3d_no_noise.png",
    )
    plt.close()

    print("\n✓ 3D plot generated for no-noise case")

    # ========== SIMULATION 2: With Increasing Noise ==========
    print("\n" + "=" * 70)
    print("SIMULATION 2: Increasing Noise Levels")
    print("=" * 70)

    for noise in NOISE_LEVELS[1:]:  # Skip 0.0, already done
        print(f"\n--- Noise Level: {noise:.0%} ---")
        df_noise = run_grid_search(noise_level=noise)
        all_results.append(df_noise)

        print(f"  Mean ARI: {df_noise['ari'].mean():.3f}")
        print(f"  Perfect (ARI=1): {(df_noise['ari'] == 1.0).sum()}/{len(df_noise)}")

    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    all_results_df.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'grid_search_results.csv'}")

    # ========== GENERATE PLOTS ==========
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # 3D comparison across noise levels
    plot_noise_comparison(
        all_results_df, save_path=RESULTS_DIR / "grid_search_3d_comparison.png"
    )
    plt.close()
    print("✓ Multi-panel 3D comparison plot generated")

    # ARI vs Noise analysis
    plot_ari_vs_noise(
        all_results_df, save_path=RESULTS_DIR / "grid_search_ari_vs_noise.png"
    )
    plt.close()
    print("✓ ARI vs Noise analysis plot generated")

    # Heatmaps for different noise levels
    for noise in [0.0, 0.10, 0.20]:
        create_heatmap(
            all_results_df,
            noise,
            save_path=RESULTS_DIR
            / f"grid_search_heatmap_noise_{int(noise * 100):02d}.png",
        )
        plt.close()
    print("✓ Heatmaps generated for noise levels 0%, 10%, 20%")

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    summary = (
        all_results_df.groupby("noise")
        .agg({"ari": ["mean", "std", "min", "max"], "k_error": ["mean", "std"]})
        .round(3)
    )

    print("\nPerformance by Noise Level:")
    print(summary.to_string())

    # Validation
    print("\n" + "-" * 40)
    print("VALIDATION")
    print("-" * 40)

    # Check expected behavior
    mean_ari_no_noise = all_results_df[all_results_df["noise"] == 0.0]["ari"].mean()
    mean_ari_high_noise = all_results_df[all_results_df["noise"] == 0.20]["ari"].mean()

    if mean_ari_no_noise > mean_ari_high_noise:
        print("✓ ARI decreases with increasing noise (as expected)")
    else:
        print("⚠ Unexpected: ARI does not decrease with noise")

    if mean_ari_no_noise > 0.7:
        print("✓ Good performance on clean data (mean ARI > 0.7)")
    else:
        print("⚠ Performance on clean data lower than expected")

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
