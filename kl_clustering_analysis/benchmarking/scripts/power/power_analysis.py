"""Power Analysis with Surface Plots.

Comprehensive evaluation of clustering performance across parameter space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from pathlib import Path
from itertools import product

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "power"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION - CRANKED UP
# ============================================================

# Larger parameter grids for comprehensive analysis
N_SAMPLES_GRID = [30, 50, 75, 100, 150, 200, 300]
N_FEATURES_GRID = [10, 20, 30, 50, 75, 100, 150]
N_CLUSTERS_GRID = [2, 3, 4, 5, 6]
NOISE_LEVELS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

N_SEEDS = 3
BASE_SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def run_clustering(data_df):
    """Run the KL-TE clustering algorithm."""
    try:
        Z = linkage(
            pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomposition = tree.decompose(leaf_data=data_df)

        cluster_assignments = decomposition.get("cluster_assignments", {})

        # Build label map
        label_map = {}
        cluster_id = 0
        for cl_key, info in cluster_assignments.items():
            for leaf in info["leaves"]:
                label_map[leaf] = cluster_id
            cluster_id += 1

        # Get labels
        labels = [label_map.get(name, 0) for name in data_df.index]
        n_found = len(cluster_assignments) if cluster_assignments else 1
        return np.array(labels), n_found

    except Exception as e:
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

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "noise": noise,
        "seed": seed,
        "ari": ari,
        "n_found": n_found,
        "perfect": 1 if ari > 0.99 else 0,
    }


def create_surface_plot(
    df,
    x_col,
    y_col,
    z_col="ari",
    fixed_params=None,
    title="",
    filename="",
):
    """Create a beautiful surface plot."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=135)

    # Filter by fixed params
    df_filtered = df.copy()
    subtitle_parts = []
    if fixed_params:
        for col, val in fixed_params.items():
            df_filtered = df_filtered[df_filtered[col] == val]
            subtitle_parts.append(f"{col}={val}")

    # Aggregate by mean
    pivot = df_filtered.groupby([x_col, y_col])[z_col].mean().reset_index()

    # Create meshgrid
    x_unique = sorted(pivot[x_col].unique())
    y_unique = sorted(pivot[y_col].unique())

    X, Y = np.meshgrid(x_unique, y_unique)
    Z = np.zeros_like(X, dtype=float)

    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            val = pivot[(pivot[x_col] == x_val) & (pivot[y_col] == y_val)][z_col]
            Z[i, j] = val.values[0] if len(val) > 0 else np.nan

    cmap = "viridis"
    norm = colors.Normalize(vmin=0, vmax=1)

    # Plot surface
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        alpha=0.95,
        antialiased=True,
    )

    # Add contour projections
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap=cmap, norm=norm, alpha=0.65)

    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12, labelpad=10)
    y_label = y_col.replace("_", " ").title()
    if y_col == "noise":
        y_label = "Noise (%)"
        ax.set_yticks(y_unique)
        ax.set_yticklabels([f"{v * 100:.0f}" for v in y_unique])
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax.set_zlabel("ARI Score", fontsize=12, labelpad=10)
    ax.set_zlim(0, 1)

    if np.isfinite(Z).any():
        max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter(
            X[max_idx],
            Y[max_idx],
            Z[max_idx],
            color="black",
            s=25,
            depthshade=False,
        )
        ax.text(
            X[max_idx],
            Y[max_idx],
            Z[max_idx] + 0.03,
            f"max {Z[max_idx]:.2f}",
            color="black",
            fontsize=9,
        )

    full_title = title
    if subtitle_parts:
        full_title += f"\n({', '.join(subtitle_parts)})"
    ax.set_title(full_title, fontsize=14, fontweight="bold")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="ARI (0-1)")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close()


def create_heatmap(
    df,
    x_col,
    y_col,
    z_col="ari",
    fixed_params=None,
    title="",
    filename="",
    cmap="RdYlGn",
):
    """Create a heatmap for cleaner 2D visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter
    df_filtered = df.copy()
    subtitle_parts = []
    if fixed_params:
        for col, val in fixed_params.items():
            df_filtered = df_filtered[df_filtered[col] == val]
            subtitle_parts.append(f"{col}={val}")

    # Pivot
    pivot = df_filtered.groupby([y_col, x_col])[z_col].mean().unstack()

    # Plot
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    full_title = title
    if subtitle_parts:
        full_title += f"\n({', '.join(subtitle_parts)})"
    ax.set_title(full_title, fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, label="ARI Score")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("POWER ANALYSIS - COMPREHENSIVE EVALUATION")
    print("=" * 70)

    total_configs = (
        len(N_SAMPLES_GRID)
        * len(N_FEATURES_GRID)
        * len(N_CLUSTERS_GRID)
        * len(NOISE_LEVELS)
        * N_SEEDS
    )
    print(f"\nTotal configurations: {total_configs}")
    print(f"  Samples:  {N_SAMPLES_GRID}")
    print(f"  Features: {N_FEATURES_GRID}")
    print(f"  Clusters: {N_CLUSTERS_GRID}")
    print(f"  Noise:    {NOISE_LEVELS}")
    print(f"  Seeds:    {N_SEEDS}")

    # Run evaluation
    results = []
    count = 0

    for noise in NOISE_LEVELS:
        print(f"\n--- Noise: {noise * 100:.0f}% ---")

        for n_samples, n_features, n_clusters in product(
            N_SAMPLES_GRID, N_FEATURES_GRID, N_CLUSTERS_GRID
        ):
            for seed in range(BASE_SEED, BASE_SEED + N_SEEDS):
                result = evaluate_single(n_samples, n_features, n_clusters, noise, seed)
                results.append(result)
                count += 1

            if count % 100 == 0:
                print(f"    {count}/{total_configs} done...")

        # Quick stats for this noise level
        noise_results = [r for r in results if r["noise"] == noise]
        mean_ari = np.mean([r["ari"] for r in noise_results])
        perfect_pct = 100 * np.mean([r["perfect"] for r in noise_results])
        print(f"  Mean ARI: {mean_ari:.3f}, Perfect: {perfect_pct:.1f}%")

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "power_analysis_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'power_analysis_results.csv'}")

    # ============================================================
    # GENERATE VISUALIZATIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Surface plots: Samples vs Features for different noise levels
    print("\n1. Samples vs Features (by noise level)")
    for noise in [0.02, 0.05, 0.10, 0.15]:
        create_surface_plot(
            df,
            "n_samples",
            "n_features",
            "ari",
            fixed_params={"n_clusters": 3, "noise": noise},
            title=f"Clustering Power: Samples × Features",
            filename=RESULTS_DIR
            / f"surface_samples_features_noise{int(noise * 100):02d}.png",
        )

    # 2. Surface plots: Samples vs Clusters for different noise levels
    print("\n2. Samples vs Clusters (by noise level)")
    for noise in [0.02, 0.05, 0.10, 0.15]:
        create_surface_plot(
            df,
            "n_samples",
            "n_clusters",
            "ari",
            fixed_params={"n_features": 50, "noise": noise},
            title=f"Clustering Power: Samples × Clusters",
            filename=RESULTS_DIR
            / f"surface_samples_clusters_noise{int(noise * 100):02d}.png",
        )

    # 3. Surface plots: Features vs Noise
    print("\n3. Features vs Noise")
    for n_clusters in [2, 3, 4]:
        create_surface_plot(
            df,
            "n_features",
            "noise",
            "ari",
            fixed_params={"n_samples": 100, "n_clusters": n_clusters},
            title=f"Clustering Power: Features × Noise",
            filename=RESULTS_DIR / f"surface_features_noise_k{n_clusters}.png",
        )

    # 4. Heatmaps for cleaner visualization
    print("\n4. Heatmaps")
    for noise in [0.05, 0.10, 0.15]:
        create_heatmap(
            df,
            "n_samples",
            "n_features",
            "ari",
            fixed_params={"n_clusters": 3, "noise": noise},
            title=f"ARI Heatmap: Samples × Features",
            filename=RESULTS_DIR
            / f"heatmap_samples_features_noise{int(noise * 100):02d}.png",
        )

    # 5. Summary plot: Mean ARI by noise level
    print("\n5. Summary plots")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ARI vs Noise by cluster count
    ax = axes[0, 0]
    for k in N_CLUSTERS_GRID:
        subset = df[df["n_clusters"] == k]
        means = subset.groupby("noise")["ari"].mean()
        ax.plot(
            means.index * 100,
            means.values,
            "o-",
            label=f"k={k}",
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Mean ARI", fontsize=12)
    ax.set_title("Noise Tolerance by Cluster Count", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ARI vs Samples by noise
    ax = axes[0, 1]
    for noise in [0.02, 0.05, 0.10, 0.15]:
        subset = df[df["noise"] == noise]
        means = subset.groupby("n_samples")["ari"].mean()
        ax.plot(
            means.index,
            means.values,
            "o-",
            label=f"{noise * 100:.0f}% noise",
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Mean ARI", fontsize=12)
    ax.set_title("Sample Size Effect", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ARI vs Features by noise
    ax = axes[1, 0]
    for noise in [0.02, 0.05, 0.10, 0.15]:
        subset = df[df["noise"] == noise]
        means = subset.groupby("n_features")["ari"].mean()
        ax.plot(
            means.index,
            means.values,
            "o-",
            label=f"{noise * 100:.0f}% noise",
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Mean ARI", fontsize=12)
    ax.set_title("Feature Dimensionality Effect", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Perfect clustering rate
    ax = axes[1, 1]
    perfect_by_noise = df.groupby("noise")["perfect"].mean() * 100
    bars = ax.bar(
        perfect_by_noise.index * 100,
        perfect_by_noise.values,
        color=plt.cm.RdYlGn(perfect_by_noise.values / 100),
        edgecolor="black",
    )
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Perfect Clustering Rate (%)", fontsize=12)
    ax.set_title(
        "Rate of Perfect Clustering (ARI > 0.99)", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, perfect_by_noise.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "power_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {RESULTS_DIR / 'power_summary.png'}")
    plt.close()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\nOverall:")
    print(f"  Mean ARI: {df['ari'].mean():.3f}")
    print(f"  Perfect Clustering Rate: {df['perfect'].mean() * 100:.1f}%")
    print(f"  Best ARI: {df['ari'].max():.3f}")

    print("\nBy Noise Level:")
    summary = (
        df.groupby("noise")
        .agg({"ari": ["mean", "std", "min", "max"], "perfect": "mean"})
        .round(3)
    )
    summary.columns = ["Mean ARI", "Std", "Min", "Max", "Perfect Rate"]
    summary["Perfect Rate"] = (summary["Perfect Rate"] * 100).round(1).astype(str) + "%"
print(summary.to_string())

print("\n" + "=" * 70)
print(f"DONE - Check {RESULTS_DIR} for visualizations")
print("=" * 70)
