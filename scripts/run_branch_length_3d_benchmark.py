#!/usr/bin/env python
"""
3D Branch length benchmark - varies both branch length and number of features.

Creates a surface plot showing ARI as a function of:
- X axis: Branch length (evolutionary divergence)
- Y axis: Number of features
- Z axis (color): ARI score

Usage:
    python scripts/run_branch_length_3d_benchmark.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from kl_clustering_analysis.benchmarking.branch_length_benchmark import (
    run_branch_length_benchmark,
)


def run_3d_benchmark(
    n_leaves: int = 200,
    branch_lengths: list = None,
    feature_counts: list = None,
    n_categories: int = 4,
    mutation_rate: float = 0.30,
    shift_strength: tuple = (0.2, 0.5),
    n_replicates: int = 3,
    base_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark varying both branch length and feature count.
    
    Returns DataFrame with columns: branch_length, n_features, ari, nmi, js_divergence, etc.
    """
    if branch_lengths is None:
        branch_lengths = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    
    if feature_counts is None:
        feature_counts = [50, 100, 150, 200, 300, 500]
    
    all_results = []
    total = len(feature_counts) * n_replicates
    current = 0
    
    for n_features in feature_counts:
        for rep in range(n_replicates):
            current += 1
            if verbose:
                print(f"[{current}/{total}] Features={n_features}, Replicate {rep + 1}/{n_replicates}")
            
            df = run_branch_length_benchmark(
                n_leaves=n_leaves,
                n_features=n_features,
                n_categories=n_categories,
                branch_lengths=branch_lengths,
                mutation_rate=mutation_rate,
                shift_strength=shift_strength,
                random_seed=base_seed + rep * 1000 + n_features,
                method="kl",
                verbose=False,
            )
            df["n_features"] = n_features
            df["replicate"] = rep
            all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)


def compute_grid_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ARI for each (branch_length, n_features) combination."""
    summary = combined.groupby(["branch_length", "n_features"]).agg({
        "ari": ["mean", "std"],
        "nmi": ["mean", "std"],
        "js_divergence": "mean",
        "n_clusters_found": "mean",
    }).reset_index()
    
    summary.columns = [
        "branch_length", "n_features", 
        "ari_mean", "ari_std", "nmi_mean", "nmi_std",
        "js_div_mean", "n_clusters_mean"
    ]
    
    return summary


def plot_3d_surface(summary: pd.DataFrame, n_leaves: int, n_replicates: int) -> plt.Figure:
    """Create 3D surface plot of ARI vs branch_length and n_features."""
    # Pivot to create grid
    pivot = summary.pivot(index="n_features", columns="branch_length", values="ari_mean")
    
    X = pivot.columns.values  # branch_length
    Y = pivot.index.values    # n_features
    X, Y = np.meshgrid(X, Y)
    Z = pivot.values
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', 
                           linewidth=0.3, alpha=0.9, antialiased=True)
    
    ax.set_xlabel('Branch Length\n(evolution steps)', fontsize=11, labelpad=10)
    ax.set_ylabel('Number of Features', fontsize=11, labelpad=10)
    ax.set_zlabel('ARI', fontsize=11, labelpad=10)
    ax.set_zlim(0, 1)
    
    ax.set_title(f'Clustering Performance Surface\n({n_leaves} leaves, {n_replicates} replicates per point)',
                 fontsize=14, weight='bold', pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('ARI', fontsize=11)
    
    # Set viewing angle - more orthographic view from above
    ax.view_init(elev=35, azim=225)
    
    # Make it more orthographic by adjusting the projection
    ax.set_box_aspect([1, 1, 0.5])  # Flatten Z axis slightly
    
    plt.tight_layout()
    return fig


def plot_heatmap(summary: pd.DataFrame, n_leaves: int, n_replicates: int) -> plt.Figure:
    """Create heatmap of ARI vs branch_length and n_features."""
    pivot = summary.pivot(index="n_features", columns="branch_length", values="ari_mean")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', 
                   vmin=0, vmax=1, origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('Branch Length (evolution steps)', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title(f'ARI Heatmap: Branch Length × Features\n({n_leaves} leaves, {n_replicates} replicates)',
                 fontsize=14, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ARI', fontsize=11)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, weight='bold')
    
    plt.tight_layout()
    return fig


def plot_line_by_features(summary: pd.DataFrame, n_leaves: int) -> plt.Figure:
    """Create line plot showing ARI vs branch_length for each feature count."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_counts = sorted(summary["n_features"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_counts)))
    
    for n_feat, color in zip(feature_counts, colors):
        subset = summary[summary["n_features"] == n_feat].sort_values("branch_length")
        ax.plot(subset["branch_length"], subset["ari_mean"], 
                '-o', color=color, label=f'{n_feat} features', 
                linewidth=2, markersize=6)
        ax.fill_between(subset["branch_length"],
                       subset["ari_mean"] - subset["ari_std"],
                       subset["ari_mean"] + subset["ari_std"],
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Branch Length (evolutionary distance)', fontsize=12)
    ax.set_ylabel('ARI', fontsize=12)
    ax.set_title(f'ARI vs Branch Length by Feature Count\n({n_leaves} leaves)',
                 fontsize=14, weight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_correct_split_rate(combined: pd.DataFrame, n_leaves: int) -> plt.Figure:
    """Plot the rate of finding the correct 2-cluster split."""
    # Compute correct split rate for each (branch_length, n_features) combination
    summary = combined.groupby(["branch_length", "n_features"]).agg({
        "correct_split": "mean",
        "ari": "mean",
    }).reset_index()
    summary.columns = ["branch_length", "n_features", "correct_split_rate", "ari_mean"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Line plot of correct split rate
    ax = axes[0]
    feature_counts = sorted(summary["n_features"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_counts)))
    
    for n_feat, color in zip(feature_counts, colors):
        subset = summary[summary["n_features"] == n_feat].sort_values("branch_length")
        ax.plot(subset["branch_length"], subset["correct_split_rate"], 
                '-o', color=color, label=f'{n_feat} features', 
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Branch Length (evolutionary distance)', fontsize=12)
    ax.set_ylabel('Correct Split Rate', fontsize=12)
    ax.set_title(f'Rate of Finding Correct 2-Cluster Split\n({n_leaves} leaves)',
                 fontsize=14, weight='bold')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale('log')
    
    # Plot 2: Heatmap of correct split rate
    ax = axes[1]
    pivot = summary.pivot(index="n_features", columns="branch_length", values="correct_split_rate")
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                   vmin=0, vmax=1, origin='lower')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('Branch Length', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Correct Split Rate Heatmap', fontsize=14, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correct Split Rate', fontsize=11)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', 
                   color=color, fontsize=7, weight='bold')
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("3D BRANCH LENGTH BENCHMARK (Jukes-Cantor Model)")
    print("=" * 70)
    print()
    
    # Configuration - branch_length is now evolutionary distance (expected substitutions/site)
    # Extended to 10 to see saturation effects
    n_leaves = 200
    branch_lengths = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    feature_counts = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
    n_categories = 4
    mutation_rate = 0.30  # Ignored in new model
    shift_strength = (0.2, 0.5)  # Ignored in new model
    n_replicates = 10
    base_seed = 42
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Configuration:")
    print(f"  - Leaves (samples): {n_leaves}")
    print(f"  - Branch lengths: {branch_lengths}")
    print(f"  - Feature counts: {feature_counts}")
    print(f"  - Replicates: {n_replicates}")
    print(f"  - Total runs: {len(branch_lengths) * len(feature_counts) * n_replicates}")
    print()
    
    # Run benchmark
    combined = run_3d_benchmark(
        n_leaves=n_leaves,
        branch_lengths=branch_lengths,
        feature_counts=feature_counts,
        n_categories=n_categories,
        mutation_rate=mutation_rate,
        shift_strength=shift_strength,
        n_replicates=n_replicates,
        base_seed=base_seed,
        verbose=True,
    )
    
    # Compute summary
    summary = compute_grid_summary(combined)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Pivot table for display
    pivot = summary.pivot(index="n_features", columns="branch_length", values="ari_mean")
    print("Mean ARI by (Features × Branch Length):")
    print(pivot.round(2).to_string())
    
    # Save data
    combined.to_csv(output_dir / "branch_length_3d_all.csv", index=False)
    summary.to_csv(output_dir / "branch_length_3d_summary.csv", index=False)
    print()
    print(f"Saved: {output_dir / 'branch_length_3d_all.csv'}")
    print(f"Saved: {output_dir / 'branch_length_3d_summary.csv'}")
    
    # Create plots
    fig1 = plot_3d_surface(summary, n_leaves, n_replicates)
    fig1.savefig(output_dir / "branch_length_3d_surface.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'branch_length_3d_surface.png'}")
    
    fig2 = plot_heatmap(summary, n_leaves, n_replicates)
    fig2.savefig(output_dir / "branch_length_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'branch_length_heatmap.png'}")
    
    fig3 = plot_line_by_features(summary, n_leaves)
    fig3.savefig(output_dir / "branch_length_by_features.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'branch_length_by_features.png'}")
    
    # Plot correct split rate
    fig4 = plot_correct_split_rate(combined, n_leaves)
    fig4.savefig(output_dir / "correct_split_rate.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'correct_split_rate.png'}")
    
    # Print correct split summary
    print()
    print("=" * 70)
    print("CORRECT SPLIT RATE")
    print("=" * 70)
    correct_summary = combined.groupby(["branch_length", "n_features"])["correct_split"].mean().unstack()
    print(correct_summary.round(2).to_string())
    
    plt.close("all")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
