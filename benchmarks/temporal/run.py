#!/usr/bin/env python
"""
Demo script for running the incremental temporal clustering benchmark.

This script demonstrates how clustering performance changes as sequences
evolve along a growing branch over time.

Usage:
    python benchmarks/temporal/run.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np

from kl_clustering_analysis.benchmarking import (
    run_incremental_temporal_benchmark,
    run_temporal_benchmark_suite,
)


def plot_temporal_results(df, title="Incremental Temporal Clustering"):
    """Plot ARI/NMI and cluster counts over time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ARI over time
    ax = axes[0, 0]
    ax.plot(df["time_step"], df["ari"], "b-o", label="ARI", linewidth=2)
    ax.plot(df["time_step"], df["nmi"], "g-s", label="NMI", linewidth=2)
    ax.set_xlabel("Time Step (True Clusters)")
    ax.set_ylabel("Score")
    ax.set_title("Clustering Performance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Cluster counts
    ax = axes[0, 1]
    ax.plot(df["time_step"], df["n_clusters_true"], "k--", label="True", linewidth=2)
    ax.plot(df["time_step"], df["n_clusters_found"], "r-o", label="Found", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cluster Count: True vs Found")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Divergence over time
    ax = axes[1, 0]
    ax.plot(df["time_step"], df["divergence_from_ancestor"], "m-o", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("JS Divergence from Ancestor")
    ax.set_title("Sequence Divergence Over Time")
    ax.grid(True, alpha=0.3)

    # ARI vs Divergence
    ax = axes[1, 1]
    ax.scatter(
        df["divergence_from_ancestor"],
        df["ari"],
        c=df["time_step"],
        cmap="viridis",
        s=100,
        edgecolors="black",
    )
    ax.set_xlabel("JS Divergence from Ancestor")
    ax.set_ylabel("ARI")
    ax.set_title("Performance vs Divergence")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Time Step")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("INCREMENTAL TEMPORAL CLUSTERING BENCHMARK")
    print("=" * 60)

    # Create output directory relative to script
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Run single benchmark with fast evolution
    print("\n--- Fast Evolution (mutation_rate=0.35) ---")
    df_fast = run_incremental_temporal_benchmark(
        n_time_points=10,
        n_features=200,
        n_categories=4,
        samples_per_time=20,
        mutation_rate=0.35,
        shift_strength=(0.2, 0.5),
        random_seed=42,
        method="kl",
        verbose=True,
    )

    print("\n--- Moderate Evolution (mutation_rate=0.20) ---")
    df_moderate = run_incremental_temporal_benchmark(
        n_time_points=10,
        n_features=200,
        n_categories=4,
        samples_per_time=20,
        mutation_rate=0.20,
        shift_strength=(0.15, 0.4),
        random_seed=43,
        method="kl",
        verbose=True,
    )

    print("\n--- Slow Evolution (mutation_rate=0.10) ---")
    df_slow = run_incremental_temporal_benchmark(
        n_time_points=10,
        n_features=200,
        n_categories=4,
        samples_per_time=20,
        mutation_rate=0.10,
        shift_strength=(0.1, 0.3),
        random_seed=44,
        method="kl",
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, df in [("Fast", df_fast), ("Moderate", df_moderate), ("Slow", df_slow)]:
        mean_ari = df["ari"].mean()
        final_ari = df["ari"].iloc[-1]
        mean_cluster_diff = (
            (df["n_clusters_found"] - df["n_clusters_true"]).abs().mean()
        )
        print(f"\n{name} Evolution:")
        print(f"  Mean ARI: {mean_ari:.3f}")
        print(f"  Final ARI (t={len(df) + 1}): {final_ari:.3f}")
        print(f"  Mean |Found - True| clusters: {mean_cluster_diff:.1f}")

    # Plot results
    fig_fast = plot_temporal_results(df_fast, "Fast Evolution (mutation_rate=0.35)")
    fig_moderate = plot_temporal_results(
        df_moderate, "Moderate Evolution (mutation_rate=0.20)"
    )
    fig_slow = plot_temporal_results(df_slow, "Slow Evolution (mutation_rate=0.10)")

    # Save plots
    fig_fast.savefig(
        output_dir / "temporal_benchmark_fast.png", dpi=150, bbox_inches="tight"
    )
    fig_moderate.savefig(
        output_dir / "temporal_benchmark_moderate.png", dpi=150, bbox_inches="tight"
    )
    fig_slow.savefig(
        output_dir / "temporal_benchmark_slow.png", dpi=150, bbox_inches="tight"
    )

    print(f"\nPlots saved to {output_dir}/temporal_benchmark_*.png")


if __name__ == "__main__":
    main()
