#!/usr/bin/env python
"""
Branch length benchmark script.

Runs the branch length benchmark with multiple replicates to measure
how clustering performance changes as evolutionary divergence increases,
while keeping the number of leaves (samples) fixed.

Usage:
    python benchmarks/branch_length/run.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.branch_length.logic import (
    run_branch_length_benchmark,
    plot_branch_length_results,
    plot_embedding_by_branch_length,
)


def run_replicated_benchmark(
    n_leaves: int = 200,
    n_features: int = 200,
    n_categories: int = 4,
    branch_lengths: list = None,
    mutation_rate: float = 0.30,
    shift_strength: tuple = (0.2, 0.5),
    n_replicates: int = 5,
    base_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark with multiple replicates per branch length.

    Returns combined DataFrame with all replicates.
    """
    if branch_lengths is None:
        # Finer-grained range for Jukes-Cantor model
        # 0.01 to 0.1: Very hard (close to within-group variation)
        # 0.1 to 0.5: Transition
        # 0.5 to 2.0: Easy (saturation)
        branch_lengths = [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.07,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
            1.50,
            2.00,
        ]

    all_results = []

    for rep in range(n_replicates):
        if verbose:
            print(f"Replicate {rep + 1}/{n_replicates}")

        df = run_branch_length_benchmark(
            n_leaves=n_leaves,
            n_features=n_features,
            n_categories=n_categories,
            branch_lengths=branch_lengths,
            mutation_rate=mutation_rate,
            shift_strength=shift_strength,
            random_seed=base_seed + rep * 1000,
            method="kl",
            verbose=False,
        )
        df["replicate"] = rep
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def compute_summary_stats(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std per branch length."""
    summary = (
        combined.groupby("branch_length")
        .agg(
            {
                "ari": ["mean", "std"],
                "nmi": ["mean", "std"],
                "js_divergence": "mean",
                "n_clusters_found": ["mean", "std"],
            }
        )
        .reset_index()
    )

    summary.columns = [
        "branch_length",
        "ari_mean",
        "ari_std",
        "nmi_mean",
        "nmi_std",
        "js_div_mean",
        "n_clusters_mean",
        "n_clusters_std",
    ]

    return summary


def plot_averaged_results(
    summary: pd.DataFrame, n_replicates: int, n_leaves: int
) -> plt.Figure:
    """Create plot with error bars from replicated results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ARI and NMI with error bars
    ax = axes[0, 0]
    ax.errorbar(
        summary["branch_length"],
        summary["ari_mean"],
        yerr=summary["ari_std"],
        fmt="b-o",
        capsize=3,
        label="ARI",
        linewidth=2,
        markersize=6,
    )

    ax.errorbar(
        summary["branch_length"],
        summary["nmi_mean"],
        yerr=summary["nmi_std"],
        fmt="g-s",
        capsize=3,
        label="NMI",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Clustering Performance (mean ± std, n={n_replicates})",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Cluster counts with error bars
    ax = axes[0, 1]
    ax.errorbar(
        summary["branch_length"],
        summary["n_clusters_mean"],
        yerr=summary["n_clusters_std"],
        fmt="r-o",
        capsize=3,
        label="Found",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("Number of Clusters", fontsize=11)
    ax.axhline(y=2, color="k", linestyle="--", linewidth=2, label="True (2)")
    ax.set_title("Cluster Count (mean ± std)", fontsize=12, weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # JS Divergence
    ax = axes[1, 0]
    ax.plot(
        summary["branch_length"],
        summary["js_div_mean"],
        "m-o",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("JS Divergence", fontsize=11)
    ax.set_title("Mean Divergence Between Groups", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)

    # ARI vs JS Divergence
    ax = axes[1, 1]
    scatter = ax.scatter(
        summary["js_div_mean"],
        summary["ari_mean"],
        c=summary["branch_length"],
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidth=1,
    )
    ax.set_xlabel("JS Divergence", fontsize=11)
    ax.set_ylabel("ARI", fontsize=11)
    ax.set_title("Performance vs Divergence", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Branch Length")

    fig.suptitle(
        f"Branch Length Benchmark ({n_replicates} replicates, {n_leaves} leaves fixed)",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("BRANCH LENGTH BENCHMARK")
    print("=" * 70)
    print()

    # Configuration
    n_leaves = 200
    n_features = 200
    n_categories = 4
    branch_lengths = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.07,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]
    mutation_rate = 0.30
    shift_strength = (0.2, 0.5)
    n_replicates = 5
    base_seed = 42

    # Use a single benchmark results root for all suites
    output_dir = repo_root / "benchmarks" / "results"
    output_dir.mkdir(exist_ok=True)

    # Run replicated benchmark
    print(f"Running {n_replicates} replicates for branch lengths {branch_lengths}...")
    print()

    combined = run_replicated_benchmark(
        n_leaves=n_leaves,
        n_features=n_features,
        n_categories=n_categories,
        branch_lengths=branch_lengths,
        mutation_rate=mutation_rate,
        shift_strength=shift_strength,
        n_replicates=n_replicates,
        base_seed=base_seed,
        verbose=True,
    )

    # Compute summary statistics
    summary = compute_summary_stats(combined)

    print()
    print("=" * 70)
    print("AVERAGED RESULTS")
    print("=" * 70)
    print()
    print(
        summary[
            ["branch_length", "js_div_mean", "n_clusters_mean", "ari_mean", "ari_std"]
        ].to_string(index=False)
    )

    # Save results to CSV
    combined.to_csv(output_dir / "branch_length_all_replicates.csv", index=False)
    summary.to_csv(output_dir / "branch_length_summary.csv", index=False)
    print()
    print(f"Saved: {output_dir / 'branch_length_all_replicates.csv'}")
    print(f"Saved: {output_dir / 'branch_length_summary.csv'}")

    # Create averaged plot
    fig1 = plot_averaged_results(summary, n_replicates, n_leaves)
    fig1.savefig(
        output_dir / "branch_length_averaged.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'branch_length_averaged.png'}")

    # Create single-run detailed plot
    single_run = combined[combined["replicate"] == 0].copy()
    fig2 = plot_branch_length_results(
        single_run, title="Branch Length Benchmark (Single Run)"
    )
    fig2.savefig(
        output_dir / "branch_length_single_run.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'branch_length_single_run.png'}")

    # Create embedding plots with KL clustering
    fig3 = plot_embedding_by_branch_length(
        n_leaves=n_leaves,
        n_features=n_features,
        branch_lengths=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        mutation_rate=mutation_rate,
        random_seed=base_seed,
        show_kl_clustering=True,
    )
    fig3.savefig(
        output_dir / "branch_length_embeddings.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'branch_length_embeddings.png'}")

    plt.close("all")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"- Tested branch lengths: {branch_lengths}")
    print(f"- Replicates per branch length: {n_replicates}")
    print(f"- Fixed leaves (samples): {n_leaves}")
    print(f"- Features: {n_features}")
    print(f"- Mutation rate: {mutation_rate}")
    print()

    # Key findings
    high_ari = summary[summary["ari_mean"] > 0.9]
    if len(high_ari) > 0:
        min_bl = high_ari["branch_length"].min()
        min_js = high_ari["js_div_mean"].min()
        print(
            f"- First branch length with mean ARI > 0.9: {min_bl} (JS div ≈ {min_js:.3f})"
        )

    perfect = summary[summary["ari_mean"] > 0.95]
    if len(perfect) > 0:
        print(
            f"- Branch lengths with mean ARI > 0.95: {list(perfect['branch_length'])}"
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
