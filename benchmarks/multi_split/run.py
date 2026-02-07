#!/usr/bin/env python
"""
Multi-split benchmark - tests clustering with varying numbers of true clusters.

Creates balanced phylogenies with k groups, each diverging from a common ancestor.
Tests how well the method recovers the correct number of splits.

Usage:
    python benchmarks/multi_split/run.py
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
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.evolution import (
    jukes_cantor_transition_matrix,
    generate_ancestral_sequence,
    evolve_sequence,
)
from kl_clustering_analysis import config


def generate_multi_group_data(
    n_groups: int,
    n_samples_per_group: int,
    n_features: int,
    n_categories: int,
    between_group_branch: float,
    within_group_branch: float,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Generate multi-group data with a star phylogeny.
    
    Creates a star tree where all groups diverge from a common ancestor:
    
                        root
                    /  |  |  \\
                 (bl) (bl) ... (bl)
                  |    |       |
               grp0  grp1 ... grpK
               /|\\   /|\\     /|\\
            samples samples  samples
    
    Args:
        n_groups: Number of groups/clusters
        n_samples_per_group: Samples per group
        n_features: Number of sites
        n_categories: Number of states (4 for DNA)
        between_group_branch: Branch length from root to each group ancestor
        within_group_branch: Terminal branch length for within-group variation
        rng: Random number generator
        
    Returns:
        sample_dict: Dict mapping sample names to sequences
        cluster_assignments: Dict mapping sample names to cluster labels
    """
    # Generate root sequence
    root = generate_ancestral_sequence(n_features, n_categories, rng)

    sample_dict = {}
    cluster_assignments = {}

    for g in range(n_groups):
        # Create group ancestor
        group_ancestor = evolve_sequence(root, between_group_branch, n_categories, rng)

        # Generate samples for this group
        for i in range(n_samples_per_group):
            name = f"G{g}_S{i}"
            sample_dict[name] = evolve_sequence(
                group_ancestor, within_group_branch, n_categories, rng
            )
            cluster_assignments[name] = g

    return sample_dict, cluster_assignments


def _run_clustering(
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int, str]:
    """Run clustering and return (labels, n_clusters, status)."""
    spec = METHOD_SPECS[method_id]
    distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    distance_matrix = squareform(distance_condensed)

    try:
        if method_id == "kl":
            result = spec.runner(
                data_df,
                distance_condensed,
                config.SIBLING_ALPHA,
                tree_linkage_method=params.get(
                    "tree_linkage_method", config.TREE_LINKAGE_METHOD
                ),
            )
        elif method_id in {"leiden", "louvain"}:
            result = spec.runner(distance_matrix, params, seed)
        else:
            result = spec.runner(distance_matrix, params)

        if result.status == "ok" and result.labels is not None:
            return result.labels, result.found_clusters, "ok"
        return None, 0, result.status
    except Exception as e:
        return None, 0, f"error: {str(e)}"


def run_multi_split_benchmark(
    n_total_samples: int = 200,
    n_groups_list: List[int] = None,
    n_features: int = 200,
    n_categories: int = 4,
    between_group_branch: float = 0.3,
    within_group_branch: float = 0.05,
    n_replicates: int = 10,
    base_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark with varying number of groups.

    Args:
        n_total_samples: Total number of samples (divided equally among groups)
        n_groups_list: List of group counts to test
        n_features: Number of features
        n_categories: Number of categories per feature
        between_group_branch: Branch length separating groups
        within_group_branch: Within-group terminal branch length
        n_replicates: Number of replicates per configuration
        base_seed: Base random seed
        verbose: Print progress

    Returns:
        DataFrame with results
    """
    if n_groups_list is None:
        n_groups_list = [2, 4, 6, 7, 8, 10, 12]

    results = []
    total = len(n_groups_list) * n_replicates
    current = 0

    for n_groups in n_groups_list:
        n_samples_per_group = n_total_samples // n_groups
        actual_total = n_samples_per_group * n_groups

        for rep in range(n_replicates):
            current += 1
            if verbose:
                print(
                    f"[{current}/{total}] Groups={n_groups}, Replicate {rep + 1}/{n_replicates}"
                )

            rng = np.random.RandomState(base_seed + rep * 1000 + n_groups * 100)

            # Generate data
            sample_dict, cluster_assignments = generate_multi_group_data(
                n_groups=n_groups,
                n_samples_per_group=n_samples_per_group,
                n_features=n_features,
                n_categories=n_categories,
                between_group_branch=between_group_branch,
                within_group_branch=within_group_branch,
                rng=rng,
            )

            # Build dataframe
            sample_names = list(sample_dict.keys())
            matrix = np.array([sample_dict[name] for name in sample_names], dtype=int)
            feature_names = [f"F{j}" for j in range(n_features)]
            data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)
            true_labels = np.array([cluster_assignments[name] for name in sample_names])

            # Run clustering
            pred_labels, n_found, status = _run_clustering(
                data_df, "kl", {}, base_seed + rep
            )

            if pred_labels is not None and status == "ok":
                ari = adjusted_rand_score(true_labels, pred_labels)
                nmi = normalized_mutual_info_score(true_labels, pred_labels)
                correct_k = n_found == n_groups
            else:
                ari = np.nan
                nmi = np.nan
                correct_k = False

            results.append(
                {
                    "n_groups_true": n_groups,
                    "n_groups_found": n_found,
                    "n_samples_per_group": n_samples_per_group,
                    "n_total_samples": actual_total,
                    "n_features": n_features,
                    "between_branch": between_group_branch,
                    "within_branch": within_group_branch,
                    "ari": ari,
                    "nmi": nmi,
                    "correct_k": correct_k,
                    "replicate": rep,
                    "status": status,
                }
            )

    return pd.DataFrame(results)


def plot_multi_split_results(df: pd.DataFrame) -> plt.Figure:
    """Plot results of multi-split benchmark."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Compute summary statistics
    summary = (
        df.groupby("n_groups_true")
        .agg(
            {
                "ari": ["mean", "std"],
                "nmi": ["mean", "std"],
                "n_groups_found": ["mean", "std"],
                "correct_k": "mean",
            }
        )
        .reset_index()
    )
    summary.columns = [
        "n_groups",
        "ari_mean",
        "ari_std",
        "nmi_mean",
        "nmi_std",
        "k_found_mean",
        "k_found_std",
        "correct_k_rate",
    ]

    # Plot 1: ARI and NMI vs number of groups
    ax = axes[0, 0]
    ax.errorbar(
        summary["n_groups"],
        summary["ari_mean"],
        yerr=summary["ari_std"],
        fmt="-o",
        label="ARI",
        linewidth=2,
        markersize=8,
        capsize=4,
    )
    ax.errorbar(
        summary["n_groups"],
        summary["nmi_mean"],
        yerr=summary["nmi_std"],
        fmt="-s",
        label="NMI",
        linewidth=2,
        markersize=8,
        capsize=4,
    )
    ax.set_xlabel("True Number of Groups", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Clustering Performance vs Number of Groups", fontsize=14, weight="bold"
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(summary["n_groups"])

    # Plot 2: Found vs True number of clusters
    ax = axes[0, 1]
    ax.errorbar(
        summary["n_groups"],
        summary["k_found_mean"],
        yerr=summary["k_found_std"],
        fmt="-o",
        color="green",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Found",
    )
    ax.plot(
        summary["n_groups"],
        summary["n_groups"],
        "k--",
        linewidth=2,
        label="True (diagonal)",
    )
    ax.set_xlabel("True Number of Groups", fontsize=12)
    ax.set_ylabel("Found Number of Groups", fontsize=12)
    ax.set_title("Cluster Count: Found vs True", fontsize=14, weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(summary["n_groups"])

    # Plot 3: Rate of finding correct K
    ax = axes[1, 0]
    bars = ax.bar(
        summary["n_groups"],
        summary["correct_k_rate"],
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("True Number of Groups", fontsize=12)
    ax.set_ylabel("Rate of Finding Correct K", fontsize=12)
    ax.set_title("Correct K Detection Rate", fontsize=14, weight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(summary["n_groups"])
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars
    for bar, val in zip(bars, summary["correct_k_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Plot 4: Distribution of found clusters for each true K
    ax = axes[1, 1]
    n_groups_list = sorted(df["n_groups_true"].unique())
    positions = []
    labels = []
    for i, k in enumerate(n_groups_list):
        subset = df[df["n_groups_true"] == k]["n_groups_found"]
        bp = ax.boxplot([subset], positions=[i], widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        positions.append(i)
        labels.append(str(k))

    # Add diagonal reference
    ax.plot(
        range(len(n_groups_list)), n_groups_list, "r--", linewidth=2, label="True K"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True Number of Groups", fontsize=12)
    ax.set_ylabel("Found Number of Groups", fontsize=12)
    ax.set_title("Distribution of Found Clusters", fontsize=14, weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_multi_split_heatmap(
    df: pd.DataFrame, branch_lengths: List[float], n_groups_list: List[int]
) -> plt.Figure:
    """Plot heatmap of ARI across branch lengths and group counts."""
    # Pivot to create grid
    pivot_ari = df.pivot_table(
        index="n_groups_true", columns="between_branch", values="ari", aggfunc="mean"
    )
    pivot_correct = df.pivot_table(
        index="n_groups_true",
        columns="between_branch",
        values="correct_k",
        aggfunc="mean",
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ARI heatmap
    ax = axes[0]
    im = ax.imshow(
        pivot_ari.values, cmap="viridis", aspect="auto", vmin=0, vmax=1, origin="lower"
    )
    ax.set_xticks(range(len(pivot_ari.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot_ari.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_ari.index)))
    ax.set_yticklabels(pivot_ari.index)
    ax.set_xlabel("Between-Group Branch Length", fontsize=12)
    ax.set_ylabel("Number of Groups", fontsize=12)
    ax.set_title("ARI Heatmap", fontsize=14, weight="bold")
    plt.colorbar(im, ax=ax, label="ARI")

    # Add annotations
    for i in range(len(pivot_ari.index)):
        for j in range(len(pivot_ari.columns)):
            val = pivot_ari.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                weight="bold",
            )

    # Correct K heatmap
    ax = axes[1]
    im = ax.imshow(
        pivot_correct.values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=0,
        vmax=1,
        origin="lower",
    )
    ax.set_xticks(range(len(pivot_correct.columns)))
    ax.set_xticklabels(
        [f"{x:.2f}" for x in pivot_correct.columns], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(pivot_correct.index)))
    ax.set_yticklabels(pivot_correct.index)
    ax.set_xlabel("Between-Group Branch Length", fontsize=12)
    ax.set_ylabel("Number of Groups", fontsize=12)
    ax.set_title("Correct K Rate Heatmap", fontsize=14, weight="bold")
    plt.colorbar(im, ax=ax, label="Correct K Rate")

    # Add annotations
    for i in range(len(pivot_correct.index)):
        for j in range(len(pivot_correct.columns)):
            val = pivot_correct.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                weight="bold",
            )

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("MULTI-SPLIT BENCHMARK")
    print("=" * 70)
    print()

    # Configuration
    n_total_samples = 200
    n_groups_list = [2, 4, 6, 7, 8, 10, 12]
    n_features = 200
    n_categories = 4
    branch_lengths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    within_group_branch = 0.05
    n_replicates = 10
    base_seed = 42

    # Create output directory relative to script
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("Configuration:")
    print(f"  - Total samples: {n_total_samples}")
    print(f"  - Group counts: {n_groups_list}")
    print(f"  - Features: {n_features}")
    print(f"  - Between-group branch lengths: {branch_lengths}")
    print(f"  - Within-group branch: {within_group_branch}")
    print(f"  - Replicates: {n_replicates}")
    print()

    # Run benchmark for each branch length
    all_results = []

    for bl in branch_lengths:
        print(f"\n--- Branch length: {bl} ---")
        df = run_multi_split_benchmark(
            n_total_samples=n_total_samples,
            n_groups_list=n_groups_list,
            n_features=n_features,
            n_categories=n_categories,
            between_group_branch=bl,
            within_group_branch=within_group_branch,
            n_replicates=n_replicates,
            base_seed=base_seed,
            verbose=True,
        )
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = (
        combined.groupby(["between_branch", "n_groups_true"])
        .agg(
            {
                "ari": "mean",
                "correct_k": "mean",
                "n_groups_found": "mean",
            }
        )
        .round(2)
    )
    print(summary.to_string())

    # Save data
    combined.to_csv(output_dir / "multi_split_all.csv", index=False)
    print(f"\nSaved: {output_dir / 'multi_split_all.csv'}")

    # Create plots for the middle branch length (0.3)
    df_mid = combined[combined["between_branch"] == 0.3]
    fig1 = plot_multi_split_results(df_mid)
    fig1.savefig(output_dir / "multi_split_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'multi_split_results.png'}")

    # Create heatmap across all branch lengths
    fig2 = plot_multi_split_heatmap(combined, branch_lengths, n_groups_list)
    fig2.savefig(output_dir / "multi_split_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'multi_split_heatmap.png'}")

    plt.close("all")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
