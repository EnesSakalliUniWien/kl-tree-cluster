"""
Branch length benchmark with fixed number of leaves.

Tests how clustering performance changes as one branch gets longer (more divergent)
while keeping the total number of samples (leaves) constant.

Uses a proper Jukes-Cantor substitution model for sequence evolution.

The setup:
- Two groups: "ancestral" and "evolved"
- Both groups have the same number of samples
- The evolved group diverges more as branch_length increases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.runners.dispatch import run_clustering
from benchmarks.shared.evolution import (
    generate_ancestral_sequence,
    evolve_sequence,
    compute_expected_divergence,
)


def generate_two_group_data(
    n_samples_per_group: int,
    n_features: int,
    n_categories: int,
    branch_length: float,
    random_state: np.random.RandomState = None,
    within_group_branch: float = 0.1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], float]:
    """Generate two-group data using Jukes-Cantor substitution model.
    
    Creates a phylogeny with realistic within-group variation:
    
                        root
                       /    \\
                  (0.01)    (branch_length)
                    |           |
               group0_anc   group1_anc
                  /|\\          /|\\
           (within)  ...  (within)  ...
              |              |
           samples        samples
    
    Each sample has its own terminal branch (within_group_branch) to create
    realistic within-group variation.
    
    Args:
        n_samples_per_group: Number of samples per group
        n_features: Number of sites/features
        n_categories: Number of states (4 for DNA)
        branch_length: Evolutionary distance separating the two groups
        random_state: Random number generator
        within_group_branch: Terminal branch length for within-group variation
        
    Returns:
        sample_dict: Dict mapping sample names to sequences
        cluster_assignments: Dict mapping sample names to cluster labels (0 or 1)
        expected_divergence: Expected proportion of differing sites between groups
    """
    if random_state is None:
        random_state = np.random.RandomState()

    # Generate root sequence
    root = generate_ancestral_sequence(n_features, n_categories, random_state)

    # Create group ancestors (MRCA of each group)
    # Symmetric divergence: Each group evolves by branch_length / 2 from root
    # Total expected distance between ancestors ≈ branch_length
    dist_to_ancestor = branch_length / 2.0
    group0_ancestor = evolve_sequence(
        root, dist_to_ancestor, n_categories, random_state
    )
    group1_ancestor = evolve_sequence(
        root, dist_to_ancestor, n_categories, random_state
    )

    # Compute expected divergence between group ancestors
    expected_div = compute_expected_divergence(branch_length, n_categories)

    sample_dict = {}
    cluster_assignments = {}

    # Group 0: Each sample evolves independently from group0_ancestor
    # with its own terminal branch
    for i in range(n_samples_per_group):
        name = f"A{i}"
        sample_dict[name] = evolve_sequence(
            group0_ancestor, within_group_branch, n_categories, random_state
        )
        cluster_assignments[name] = 0

    # Group 1: Each sample evolves independently from group1_ancestor
    for i in range(n_samples_per_group):
        name = f"E{i}"
        sample_dict[name] = evolve_sequence(
            group1_ancestor, within_group_branch, n_categories, random_state
        )
        cluster_assignments[name] = 1

    return sample_dict, cluster_assignments, expected_div


def run_branch_length_benchmark(
    n_leaves: int = 200,
    n_features: int = 200,
    n_categories: int = 4,
    branch_lengths: Optional[List[int]] = None,
    mutation_rate: float = 0.30,
    shift_strength: Tuple[float, float] = (0.2, 0.5),
    root_concentration: float = 1.0,
    random_seed: Optional[int] = None,
    method: str = "kl",
    method_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark with fixed leaves, varying branch length.

    Args:
        n_leaves: Total number of samples (fixed). Split 50/50 between groups.
        n_features: Number of features/sites
        n_categories: Categories per feature (4 for DNA)
        branch_lengths: List of branch lengths to test (evolution steps)
        mutation_rate: Probability of mutation per feature per step
        shift_strength: (min, max) distribution shift on mutation
        root_concentration: Dirichlet concentration for ancestor
        random_seed: Seed for reproducibility
        method: Clustering method to use
        method_params: Parameters for the clustering method
        verbose: Print progress

    Returns:
        DataFrame with results for each branch length
    """
    if method not in METHOD_SPECS:
        raise ValueError(f"Unknown method: {method}")

        branch_lengths = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    method_params = method_params or {}
    n_samples_per_group = n_leaves // 2

    results = []

    for bl in branch_lengths:
        # Use SAME base seed for all branch lengths for fair comparison
        rng = np.random.RandomState(random_seed if random_seed else None)

        # Generate data
        sample_dict, cluster_assignments, js_div = generate_two_group_data(
            n_samples_per_group=n_samples_per_group,
            n_features=n_features,
            n_categories=n_categories,
            branch_length=bl,
            random_state=rng,
        )

        # Build dataframe
        sample_names = list(sample_dict.keys())
        matrix = np.array([sample_dict[name] for name in sample_names], dtype=int)
        feature_names = [f"F{j}" for j in range(n_features)]
        data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)
        true_labels = np.array([cluster_assignments[name] for name in sample_names])

        if verbose:
            print(f"Branch length {bl}: {n_leaves} leaves, JS divergence={js_div:.4f}")

        # Run clustering
        pred_labels, n_found, status = run_clustering(
            data_df, method, method_params, random_seed
        )

        if pred_labels is not None and status == "ok":
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            # Check if we found the correct 2-cluster split
            correct_split = n_found == 2 and ari > 0.95
        else:
            ari = np.nan
            nmi = np.nan
            correct_split = False

        results.append(
            {
                "branch_length": bl,
                "n_leaves": n_leaves,
                "n_clusters_true": 2,
                "n_clusters_found": n_found,
                "ari": ari,
                "nmi": nmi,
                "js_divergence": js_div,
                "status": status,
                "correct_split": correct_split,
            }
        )

        if verbose and status == "ok":
            print(f"  Found {n_found} clusters, ARI={ari:.3f}, NMI={nmi:.3f}")

    return pd.DataFrame(results)


__all__ = [
    "run_branch_length_benchmark",
    "generate_two_group_data",
    "plot_branch_length_results",
    "plot_embedding_by_branch_length",
]


def plot_branch_length_results(
    df: pd.DataFrame, title: str = "Branch Length Benchmark"
) -> plt.Figure:
    """Plot ARI/NMI and cluster counts vs branch length.

    Args:
        df: Results DataFrame from run_branch_length_benchmark
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ARI and NMI vs branch length
    ax = axes[0, 0]
    ax.plot(
        df["branch_length"], df["ari"], "b-o", label="ARI", linewidth=2, markersize=8
    )
    ax.plot(
        df["branch_length"], df["nmi"], "g-s", label="NMI", linewidth=2, markersize=8
    )
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Clustering Performance vs Branch Length", fontsize=12, weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Cluster counts
    ax = axes[0, 1]
    ax.axhline(y=2, color="k", linestyle="--", linewidth=2, label="True (2)")
    ax.plot(
        df["branch_length"],
        df["n_clusters_found"],
        "r-o",
        label="Found",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("Number of Clusters", fontsize=11)
    ax.set_title("Cluster Count: True vs Found", fontsize=12, weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # JS Divergence vs branch length
    ax = axes[1, 0]
    ax.plot(df["branch_length"], df["js_divergence"], "m-o", linewidth=2, markersize=8)
    ax.set_xlabel("Branch Length (evolution steps)", fontsize=11)
    ax.set_ylabel("JS Divergence", fontsize=11)
    ax.set_title("Divergence Between Groups", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)

    # ARI vs JS Divergence
    ax = axes[1, 1]
    scatter = ax.scatter(
        df["js_divergence"],
        df["ari"],
        c=df["branch_length"],
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

    fig.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    return fig


def plot_embedding_by_branch_length(
    n_leaves: int = 200,
    n_features: int = 200,
    n_categories: int = 4,
    branch_lengths: Optional[List[int]] = None,
    mutation_rate: float = 0.30,
    shift_strength: Tuple[float, float] = (0.2, 0.5),
    root_concentration: float = 1.0,
    random_seed: Optional[int] = None,
    show_kl_clustering: bool = True,
) -> plt.Figure:
    """Create UMAP/PCA embeddings showing data at different branch lengths.

    Args:
        n_leaves: Total number of samples (fixed)
        n_features: Number of features/sites
        n_categories: Categories per feature
        branch_lengths: List of branch lengths to visualize
        mutation_rate: Probability of mutation per feature per step
        shift_strength: (min, max) distribution shift on mutation
        root_concentration: Dirichlet concentration for ancestor
        random_seed: Seed for reproducibility
        show_kl_clustering: If True, color by KL clustering results instead of true labels

    Returns:
        matplotlib Figure with embedding plots
    """
    if branch_lengths is None:
        branch_lengths = [1, 5, 10, 20]

    n_plots = len(branch_lengths)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # If showing both true and KL, double the rows
    if show_kl_clustering:
        n_rows *= 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    n_samples_per_group = n_leaves // 2

    # Try to use UMAP, fall back to PCA
    try:
        import umap

        use_umap = True
    except ImportError:
        use_umap = False

    # Color palette for multiple clusters
    from matplotlib import cm

    cmap = cm.get_cmap("tab10")

    for idx, bl in enumerate(branch_lengths):
        col = idx % n_cols
        row_true = (idx // n_cols) * 2 if show_kl_clustering else idx // n_cols

        rng = np.random.RandomState(
            int(random_seed + bl * 10000) if random_seed else None
        )

        sample_dict, cluster_assignments, js_div = generate_two_group_data(
            n_samples_per_group=n_samples_per_group,
            n_features=n_features,
            n_categories=n_categories,
            branch_length=bl,
            random_state=rng,
        )

        sample_names = list(sample_dict.keys())
        matrix = np.array([sample_dict[name] for name in sample_names], dtype=int)
        true_labels = np.array([cluster_assignments[name] for name in sample_names])

        # Build dataframe for clustering
        feature_names = [f"F{j}" for j in range(n_features)]
        data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)

        # Scale and embed
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(matrix.astype(float))

        if use_umap:
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(X_scaled) - 1),
                min_dist=0.1,
            )
            embedding = reducer.fit_transform(X_scaled)
        else:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            embedding = pca.fit_transform(X_scaled)

        # Plot true labels
        ax_true = axes[row_true, col]
        colors_true = ["#1f77b4", "#ff7f0e"]  # Blue for ancestral, orange for evolved

        for label in [0, 1]:
            mask = true_labels == label
            label_name = "Ancestral" if label == 0 else "Evolved"
            ax_true.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colors_true[label],
                label=label_name,
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        ax_true.set_title(
            f"True Labels (BL={bl})\nJS={js_div:.4f}", fontsize=10, weight="bold"
        )
        ax_true.set_xlabel("Dim 1", fontsize=8)
        ax_true.set_ylabel("Dim 2", fontsize=8)
        ax_true.legend(frameon=False, fontsize=7, loc="upper right")
        ax_true.grid(True, alpha=0.3)

        # Run KL clustering and plot
        if show_kl_clustering:
            row_kl = row_true + 1
            ax_kl = axes[row_kl, col]

            pred_labels, n_found, status = run_clustering(
                data_df, "kl", {}, random_seed
            )

            if pred_labels is not None and status == "ok":
                unique_labels = np.unique(pred_labels)
                for i, label in enumerate(unique_labels):
                    mask = pred_labels == label
                    color = cmap(i % 10)
                    ax_kl.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        c=[color],
                        label=f"C{label}",
                        alpha=0.7,
                        s=50,
                        edgecolors="black",
                        linewidth=0.5,
                    )

                ari = adjusted_rand_score(true_labels, pred_labels)
                ax_kl.set_title(
                    f"KL Clustering (BL={bl})\nFound {n_found}, ARI={ari:.2f}",
                    fontsize=10,
                    weight="bold",
                )
            else:
                ax_kl.set_title(
                    f"KL Clustering (BL={bl})\nFailed: {status}",
                    fontsize=10,
                    weight="bold",
                )

            ax_kl.set_xlabel("Dim 1", fontsize=8)
            ax_kl.set_ylabel("Dim 2", fontsize=8)
            if n_found <= 6:
                ax_kl.legend(frameon=False, fontsize=7, loc="upper right")
            ax_kl.grid(True, alpha=0.3)

    # Hide unused axes
    for row in range(n_rows):
        for col in range(n_cols):
            idx = (row // (2 if show_kl_clustering else 1)) * n_cols + col
            if idx >= len(branch_lengths):
                axes[row, col].axis("off")

    embed_type = "UMAP" if use_umap else "PCA"
    title = f"{embed_type} Embeddings: True Labels vs KL Clustering\n({n_leaves} leaves fixed)"
    fig.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    return fig
