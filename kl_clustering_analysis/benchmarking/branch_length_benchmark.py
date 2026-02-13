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

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kl_clustering_analysis import config
from benchmarks.shared.runners.method_registry import METHOD_SPECS


def _jukes_cantor_transition_matrix(
    n_categories: int,
    branch_length: float,
) -> np.ndarray:
    """Compute Jukes-Cantor transition probability matrix.

    P(j|i, t) = (1/k) + (1 - 1/k) * exp(-k*mu*t)  if i == j
              = (1/k) * (1 - exp(-k*mu*t))        if i != j

    where k = n_categories, mu = 1 (normalized), t = branch_length

    Args:
        n_categories: Number of states (e.g., 4 for DNA)
        branch_length: Evolutionary distance (expected substitutions per site)

    Returns:
        Transition probability matrix of shape (n_categories, n_categories)
    """
    k = n_categories
    # Probability of staying in same state
    p_same = (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * branch_length / (k - 1))
    # Probability of changing to any other state
    p_diff = (1.0 / k) * (1 - np.exp(-k * branch_length / (k - 1)))

    # Build transition matrix
    P = np.full((k, k), p_diff)
    np.fill_diagonal(P, p_same)

    return P


def _generate_ancestral_sequence(
    n_features: int,
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate a random ancestral sequence with uniform base frequencies."""
    return random_state.randint(0, n_categories, size=n_features)


def _evolve_sequence(
    ancestor: np.ndarray,
    branch_length: float,
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Evolve a sequence along a branch using Jukes-Cantor model.

    Args:
        ancestor: Ancestral sequence (array of integers 0 to n_categories-1)
        branch_length: Evolutionary distance
        n_categories: Number of states
        random_state: Random number generator

    Returns:
        Evolved sequence
    """
    P = _jukes_cantor_transition_matrix(n_categories, branch_length)
    evolved = np.zeros_like(ancestor)

    for i, state in enumerate(ancestor):
        evolved[i] = random_state.choice(n_categories, p=P[state])

    return evolved


def _compute_expected_divergence(branch_length: float, n_categories: int) -> float:
    """Compute expected proportion of differing sites under Jukes-Cantor.

    d = (k-1)/k * (1 - exp(-k*t/(k-1)))

    where k = n_categories, t = branch_length
    """
    k = n_categories
    return ((k - 1.0) / k) * (1.0 - np.exp(-k * branch_length / (k - 1)))


def generate_two_group_data(
    n_samples_per_group: int,
    n_features: int,
    n_categories: int,
    branch_length: float,
    random_state: Optional[np.random.RandomState] = None,
    within_group_branch: float = 0.05,
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
    root = _generate_ancestral_sequence(n_features, n_categories, random_state)

    # Create group ancestors (MRCA of each group)
    # Group 0: Very short stem branch from root
    group0_ancestor = _evolve_sequence(root, 0.01, n_categories, random_state)

    # Group 1: Longer stem branch from root (this is the key parameter)
    group1_ancestor = _evolve_sequence(root, branch_length, n_categories, random_state)

    # Compute expected divergence between group ancestors
    expected_div = _compute_expected_divergence(branch_length, n_categories)

    sample_dict = {}
    cluster_assignments = {}

    # Group 0: Each sample evolves independently from group0_ancestor
    # with its own terminal branch
    for i in range(n_samples_per_group):
        name = f"A{i}"
        sample_dict[name] = _evolve_sequence(
            group0_ancestor, within_group_branch, n_categories, random_state
        )
        cluster_assignments[name] = 0

    # Group 1: Each sample evolves independently from group1_ancestor
    for i in range(n_samples_per_group):
        name = f"E{i}"
        sample_dict[name] = _evolve_sequence(
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

    if branch_lengths is None:
        branch_lengths = [1, 2, 3, 5, 8, 10, 15, 20, 30]

    method_params = method_params or {}
    n_samples_per_group = n_leaves // 2

    results = []

    for bl in branch_lengths:
        # Use SAME base seed for all branch lengths for fair comparison
        rng = np.random.RandomState(random_seed if random_seed is not None else None)

        # Generate data
        sample_dict, cluster_assignments, _ = generate_two_group_data(
            n_samples_per_group=n_samples_per_group,
            n_features=n_features,
            n_categories=n_categories,
            branch_length=bl,
            random_state=rng,
        )

        # Build dataframe
        sample_names = list(sample_dict.keys())
        X = np.array([sample_dict[name] for name in sample_names])
        data_df = pd.DataFrame(X, index=sample_names)

        true_labels = np.array([cluster_assignments[name] for name in sample_names])
        true_k = len(np.unique(true_labels))

        # Run clustering
        labels, found_k, status = run_clustering(data_df, method, method_params, seed=random_seed)

        if labels is None:
            if verbose:
                print(f"  BL={bl:.2f}: {status}")
            continue

        # Compute metrics
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        # Purity
        purity = 0.0
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            if np.sum(mask) > 0:
                cluster_true = true_labels[mask]
                most_common = np.bincount(cluster_true).argmax()
                purity += np.sum(cluster_true == most_common)
        purity /= len(labels)

        results.append(
            {
                "Test": "branch_length",
                "Case_Name": f"bl_{bl:.2f}",
                "Method": method,
                "Params": str(method_params),
                "True": true_k,
                "Found": found_k,
                "Samples": n_leaves,
                "Features": n_features,
                "Noise": 0.0,
                "Branch_length": bl,
                "Expected_Divergence": js_div,
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
                "Status": status,
            }
        )

        if verbose:
            print(f"  BL={bl:.2f}: K={found_k}, ARI={ari:.3f}, NMI={nmi:.3f}, Purity={purity:.3f}")

    return pd.DataFrame(results)


__all__ = [
    "run_branch_length_benchmark",
    "generate_two_group_data",
    "plot_branch_length_results",
    "plot_embedding_by_branch_length",
]


def plot_branch_length_results(
    results_df: pd.DataFrame,
    n_leaves: int,
    n_features: int,
    n_categories: int,
    branch_lengths: Optional[List[float]] = None,
    random_seed: Optional[int] = None,
    show_kl_clustering: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot branch length benchmark results.

    Creates a grid of plots showing:
    - True labels (top row if show_kl_clustering)
    - KL clustering results (bottom row if show_kl_clustering)

    Each column corresponds to a different branch length.

    """
    _ = results_df  # Reserved for future summary overlays; currently unused.

    if branch_lengths is None:
        branch_lengths = [1, 2, 3, 5, 8, 10, 15, 20, 30]

    n_plots = len(branch_lengths)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
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

        rng = np.random.RandomState((random_seed + bl) if random_seed is not None else None)

        sample_dict, cluster_assignments, _ = generate_two_group_data(
            n_samples_per_group=n_samples_per_group,
            n_features=n_features,
            n_categories=n_categories,
            branch_length=bl,
            random_state=rng,
        )

        sample_names = list(sample_dict.keys())
        X = np.array([sample_dict[name] for name in sample_names])
        data_df = pd.DataFrame(X, index=sample_names)

        true_labels = np.array([cluster_assignments[name] for name in sample_names])

        # Plot true labels
        if use_umap:
            reducer = umap.UMAP(n_components=2, random_state=random_seed)
            X_2d = reducer.fit_transform(X)
        else:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=random_seed)
            X_2d = pca.fit_transform(X)

        ax = axes[row_true, col]
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=[cmap(label) for label in true_labels],
            s=50,
            alpha=0.7,
        )
        ax.set_title(f"True Labels\nBL={bl:.2f}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        # Plot KL clustering results if requested
        if show_kl_clustering:
            row_kl = row_true + 1
            labels, found_k, status = run_clustering(data_df, "kl", {}, seed=random_seed)

            if labels is not None:
                ax_kl = axes[row_kl, col]
                ax_kl.scatter(
                    X_2d[:, 0],
                    X_2d[:, 1],
                    c=[cmap(label) for label in labels],
                    s=50,
                    alpha=0.7,
                )
                ax_kl.set_title(f"KL Clustering\nK={found_k}")
                ax_kl.set_xlabel("Component 1")
                ax_kl.set_ylabel("Component 2")
            else:
                axes[row_kl, col].text(
                    0.5,
                    0.5,
                    f"Failed: {status}",
                    ha="center",
                    va="center",
                    transform=axes[row_kl, col].transAxes,
                )
                axes[row_kl, col].set_title(f"KL Clustering\nBL={bl:.2f}")

    # Hide unused subplots
    for idx in range(len(branch_lengths), n_cols * n_rows):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def run_clustering(
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int, str]:
    """Run clustering and return (labels, n_clusters, status)."""
    spec = METHOD_SPECS[method_id]
    distance_condensed = pdist(data_df.to_numpy(), metric=config.TREE_DISTANCE_METRIC)
    distance_matrix = squareform(distance_condensed)

    try:
        if method_id == "kl":
            result = spec.runner(
                data_df,
                distance_condensed,
                config.SIBLING_ALPHA,
                tree_linkage_method=params.get("tree_linkage_method", config.TREE_LINKAGE_METHOD),
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


def main() -> None:
    """Main entry point for branch length benchmark."""
    # Default parameters
    n_leaves = 40
    n_features = 100
    n_categories = 4  # DNA-like
    branch_lengths = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    random_seed = 42

    print("Running branch length benchmark...")
    print(f"  n_leaves: {n_leaves}")
    print(f"  n_features: {n_features}")
    print(f"  n_categories: {n_categories}")
    print(f"  branch_lengths: {branch_lengths}")
    print()

    # Run benchmark
    results_df = run_branch_length_benchmark(
        n_leaves=n_leaves,
        n_features=n_features,
        n_categories=n_categories,
        branch_lengths=branch_lengths,
        random_seed=random_seed,
        method="kl",
        verbose=True,
    )

    # Plot results
    plot_branch_length_results(
        results_df=results_df,
        n_leaves=n_leaves,
        n_features=n_features,
        n_categories=n_categories,
        branch_lengths=branch_lengths,
        random_seed=random_seed,
        show_kl_clustering=True,
    )


if __name__ == "__main__":
    main()
