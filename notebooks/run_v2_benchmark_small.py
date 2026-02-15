"""
V2 Benchmark on Penguins Dataset (Small).
Tests the new signal localization logic.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)


def load_penguins():
    print("Loading Palmer Penguins...")
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
    df = pd.read_csv(url).dropna()
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    X = df[feature_cols].values
    species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    y = df["species"].map(species_map).values
    return X, y


def binarize_data(X, threshold=0.0):
    thresholds = np.median(X, axis=0) if threshold is None else threshold
    return (X > thresholds).astype(int)


def run_kl_clustering_v2(X, sample_names=None, verbose=True):
    if sample_names is None:
        sample_names = [f"S{i}" for i in range(X.shape[0])]

    data = pd.DataFrame(X, index=sample_names)

    # Build tree
    Z = linkage(pdist(data.values, metric="hamming"), method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Initialize decomposer manually to access v2
    # Populate stats first
    tree.populate_node_divergences(data)

    from kl_clustering_analysis.hierarchy_analysis.statistics import (
        annotate_child_parent_divergence,
        annotate_sibling_divergence,
    )

    # Annotate and UPDATE stats_df
    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )

    decomposer = TreeDecomposition(
        tree=tree,
        results_df=tree.stats_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
        use_signal_localization=True,  # Enable localization
        localization_max_depth=5,  # Set depth limit
    )

    print("Running decompose_tree_v2()...")
    results = decomposer.decompose_tree_v2()

    n_clusters = results.get("num_clusters", 0)
    print(f"  Found {n_clusters} clusters")

    # Localization stats
    loc_results = results.get("localization_results", {})
    soft_boundaries = sum(1 for r in loc_results.values() if r.has_soft_boundaries)
    print(f"  Split points with soft boundaries: {soft_boundaries}/{len(loc_results)}")

    cluster_assignments = results.get("cluster_assignments", {})
    label_map = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id

    labels = np.array([label_map.get(name, -1) for name in sample_names])
    return labels


def plot_results(X, y, labels, title="Clustering Results"):
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        print("UMAP or Matplotlib not installed, skipping plot.")
        return

    print("Generating UMAP plot...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True Labels
    scatter1 = axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=15, alpha=0.7
    )
    axes[0].set_title(f"{title} - True Labels")

    # Predicted Clusters
    mask = labels >= 0
    if mask.any():
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=labels[mask],
            cmap="tab20",
            s=15,
            alpha=0.7,
        )
    if (~mask).any():
        axes[1].scatter(
            embedding[~mask, 0],
            embedding[~mask, 1],
            c="gray",
            s=10,
            alpha=0.3,
            marker="x",
            label="Unclustered",
        )

    axes[1].set_title(f"{title} - Predicted Clusters")

    plt.tight_layout()
    output_file = "penguins_v2_benchmark.png"
    # Save to absolute path so we can locate it easily
    output_path = Path(repo_root) / output_file
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    X, y = load_penguins()

    # Preprocessing
    X_proc = binarize_data(StandardScaler().fit_transform(X))

    print("\nRunning V2 Benchmark on Penguins (333 samples)...")
    try:
        labels = run_kl_clustering_v2(X_proc)

        # Metrics
        mask = labels >= 0
        if mask.any():
            ari = adjusted_rand_score(y[mask], labels[mask])
            nmi = normalized_mutual_info_score(y[mask], labels[mask])
            print(f"\nResults:")
            print(f"  ARI: {ari:.4f}")
            print(f"  NMI: {nmi:.4f}")

            # Plot
            plot_results(X, y, labels, title="Penguins (V2 Decomposition)")

        else:
            print("No clusters found.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
