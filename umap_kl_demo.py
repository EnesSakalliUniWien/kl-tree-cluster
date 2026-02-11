"""
UMAP + KL-TE clustering demo.

What this script shows:
- Generate synthetic data with known clusters.
- Binarize features to match KL-TE assumptions.
- Build a hierarchy and decompose it into clusters.
- Visualize the clustering result with a UMAP embedding.

Run:
  python debug_scripts/umap_kl_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def main() -> None:
    # 1) Synthetic data with known labels
    X, y_true = make_blobs(
        n_samples=200,
        n_features=20,
        centers=4,
        cluster_std=1.1,
        random_state=42,
    )

    # 2) Binarize features for the KL-TE pipeline
    X_binary = (X > np.median(X, axis=0)).astype(int)
    data = pd.DataFrame(
        X_binary,
        index=[f"Sample_{j}" for j in range(X.shape[0])],
    )

    # 3) Build hierarchy using the library config (distance + linkage)
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

    # 4) Decompose the tree into clusters
    results = tree.decompose(
        leaf_data=data,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    # Map each leaf to its assigned cluster id
    cluster_assignments = results.get("cluster_assignments", {})
    predicted_labels: dict[str, int] = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            predicted_labels[leaf] = cluster_id

    # Convert the mapping to a label per sample (preserves sample order)
    labels = np.array([predicted_labels.get(name, -1) for name in data.index])

    # Print the assigned cluster for each sample (one line per sample)
    print("Cluster labels:")
    for name, label in zip(data.index, labels, strict=True):
        print(f"  {name}: {label}")

    # Also show a compact table view for quick inspection
    cluster_table = (
        pd.DataFrame({"sample": data.index, "cluster": labels})
        .sort_values(["cluster", "sample"], kind="mergesort")
        .reset_index(drop=True)
    )
    print("\nCluster table (sorted by cluster, then sample):")
    print(cluster_table.to_string(index=False))

    # Merge clusters with the original feature data (binary)
    merged_binary = data.copy()
    merged_binary.insert(0, "cluster", labels)
    print("\nMerged (cluster + binary features) preview:")
    print(merged_binary.head(10).to_string())

    # Merge clusters with the original continuous features
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    merged_continuous = pd.DataFrame(X, index=data.index, columns=feature_cols)
    merged_continuous.insert(0, "cluster", labels)
    print("\nMerged (cluster + continuous features) preview:")
    print(merged_continuous.head(10).to_string())

    # Save the table to CSV for easy inspection/reuse
    table_path = Path("cluster_assignments.csv")
    cluster_table.to_csv(table_path, index=False)
    print(f"Saved cluster table to {table_path}")

    # 5) Create a 2D UMAP embedding for visualization
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title("KL-TE Clusters (UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    output_path = Path("umap_kl_clusters.png")
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved UMAP plot to {output_path}")

    # 6) Quick quality check using ARI against ground truth
    ari = adjusted_rand_score(y_true, labels)
    print(f"ARI: {ari:.4f}")


if __name__ == "__main__":
    main()
