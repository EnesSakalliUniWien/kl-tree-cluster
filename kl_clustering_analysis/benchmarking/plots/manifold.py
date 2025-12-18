"""Manifold alignment plots (UMAP vs Isomap) with KL cluster coloring."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from skbio.stats.distance import mantel

from kl_clustering_analysis.plot.cluster_color_mapping import build_cluster_color_spec
from .embedding import _color_cluster_count


def _pad_to_three_dims(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] >= 3:
        return embedding[:, :3]
    pad = np.zeros((embedding.shape[0], 3 - embedding.shape[1]))
    return np.hstack([embedding, pad])


def create_manifold_alignment_plot(
    X_original: np.ndarray,
    y_kl: np.ndarray,
    test_case_num: int,
    meta: dict,
    y_true: np.ndarray | None = None,
):
    """Compare UMAP and Isomap manifolds with KL cluster coloring."""
    labels_array = np.asarray(y_kl)
    X = np.asarray(X_original, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        import umap

        n_neighbors = min(15, len(X_scaled) - 1) if len(X_scaled) > 1 else 1
        reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            n_neighbors=max(2, n_neighbors),
            min_dist=0.05,
            metric="euclidean",
        )
        embedding_umap = reducer.fit_transform(X_scaled)
    except Exception:
        pca = PCA(n_components=min(3, X_scaled.shape[1], len(X_scaled)))
        embedding_umap = pca.fit_transform(X_scaled)

    embedding_umap = _pad_to_three_dims(embedding_umap)

    n_neighbors_iso = min(10, len(X_scaled) - 1) if len(X_scaled) > 1 else 1
    embedding_iso = Isomap(
        n_neighbors=max(2, n_neighbors_iso),
        n_components=3,
        metric="euclidean",
    ).fit_transform(X_scaled)
    embedding_iso = _pad_to_three_dims(embedding_iso)

    dist_umap = pairwise_distances(embedding_umap)
    dist_iso = pairwise_distances(embedding_iso)
    mantel_stat, mantel_p, _ = mantel(
        dist_umap,
        dist_iso,
        method="pearson",
        permutations=199,
    )

    upper_idx = np.triu_indices_from(dist_umap, k=1)

    found_clusters = int(meta.get("found_clusters", meta["n_clusters"]))
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        (
            f"Test Case {test_case_num}: Manifold Alignment "
            f"(expected={meta['n_clusters']}, KL found={found_clusters})\n"
            f"UMAP vs Isomap concordance r={mantel_stat:.2f}, p={mantel_p:.3f}"
        ),
        fontsize=15,
        weight="bold",
    )

    color_clusters = _color_cluster_count(
        int(meta["n_clusters"]), found_clusters, labels_array
    )
    cluster_spec = build_cluster_color_spec(color_clusters, unassigned_color="#CCCCCC")

    def _scatter3d(ax, embedding, title: str) -> None:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=labels_array,
            cmap=cluster_spec.cmap,
            norm=cluster_spec.norm,
            s=35,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_zlabel("dim 3")
        if y_true is not None:
            ax.text(0.02, 0.92, "color = KL labels", transform=ax.transAxes)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _scatter3d(ax1, embedding_umap, "3D UMAP embedding")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _scatter3d(ax2, embedding_iso, "3D Isomap embedding")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(
        dist_umap[upper_idx],
        dist_iso[upper_idx],
        s=6,
        alpha=0.35,
        color="#1f77b4",
    )
    ax3.axline((0, 0), slope=1.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_xlabel("UMAP pairwise distance")
    ax3.set_ylabel("Isomap pairwise distance")
    ax3.set_title("Distance concordance")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, float(mantel_stat), float(mantel_p)


__all__ = ["create_manifold_alignment_plot"]
