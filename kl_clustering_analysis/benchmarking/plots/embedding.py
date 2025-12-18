"""Embedding comparison plots (UMAP/t-SNE) with cluster coloring."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

from kl_clustering_analysis.plot.cluster_color_mapping import (
    build_cluster_color_spec,
    present_cluster_ids,
)


def _max_assigned_label(labels: np.ndarray | None) -> int:
    """Maximum non-negative cluster label in an array (or -1 if none)."""
    if labels is None:
        return -1
    arr = np.asarray(labels)
    if arr.size == 0:
        return -1
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        return -1
    return int(arr.max())


def _color_cluster_count(expected: int, found: int, *label_arrays: np.ndarray) -> int:
    """
    Determine how many cluster colors are needed to cover all label arrays.

    Uses the maximum label value (>=0) plus one so colors line up with the
    actual cluster IDs returned by the algorithms, not just the expected count.
    """
    max_label_seen = max(_max_assigned_label(arr) for arr in label_arrays)
    base = max(expected, found)
    if max_label_seen >= 0:
        base = max(base, max_label_seen + 1)
    return base


def create_clustering_comparison_plot(
    X_original: np.ndarray,
    y_true: np.ndarray,
    y_kl: np.ndarray,
    test_case_num: int,
    meta: dict,
):
    """Create a 2D embedding plot comparing clusterings."""
    X = np.asarray(X_original, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    expected_clusters = int(meta["n_clusters"])
    found_clusters = int(meta.get("found_clusters", expected_clusters))

    # UMAP if available, else t-SNE
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(X_scaled) - 1),
            min_dist=0.1,
        )
        X_embedded = reducer.fit_transform(X_scaled)
    except Exception:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1)
        )
        X_embedded = tsne.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=expected_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=expected_clusters,
        random_state=42,
        affinity="nearest_neighbors",
        assign_labels="cluster_qr",
    )
    y_spectral = spectral.fit_predict(X_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    meta_text = (
        f"samples={meta.get('n_samples', '?')}, "
        f"features={meta.get('n_features', '?')}, "
        f"generator={meta.get('generator', 'unknown')}, "
        f"noise={meta.get('noise', 'n/a')}"
    )
    fig.suptitle(
        (
            f"Test Case {test_case_num}: expected {expected_clusters} clusters "
            f"(KL found {found_clusters})\n{meta_text}"
        ),
        fontsize=16,
        weight="bold",
        y=0.99,
    )

    methods = [
        ("Ground Truth", y_true, "True cluster labels"),
        ("KL Divergence", y_kl, "KL clustering"),
        ("K-Means", y_kmeans, "K-means clustering"),
        ("Spectral", y_spectral, "Spectral clustering"),
        ("Embedding X", X_embedded[:, 0], "Embedding dimension 1"),
        ("Embedding Y", X_embedded[:, 1], "Embedding dimension 2"),
    ]

    color_clusters = _color_cluster_count(
        expected_clusters, found_clusters, y_true, y_kl, y_kmeans, y_spectral
    )
    spec = build_cluster_color_spec(color_clusters, unassigned_color="#CCCCCC")
    colors = spec.colors

    for i, (method_name, labels, description) in enumerate(methods):
        ax = axes[i // 3, i % 3]
        if method_name in {"Ground Truth", "KL Divergence", "K-Means", "Spectral"}:
            labels_array = np.asarray(labels)
            ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels_array,
                cmap=spec.cmap,
                norm=spec.norm,
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            for cluster_id in present_cluster_ids(labels_array):
                if cluster_id >= len(colors):
                    continue
                mask = labels_array == cluster_id
                if np.sum(mask) > 0:
                    center_x = np.mean(X_embedded[mask, 0])
                    center_y = np.mean(X_embedded[mask, 1])
                    ax.scatter(
                        center_x,
                        center_y,
                        c=[colors[cluster_id]],
                        marker="x",
                        s=100,
                        linewidth=3,
                    )
        else:
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            plt.colorbar(scatter, ax=ax, shrink=0.8)

        ax.set_title(f"{method_name}\n{description}", fontsize=11, weight="bold")
        ax.set_xlabel("Embedding Dimension 1", fontsize=9)
        ax.set_ylabel("Embedding Dimension 2", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


__all__ = [
    "create_clustering_comparison_plot",
    "_color_cluster_count",
]
