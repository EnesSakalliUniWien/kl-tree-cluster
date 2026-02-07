"""Embedding comparison plots (UMAP/t-SNE) with cluster coloring."""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

from kl_clustering_analysis.plot.cluster_color_mapping import (
    build_cluster_color_spec,
    present_cluster_ids,
)

# Reduce noisy but expected warnings emitted during visualization
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The spectral clustering API has changed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected, spectral embedding may not work as expected.*",
    category=UserWarning,
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
    labels_dict: dict[str, np.ndarray],
    test_case_num: int,
    meta: dict,
):
    """Create a 2D embedding plot comparing clusterings."""
    X = np.asarray(X_original, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    expected_clusters = int(meta["n_clusters"])

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

    # Add K-Means and Spectral Clustering to the labels_dict
    kmeans = KMeans(n_clusters=expected_clusters, random_state=42, n_init=10)
    labels_dict["K-Means"] = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=expected_clusters,
        random_state=42,
        affinity="nearest_neighbors",
        assign_labels="cluster_qr",
    )
    labels_dict["Spectral"] = spectral.fit_predict(X_scaled)

    n_methods = len(labels_dict)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()

    meta_text = (
        f"samples={meta.get('n_samples', '?')}, "
        f"features={meta.get('n_features', '?')}, "
        f"generator={meta.get('generator', 'unknown')}, "
        f"noise={meta.get('noise', 'n/a')}"
    )
    fig.suptitle(
        (
            f"Test Case {test_case_num}: expected {expected_clusters} clusters\n"
            f"{meta_text}"
        ),
        fontsize=16,
        weight="bold",
        y=0.99,
    )

    all_labels = [labels for labels in labels_dict.values() if labels is not None]
    
    color_clusters = _color_cluster_count(
        expected_clusters, -1, *all_labels # -1 as found_clusters is no longer single
    )
    spec = build_cluster_color_spec(color_clusters, unassigned_color="#CCCCCC")
    colors = spec.colors

    for i, (method_name, labels) in enumerate(labels_dict.items()):
        ax = axes[i]
        if labels is None:
            ax.set_title(f"{method_name}\n(Labels Missing)", fontsize=11, weight="bold")
            ax.axis('off')
            continue

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
        ax.set_title(f"{method_name}\nClustering", fontsize=11, weight="bold")
        ax.set_xlabel("Embedding Dimension 1", fontsize=9)
        ax.set_ylabel("Embedding Dimension 2", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


__all__ = [
    "create_clustering_comparison_plot",
    "_color_cluster_count",
    "create_clustering_comparison_plot_3d",
]


def _fit_embedding_3d(X_scaled: np.ndarray) -> np.ndarray:
    """Return a 3D embedding using UMAP when available, else t-SNE, else PCA."""
    try:
        import umap

        reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            n_neighbors=min(15, len(X_scaled) - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(X_scaled)
    except Exception:
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=min(30, len(X_scaled) - 1),
            )
            return tsne.fit_transform(X_scaled)
        except Exception:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(3, X_scaled.shape[1], len(X_scaled)))
            return pca.fit_transform(X_scaled)


def create_clustering_comparison_plot_3d(
    X_original: np.ndarray,
    y_true: np.ndarray,
    y_kl: np.ndarray,
    test_case_num: int,
    meta: dict,
):
    """Create a 3D embedding plot (UMAP/TSNE/PCA) comparing clusterings."""
    X = np.asarray(X_original, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    expected_clusters = int(meta["n_clusters"])
    found_clusters = int(meta.get("found_clusters", expected_clusters))

    embedding = _fit_embedding_3d(X_scaled)

    kmeans = KMeans(n_clusters=expected_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=expected_clusters,
        random_state=42,
        affinity="nearest_neighbors",
        assign_labels="cluster_qr",
    )
    y_spectral = spectral.fit_predict(X_scaled)

    fig = plt.figure(figsize=(18, 12))
    meta_text = (
        f"samples={meta.get('n_samples', '?')}, "
        f"features={meta.get('n_features', '?')}, "
        f"generator={meta.get('generator', 'unknown')}, "
        f"noise={meta.get('noise', 'n/a')}"
    )
    fig.suptitle(
        (
            f"3D Embedding – Test {test_case_num}: expected {expected_clusters} clusters "
            f"(KL found {found_clusters})\n{meta_text}"
        ),
        fontsize=16,
        weight="bold",
        y=0.97,
    )

    methods = [
        ("Ground Truth", y_true),
        ("KL Divergence", y_kl),
        ("K-Means", y_kmeans),
        ("Spectral", y_spectral),
    ]

    color_clusters = _color_cluster_count(
        expected_clusters, found_clusters, y_true, y_kl, y_kmeans, y_spectral
    )
    spec = build_cluster_color_spec(color_clusters, unassigned_color="#CCCCCC")
    colors = spec.colors

    for i, (method_name, labels) in enumerate(methods, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        labels_array = np.asarray(labels)
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=labels_array,
            cmap=spec.cmap,
            norm=spec.norm,
            alpha=0.7,
            s=40,
            edgecolors="black",
            linewidth=0.3,
        )
        for cluster_id in present_cluster_ids(labels_array):
            if cluster_id >= len(colors):
                continue
            mask = labels_array == cluster_id
            if np.sum(mask) > 0:
                center_x = np.mean(embedding[mask, 0])
                center_y = np.mean(embedding[mask, 1])
                center_z = np.mean(embedding[mask, 2])
                ax.scatter(
                    center_x,
                    center_y,
                    center_z,
                    c=[colors[cluster_id]],
                    marker="x",
                    s=90,
                    linewidth=3,
                )
        ax.set_title(method_name, fontsize=11, weight="bold")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_zlabel("dim 3")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig
