"""
Matplotlib-based validation visualizations.

Legacy HoloViews/HoloViz rendering has been removed. This module contains simple
matplotlib plots that can be saved to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from skbio.stats.distance import mantel

from .cluster_color_mapping import build_cluster_color_spec


def create_validation_plot(df_results):
    """Create a compact summary plot for validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    if df_results.empty:
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Validation Results (empty)")
        return fig

    axes[0].plot(df_results["True"], label="True", marker="o")
    axes[0].plot(df_results["Found"], label="Found", marker="o")
    axes[0].set_title("Clusters: True vs Found")
    axes[0].set_xlabel("Test case")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(df_results["ARI"], marker="o")
    axes[1].set_title("ARI")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_results["NMI"], marker="o")
    axes[2].set_title("NMI")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df_results["Purity"], marker="o")
    axes[3].set_title("Purity (Homogeneity)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Cluster Validation Summary", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig


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

    n_clusters = int(meta["n_clusters"])

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

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity="nearest_neighbors",
        assign_labels="cluster_qr",
    )
    y_spectral = spectral.fit_predict(X_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Test Case {test_case_num}: {n_clusters} Clusters - Clustering Comparison",
        fontsize=16,
        weight="bold",
        y=0.995,
    )

    methods = [
        ("Ground Truth", y_true, "True cluster labels"),
        ("KL Divergence", y_kl, "KL clustering"),
        ("K-Means", y_kmeans, "K-means clustering"),
        ("Spectral", y_spectral, "Spectral clustering"),
        ("Embedding X", X_embedded[:, 0], "Embedding dimension 1"),
        ("Embedding Y", X_embedded[:, 1], "Embedding dimension 2"),
    ]

    spec = build_cluster_color_spec(n_clusters, unassigned_color="#CCCCCC")
    colors = spec.colors

    for i, (method_name, labels, description) in enumerate(methods):
        ax = axes[i // 3, i % 3]
        if method_name in {"Ground Truth", "KL Divergence", "K-Means", "Spectral"}:
            ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap=spec.cmap,
                norm=spec.norm,
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
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

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        (
            f"Test Case {test_case_num}: Manifold Alignment (n_clusters={meta['n_clusters']})\n"
            f"UMAP vs Isomap concordance r={mantel_stat:.2f}, p={mantel_p:.3f}"
        ),
        fontsize=15,
        weight="bold",
    )

    cluster_spec = build_cluster_color_spec(
        int(meta["n_clusters"]), unassigned_color="#CCCCCC"
    )

    def _scatter3d(ax, embedding, title: str) -> None:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=y_kl,
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


def create_umap_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        i = result["test_case_num"]
        meta = result["meta"]
        if verbose:
            print(f"  Creating UMAP comparison plot for test case {i}...")
        fig = create_clustering_comparison_plot(
            result["X_original"],
            result.get("y_true"),
            np.asarray(result["kl_labels"]),
            test_case_num=i,
            meta=meta,
        )
        fig.savefig(
            output_dir / f"umap_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def create_manifold_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        i = result["test_case_num"]
        meta = result["meta"]
        if verbose:
            print(f"  Creating manifold plot for test case {i}...")
        fig, mantel_r, mantel_p = create_manifold_alignment_plot(
            result["X_original"],
            np.asarray(result["kl_labels"]),
            test_case_num=i,
            meta=meta,
            y_true=result.get("y_true"),
        )
        fig.savefig(
            output_dir
            / f"manifold_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        if verbose:
            print(
                f"    Saved manifold diagnostics (r={mantel_r:.2f}, p={mantel_p:.3f})"
            )


def create_tree_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    from .cluster_tree_visualization import plot_tree_with_clusters

    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        tree_t = result["tree"]
        decomp_t = result["decomposition"]
        meta = result["meta"]
        i = result["test_case_num"]
        if verbose:
            print(f"  Creating tree plot for test case {i}...")
        fig, _ax = plot_tree_with_clusters(
            tree=tree_t,
            decomposition_results=decomp_t,
            use_labels=True,
            width=1000,
            height=700,
            node_size=20,
            font_size=9,
            title=f"Hierarchical Tree with KL Divergence Clusters\nTest Case {i}",
            show=False,
        )
        fig.savefig(
            output_dir / f"tree_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
