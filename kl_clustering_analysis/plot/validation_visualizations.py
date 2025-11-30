"""
Validation and comparison visualizations for clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from skbio.stats.distance import mantel
import warnings


def create_validation_plot(df_results: pd.DataFrame) -> plt.Figure:
    """Create 4-panel visualization of validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Algorithm Performance Across Test Cases", fontsize=16, weight="bold", y=0.995
    )

    df_results = df_results.reset_index(drop=True)
    x = np.arange(len(df_results))
    xtick_labels = [f"T{row.Test}" for row in df_results.itertuples(index=False)]

    # 1. Cluster Detection Accuracy
    ax1 = axes[0, 0]
    width = 0.35
    ax1.bar(
        x - width / 2,
        df_results["True"],
        width,
        label="True Clusters",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
    )
    ax1.bar(
        x + width / 2,
        df_results["Found"],
        width,
        label="Found Clusters",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
    )
    for idx, row in df_results.iterrows():
        match = "✓" if row["True"] == row["Found"] else "✗"
        color = "green" if row["True"] == row["Found"] else "red"
        ax1.text(
            x[idx],
            max(row["True"], row["Found"]) + 0.3,
            match,
            ha="center",
            fontsize=20,
            weight="bold",
            color=color,
        )
    ax1.set_xlabel("Test Case", fontsize=11, weight="bold")
    ax1.set_ylabel("Number of Clusters", fontsize=11, weight="bold")
    ax1.set_title("Cluster Count: Expected vs Found", fontsize=12, weight="bold")
    ax1.legend(loc="upper left")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=40, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Quality Metrics
    ax2 = axes[0, 1]
    x_pos = x
    ax2.plot(
        x_pos,
        df_results["ARI"],
        "o-",
        linewidth=2.5,
        markersize=10,
        label="ARI",
        color="#e74c3c",
    )
    ax2.plot(
        x_pos,
        df_results["NMI"],
        "s-",
        linewidth=2.5,
        markersize=10,
        label="NMI",
        color="#9b59b6",
    )
    ax2.plot(
        x_pos,
        df_results["Purity"],
        "^-",
        linewidth=2.5,
        markersize=10,
        label="Purity",
        color="#f39c12",
    )
    ax2.axhline(
        y=1.0, color="green", linestyle="--", alpha=0.3, linewidth=2, label="Perfect"
    )
    ax2.set_xlabel("Test Case", fontsize=11, weight="bold")
    ax2.set_ylabel("Score", fontsize=11, weight="bold")
    ax2.set_title("Clustering Quality Metrics", fontsize=12, weight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(xtick_labels, rotation=40, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # 3. Dataset Complexity
    ax3 = axes[1, 0]
    colors = [
        "#2ecc71" if row["True"] == row["Found"] else "#e74c3c"
        for _, row in df_results.iterrows()
    ]
    ax3.scatter(
        df_results["Samples"],
        df_results["Features"],
        s=df_results["True"] * 100,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=2,
    )
    for i, row in df_results.iterrows():
        ax3.annotate(
            f"Test {row['Test']}\n{row['True']}→{row['Found']}",
            (row["Samples"], row["Features"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
    ax3.set_xlabel("Number of Samples", fontsize=11, weight="bold")
    ax3.set_ylabel("Number of Features", fontsize=11, weight="bold")
    ax3.set_title(
        "Dataset Complexity (size = true clusters, color = match)",
        fontsize=12,
        weight="bold",
    )
    ax3.grid(alpha=0.3)

    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate summary
    matches = sum(df_results["True"] == df_results["Found"])
    accuracy = matches / len(df_results) * 100
    avg_ari = df_results["ARI"].mean()
    avg_nmi = df_results["NMI"].mean()
    avg_purity = df_results["Purity"].mean()
    run_count = len(df_results)

    summary_text = f"""
SUMMARY STATISTICS

Cluster Detection
  Correct:     {matches}/{len(df_results)} ({accuracy:.0f}%)
  Runs:        {run_count}
  
Quality Metrics (Average)
  ARI:         {avg_ari:.3f}
  NMI:         {avg_nmi:.3f}
  Purity:      {avg_purity:.3f}
  
Test Complexity
  Samples:     {df_results["Samples"].min()}-{df_results["Samples"].max()}
  Features:    {df_results["Features"].min()}-{df_results["Features"].max()}
  Clusters:    {df_results["True"].min()}-{df_results["True"].max()}
  Noise (σ):   {df_results["Noise"].min()}-{df_results["Noise"].max()}
"""

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    return fig


def create_umap_comparison_plot(
    X_original: np.ndarray,
    y_true: np.ndarray,
    y_kl: np.ndarray,
    test_case_num: int,
    n_clusters: int,
) -> plt.Figure:
    """Create UMAP visualization comparing KL clustering with other methods."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Standardize the data for better UMAP performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)

        # Reduce dimensionality if too high for UMAP
        if X_scaled.shape[1] > 50:
            pca = PCA(
                n_components=min(50, X_scaled.shape[0] - 1, X_scaled.shape[1]),
                random_state=42,
            )
            X_scaled = pca.fit_transform(X_scaled)

        # Apply UMAP for dimensionality reduction
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(X_scaled) - 1),
                min_dist=0.1,
            )
            X_embedded = reducer.fit_transform(X_scaled)
        except ImportError:
            # Fallback to t-SNE if UMAP not available
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1)
            )
            X_embedded = tsne.fit_transform(X_scaled)

        # Apply comparison clustering methods
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X_scaled)

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="nearest_neighbors",
            assign_labels="cluster_qr",
        )
        y_spectral = spectral.fit_predict(X_scaled)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Test Case {test_case_num}: {n_clusters} Clusters - Clustering Comparison",
        fontsize=16,
        weight="bold",
        y=0.995,
    )

    methods = [
        ("Ground Truth", y_true, "True cluster labels"),
        ("KL Divergence", y_kl, "Local KL divergence clustering"),
        ("K-Means", y_kmeans, "K-means clustering"),
        ("Spectral", y_spectral, "Spectral clustering"),
        ("Embedding X", X_embedded[:, 0], "Embedding dimension 1 (no clustering)"),
        ("Embedding Y", X_embedded[:, 1], "Embedding dimension 2 (no clustering)"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i, (method_name, labels, description) in enumerate(methods):
        ax = axes[i // 3, i % 3]

        if method_name in ["Ground Truth", "KL Divergence", "K-Means", "Spectral"]:
            # Clustering results - color by cluster
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            # Add cluster centers
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
            # Raw embedding dimensions - color by value
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
    """Ensure the embedding has three columns for 3D visualization."""

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
) -> tuple[plt.Figure, float, float]:
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
    except ImportError:
        # Fallback to PCA if UMAP unavailable
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

    def _scatter3d(ax, embedding, title: str) -> None:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=y_kl,
            cmap="tab10",
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


def create_manifold_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    """Generate manifold alignment plots for each computed test case."""

    output_dir.mkdir(exist_ok=True)

    for result in test_results:
        i = result["test_case_num"]
        meta = result["meta"]
        mode_label = result.get("local_mode")
        mode_prefix = f"{mode_label}_" if mode_label else ""
        if verbose:
            label_str = f" [{mode_label}]" if mode_label else ""
            print(f"  Creating manifold plot for test case {i}{label_str}...")
        fig, mantel_r, mantel_p = create_manifold_alignment_plot(
            result["X_original"],
            result["kl_labels"],
            test_case_num=i,
            meta=meta,
            y_true=result.get("y_true"),
        )
        manifold_filename = (
            f"{mode_prefix}manifold_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png"
        )
        fig.savefig(output_dir / manifold_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(
                f"    Saved manifold diagnostics (Mantel r={mantel_r:.2f}, p={mantel_p:.3f})"
            )


def create_tree_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    """
    Create tree visualizations from already computed test results.

    Parameters
    ----------
    test_results : list
        List of dictionaries containing test results with keys:
        - tree, decomposition, meta, test_case_num
    output_dir : Path
        Directory to save plots
    timestamp : str
        Timestamp string for filenames
    verbose : bool
        Whether to print progress
    """
    from .cluster_tree_visualization import (
        plot_tree_with_clusters,
        plot_cluster_summary,
    )

    output_dir.mkdir(exist_ok=True)

    for result in test_results:
        tree_t = result["tree"]
        decomp_t = result["decomposition"]
        meta = result["meta"]
        i = result["test_case_num"]
        mode_label = result.get("local_mode")
        mode_prefix = f"{mode_label}_" if mode_label else ""

        if verbose:
            label_str = f" [{mode_label}]" if mode_label else ""
            print(f"  Creating tree visualization for test case {i}{label_str}...")

        # Create tree visualization
        noise_label = "σ" if meta["generator"] == "blobs" else "entropy"
        noise_value = meta["noise"]
        if isinstance(noise_value, (int, float, np.floating, np.integer)):
            noise_value_str = f"{float(noise_value):.2f}"
        else:
            noise_value_str = str(noise_value)
        test_case_name = (
            f"Test Case {i}: {meta['n_clusters']} Clusters "
            f"({noise_label}={noise_value_str})"
        )
        if meta.get("name"):
            test_case_name += f" [{meta['name']}]"

        tree_fig, _, _ = plot_tree_with_clusters(
            tree=tree_t,
            decomposition_results=decomp_t,
            use_labels=True,
            figsize=(20, 14),
            node_size=2500,
            font_size=9,
            show_cluster_boundaries=True,
            title=f"Hierarchical Tree with KL Divergence Clusters\n{test_case_name}",
        )

        tree_filename = (
            f"{mode_prefix}tree_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png"
        )
        tree_fig.savefig(
            output_dir / tree_filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(tree_fig)

        # Create cluster summary visualization
        if decomp_t["num_clusters"] > 0:
            summary_fig, _ = plot_cluster_summary(decomp_t, figsize=(14, 6))
            summary_fig.suptitle(
                f"Cluster Analysis Summary - {test_case_name}",
                fontsize=16,
                weight="bold",
                y=0.98,
            )

            summary_filename = (
                f"{mode_prefix}summary_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png"
            )
            summary_fig.savefig(
                output_dir / summary_filename,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(summary_fig)


def create_umap_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    """
    Create UMAP comparison plots from already computed test results.

    Parameters
    ----------
    test_results : list
        List of dictionaries containing test results with keys:
        - X_original, y_true, kl_labels, meta, test_case_num
    output_dir : Path
        Directory to save plots
    timestamp : str
        Timestamp string for filenames
    verbose : bool
        Whether to print progress
    """
    output_dir.mkdir(exist_ok=True)

    for result in test_results:
        X_original = result["X_original"]
        y_true = result["y_true"]
        kl_labels = result["kl_labels"]
        meta = result["meta"]
        i = result["test_case_num"]
        mode_label = result.get("local_mode")
        mode_prefix = f"{mode_label}_" if mode_label else ""

        if verbose:
            label_str = f" [{mode_label}]" if mode_label else ""
            print(f"  Creating UMAP visualization for test case {i}{label_str}...")

        # Create UMAP comparison visualization
        umap_fig = create_umap_comparison_plot(
            X_original, y_true, kl_labels, i, meta["n_clusters"]
        )

        umap_filename = (
            f"{mode_prefix}umap_test_{i}_{meta['n_clusters']}_clusters_{timestamp}.png"
        )
        umap_fig.savefig(
            output_dir / umap_filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(umap_fig)
