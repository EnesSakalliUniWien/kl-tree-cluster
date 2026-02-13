"""Embedding comparison plots (UMAP/t-SNE) with cluster coloring."""

from __future__ import annotations

import os
import textwrap
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kl_clustering_analysis.plot.cluster_color_mapping import (
    build_cluster_color_spec,
    present_cluster_ids,
)
from benchmarks.shared.pdf_utils import PDF_PAGE_SIZE_INCHES

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


def _fit_embedding_2d(X_scaled: np.ndarray) -> np.ndarray:
    """Return a 2D embedding with a stability-first backend strategy."""
    backend = (os.getenv("KL_TE_EMBEDDING_BACKEND") or "umap").strip().lower()

    if backend in {"umap", "auto"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, max(2, len(X_scaled) - 1)),
                min_dist=0.1,
                low_memory=True,
            )
            return reducer.fit_transform(X_scaled)
        except Exception as exc:
            raise RuntimeError(
                "UMAP embedding was requested but failed to initialize/fit."
            ) from exc

    if backend == "tsne":
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, max(2, len(X_scaled) - 1)),
            )
            return tsne.fit_transform(X_scaled)
        except Exception:
            pass

    # Stable fallback: PCA
    pca = PCA(n_components=min(2, X_scaled.shape[1], len(X_scaled)))
    return pca.fit_transform(X_scaled)


def _format_method_subplot_title(method_name: str, max_line_chars: int = 26) -> str:
    """Compact long method/param labels for subplot titles."""
    name = str(method_name).strip()
    if " (" in name and name.endswith(")"):
        method_part, param_part = name.split(" (", 1)
        param_part = param_part[:-1]
        compact_params: list[str] = []
        for token in param_part.split(","):
            token = token.strip()
            if token.startswith("tree_distance_metric="):
                compact_params.append(token.replace("tree_distance_metric=", "metric="))
            elif token.startswith("tree_linkage_method="):
                compact_params.append(token.replace("tree_linkage_method=", "linkage="))
            elif token:
                compact_params.append(token)
        compact_param_str = ", ".join(compact_params)
        if len(compact_param_str) > 40:
            compact_param_str = compact_param_str[:37] + "..."
        name = f"{method_part} ({compact_param_str})"

    if len(name) > max_line_chars * 2:
        name = name[: max_line_chars * 2 - 3] + "..."
    wrapped = "\n".join(
        textwrap.wrap(name, width=max_line_chars, break_long_words=False)
    )
    return f"{wrapped}\nClustering"


def create_clustering_comparison_plot(
    X_original: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    test_case_num: int,
    meta: dict,
):
    """Create a single 2D embedding plot page (backward-compatible wrapper)."""
    figs = create_clustering_comparison_plots(
        X_original=X_original,
        labels_dict=labels_dict,
        test_case_num=test_case_num,
        meta=meta,
    )
    return figs[0]


def create_clustering_comparison_plots(
    X_original: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    test_case_num: int,
    meta: dict,
    *,
    n_cols: int = 3,
    max_panels_per_page: int = 6,
) -> list[plt.Figure]:
    """Create paginated 2D embedding plots comparing clusterings.

    Long method lists are split across multiple pages so subplots stay readable.
    """
    X = np.asarray(X_original, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    expected_clusters = int(meta["n_clusters"])

    X_embedded = _fit_embedding_2d(X_scaled)

    labels_to_plot = dict(labels_dict)

    # Add K-Means and Spectral Clustering to the plotted methods.
    kmeans = KMeans(n_clusters=expected_clusters, random_state=42, n_init=10)
    labels_to_plot["K-Means"] = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=expected_clusters,
        random_state=42,
        affinity="nearest_neighbors",
        assign_labels="cluster_qr",
    )
    labels_to_plot["Spectral"] = spectral.fit_predict(X_scaled)

    all_labels = [labels for labels in labels_to_plot.values() if labels is not None]
    color_clusters = _color_cluster_count(expected_clusters, -1, *all_labels)
    spec = build_cluster_color_spec(color_clusters, unassigned_color="#CCCCCC")
    colors = spec.colors

    entries = list(labels_to_plot.items())
    max_panels_per_page = max(1, int(max_panels_per_page))
    n_pages = (len(entries) + max_panels_per_page - 1) // max_panels_per_page
    figures: list[plt.Figure] = []

    meta_text = (
        f"samples={meta.get('n_samples', '?')}, "
        f"features={meta.get('n_features', '?')}, "
        f"generator={meta.get('generator', 'unknown')}, "
        f"noise={meta.get('noise', 'n/a')}"
    )

    for page_idx in range(n_pages):
        start = page_idx * max_panels_per_page
        end = min(start + max_panels_per_page, len(entries))
        page_entries = entries[start:end]

        n_panels = len(page_entries)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=PDF_PAGE_SIZE_INCHES,
            squeeze=False,
        )
        flat_axes = axes.flatten()

        page_suffix = f" (page {page_idx + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            f"Test Case {test_case_num}: expected {expected_clusters} clusters{page_suffix}",
            fontsize=17,
            weight="bold",
            y=0.985,
        )
        fig.text(0.5, 0.945, meta_text, ha="center", va="center", fontsize=10)

        for i, (method_name, labels) in enumerate(page_entries):
            ax = flat_axes[i]
            if labels is None:
                ax.set_title(
                    f"{_format_method_subplot_title(method_name)}\n(Labels Missing)",
                    fontsize=10,
                    weight="bold",
                    pad=8,
                )
                ax.axis("off")
                continue

            labels_array = np.asarray(labels)
            ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels_array,
                cmap=spec.cmap,
                norm=spec.norm,
                alpha=0.75,
                s=48,
                edgecolors="black",
                linewidth=0.45,
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
                        s=92,
                        linewidth=2.7,
                    )
            ax.set_title(
                _format_method_subplot_title(method_name),
                fontsize=10,
                weight="bold",
                pad=8,
            )
            ax.set_xlabel("Embedding Dimension 1", fontsize=9)
            ax.set_ylabel("Embedding Dimension 2", fontsize=9)
            ax.tick_params(labelsize=8.5)
            ax.grid(True, alpha=0.3)

        for j in range(n_panels, len(flat_axes)):
            flat_axes[j].axis("off")

        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.90), h_pad=1.5, w_pad=1.15)
        figures.append(fig)

    return figures


__all__ = [
    "create_clustering_comparison_plots",
    "create_clustering_comparison_plot",
    "_color_cluster_count",
    "create_clustering_comparison_plot_3d",
]


def _fit_embedding_3d(X_scaled: np.ndarray) -> np.ndarray:
    """Return a 3D embedding using UMAP when available, else t-SNE, else PCA."""
    backend = (os.getenv("KL_TE_EMBEDDING_BACKEND_3D") or "umap").strip().lower()

    if backend in {"umap", "auto"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=min(15, max(2, len(X_scaled) - 1)),
                min_dist=0.1,
                low_memory=True,
            )
            return reducer.fit_transform(X_scaled)
        except Exception as exc:
            raise RuntimeError(
                "3D UMAP embedding was requested but failed to initialize/fit."
            ) from exc

    if backend == "tsne":
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=min(30, max(2, len(X_scaled) - 1)),
            )
            return tsne.fit_transform(X_scaled)
        except Exception:
            pass

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
