"""Tests for UMAP comparison report layout safeguards."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.shared.plots import embedding
from benchmarks.shared.util.pdf.layout import PDF_WIDE_PAGE_SIZE_INCHES


def test_method_subplot_title_compacts_parameters_and_limits_lines():
    title = embedding._format_method_subplot_title(
        (
            "KL diffusion tree (edge_alpha=0.001, sibling_alpha=0.01, "
            "tree_distance_metric=tree_distribution_kl, "
            "tree_linkage_method=average) [K=64, A=0.123, N=0.456]"
        )
    )

    lines = title.splitlines()
    assert len(lines) <= 3
    assert "tree_distance_metric" not in title
    assert "sibling_alpha" not in title
    assert "metric=tree_kl" in title
    assert all(len(line) <= 24 for line in lines)


def test_long_titles_use_relaxed_wide_grid(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4))
    labels = np.array([0, 1] * 6)

    monkeypatch.setattr(
        embedding,
        "_fit_embedding_2d",
        lambda X_scaled, *, cache_key=None: X_scaled[:, :2],
    )

    figs = embedding.create_clustering_comparison_plots(
        X_original=X,
        labels_dict={
            "Ground Truth": labels,
            "KL tree (tree_distance_metric=tree_distribution_kl, tree_linkage_method=average)": labels,
            "KL diffusion tree (edge_alpha=0.001, sibling_alpha=0.01)": labels,
            "K-Means": labels,
            "Spectral": labels,
        },
        test_case_num=1,
        meta={
            "n_clusters": 2,
            "n_samples": 12,
            "n_features": 4,
            "generator": "unit",
            "noise": 0.0,
        },
    )

    assert len(figs) == 2
    assert tuple(figs[0].get_size_inches()) == PDF_WIDE_PAGE_SIZE_INCHES

    for fig in figs:
        plt.close(fig)
