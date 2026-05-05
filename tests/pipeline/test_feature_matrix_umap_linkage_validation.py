from __future__ import annotations

import pandas as pd
import pytest
from scripts.run_feature_matrix_with_umap import _build_linkage_tree


def _small_feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        index=["A", "B", "C", "D"],
    )


def test_build_linkage_tree_rejects_ward_with_non_euclidean_metric() -> None:
    with pytest.raises(ValueError, match="Ward linkage requires euclidean distance"):
        _build_linkage_tree(
            data_df=_small_feature_matrix(),
            tree_method="kl",
            tree_distance_metric="cosine",
            tree_linkage_method="ward",
            diffusion_k_neighbors=10,
            diffusion_time=1,
            diffusion_components=10,
            adaptive_neighbor_k=None,
            adaptive_bandwidth_type="local",
            adaptive_epsilon="auto",
            adaptive_metric="hamming",
        )


def test_build_linkage_tree_allows_ward_with_euclidean_metric() -> None:
    linkage_matrix, distance_metric, linkage_method, adaptive_metadata = _build_linkage_tree(
        data_df=_small_feature_matrix(),
        tree_method="kl",
        tree_distance_metric="euclidean",
        tree_linkage_method="ward",
        diffusion_k_neighbors=10,
        diffusion_time=1,
        diffusion_components=10,
        adaptive_neighbor_k=None,
        adaptive_bandwidth_type="local",
        adaptive_epsilon="auto",
        adaptive_metric="hamming",
    )

    assert linkage_matrix.shape == (3, 4)
    assert distance_metric == "euclidean"
    assert linkage_method == "ward"
    assert adaptive_metadata is None
