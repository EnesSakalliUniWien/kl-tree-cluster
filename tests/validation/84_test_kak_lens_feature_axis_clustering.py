from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    cosine_eigendecomposition,
)
from benchmarks.diagnostics.spectral.kak_lens_feature_axis_clustering import (
    feature_axes_from_cosine_eigenvectors,
    feature_axis_debug_metrics,
    feature_axis_scores,
    row_normalized_weighted_matrix,
)
from sklearn.preprocessing import normalize


def test_feature_axes_reconstruct_raw_cosine_sample_coordinates() -> None:
    data = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        index=["g1", "g2", "g3", "g4"],
        columns=["f1", "f2", "f3", "f4"],
    )
    z = row_normalized_weighted_matrix(data, "binary")
    eigvals, eigvecs = cosine_eigendecomposition(data.to_numpy(dtype=float), max_rank=3)

    feature_axes = feature_axes_from_cosine_eigenvectors(z, eigvals, eigvecs)
    debug = feature_axis_debug_metrics(z, eigvals, eigvecs, feature_axes)

    reconstructed = z @ feature_axes[:, :3]
    expected = eigvecs[:, :3] * np.sqrt(eigvals[:3])[np.newaxis, :]
    assert np.allclose(reconstructed, expected)
    assert np.allclose(np.linalg.norm(feature_axes, axis=0), 1.0)
    assert debug["feature_axis_reconstruction_max_abs_error"] < 1e-12
    assert debug["feature_axis_orthogonality_max_abs_error"] < 1e-12


def test_feature_axis_scores_connect_common_and_variant_loadings() -> None:
    features = pd.Index(["a", "b", "c"])
    eigvals = np.array([4.0, 2.0, 1.0])
    feature_axes = normalize(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [0.5, -1.0, 1.0],
                [0.1, 0.0, -1.0],
            ]
        ),
        norm="l2",
        axis=0,
    )

    scores = feature_axis_scores(
        features=features,
        eigvals=eigvals,
        feature_axes=feature_axes,
        block_start=2,
        block_end=3,
    )

    assert set(scores.columns) >= {
        "common_axis_loading",
        "variant_loading_energy_fraction",
        "axis_connection_score",
    }
    assert scores.iloc[0]["axis_connection_rank"] == 1
    assert scores["variant_loading_energy_fraction"].sum() == pytest.approx(1.0)
