from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.spectral.kak_feature_subspace_clustering import (
    build_feature_tree,
    feature_coordinates_for_block,
    select_active_feature_coordinates,
)
from benchmarks.diagnostics.spectral.kak_lens_feature_axis_clustering import (
    feature_axes_from_cosine_eigenvectors,
    row_normalized_weighted_matrix,
)
from tree_break_selection.space_separation import (
    SpectralBlock,
    cosine_eigendecomposition,
)


def test_feature_subspace_coordinates_reconstruct_restricted_feature_gram() -> None:
    data = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        index=["g1", "g2", "g3", "g4", "g5"],
        columns=["f1", "f2", "f3", "f4", "f5"],
    )
    z = row_normalized_weighted_matrix(data, "binary")
    eigvals, eigvecs = cosine_eigendecomposition(data.to_numpy(dtype=float), max_rank=4)
    feature_axes = feature_axes_from_cosine_eigenvectors(z, eigvals, eigvecs)
    block = SpectralBlock(
        block_id=0,
        block_name="adaptive_modes_02_04",
        block_start=2,
        block_end=4,
        block_type="adaptive_decay_regime",
    )

    coords = feature_coordinates_for_block(
        features=data.columns,
        eigvals=eigvals,
        feature_axes=feature_axes,
        block=block,
    )

    start = block.block_start - 1
    end = block.block_end
    expected = (
        feature_axes[:, start:end]
        @ np.diag(eigvals[start:end])
        @ feature_axes[:, start:end].T
    )
    assert np.allclose(coords.to_numpy() @ coords.to_numpy().T, expected)
    assert list(coords.index) == list(data.columns)
    assert list(coords.columns) == ["mode_02", "mode_03", "mode_04"]


def test_select_active_feature_coordinates_keeps_high_energy_rows() -> None:
    coordinates = pd.DataFrame(
        {
            "mode_02": [3.0, 1.0, 0.0, 0.2],
            "mode_03": [0.0, 1.0, 0.0, 0.1],
        },
        index=["strong", "medium", "zero", "weak"],
    )

    active, energy, fraction = select_active_feature_coordinates(
        coordinates,
        max_active_features=2,
    )

    assert list(active.index) == ["strong", "medium"]
    assert list(energy.index) == ["strong", "medium"]
    assert fraction == pytest.approx((9.0 + 2.0) / (9.0 + 2.0 + 0.05))


def test_build_feature_tree_accepts_signed_continuous_feature_coordinates() -> None:
    coordinates = pd.DataFrame(
        [
            [1.0, -0.2],
            [0.9, -0.1],
            [-0.8, 0.3],
            [-0.7, 0.4],
        ],
        index=["f1", "f2", "f3", "f4"],
        columns=["mode_02", "mode_03"],
    )

    tree = build_feature_tree(coordinates, tree_linkage_method="average")

    assert tree.number_of_nodes() == 7
