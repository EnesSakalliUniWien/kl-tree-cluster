"""Tests for reusable space-separation method interfaces."""

import numpy as np
import pandas as pd
from tree_break_selection.space_separation import (
    SpectralBlock,
    block_diffusion_distance,
    coordinates_for_block,
    decompose_invariant_equivariant_space,
    hamming_knn_diffusion_distance,
    separate_adaptive_cosine_space,
)


def test_invariant_equivariant_decomposition_is_finite_and_energy_ordered() -> None:
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 4.0],
            [3.0, 2.0, 5.0],
            [4.0, 2.0, 7.0],
        ]
    )

    result = decompose_invariant_equivariant_space(values, equivariant_dim=2)

    assert result.coordinates.shape == (4, 3)
    assert result.axes.shape == (3, 3)
    assert np.isfinite(result.coordinates).all()
    assert np.all(np.diff(result.singular_values) <= 0.0)
    assert result.invariant_coordinates.shape == (4,)
    assert result.equivariant_coordinates.shape == (4, 2)


def test_adaptive_cosine_space_exposes_selected_block_coordinates() -> None:
    data = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )

    result = separate_adaptive_cosine_space(
        data,
        weighting="binary",
        max_rank=4,
        min_segment_length=2,
        max_segments=2,
    )

    assert result.blocks
    coordinates = result.coordinates(result.blocks[0])
    assert coordinates.shape[0] == len(data)
    assert np.isfinite(coordinates).all()


def test_coordinates_for_block_uses_one_based_inclusive_modes() -> None:
    eigenvalues = np.array([9.0, 4.0, 1.0])
    eigenvectors = np.eye(3)
    block = SpectralBlock(0, "modes_02_03", 2, 3, "test")

    coordinates = coordinates_for_block(eigenvalues, eigenvectors, block)

    np.testing.assert_allclose(
        coordinates,
        np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )


def test_block_diffusion_distance_is_a_public_separation_interface() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]],
        dtype=float,
    )

    distances, metadata = block_diffusion_distance(
        coordinates,
        k_neighbors=2,
        diffusion_time=2,
        n_components=3,
    )

    assert distances.shape == (6,)
    assert np.isfinite(distances).all()
    assert metadata["kernel"] == "knn_gaussian"


def test_hamming_diffusion_distance_is_a_public_separation_interface() -> None:
    binary = pd.DataFrame(
        [
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
        ]
    )

    distances = hamming_knn_diffusion_distance(
        binary,
        k_neighbors=2,
        diffusion_time=2,
        n_components=3,
    )

    assert distances.shape == (6,)
    assert np.isfinite(distances).all()


def test_hamming_diffusion_distance_rejects_continuous_values() -> None:
    with np.testing.assert_raises_regex(ValueError, "binary or one-hot"):
        hamming_knn_diffusion_distance(
            np.array([[0.0, 0.5], [1.0, 0.0]]),
            k_neighbors=1,
            diffusion_time=1,
            n_components=1,
        )
