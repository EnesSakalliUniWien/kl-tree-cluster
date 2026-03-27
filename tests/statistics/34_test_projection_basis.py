from __future__ import annotations

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_basis import (
    build_projection_basis_with_padding,
)


def test_projection_basis_without_pca_uses_random_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_generate_projection_matrix_backend(
        n_features: int,
        n_components: int,
        random_state: int | None = None,
        *,
        use_cache: bool = True,
    ) -> np.ndarray:
        captured["args"] = (n_features, n_components, random_state, use_cache)
        return np.full((n_components, n_features), 7.0, dtype=np.float64)

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_basis.generate_projection_matrix_backend",
        _fake_generate_projection_matrix_backend,
    )

    basis, eigenvalues = build_projection_basis_with_padding(
        n_features=5,
        k=3,
        random_state=11,
    )

    np.testing.assert_array_equal(basis, np.full((3, 5), 7.0, dtype=np.float64))
    assert eigenvalues is None
    assert captured["args"] == (5, 3, 11, False)


def test_projection_basis_truncates_parent_pca_to_target_dimension() -> None:
    pca_projection = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pca_eigenvalues = np.array([5.0, 3.0, 1.0], dtype=np.float64)

    basis, eigenvalues = build_projection_basis_with_padding(
        n_features=3,
        k=2,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )

    np.testing.assert_array_equal(basis, pca_projection[:2])
    np.testing.assert_array_equal(eigenvalues, pca_eigenvalues[:2])


def test_projection_basis_pads_short_parent_pca_with_random_rows(monkeypatch) -> None:
    def _fake_generate_projection_matrix_backend(
        n_features: int,
        n_components: int,
        random_state: int | None = None,
        *,
        use_cache: bool = True,
    ) -> np.ndarray:
        assert (n_features, n_components, random_state, use_cache) == (4, 2, 19, False)
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_basis.generate_projection_matrix_backend",
        _fake_generate_projection_matrix_backend,
    )

    pca_projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pca_eigenvalues = np.array([9.0, 4.0], dtype=np.float64)

    basis, eigenvalues = build_projection_basis_with_padding(
        n_features=4,
        k=4,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        random_state=19,
    )

    np.testing.assert_array_equal(
        basis,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(eigenvalues, pca_eigenvalues)
