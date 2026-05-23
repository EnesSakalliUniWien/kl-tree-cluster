from __future__ import annotations

import numpy as np
import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_projection_basis import (
    build_pca_projection_basis,
)


def test_projection_basis_requires_parent_pca() -> None:
    with pytest.raises(ValueError, match="require a PCA projection basis"):
        build_pca_projection_basis(
            k=3,
        )


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

    basis, eigenvalues = build_pca_projection_basis(
        k=2,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )

    np.testing.assert_array_equal(basis, pca_projection[:2])
    np.testing.assert_array_equal(eigenvalues, pca_eigenvalues[:2])


def test_projection_basis_rejects_short_parent_pca() -> None:
    pca_projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="2 row\\(s\\), but 4 projection dimension"):
        build_pca_projection_basis(
            k=4,
            pca_projection=pca_projection,
            pca_eigenvalues=np.array([9.0, 4.0], dtype=np.float64),
        )


def test_projection_basis_requires_matching_eigenvalues() -> None:
    pca_projection = np.eye(3, dtype=np.float64)

    with pytest.raises(ValueError, match="requires matching eigenvalues"):
        build_pca_projection_basis(
            k=2,
            pca_projection=pca_projection,
            pca_eigenvalues=None,
        )

    with pytest.raises(ValueError, match="1 value\\(s\\), but 2 projection dimension"):
        build_pca_projection_basis(
            k=2,
            pca_projection=pca_projection,
            pca_eigenvalues=np.array([5.0], dtype=np.float64),
        )
