"""Projection basis construction (PCA, random, and hybrid padding)."""

from __future__ import annotations

import numpy as np

from ...statistics.projection.eigen_decomposition import (
    build_pca_projection,
    eigendecompose_correlation,
)
from ...statistics.projection.random_projection import generate_projection_matrix


def build_pca_projection_basis(
    data_sub: np.ndarray,
    *,
    k: int,
    d_full: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a PCA basis and associated eigenvalues for whitening."""
    X = np.asarray(data_sub, dtype=np.float64)
    d = int(d_full) if d_full is not None else int(X.shape[1])
    eig = eigendecompose_correlation(X, need_eigh=True)
    if eig is None:
        return None, None
    return build_pca_projection(eig, k=int(k), d=d)


def build_random_orthonormal_basis(
    n_features: int,
    k: int,
    *,
    random_state: int | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Build a random orthonormal projection basis."""
    return generate_projection_matrix(
        int(n_features),
        int(k),
        random_state=random_state,
        use_cache=use_cache,
    )


def build_projection_basis_with_padding(
    n_features: int,
    k: int,
    *,
    pca_projection: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, int]:
    """Construct a basis with optional PCA head and random padded tail.

    Returns
    -------
    tuple[np.ndarray, int]
        Projection matrix and number of PCA rows included at the top.
    """
    if pca_projection is None:
        R = build_random_orthonormal_basis(
            n_features=n_features,
            k=k,
            random_state=random_state,
            use_cache=False,
        )
        return R, 0

    pca = np.asarray(pca_projection, dtype=np.float64)
    k_pca = min(int(pca.shape[0]), int(k))
    if k_pca >= int(k):
        return pca[: int(k)], int(k)

    R_pad = build_random_orthonormal_basis(
        n_features=n_features,
        k=int(k) - k_pca,
        random_state=random_state,
        use_cache=False,
    )
    return np.vstack([pca[:k_pca], R_pad]), k_pca


__all__ = [
    "build_pca_projection_basis",
    "build_random_orthonormal_basis",
    "build_projection_basis_with_padding",
]

