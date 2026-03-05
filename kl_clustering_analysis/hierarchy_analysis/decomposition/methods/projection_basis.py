"""Projection basis construction (PCA, random, and hybrid padding)."""

from __future__ import annotations

import numpy as np

from ..backends.eigen_backend import (
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
)
from ..backends.random_projection_backend import generate_projection_matrix_backend


def build_pca_projection_basis(
    data_sub: np.ndarray,
    *,
    k: int,
    d_full: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a PCA basis and associated eigenvalues for whitening."""
    X = np.asarray(data_sub, dtype=np.float64)
    d = int(d_full) if d_full is not None else int(X.shape[1])
    eig = eigendecompose_correlation_backend(X, need_eigh=True)
    if eig is None:
        return None, None
    return build_pca_projection_backend(eig, k=int(k), d=d)


def build_random_orthonormal_basis(
    n_features: int,
    k: int,
    *,
    random_state: int | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Build a random orthonormal projection basis."""
    return generate_projection_matrix_backend(
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
    pca_eigenvalues: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct a basis with optional PCA head and random padded tail.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        ``(projection_matrix, eigenvalues_for_whitening)``.
        The returned eigenvalue array aligns with the leading PCA rows only:
        - PCA-only/truncated: ``pca_eigenvalues[:k]``
        - PCA+random padding: full ``pca_eigenvalues`` (padding rows unwhitened)
        - Random-only: ``None``
    """
    target_projection_dim = int(k)

    if pca_projection is None:
        random_projection_basis = build_random_orthonormal_basis(
            n_features=n_features,
            k=target_projection_dim,
            random_state=random_state,
            use_cache=False,
        )
        return random_projection_basis, None

    pca_basis = np.asarray(pca_projection, dtype=np.float64)
    n_available_pca_rows = int(pca_basis.shape[0])
    n_pca_rows_used = min(n_available_pca_rows, target_projection_dim)

    if n_pca_rows_used >= target_projection_dim:
        eigenvalues_for_whitening = (
            np.asarray(pca_eigenvalues[:target_projection_dim], dtype=np.float64)
            if pca_eigenvalues is not None
            else None
        )
        return pca_basis[:target_projection_dim], eigenvalues_for_whitening

    n_padding_rows = target_projection_dim - n_pca_rows_used

    random_padding_basis = build_random_orthonormal_basis(
        n_features=n_features,
        k=n_padding_rows,
        random_state=random_state,
        use_cache=False,
    )

    eigenvalues_for_whitening = (
        np.asarray(pca_eigenvalues, dtype=np.float64) if pca_eigenvalues is not None else None
    )

    padded_projection_basis = np.vstack([pca_basis[:n_pca_rows_used], random_padding_basis])

    return padded_projection_basis, eigenvalues_for_whitening


__all__ = [
    "build_pca_projection_basis",
    "build_random_orthonormal_basis",
    "build_projection_basis_with_padding",
]
