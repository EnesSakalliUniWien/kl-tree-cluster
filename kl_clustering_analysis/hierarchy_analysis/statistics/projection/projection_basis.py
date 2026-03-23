"""Projection basis construction (PCA, random, and hybrid padding)."""

from __future__ import annotations

import numpy as np

from ...decomposition.backends.random_projection_backend import generate_projection_matrix_backend


def _build_random_orthonormal_basis(
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


def _resolve_pca_component(
    pca_projection: np.ndarray,
    pca_eigenvalues: np.ndarray | None,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Truncate PCA basis to target_dim rows."""
    pca_basis = np.asarray(pca_projection, dtype=np.float64)
    n_used = min(int(pca_basis.shape[0]), target_dim)
    eigenvalues = (
        np.asarray(pca_eigenvalues[:n_used], dtype=np.float64)
        if pca_eigenvalues is not None
        else None
    )
    return pca_basis[:n_used], eigenvalues, target_dim - n_used


def build_projection_basis_with_padding(
    n_features: int,
    k: int,
    *,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    child_pca_projections: list[np.ndarray] | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct a basis with optional PCA head and random padded tail."""
    del child_pca_projections

    target_projection_dim = int(k)

    if pca_projection is None:
        random_projection_basis = _build_random_orthonormal_basis(
            n_features=n_features,
            k=target_projection_dim,
            random_state=random_state,
            use_cache=False,
        )
        return random_projection_basis, None

    pca_basis, eigenvalues_for_whitening, n_padding_rows = _resolve_pca_component(
        pca_projection, pca_eigenvalues, target_projection_dim
    )

    if n_padding_rows == 0:
        return pca_basis, eigenvalues_for_whitening

    random_padding_basis = _build_random_orthonormal_basis(
        n_features=n_features,
        k=n_padding_rows,
        random_state=random_state,
        use_cache=False,
    )

    return np.vstack([pca_basis, random_padding_basis]), eigenvalues_for_whitening


__all__ = [
    "build_projection_basis_with_padding",
]
