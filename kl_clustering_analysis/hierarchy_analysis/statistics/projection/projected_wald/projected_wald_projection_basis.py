"""Projection basis construction (PCA, random, and hybrid padding)."""

from __future__ import annotations

import numpy as np

from ....decomposition.backends.random_projection.matrix import generate_projection_matrix


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


def _orthogonal_complement_basis(
    fixed_basis: np.ndarray,
    *,
    n_features: int,
) -> np.ndarray:
    """Return orthonormal rows spanning the complement of fixed_basis."""
    if fixed_basis.size == 0:
        return np.eye(int(n_features), dtype=np.float64)

    _, singular_values, vh = np.linalg.svd(
        np.asarray(fixed_basis, dtype=np.float64),
        full_matrices=True,
    )
    if singular_values.size == 0:
        rank = 0
    else:
        tolerance = (
            np.finfo(np.float64).eps
            * max(fixed_basis.shape)
            * float(np.max(singular_values))
        )
        rank = int(np.sum(singular_values > tolerance))
    return vh[rank:]


def _generate_padding_basis(
    pca_basis: np.ndarray,
    *,
    n_features: int,
    n_padding_rows: int,
    random_state: int | None,
) -> np.ndarray:
    """Generate random rows inside the complement of the PCA row space."""
    complement_basis = _orthogonal_complement_basis(pca_basis, n_features=n_features)
    complement_dimension = int(complement_basis.shape[0])
    if n_padding_rows > complement_dimension:
        raise ValueError(
            "Cannot pad PCA projection with more rows than the orthogonal "
            f"complement provides: requested {n_padding_rows}, "
            f"available {complement_dimension}."
        )

    random_complement_rows = generate_projection_matrix(
        complement_dimension,
        int(n_padding_rows),
        random_state=random_state,
        use_cache=False,
    )
    return random_complement_rows @ complement_basis


def build_projection_basis_with_padding(
    n_features: int,
    k: int,
    *,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct a basis from parent PCA rows with optional random padding."""

    target_projection_dim = int(k)

    if pca_projection is None:
        random_projection_basis = generate_projection_matrix(
            int(n_features),
            int(target_projection_dim),
            random_state=random_state,
            use_cache=False,
        )
        return random_projection_basis, None

    pca_basis, eigenvalues_for_whitening, n_padding_rows = _resolve_pca_component(
        pca_projection, pca_eigenvalues, target_projection_dim
    )

    if n_padding_rows == 0:
        return pca_basis, eigenvalues_for_whitening

    random_padding_basis = _generate_padding_basis(
        pca_basis,
        n_features=int(n_features),
        n_padding_rows=int(n_padding_rows),
        random_state=random_state,
    )

    return np.vstack([pca_basis, random_padding_basis]), eigenvalues_for_whitening


__all__ = [
    "build_projection_basis_with_padding",
]
