"""Projection basis construction (PCA with orthogonal-complement padding)."""

from __future__ import annotations

import numpy as np


def _resolve_pca_component(
    pca_projection: np.ndarray,
    pca_eigenvalues: np.ndarray | None,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Truncate PCA basis to target_dim rows.

    Returns (basis[:n_used], eigenvalues[:n_used], n_padding_rows).
    n_padding_rows == 0 means PCA fully covers the target — no padding needed.
    """
    pca_basis = np.asarray(pca_projection, dtype=np.float64)
    n_used = min(int(pca_basis.shape[0]), target_dim)
    eigenvalues = (
        np.asarray(pca_eigenvalues[:n_used], dtype=np.float64)
        if pca_eigenvalues is not None
        else None
    )
    return pca_basis[:n_used], eigenvalues, target_dim - n_used


def _build_orthogonal_complement_padding(
    parent_pca_basis: np.ndarray,
    child_pca_projections: list[np.ndarray],
    n_padding_rows: int,
) -> np.ndarray:
    """Build padding rows from child PCA directions orthogonal to the parent basis.

    Stacks child eigenvectors, projects out the parent PCA subspace via
    ``(I - V_P V_P^T)``, and returns the top SVD directions of the residual.
    """
    child_rows = np.vstack(child_pca_projections)  # (sum_k_children, d)

    # Project out parent PCA subspace: residual = (I - V_P V_P^T) @ child_rows^T
    residual = child_rows - child_rows @ parent_pca_basis.T @ parent_pca_basis

    # SVD of residual to get orthonormal directions
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    # Keep only directions with non-trivial singular values
    tol = max(S[0] * 1e-10, 1e-15) if len(S) > 0 else 1e-15
    n_valid = int(np.sum(S > tol))
    n_take = min(n_padding_rows, n_valid)

    if n_take == 0:
        return np.empty((0, parent_pca_basis.shape[1]), dtype=np.float64)

    return Vt[:n_take]


def build_projection_basis_with_padding(
    n_features: int,
    k: int,
    *,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    child_pca_projections: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct a basis with PCA head and optional orthogonal-complement padding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        ``(projection_matrix, eigenvalues_for_whitening)``.
        The returned eigenvalue array aligns with the leading PCA rows only:
        - PCA-only/truncated: ``pca_eigenvalues[:k]``
        - PCA + orthogonal-complement padding: full ``pca_eigenvalues`` (padding rows unwhitened)
        - No PCA available: raises ValueError (spectral k > 0 requires a PCA basis)
    """
    target_projection_dim = int(k)

    if pca_projection is None:
        raise ValueError("pca_projection is required — spectral k > 0 implies a PCA basis exists.")

    pca_basis, eigenvalues_for_whitening, n_padding_rows = _resolve_pca_component(
        pca_projection, pca_eigenvalues, target_projection_dim
    )

    if n_padding_rows == 0:
        return pca_basis, eigenvalues_for_whitening

    # Attempt orthogonal-complement padding from child PCA directions
    if child_pca_projections is not None and len(child_pca_projections) > 0:
        padding = _build_orthogonal_complement_padding(
            pca_basis, child_pca_projections, n_padding_rows
        )
        if padding.shape[0] > 0:
            padded = np.vstack([pca_basis, padding])
            return padded, eigenvalues_for_whitening

    # No child PCA available or no valid orthogonal-complement directions —
    # truncate to available PCA rows (no random padding).
    return pca_basis, eigenvalues_for_whitening


__all__ = [
    "build_projection_basis_with_padding",
]
