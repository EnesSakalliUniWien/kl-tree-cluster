"""Projection basis construction from PCA components."""

from __future__ import annotations

import numpy as np


def _resolve_pca_component(
    pca_projection: np.ndarray,
    pca_eigenvalues: np.ndarray | None,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate PCA basis to target_dim rows."""
    pca_basis = np.asarray(pca_projection, dtype=np.float64)
    if pca_basis.shape[0] < target_dim:
        raise ValueError(
            f"PCA projection has {pca_basis.shape[0]} row(s), but {target_dim} "
            "projection dimension(s) were requested."
        )
    if pca_eigenvalues is None:
        raise ValueError("PCA projection requires matching eigenvalues.")
    eigenvalues = np.asarray(pca_eigenvalues[:target_dim], dtype=np.float64)
    if eigenvalues.shape[0] != target_dim:
        raise ValueError(
            f"PCA eigenvalues have {eigenvalues.shape[0]} value(s), but {target_dim} "
            "projection dimension(s) were requested."
        )
    return pca_basis[:target_dim], eigenvalues


def build_pca_projection_basis(
    k: int,
    *,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a projection basis from parent PCA rows."""

    target_projection_dim = int(k)

    if pca_projection is None:
        raise ValueError("Projected Wald tests require a PCA projection basis.")

    return _resolve_pca_component(
        pca_projection, pca_eigenvalues, target_projection_dim
    )


__all__ = [
    "build_pca_projection_basis",
]
