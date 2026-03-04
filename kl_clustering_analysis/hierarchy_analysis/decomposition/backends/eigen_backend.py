"""Eigen-decomposition backend wrappers."""

from __future__ import annotations

import numpy as np

from ...statistics.projection.eigen_decomposition import (
    EigenResult,
    build_pca_projection,
    eigendecompose_correlation,
    estimate_spectral_k,
)


def eigendecompose_correlation_backend(
    data_sub: np.ndarray,
    *,
    need_eigh: bool,
) -> EigenResult | None:
    """Backend wrapper for correlation eigendecomposition."""
    return eigendecompose_correlation(
        np.asarray(data_sub, dtype=np.float64),
        need_eigh=need_eigh,
    )


def estimate_spectral_k_backend(
    eigenvalues: np.ndarray,
    *,
    method: str,
    n_desc: int,
    d_active: int,
    min_k: int,
) -> int:
    """Backend wrapper for spectral-k estimation."""
    return estimate_spectral_k(
        np.asarray(eigenvalues, dtype=np.float64),
        method=method,
        n_desc=n_desc,
        d_active=d_active,
        min_k=min_k,
    )


def build_pca_projection_backend(
    eig: EigenResult,
    *,
    k: int,
    d: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Backend wrapper for PCA projection reconstruction."""
    return build_pca_projection(eig, k=k, d=d)


__all__ = [
    "eigendecompose_correlation_backend",
    "estimate_spectral_k_backend",
    "build_pca_projection_backend",
]

