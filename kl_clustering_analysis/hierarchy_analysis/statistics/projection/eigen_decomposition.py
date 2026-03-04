"""Compatibility wrappers for correlation eigendecomposition and PCA projection.

The numerical implementations now live in
``hierarchy_analysis.decomposition.backends.eigen_backend``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...decomposition.backends.eigen_backend import (
    EigenResult,
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
    estimate_spectral_k_backend,
)


def eigendecompose_correlation(
    data_sub: np.ndarray,
    need_eigh: bool,
) -> Optional[EigenResult]:
    """Eigendecompose the correlation matrix (primal/dual form)."""
    return eigendecompose_correlation_backend(data_sub, need_eigh=need_eigh)


def estimate_spectral_k(
    eigenvalues: np.ndarray,
    method: str,
    n_desc: int,
    d_active: int,
    min_k: int,
) -> int:
    """Estimate projection dimension from eigenvalues."""
    return estimate_spectral_k_backend(
        eigenvalues,
        method=method,
        n_desc=n_desc,
        d_active=d_active,
        min_k=min_k,
    )


def build_pca_projection(
    eig: EigenResult,
    k: int,
    d: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build ``(k_avail × d)`` PCA projection matrix and eigenvalues."""
    return build_pca_projection_backend(eig, k=k, d=d)


__all__ = [
    "EigenResult",
    "eigendecompose_correlation",
    "estimate_spectral_k",
    "build_pca_projection",
]

