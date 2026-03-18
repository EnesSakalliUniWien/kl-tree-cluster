"""Numerical backend wrappers used by decomposition methods."""

from ..core.eigen_result import EigenResult
from .eigen_backend import (
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
    estimate_spectral_k_backend,
)
from .random_projection_backend import (
    derive_projection_seed_backend,
)

__all__ = [
    "EigenResult",
    "eigendecompose_correlation_backend",
    "estimate_spectral_k_backend",
    "build_pca_projection_backend",
    "derive_projection_seed_backend",
]
