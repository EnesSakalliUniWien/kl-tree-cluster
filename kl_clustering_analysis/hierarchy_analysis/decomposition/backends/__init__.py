"""Numerical backend wrappers used by decomposition methods."""

from .eigen_backend import (
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
    estimate_spectral_k_backend,
)
from .random_projection_backend import (
    compute_projection_dimension_backend,
    derive_projection_seed_backend,
    generate_projection_matrix_backend,
    resolve_min_k_backend,
)

__all__ = [
    "eigendecompose_correlation_backend",
    "estimate_spectral_k_backend",
    "build_pca_projection_backend",
    "compute_projection_dimension_backend",
    "generate_projection_matrix_backend",
    "derive_projection_seed_backend",
    "resolve_min_k_backend",
]

