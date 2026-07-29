"""Projection statistics package.

Subpackages are organized by statistical responsibility:

- ``projected_wald``: projected-test reference distributions, projection-basis
  construction, and the shared projected Wald kernel.
- ``projection_dimension_estimation``: Marchenko-Pastur projection-dimension
  estimators.
- ``spectral``: per-node spectral decomposition over tree substructures.
"""

from .projected_wald import (
    ProjectedQuadraticReference,
    ProjectedWaldResult,
    build_pca_projection_basis,
    compute_projected_pvalue,
    run_projected_wald_kernel,
)
from .projection_dimension_estimation import (
    MarchenkoPasturDimensionEstimate,
    estimate_marchenko_pastur_dimension,
    marchenko_pastur_signal_count,
)
from .spectral.spectral_decomposition_result import SpectralDecompositionResult
from .spectral.tree_estimator import compute_spectral_decomposition
from .spectral.tree_helpers import (
    is_leaf,
    precompute_descendants,
)

__all__ = [
    # projection_dimension_estimation
    "MarchenkoPasturDimensionEstimate",
    "estimate_marchenko_pastur_dimension",
    "marchenko_pastur_signal_count",
    # projected_wald
    "ProjectedQuadraticReference",
    "ProjectedWaldResult",
    "run_projected_wald_kernel",
    # projected_wald
    "build_pca_projection_basis",
    # tree_helpers
    "is_leaf",
    "precompute_descendants",
    # projected_wald_reference
    "compute_projected_pvalue",
    # spectral_dimension
    "SpectralDecompositionResult",
    "compute_spectral_decomposition",
]
