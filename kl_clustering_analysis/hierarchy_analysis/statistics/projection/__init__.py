"""Projection statistics package.

Subpackages are organized by statistical responsibility:

- ``projected_wald``: projected-test reference distributions, projection-basis
  construction, and the shared projected Wald kernel.
- ``projection_dimension_estimation``: effective-rank and Marchenko-Pastur
  projection-dimension estimators.
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
    effective_rank,
    estimate_k_marchenko_pastur,
    marchenko_pastur_signal_count,
)
from .spectral.tree_estimator import compute_spectral_decomposition
from .spectral.tree_helpers import (
    build_subtree_data,
    is_leaf,
    precompute_descendants,
)

__all__ = [
    # k_estimators
    "effective_rank",
    "estimate_k_marchenko_pastur",
    "marchenko_pastur_signal_count",
    # projected_wald
    "ProjectedQuadraticReference",
    "ProjectedWaldResult",
    "run_projected_wald_kernel",
    # projection_basis
    "build_pca_projection_basis",
    # tree_helpers
    "build_subtree_data",
    "is_leaf",
    "precompute_descendants",
    # satterthwaite
    "compute_projected_pvalue",
    # spectral_dimension
    "compute_spectral_decomposition",
]
