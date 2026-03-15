"""Projection and spectral dimension subpackage.

Provides projected-test p-value utilities, projection basis construction,
projected Wald test kernel, dimension estimators, and spectral decomposition helpers.

Submodules
----------
k_estimators
    Projection-dimension estimators (effective rank, Marchenko-Pastur).
projected_wald
    Shared projected Wald test kernel.
projection_basis
    Projection basis construction (PCA, random, hybrid padding).
tree_helpers
    Tree traversal helpers for spectral dimension estimation.
spectral_dimension
    Public orchestrator for per-node spectral decomposition.
"""

from .chi2_pvalue import compute_projected_pvalue
from .k_estimators import (
    effective_rank,
    estimate_k_marchenko_pastur,
    marchenko_pastur_signal_count,
)
from .projected_wald import run_projected_wald_kernel
from .projection_basis import build_projection_basis_with_padding
from .spectral import (
    compute_spectral_decomposition,
)
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
    "run_projected_wald_kernel",
    # projection_basis
    "build_projection_basis_with_padding",
    # tree_helpers
    "build_subtree_data",
    "is_leaf",
    "precompute_descendants",
    # satterthwaite
    "compute_projected_pvalue",
    # spectral_dimension
    "compute_spectral_decomposition",
]
