"""Projection and spectral dimension subpackage.

Provides random projection utilities (JL-based dimension selection,
orthonormal projection matrices) and spectral dimension estimation
(effective rank, Marchenko-Pastur, PCA projections).

Submodules
----------
estimators
    Pure eigenvalue-based dimension estimators (no tree dependency).
eigen_decomposition
    Correlation-matrix eigendecomposition and PCA projection construction.
tree_helpers
    Tree traversal helpers for spectral dimension estimation.
spectral_dimension
    Public orchestrators that combine the above into per-node computation.
random_projection
    JL-based random projection utilities.
"""

from .estimators import (
    count_active_features,
    effective_rank,
    marchenko_pastur_signal_count,
)
from .eigen_decomposition import (
    EigenResult,
    build_pca_projection,
    eigendecompose_correlation,
    estimate_spectral_k,
)
from .random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    estimate_min_projection_dimension,
    generate_projection_matrix,
    resolve_min_k,
)
from .sibling_spectral_dimension import compute_sibling_spectral_dimensions
from .spectral_dimension import (
    compute_node_pca_projections,
    compute_node_spectral_dimensions,
    compute_spectral_decomposition,
)
from .tree_helpers import (
    build_subtree_data,
    is_leaf,
    precompute_descendants,
)

__all__ = [
    # estimators
    "count_active_features",
    "effective_rank",
    "marchenko_pastur_signal_count",
    # eigen_decomposition
    "EigenResult",
    "build_pca_projection",
    "eigendecompose_correlation",
    "estimate_spectral_k",
    # tree_helpers
    "build_subtree_data",
    "is_leaf",
    "precompute_descendants",
    # random_projection
    "compute_projection_dimension",
    "derive_projection_seed",
    "estimate_min_projection_dimension",
    "generate_projection_matrix",
    "resolve_min_k",
    # spectral_dimension (orchestrators)
    "compute_node_pca_projections",
    "compute_node_spectral_dimensions",
    "compute_sibling_spectral_dimensions",
    "compute_spectral_decomposition",
]
