"""Projection and spectral dimension subpackage.

Provides projected-test p-value utilities and spectral decomposition helpers.

Submodules
----------
tree_helpers
    Tree traversal helpers for spectral dimension estimation.
spectral_dimension
    Public orchestrator for per-node spectral decomposition.
"""

from .satterthwaite import compute_projected_pvalue
from .spectral_dimension import (
    compute_spectral_decomposition,
)
from .tree_helpers import (
    build_subtree_data,
    is_leaf,
    precompute_descendants,
)

__all__ = [
    # tree_helpers
    "build_subtree_data",
    "is_leaf",
    "precompute_descendants",
    # satterthwaite
    "compute_projected_pvalue",
    # spectral_dimension
    "compute_spectral_decomposition",
]
