"""Pluggable decomposition methods (k, basis, tests, calibration)."""

from .k_estimators import (
    estimate_k_active_features,
    estimate_k_effective_rank,
    estimate_k_marchenko_pastur,
)
from .projection_basis import (
    build_pca_projection_basis,
    build_projection_basis_with_padding,
    build_random_orthonormal_basis,
)
from .projected_wald import (
    compute_projected_pvalue,
    run_projected_wald_kernel,
    run_projected_wald_test,
)
from .sibling_calibration import (
    apply_sibling_calibration,
    fit_cousin_adjusted_wald,
    fit_cousin_tree_guided,
    fit_cousin_weighted_wald,
)

__all__ = [
    "estimate_k_effective_rank",
    "estimate_k_marchenko_pastur",
    "estimate_k_active_features",
    "build_pca_projection_basis",
    "build_random_orthonormal_basis",
    "build_projection_basis_with_padding",
    "run_projected_wald_kernel",
    "run_projected_wald_test",
    "compute_projected_pvalue",
    "fit_cousin_weighted_wald",
    "fit_cousin_adjusted_wald",
    "fit_cousin_tree_guided",
    "apply_sibling_calibration",
]
