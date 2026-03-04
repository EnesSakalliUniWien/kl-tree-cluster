"""Method namespace with lazy exports to keep import side-effects minimal."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
    "estimate_k_effective_rank",
    "estimate_k_marchenko_pastur",
    "estimate_k_active_features",
    "build_pca_projection_basis",
    "build_random_orthonormal_basis",
    "build_projection_basis_with_padding",
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
    "fit_cousin_weighted_wald",
    "fit_cousin_adjusted_wald",
    "fit_cousin_tree_guided",
]

_NAME_TO_MODULE = {
    "effective_rank": ".k_estimators",
    "marchenko_pastur_signal_count": ".k_estimators",
    "count_active_features": ".k_estimators",
    "estimate_k_effective_rank": ".k_estimators",
    "estimate_k_marchenko_pastur": ".k_estimators",
    "estimate_k_active_features": ".k_estimators",
    "build_pca_projection_basis": ".projection_basis",
    "build_random_orthonormal_basis": ".projection_basis",
    "build_projection_basis_with_padding": ".projection_basis",
    "run_projected_wald_kernel": ".projected_wald",
    "compute_projected_pvalue": ".projected_wald",
    "fit_cousin_weighted_wald": ".sibling_calibration",
    "fit_cousin_adjusted_wald": ".sibling_calibration",
    "fit_cousin_tree_guided": ".sibling_calibration",
}


def __getattr__(name: str) -> Any:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
