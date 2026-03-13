"""Method namespace with lazy exports to keep import side-effects minimal."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "estimate_k_marchenko_pastur",
    "build_pca_projection_basis",
    "build_random_orthonormal_basis",
    "build_projection_basis_with_padding",
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
    "annotate_wald",
    "annotate_cousin_adjusted_wald",
]

_NAME_TO_MODULE = {
    "effective_rank": ".k_estimators",
    "marchenko_pastur_signal_count": ".k_estimators",
    "estimate_k_marchenko_pastur": ".k_estimators",
    "build_pca_projection_basis": ".projection_basis",
    "build_random_orthonormal_basis": ".projection_basis",
    "build_projection_basis_with_padding": ".projection_basis",
    "run_projected_wald_kernel": ".projected_wald",
    "compute_projected_pvalue": ".projected_wald",
    "annotate_wald": ".sibling_calibration",
    "annotate_cousin_adjusted_wald": ".sibling_calibration",
}


def __getattr__(name: str) -> Any:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
