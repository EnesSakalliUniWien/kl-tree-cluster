"""Method registry and resolvers for decomposition components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod
from .errors import DecompositionMethodError


@dataclass
class MethodRegistry:
    """Typed registry for method callables used by decomposition."""

    k_estimators: dict[str, Callable] = field(default_factory=dict)
    projection_builders: dict[str, Callable] = field(default_factory=dict)
    sibling_calibrators: dict[str, Callable] = field(default_factory=dict)


def resolve_k_estimator(method: SpectralKMethod | str) -> Callable:
    """Resolve a spectral-k estimator callable."""
    from ..methods.k_estimators import (
        estimate_k_active_features,
        estimate_k_effective_rank,
        estimate_k_marchenko_pastur,
    )

    name = method.value if isinstance(method, SpectralKMethod) else str(method)
    mapping = {
        SpectralKMethod.EFFECTIVE_RANK.value: estimate_k_effective_rank,
        SpectralKMethod.MARCHENKO_PASTUR.value: estimate_k_marchenko_pastur,
        SpectralKMethod.ACTIVE_FEATURES.value: estimate_k_active_features,
    }
    if name not in mapping:
        raise DecompositionMethodError(f"Unknown spectral k estimator: {name!r}.")
    return mapping[name]


def normalize_spectral_k_method(method: SpectralKMethod | str | None) -> str | None:
    """Normalize and validate configured spectral method names.

    Returns ``None`` when no spectral method should be used, accepting both
    ``None`` and the stable config alias ``\"none\"``.
    """
    if method is None:
        return None
    name = method.value if isinstance(method, SpectralKMethod) else str(method).strip()
    if name == SpectralKMethod.NONE.value:
        return None
    resolve_k_estimator(name)
    return name


def resolve_projection_builder(kind: ProjectionBasisKind | str) -> Callable:
    """Resolve a projection-basis builder callable."""
    from ..methods.projection_basis import (
        build_pca_projection_basis,
        build_projection_basis_with_padding,
        build_random_orthonormal_basis,
    )

    name = kind.value if isinstance(kind, ProjectionBasisKind) else str(kind)
    mapping = {
        ProjectionBasisKind.PCA.value: build_pca_projection_basis,
        ProjectionBasisKind.RANDOM_ORTHONORMAL.value: build_random_orthonormal_basis,
        ProjectionBasisKind.PCA_WITH_RANDOM_PADDING.value: build_projection_basis_with_padding,
    }
    if name not in mapping:
        raise DecompositionMethodError(f"Unknown projection basis kind: {name!r}.")
    return mapping[name]


def resolve_sibling_calibrator(method: SiblingCalibrationMethod | str) -> Callable:
    """Resolve a sibling calibration callable."""
    from ..methods.sibling_calibration import (
        fit_cousin_ftest,
        fit_cousin_adjusted_wald,
        fit_cousin_tree_guided,
        fit_wald,
        fit_cousin_weighted_wald,
    )

    name = method.value if isinstance(method, SiblingCalibrationMethod) else str(method)
    mapping = {
        SiblingCalibrationMethod.WALD.value: fit_wald,
        SiblingCalibrationMethod.COUSIN_FTEST.value: fit_cousin_ftest,
        SiblingCalibrationMethod.COUSIN_ADJUSTED_WALD.value: fit_cousin_adjusted_wald,
        SiblingCalibrationMethod.COUSIN_TREE_GUIDED.value: fit_cousin_tree_guided,
        SiblingCalibrationMethod.COUSIN_WEIGHTED_WALD.value: fit_cousin_weighted_wald,
    }
    if name not in mapping:
        raise DecompositionMethodError(f"Unknown sibling calibration method: {name!r}.")
    return mapping[name]


def normalize_sibling_calibration_method(method: SiblingCalibrationMethod | str) -> str:
    """Normalize and validate configured sibling calibration method names."""
    name = method.value if isinstance(method, SiblingCalibrationMethod) else str(method).strip()
    resolve_sibling_calibrator(name)
    return name
