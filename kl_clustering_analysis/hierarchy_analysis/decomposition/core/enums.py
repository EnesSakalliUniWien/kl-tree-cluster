"""Enums for method selection in decomposition."""

from __future__ import annotations

from enum import Enum


class SpectralKMethod(str, Enum):
    """Supported strategies for projection dimension selection."""

    MARCHENKO_PASTUR = "marchenko_pastur"
    NONE = "none"


class ProjectionBasisKind(str, Enum):
    """Supported projection basis families."""

    PCA = "pca"
    RANDOM_ORTHONORMAL = "random_orthonormal"
    PCA_WITH_RANDOM_PADDING = "pca_with_random_padding"


class SiblingCalibrationMethod(str, Enum):
    """Supported sibling divergence calibration controllers."""

    WALD = "wald"
    COUSIN_ADJUSTED_WALD = "cousin_adjusted_wald"

