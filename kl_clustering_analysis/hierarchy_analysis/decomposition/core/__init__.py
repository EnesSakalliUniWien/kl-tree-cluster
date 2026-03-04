"""Core contracts, enums, errors, and registry for decomposition."""

from .contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    GateAnnotationBundle,
    ProjectedTestResult,
)
from .enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod
from .errors import DecompositionMethodError, DecompositionValidationError

__all__ = [
    "LEGACY_EDGE_COLUMNS",
    "LEGACY_SIBLING_COLUMNS",
    "LEGACY_SIBLING_OPTIONAL_COLUMNS",
    "GateAnnotationBundle",
    "ProjectedTestResult",
    "SpectralKMethod",
    "ProjectionBasisKind",
    "SiblingCalibrationMethod",
    "DecompositionValidationError",
    "DecompositionMethodError",
]
