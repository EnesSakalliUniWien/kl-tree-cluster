"""Core contracts, enums, validation, and registry for decomposition."""

from .contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    DecompositionRunConfig,
    GateAnnotationBundle,
    ProjectedTestResult,
    ProjectionPlan,
)
from .enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod
from .errors import DecompositionMethodError, DecompositionValidationError

__all__ = [
    "LEGACY_EDGE_COLUMNS",
    "LEGACY_SIBLING_COLUMNS",
    "LEGACY_SIBLING_OPTIONAL_COLUMNS",
    "DecompositionRunConfig",
    "GateAnnotationBundle",
    "ProjectionPlan",
    "ProjectedTestResult",
    "SpectralKMethod",
    "ProjectionBasisKind",
    "SiblingCalibrationMethod",
    "DecompositionValidationError",
    "DecompositionMethodError",
]
