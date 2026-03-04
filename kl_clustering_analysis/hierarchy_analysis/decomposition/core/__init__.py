"""Core contracts, enums, validation, and registry for decomposition."""

from .contracts import DecompositionRunConfig, GateAnnotationBundle, ProjectedTestResult, ProjectionPlan
from .enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod
from .errors import DecompositionMethodError, DecompositionValidationError

__all__ = [
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

