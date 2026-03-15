"""Core contracts, errors, and result types for decomposition."""

from .contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    GateAnnotationBundle,
    ProjectedTestResult,
)
from .eigen_result import EigenResult
from .errors import DecompositionMethodError, DecompositionValidationError

__all__ = [
    "EigenResult",
    "LEGACY_EDGE_COLUMNS",
    "LEGACY_SIBLING_COLUMNS",
    "LEGACY_SIBLING_OPTIONAL_COLUMNS",
    "GateAnnotationBundle",
    "ProjectedTestResult",
    "DecompositionValidationError",
    "DecompositionMethodError",
]
