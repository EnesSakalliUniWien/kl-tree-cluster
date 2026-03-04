"""Decomposition package for modular gate/method/backends architecture."""

from .core.contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    GateAnnotationBundle,
    ProjectedTestResult,
)
from .core.enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod

__all__ = [
    "LEGACY_EDGE_COLUMNS",
    "LEGACY_SIBLING_COLUMNS",
    "LEGACY_SIBLING_OPTIONAL_COLUMNS",
    "GateAnnotationBundle",
    "ProjectedTestResult",
    "SpectralKMethod",
    "ProjectionBasisKind",
    "SiblingCalibrationMethod",
]
