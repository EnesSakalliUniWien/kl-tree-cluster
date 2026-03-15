"""Decomposition package for modular gate/backends architecture."""

from .core.contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    GateAnnotationBundle,
    ProjectedTestResult,
)

__all__ = [
    "LEGACY_EDGE_COLUMNS",
    "LEGACY_SIBLING_COLUMNS",
    "LEGACY_SIBLING_OPTIONAL_COLUMNS",
    "GateAnnotationBundle",
    "ProjectedTestResult",
]
