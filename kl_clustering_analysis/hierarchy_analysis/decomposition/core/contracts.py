"""Dataclass contracts used by modular decomposition components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .enums import ProjectionBasisKind, SiblingCalibrationMethod, SpectralKMethod


LEGACY_EDGE_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Invalid",
)

LEGACY_SIBLING_COLUMNS: tuple[str, ...] = (
    "Sibling_Divergence_Skipped",
    "Sibling_Test_Statistic",
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_Divergence_Invalid",
    "Sibling_BH_Different",
    "Sibling_BH_Same",
)

LEGACY_SIBLING_OPTIONAL_COLUMNS: tuple[str, ...] = ("Sibling_Test_Method",)


@dataclass(frozen=True)
class DecompositionRunConfig:
    """Top-level immutable run configuration for decomposition orchestration."""

    alpha_local: float
    sibling_alpha: float
    spectral_k_method: SpectralKMethod
    sibling_calibration_method: SiblingCalibrationMethod
    projection_basis_kind: ProjectionBasisKind = ProjectionBasisKind.RANDOM_ORTHONORMAL
    min_k: int | None = None


@dataclass
class ProjectionPlan:
    """Projection basis and dimension plan for a single hypothesis test."""

    k: int
    basis_kind: ProjectionBasisKind
    projection_matrix: np.ndarray | None = None
    pca_eigenvalues: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectedTestResult:
    """Canonical projected test output shared by edge/sibling gates."""

    statistic: float
    degrees_of_freedom: float
    p_value: float
    invalid: bool = False


@dataclass
class GateAnnotationBundle:
    """Container for gate-annotation outputs and optional run metadata."""

    annotated_df: pd.DataFrame
    local_gate_column: str = "Child_Parent_Divergence_Significant"
    sibling_gate_column: str = "Sibling_BH_Different"
    local_gate_columns: tuple[str, ...] = LEGACY_EDGE_COLUMNS
    sibling_gate_columns: tuple[str, ...] = LEGACY_SIBLING_COLUMNS
    metadata: dict[str, Any] = field(default_factory=dict)
