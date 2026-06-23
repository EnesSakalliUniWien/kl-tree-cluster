"""Dataclass contracts used by modular decomposition components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EDGE_GATE_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Invalid",
)
EDGE_GATE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Ancestor_Blocked",
)

SIBLING_GATE_COLUMNS: tuple[str, ...] = (
    "Sibling_Divergence_Skipped",
    "Sibling_Test_Statistic",
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_Divergence_Invalid",
    "Sibling_BH_Different",
    "Sibling_BH_Same",
)

SIBLING_GATE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "Sibling_Test_Method",
    "Sibling_Projection_Dimension_Source",
    "Sibling_Resolved_Projection_Dimension",
    "Sibling_Used_Parent_Principal_Component_Basis",
)

GATE_ANNOTATION_METADATA_ATTR = "_gate_annotation_metadata"


@dataclass
class SpectralContext:
    """Typed Gate 2 spectral outputs reused by Gate 3."""

    spectral_projection_dimensions_by_node: dict[str, int] | None = None
    principal_component_projections_by_node: dict[str, np.ndarray] | None = None
    principal_component_eigenvalues_by_node: dict[str, np.ndarray] | None = None
    single_feature_subtree_audit: dict[str, Any] | None = None


@dataclass
class Gate2Result:
    """Typed Gate 2 output passed into the sibling gate."""

    annotated_df: pd.DataFrame
    spectral_context: SpectralContext
    local_gate_columns: tuple[str, ...] = EDGE_GATE_COLUMNS
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateAnnotationBundle:
    """Container for gate-annotation outputs and optional run metadata."""

    annotated_df: pd.DataFrame
    local_gate_columns: tuple[str, ...] = EDGE_GATE_COLUMNS
    sibling_gate_columns: tuple[str, ...] = SIBLING_GATE_COLUMNS
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_two_result: Gate2Result | None = None
