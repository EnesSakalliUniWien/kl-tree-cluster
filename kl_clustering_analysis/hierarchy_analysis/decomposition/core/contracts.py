"""Dataclass contracts used by modular decomposition components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


LEGACY_EDGE_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Invalid",
)
LEGACY_EDGE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Ancestor_Blocked",
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
    local_gate_columns: tuple[str, ...] = LEGACY_EDGE_COLUMNS
    sibling_gate_columns: tuple[str, ...] = LEGACY_SIBLING_COLUMNS
    metadata: dict[str, Any] = field(default_factory=dict)
