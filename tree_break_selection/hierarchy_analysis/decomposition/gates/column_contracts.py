"""Shared column-prefix and contract helpers for gate annotation outputs."""

from __future__ import annotations

import pandas as pd

from ..core.errors import DecompositionValidationError

EDGE_COLUMN_PREFIX = "Child_Parent_"
SIBLING_COLUMN_PREFIX = "Sibling_"

EDGE_GATE_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_Test_Statistic",
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Invalid",
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
    "Sibling_Test_Method",
    "Sibling_Gate_P_Value_Calibration",
    "Sibling_Gate_P_Value_Role",
    "Sibling_Projection_Dimension",
    "Sibling_Fixed_Coordinate_BH_P_Value",
    "Sibling_Fixed_Block_BH_P_Value",
    "Sibling_Fixed_Global_P_Value",
    "Sibling_Sparse_Evidence_P_Value",
    "Sibling_Sparse_Evidence_Method",
    "Sibling_Sparse_Evidence_Calibration",
    "Sibling_Dense_Evidence_P_Value",
    "Sibling_Dense_Evidence_Method",
    "Sibling_Dense_Evidence_Calibration",
    "Sibling_Dense_Evidence_Test_Statistic",
    "Sibling_Dense_Evidence_Degrees_of_Freedom",
)


def prefixed_columns(df: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    """Return columns in `df` that start with `prefix` (stable order)."""
    return tuple(col for col in df.columns if col.startswith(prefix))


def edge_gate_columns(df: pd.DataFrame) -> tuple[str, ...]:
    return prefixed_columns(df, EDGE_COLUMN_PREFIX)


def sibling_gate_columns(df: pd.DataFrame) -> tuple[str, ...]:
    return prefixed_columns(df, SIBLING_COLUMN_PREFIX)


def _format_contract_detail(missing: list[str], extras: list[str]) -> str:
    detail_parts: list[str] = []
    if missing:
        detail_parts.append(f"missing={missing}")
    if extras:
        detail_parts.append(f"unexpected={extras}")
    return "; ".join(detail_parts)


def validate_edge_gate_columns(
    df: pd.DataFrame,
    *,
    error_context: str = "Edge gate columns differ from required contract",
) -> tuple[str, ...]:
    produced = edge_gate_columns(df)
    missing = [col for col in EDGE_GATE_COLUMNS if col not in produced]
    allowed = set(EDGE_GATE_COLUMNS)
    extras = [col for col in produced if col not in allowed]
    if missing or extras:
        detail = _format_contract_detail(missing, extras)
        raise DecompositionValidationError(f"{error_context}: {detail}.")
    return produced


def validate_sibling_gate_columns(
    df: pd.DataFrame,
    *,
    error_context: str = "Sibling gate columns differ from required contract",
) -> tuple[str, ...]:
    produced = sibling_gate_columns(df)
    missing = [col for col in SIBLING_GATE_COLUMNS if col not in produced]
    allowed = set(SIBLING_GATE_COLUMNS)
    extras = [col for col in produced if col not in allowed]
    if missing or extras:
        detail = _format_contract_detail(missing, extras)
        raise DecompositionValidationError(f"{error_context}: {detail}.")
    return produced


__all__ = [
    "EDGE_COLUMN_PREFIX",
    "EDGE_GATE_COLUMNS",
    "SIBLING_COLUMN_PREFIX",
    "SIBLING_GATE_COLUMNS",
    "prefixed_columns",
    "edge_gate_columns",
    "sibling_gate_columns",
    "validate_edge_gate_columns",
    "validate_sibling_gate_columns",
]
