"""Shared column-prefix and legacy-contract helpers for gate adapters."""

from __future__ import annotations

import pandas as pd

from ..core.contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
)
from ..core.errors import DecompositionValidationError

EDGE_COLUMN_PREFIX = "Child_Parent_"
SIBLING_COLUMN_PREFIX = "Sibling_"


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


def validate_legacy_edge_columns(
    df: pd.DataFrame,
    *,
    error_context: str = "Edge gate columns differ from legacy contract",
) -> tuple[str, ...]:
    produced = edge_gate_columns(df)
    missing = [col for col in LEGACY_EDGE_COLUMNS if col not in produced]
    extras = [col for col in produced if col not in LEGACY_EDGE_COLUMNS]
    if missing or extras:
        detail = _format_contract_detail(missing, extras)
        raise DecompositionValidationError(f"{error_context}: {detail}.")
    return produced


def validate_legacy_sibling_columns(
    df: pd.DataFrame,
    *,
    error_context: str = "Sibling gate columns differ from legacy contract",
) -> tuple[str, ...]:
    produced = sibling_gate_columns(df)
    missing = [col for col in LEGACY_SIBLING_COLUMNS if col not in produced]
    allowed = set(LEGACY_SIBLING_COLUMNS) | set(LEGACY_SIBLING_OPTIONAL_COLUMNS)
    extras = [col for col in produced if col not in allowed]
    if missing or extras:
        detail = _format_contract_detail(missing, extras)
        raise DecompositionValidationError(f"{error_context}: {detail}.")
    return produced


__all__ = [
    "EDGE_COLUMN_PREFIX",
    "SIBLING_COLUMN_PREFIX",
    "prefixed_columns",
    "edge_gate_columns",
    "sibling_gate_columns",
    "validate_legacy_edge_columns",
    "validate_legacy_sibling_columns",
]
