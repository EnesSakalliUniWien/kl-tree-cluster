"""Required DataFrame columns for edge-level interpolation metadata."""

from __future__ import annotations

REQUIRED_EDGE_METADATA_COLUMNS: tuple[str, ...] = (
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_P_Value_BH",
)

__all__ = ["REQUIRED_EDGE_METADATA_COLUMNS"]
