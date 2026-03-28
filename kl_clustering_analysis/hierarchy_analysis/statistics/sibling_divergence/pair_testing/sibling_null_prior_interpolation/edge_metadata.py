"""Edge metadata helpers for tree-neighborhood weight smoothing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ....multiple_testing.stopping_edge_recovery.serialization import parse_stopping_edge_attrs

_REQUIRED_EDGE_METADATA_COLUMNS = (
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_P_Value_BH",
)


@dataclass(frozen=True)
class StoppingEdgeSummary:
    """Stopping-edge support attached to a child edge."""

    stopping_edge_p_value: float
    distance_to_stopping_edge: float


@dataclass(frozen=True)
class EdgeLevelMetadata:
    """Edge-level Gate 2 metadata used for tree-neighborhood smoothing."""

    edge_child_ids: list[str]
    child_parent_edge_tested: np.ndarray
    child_parent_edge_significant: np.ndarray
    child_parent_edge_bh_p_values: np.ndarray
    edge_spectral_dims: dict[str, int] | None


def extract_stopping_edge_info(
    annotations_dataframe: pd.DataFrame,
) -> dict[str, StoppingEdgeSummary] | None:
    """Extract per-child stopping-edge information from DataFrame attrs."""
    metadata = parse_stopping_edge_attrs(cast(Mapping[str, object], annotations_dataframe.attrs))
    if metadata is None:
        return None

    stopping_edge_info_by_child: dict[str, StoppingEdgeSummary] = {}
    for index, child_id in enumerate(metadata.child_node_ids):
        stopping_edge_p_value = metadata.stopping_edge_p_values[index]
        if not np.isfinite(stopping_edge_p_value):
            continue
        stopping_edge_info_by_child[child_id] = StoppingEdgeSummary(
            stopping_edge_p_value=stopping_edge_p_value,
            distance_to_stopping_edge=metadata.distances_to_stopping_edge[index],
        )
    return stopping_edge_info_by_child if stopping_edge_info_by_child else None


def extract_edge_metadata(annotations_dataframe: pd.DataFrame) -> EdgeLevelMetadata:
    """Return edge-level tested/significant masks and BH p-values."""
    missing_columns = [
        column_name
        for column_name in _REQUIRED_EDGE_METADATA_COLUMNS
        if column_name not in annotations_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required child-parent edge metadata columns for sibling null-prior "
            f"interpolation: {missing_columns!r}."
        )

    child_parent_edge_tested = (
        annotations_dataframe["Child_Parent_Divergence_Tested"].astype(bool).to_numpy()
    )
    child_parent_edge_significant = (
        annotations_dataframe["Child_Parent_Divergence_Significant"].astype(bool).to_numpy()
    )
    child_parent_edge_bh_p_values = (
        annotations_dataframe["Child_Parent_Divergence_P_Value_BH"].astype(float).to_numpy()
    )

    return EdgeLevelMetadata(
        edge_child_ids=list(map(str, annotations_dataframe.index.tolist())),
        child_parent_edge_tested=child_parent_edge_tested,
        child_parent_edge_significant=child_parent_edge_significant,
        child_parent_edge_bh_p_values=child_parent_edge_bh_p_values,
        edge_spectral_dims=annotations_dataframe.attrs.get("_spectral_dims"),
    )


def edge_scale(
    node_id: str,
    annotations_dataframe: pd.DataFrame,
    edge_spectral_dims: dict[str, int] | None,
) -> float:
    """Return the edge-level scale used for neighborhood matching."""
    spectral_dimension = None
    if edge_spectral_dims is not None:
        spectral_dimension = edge_spectral_dims.get(str(node_id))
    if (
        spectral_dimension is not None
        and np.isfinite(spectral_dimension)
        and float(spectral_dimension) > 0
    ):
        return float(spectral_dimension)

    edge_degrees_of_freedom = None
    if "Child_Parent_Divergence_df" in annotations_dataframe.columns:
        edge_degrees_of_freedom_raw = cast(
            Any, annotations_dataframe.at[node_id, "Child_Parent_Divergence_df"]
        )
        edge_degrees_of_freedom = float(np.asarray(edge_degrees_of_freedom_raw, dtype=float).item())
    if (
        edge_degrees_of_freedom is not None
        and np.isfinite(edge_degrees_of_freedom)
        and edge_degrees_of_freedom > 0
    ):
        return float(edge_degrees_of_freedom)

    return 1.0


__all__ = [
    "EdgeLevelMetadata",
    "StoppingEdgeSummary",
    "edge_scale",
    "extract_edge_metadata",
    "extract_stopping_edge_info",
]
