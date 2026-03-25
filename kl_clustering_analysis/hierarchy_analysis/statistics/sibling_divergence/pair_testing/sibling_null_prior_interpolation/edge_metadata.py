"""Edge metadata helpers for tree-neighborhood weight smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from ....multiple_testing.stopping_edge_recovery.serialization import parse_stopping_edge_attrs

_DISTANCE_INF = 1e12


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
    annotations_df: pd.DataFrame,
) -> dict[str, StoppingEdgeSummary] | None:
    """Extract per-child stopping-edge information from DataFrame attrs."""
    metadata = parse_stopping_edge_attrs(annotations_df.attrs)
    if metadata is None:
        return None

    stopping_edge_info_by_child: dict[str, StoppingEdgeSummary] = {}
    for index, child_id in enumerate(metadata.child_node_ids):
        stopping_edge_p_value = metadata.stopping_edge_p_values[index]
        if not np.isfinite(stopping_edge_p_value):
            continue
        stopping_edge_info_by_child[str(child_id)] = StoppingEdgeSummary(
            stopping_edge_p_value=float(stopping_edge_p_value),
            distance_to_stopping_edge=float(metadata.distances_to_stopping_edge[index]),
        )
    return stopping_edge_info_by_child if stopping_edge_info_by_child else None


def extract_edge_metadata(annotations_df: pd.DataFrame) -> EdgeLevelMetadata:
    """Return edge-level tested/significant masks and BH p-values."""
    child_parent_edge_tested = (
        annotations_df["Child_Parent_Divergence_Tested"].astype(bool).to_numpy()
        if "Child_Parent_Divergence_Tested" in annotations_df.columns
        else np.ones(len(annotations_df), dtype=bool)
    )
    child_parent_edge_significant = (
        annotations_df["Child_Parent_Divergence_Significant"].astype(bool).to_numpy()
        if "Child_Parent_Divergence_Significant" in annotations_df.columns
        else np.zeros(len(annotations_df), dtype=bool)
    )
    child_parent_edge_bh_p_values = (
        annotations_df["Child_Parent_Divergence_P_Value_BH"].astype(float).to_numpy()
        if "Child_Parent_Divergence_P_Value_BH" in annotations_df.columns
        else np.full(len(annotations_df), np.nan, dtype=float)
    )
    return EdgeLevelMetadata(
        edge_child_ids=list(map(str, annotations_df.index.tolist())),
        child_parent_edge_tested=child_parent_edge_tested,
        child_parent_edge_significant=child_parent_edge_significant,
        child_parent_edge_bh_p_values=child_parent_edge_bh_p_values,
        edge_spectral_dims=annotations_df.attrs.get("_spectral_dims"),
    )


def edge_structural_dimension(
    node_id: str,
    annotations_df: pd.DataFrame,
    edge_spectral_dims: dict[str, int] | None,
) -> float:
    """Return the edge-level structural dimension used for neighborhood matching."""
    if edge_spectral_dims is not None:
        value = edge_spectral_dims.get(str(node_id))
        if value is not None and np.isfinite(value) and float(value) > 0:
            return float(value)

    if "Child_Parent_Divergence_df" in annotations_df.columns:
        value = float(annotations_df.at[node_id, "Child_Parent_Divergence_df"])
        if np.isfinite(value) and value > 0:
            return float(value)

    return 1.0


def build_tree_distance(tree: nx.DiGraph) -> Callable[[str, str], float]:
    """Build a cached shortest-path lookup over the undirected tree."""
    tree_undirected = tree.to_undirected(as_view=True)

    @lru_cache(maxsize=None)
    def tree_distance(node_a: str, node_b: str) -> float:
        if node_a == node_b:
            return 0.0
        try:
            return float(nx.shortest_path_length(tree_undirected, node_a, node_b))
        except nx.NetworkXNoPath:
            return _DISTANCE_INF

    return tree_distance


__all__ = [
    "EdgeLevelMetadata",
    "StoppingEdgeSummary",
    "build_tree_distance",
    "edge_structural_dimension",
    "extract_edge_metadata",
    "extract_stopping_edge_info",
]
