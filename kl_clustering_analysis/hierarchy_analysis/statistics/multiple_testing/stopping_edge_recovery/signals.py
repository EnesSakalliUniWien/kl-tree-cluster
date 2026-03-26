from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np

from ._tree import build_tree_distance_resolver
from .models import SignalNeighborInfo


@dataclass(frozen=True)
class _SignificantSignalNodes:
    """Tested-significant node ids aligned with their corrected p-values."""

    node_ids: tuple[str, ...]
    p_values: tuple[float, ...]


def _collect_significant_signal_nodes(
    *,
    child_node_ids: List[str],
    child_parent_edge_tested_by_tree_bh: np.ndarray,
    child_parent_edge_null_rejected_by_tree_bh: np.ndarray,
    child_parent_edge_corrected_p_values_by_tree_bh: np.ndarray,
) -> _SignificantSignalNodes:
    """Collect tested-significant node ids and corrected p-values in child order."""
    node_ids: list[str] = []
    p_values: list[float] = []

    for index, child_node_id in enumerate(child_node_ids):
        if not (
            child_parent_edge_tested_by_tree_bh[index]
            and child_parent_edge_null_rejected_by_tree_bh[index]
        ):
            continue

        node_ids.append(child_node_id)
        p_value = child_parent_edge_corrected_p_values_by_tree_bh[index]
        p_values.append(float(p_value) if np.isfinite(p_value) else 0.0)

    return _SignificantSignalNodes(tuple(node_ids), tuple(p_values))


def _find_nearest_signal_neighbor(
    *,
    untested_child_node: str,
    signal_nodes: _SignificantSignalNodes,
    tree_distance: Callable[[str, str], float],
) -> SignalNeighborInfo:
    """Return the nearest tested-significant node for one untested child."""
    if not signal_nodes.node_ids:
        return SignalNeighborInfo(
            sig_node=None,
            sig_p_value=1.0,
            distance_to_sig=float("inf"),
        )

    best_distance = float("inf")
    best_index = 0
    for sig_index, sig_node in enumerate(signal_nodes.node_ids):
        distance = tree_distance(untested_child_node, sig_node)
        if distance < best_distance:
            best_distance = float(distance)
            best_index = sig_index

    return SignalNeighborInfo(
        sig_node=signal_nodes.node_ids[best_index],
        sig_p_value=signal_nodes.p_values[best_index],
        distance_to_sig=best_distance,
    )


def recover_signal_neighbors(
    tree: nx.DiGraph,
    child_node_ids: List[str],
    child_parent_edge_null_rejected_by_tree_bh: np.ndarray,
    child_parent_edge_tested_by_tree_bh: np.ndarray,
    child_parent_edge_corrected_p_values_by_tree_bh: np.ndarray,
) -> dict[str, SignalNeighborInfo]:
    """Find the nearest tested-significant edge by tree distance for each untested child."""
    tested_hypotheses_arr = np.asarray(child_parent_edge_tested_by_tree_bh, dtype=bool)
    rejected_arr = np.asarray(child_parent_edge_null_rejected_by_tree_bh, dtype=bool)
    tree_distance = build_tree_distance_resolver(tree)
    signal_nodes = _collect_significant_signal_nodes(
        child_node_ids=child_node_ids,
        child_parent_edge_tested_by_tree_bh=tested_hypotheses_arr,
        child_parent_edge_null_rejected_by_tree_bh=rejected_arr,
        child_parent_edge_corrected_p_values_by_tree_bh=(
            child_parent_edge_corrected_p_values_by_tree_bh
        ),
    )

    result: dict[str, SignalNeighborInfo] = {}
    for index, child_node_id in enumerate(child_node_ids):
        if tested_hypotheses_arr[index]:
            continue

        result[child_node_id] = _find_nearest_signal_neighbor(
            untested_child_node=child_node_id,
            signal_nodes=signal_nodes,
            tree_distance=tree_distance,
        )

    return result
