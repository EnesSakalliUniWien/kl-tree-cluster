"""Recover blocker ancestry and signal neighborhood for TreeBH-blocked edges.

When TreeBH stops exploring at a family because a parent edge was not
rejected, all descendant edges are blocked and therefore never directly
tested. This module recovers:

1. The nearest tested non-significant ancestor edge (the blocker).
2. The nearest tested significant edge for signal-pressure discounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import networkx as nx
import numpy as np

from ..branch_length_utils import sanitize_positive_branch_length
from .tree_bh import TreeBHResult


@dataclass(frozen=True)
class BlockerInfo:
    """Ancestry metadata for a TreeBH-blocked edge."""

    blocker_node: str
    blocker_p_value: float
    distance_to_blocker: float
    generations_above: int


@dataclass(frozen=True)
class SignalNeighborInfo:
    """Nearest tested significant edge for signal-pressure computation."""

    sig_node: str | None
    sig_p_value: float
    distance_to_sig: float


def _get_unique_parent_id(tree: nx.DiGraph, node_id: str) -> str | None:
    """Return a node's unique parent or raise if the graph is not tree-shaped."""
    predecessors = [str(parent_id) for parent_id in tree.predecessors(node_id)]
    if not predecessors:
        return None
    if len(predecessors) > 1:
        raise ValueError(
            "blocker recovery expects a rooted tree with at most one parent per node; "
            f"node {node_id!r} has parents {predecessors!r}."
        )
    return predecessors[0]


def _walk_to_blocker(
    tree: nx.DiGraph,
    target_child: str,
    tree_bh_result: TreeBHResult,
    child_id_to_index: Dict[str, int],
) -> BlockerInfo | None:
    """Walk up from a blocked child to find the nearest non-rejected ancestor."""
    current = str(target_child)
    generations = 0
    cumulative_branch_length = 0.0
    all_branch_lengths_available = True

    while True:
        parent_id = _get_unique_parent_id(tree, current)
        if parent_id is None:
            return None

        branch_length = sanitize_positive_branch_length(tree.edges[parent_id, current].get("branch_length"))
        if branch_length is None:
            all_branch_lengths_available = False
        else:
            cumulative_branch_length += branch_length

        family_outcome = tree_bh_result.family_outcomes.get(parent_id)
        if family_outcome is not None:
            family_children = [str(child_id) for child_id in family_outcome.tested_child_ids]
            try:
                within_family_index = family_children.index(current)
            except ValueError:
                within_family_index = -1

            if within_family_index >= 0 and not family_outcome.reject_mask[within_family_index]:
                blocking_index = child_id_to_index.get(current)
                blocker_p = (
                    float(tree_bh_result.adjusted_p_values[blocking_index])
                    if blocking_index is not None
                    and np.isfinite(tree_bh_result.adjusted_p_values[blocking_index])
                    else float(family_outcome.raw_p_values[within_family_index])
                )
                distance = (
                    cumulative_branch_length if all_branch_lengths_available else float(generations)
                )
                return BlockerInfo(
                    blocker_node=current,
                    blocker_p_value=blocker_p,
                    distance_to_blocker=distance,
                    generations_above=generations,
                )

        current = parent_id
        generations += 1


def recover_blocker_metadata(
    tree: nx.DiGraph,
    tree_bh_result: TreeBHResult,
    child_ids: List[str],
) -> Dict[str, BlockerInfo]:
    """Recover blocker ancestry for TreeBH-blocked edges."""
    child_id_to_index = {cid: i for i, cid in enumerate(child_ids)}
    tested_mask = np.asarray(tree_bh_result.tested_mask, dtype=bool)

    result: Dict[str, BlockerInfo] = {}
    for index, child_id in enumerate(child_ids):
        if tested_mask[index]:
            continue
        info = _walk_to_blocker(tree, child_id, tree_bh_result, child_id_to_index)
        if info is not None:
            result[str(child_id)] = info
    return result


def recover_signal_neighbors(
    tree: nx.DiGraph,
    child_ids: List[str],
    reject_mask: np.ndarray,
    tested_mask: np.ndarray,
    corrected_p_values: np.ndarray,
    depths: Dict[str, int],
) -> Dict[str, SignalNeighborInfo]:
    """Find the nearest tested-significant edge by tree distance for each blocked child."""
    tested_arr = np.asarray(tested_mask, dtype=bool)
    reject_arr = np.asarray(reject_mask, dtype=bool)
    tree_undirected = tree.to_undirected(as_view=True)

    @lru_cache(maxsize=None)
    def _tree_distance(node_a: str, node_b: str) -> float:
        if node_a == node_b:
            return 0.0
        try:
            return float(nx.shortest_path_length(tree_undirected, node_a, node_b))
        except nx.NetworkXNoPath:
            return float("inf")

    sig_nodes: List[str] = []
    sig_p_values: List[float] = []
    for index, child_id in enumerate(child_ids):
        if tested_arr[index] and reject_arr[index]:
            sig_nodes.append(str(child_id))
            p_value = corrected_p_values[index]
            sig_p_values.append(float(p_value) if np.isfinite(p_value) else 0.0)

    result: Dict[str, SignalNeighborInfo] = {}
    for index, child_id in enumerate(child_ids):
        if tested_arr[index]:
            continue

        if not sig_nodes:
            result[str(child_id)] = SignalNeighborInfo(
                sig_node=None,
                sig_p_value=1.0,
                distance_to_sig=float("inf"),
            )
            continue

        blocked_child_id = str(child_id)
        best_distance = float("inf")
        best_index = 0
        for sig_index, sig_node in enumerate(sig_nodes):
            distance = _tree_distance(blocked_child_id, sig_node)
            if distance < best_distance:
                best_distance = float(distance)
                best_index = sig_index

        result[blocked_child_id] = SignalNeighborInfo(
            sig_node=sig_nodes[best_index],
            sig_p_value=sig_p_values[best_index],
            distance_to_sig=best_distance,
        )

    return result


__all__ = [
    "BlockerInfo",
    "SignalNeighborInfo",
    "recover_blocker_metadata",
    "recover_signal_neighbors",
]
