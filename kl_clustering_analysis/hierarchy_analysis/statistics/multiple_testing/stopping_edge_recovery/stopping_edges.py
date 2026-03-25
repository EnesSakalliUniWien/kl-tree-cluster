from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np

from ...branch_length_utils import sanitize_positive_branch_length
from ..tree_bh import ChildParentEdgeTreeBHResult
from ._tree import get_unique_parent_id
from .models import StoppingEdgeInfo


def _resolve_stopping_edge_p_value(
    *,
    current_child_id: str,
    within_sibling_group_index: int,
    tree_bh_result: ChildParentEdgeTreeBHResult,
    child_node_id_to_hypothesis_index: Dict[str, int],
) -> float:
    """Return the best available p-value for the tested edge that stopped descent."""
    stopping_hypothesis_index = child_node_id_to_hypothesis_index.get(current_child_id)
    if stopping_hypothesis_index is not None and np.isfinite(
        tree_bh_result.child_parent_edge_corrected_p_values_by_tree_bh[
            stopping_hypothesis_index
        ]
    ):
        return float(
            tree_bh_result.child_parent_edge_corrected_p_values_by_tree_bh[
                stopping_hypothesis_index
            ]
        )

    for sibling_group_outcome in tree_bh_result.sibling_group_outcomes.values():
        sibling_group_children = [
            str(child_id) for child_id in sibling_group_outcome.tested_child_ids
        ]
        if (
            within_sibling_group_index < len(sibling_group_children)
            and sibling_group_children[within_sibling_group_index] == current_child_id
        ):
            return float(sibling_group_outcome.raw_p_values[within_sibling_group_index])

    raise ValueError(f"Missing stopping-edge p-value for child {current_child_id!r}.")


def _resolve_stopping_edge_distance(
    *,
    cumulative_branch_length: float,
    all_branch_lengths_available: bool,
    generations: int,
) -> float:
    """Return stopping-edge distance in branch-length or generation units."""
    if all_branch_lengths_available:
        return cumulative_branch_length
    return float(generations)


def _walk_to_stopping_edge(
    tree: nx.DiGraph,
    target_child_node: str,
    tree_bh_result: ChildParentEdgeTreeBHResult,
    child_node_id_to_hypothesis_index: Dict[str, int],
) -> StoppingEdgeInfo | None:
    """Walk upward from an untested child to find the nearest stopping edge."""
    current_child_node = str(target_child_node)
    generations = 0
    cumulative_branch_length = 0.0
    all_branch_lengths_available = True

    while True:
        parent_id = get_unique_parent_id(tree, current_child_node)
        if parent_id is None:
            return None

        branch_length = sanitize_positive_branch_length(
            tree.edges[parent_id, current_child_node].get("branch_length")
        )
        if branch_length is None:
            all_branch_lengths_available = False
        else:
            cumulative_branch_length += branch_length

        sibling_group_outcome = tree_bh_result.sibling_group_outcomes.get(parent_id)
        if sibling_group_outcome is not None:
            sibling_group_children = [
                str(child_id) for child_id in sibling_group_outcome.tested_child_ids
            ]
            try:
                within_sibling_group_index = sibling_group_children.index(current_child_node)
            except ValueError:
                within_sibling_group_index = -1

            if (
                within_sibling_group_index >= 0
                and not sibling_group_outcome.child_hypotheses_rejected_by_bh[
                    within_sibling_group_index
                ]
            ):
                stopping_edge_p_value = _resolve_stopping_edge_p_value(
                    current_child_id=current_child_node,
                    within_sibling_group_index=within_sibling_group_index,
                    tree_bh_result=tree_bh_result,
                    child_node_id_to_hypothesis_index=child_node_id_to_hypothesis_index,
                )
                distance_to_stopping_edge = _resolve_stopping_edge_distance(
                    cumulative_branch_length=cumulative_branch_length,
                    all_branch_lengths_available=all_branch_lengths_available,
                    generations=generations,
                )
                return StoppingEdgeInfo(
                    stopping_child_node=current_child_node,
                    stopping_edge_p_value=stopping_edge_p_value,
                    distance_to_stopping_edge=distance_to_stopping_edge,
                    generations_above=generations,
                )

        current_child_node = parent_id
        generations += 1


def recover_stopping_edge_info(
    tree: nx.DiGraph,
    tree_bh_result: ChildParentEdgeTreeBHResult,
    child_node_ids: List[str],
) -> Dict[str, StoppingEdgeInfo]:
    """Recover nearest stopping-edge information for Tree-BH-untested edges."""
    child_node_id_to_hypothesis_index = {node_id: i for i, node_id in enumerate(child_node_ids)}
    tested_edge_flags = np.asarray(
        tree_bh_result.child_parent_edge_tested_by_tree_bh,
        dtype=bool,
    )

    result: Dict[str, StoppingEdgeInfo] = {}
    for index, child_node_id in enumerate(child_node_ids):
        if tested_edge_flags[index]:
            continue
        info = _walk_to_stopping_edge(
            tree,
            child_node_id,
            tree_bh_result,
            child_node_id_to_hypothesis_index,
        )
        if info is not None:
            result[str(child_node_id)] = info
    return result
