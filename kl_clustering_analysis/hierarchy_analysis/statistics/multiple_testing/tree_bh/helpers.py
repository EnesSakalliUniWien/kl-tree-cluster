"""Helper utilities for the Tree-BH hierarchical multiple-testing procedure."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from statsmodels.stats.multitest import multipletests

from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

from .models import TreeBHSiblingGroupOutcome


def find_root_nodes(tree: nx.DiGraph) -> list[str]:
    """Return root nodes of the directed tree."""
    return [str(node_id) for node_id in tree.nodes() if tree.in_degree(node_id) == 0]


def group_child_indices_by_parent(
    tree: nx.DiGraph,
    child_ids: list[str],
) -> dict[str, list[int]]:
    """Map each parent node to indices of its child hypotheses."""
    sibling_group_indices: dict[str, list[int]] = defaultdict(list)

    for index, child_id in enumerate(child_ids):
        predecessors = list(tree.predecessors(child_id))
        if predecessors:
            parent_id = str(predecessors[0])
            sibling_group_indices[parent_id].append(index)

    return dict(sibling_group_indices)


def compute_child_depths(tree: nx.DiGraph, child_ids: list[str]) -> np.ndarray:
    """Return tree depths aligned to child_ids."""
    node_depths = compute_node_depths(tree)
    return np.array([node_depths.get(child_id, 0) for child_id in child_ids])


def initialize_tree_bh_arrays(
    hypothesis_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize result arrays for Tree-BH."""
    child_parent_edge_null_rejected_by_tree_bh = np.zeros(hypothesis_count, dtype=bool)
    child_parent_edge_corrected_p_values_by_tree_bh = np.ones(hypothesis_count, dtype=float)
    child_parent_edge_tested_by_tree_bh = np.zeros(hypothesis_count, dtype=bool)
    return (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
    )


def collect_eligible_sibling_groups_at_depth(
    sibling_group_indices_by_parent: dict[str, list[int]],
    child_depths: np.ndarray,
    depth: int,
    eligible_parent_nodes: set[str],
) -> list[tuple[str, list[int]]]:
    """Collect sibling groups at one depth whose parent node is eligible for testing."""
    sibling_groups_at_depth: list[tuple[str, list[int]]] = []
    for parent_id, child_indices in sibling_group_indices_by_parent.items():
        depth_child_indices = [index for index in child_indices if child_depths[index] == depth]
        if depth_child_indices and parent_id in eligible_parent_nodes:
            sibling_groups_at_depth.append((parent_id, depth_child_indices))
    return sibling_groups_at_depth


def compute_sibling_group_alpha(
    tree: nx.DiGraph,
    parent_id: str,
    ancestor_sibling_group_rejections: dict[str, tuple[int, int]],
    base_alpha: float,
    root_nodes: set[str],
) -> float:
    """Compute the Tree-BH testing level for one sibling group."""
    sibling_group_alpha = base_alpha
    current = parent_id

    while current not in root_nodes:
        predecessors = list(tree.predecessors(current))
        if not predecessors:
            break

        ancestor_parent = str(predecessors[0])
        if ancestor_parent in ancestor_sibling_group_rejections:
            rejection_count, test_count = ancestor_sibling_group_rejections[ancestor_parent]
            if test_count > 0:
                sibling_group_alpha *= rejection_count / test_count

        current = ancestor_parent

    return sibling_group_alpha


def run_bh_within_sibling_group(
    sibling_group_p_values: np.ndarray,
    sibling_group_alpha: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply BH correction to one sibling group of child hypotheses."""
    if len(sibling_group_p_values) == 0 or sibling_group_alpha <= 0:
        return None

    child_hypotheses_rejected_by_bh, child_hypothesis_corrected_p_values_by_bh, _, _ = multipletests(
        sibling_group_p_values,
        alpha=sibling_group_alpha,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )
    return child_hypotheses_rejected_by_bh, child_hypothesis_corrected_p_values_by_bh


def record_sibling_group_outcome(
    *,
    parent_id: str,
    depth: int,
    sibling_group_child_indices: list[int],
    sibling_group_p_values: np.ndarray,
    child_hypotheses_rejected_by_bh: np.ndarray,
    child_hypothesis_corrected_p_values_by_bh: np.ndarray,
    child_ids: list[str],
    child_parent_edge_null_rejected_by_tree_bh: np.ndarray,
    child_parent_edge_corrected_p_values_by_tree_bh: np.ndarray,
    child_parent_edge_tested_by_tree_bh: np.ndarray,
    eligible_parent_nodes: set[str],
    ancestor_sibling_group_rejections: dict[str, tuple[int, int]],
    sibling_group_outcomes: dict[str, TreeBHSiblingGroupOutcome],
    sibling_group_alpha: float,
) -> tuple[int, int]:
    """Record one sibling group's BH result into the aggregate Tree-BH state."""
    for within_sibling_group_index, global_index in enumerate(sibling_group_child_indices):
        child_parent_edge_null_rejected_by_tree_bh[global_index] = child_hypotheses_rejected_by_bh[
            within_sibling_group_index
        ]
        child_parent_edge_corrected_p_values_by_tree_bh[global_index] = (
            child_hypothesis_corrected_p_values_by_bh[within_sibling_group_index]
        )
        child_parent_edge_tested_by_tree_bh[global_index] = True
        if child_hypotheses_rejected_by_bh[within_sibling_group_index]:
            eligible_parent_nodes.add(str(child_ids[global_index]))

    outcome = TreeBHSiblingGroupOutcome(
        depth=depth,
        sibling_group_alpha=sibling_group_alpha,
        tested_child_ids=[str(child_ids[index]) for index in sibling_group_child_indices],
        raw_p_values=sibling_group_p_values.tolist(),
        child_hypotheses_rejected_by_bh=child_hypotheses_rejected_by_bh.tolist(),
    )
    sibling_group_outcomes[parent_id] = outcome
    ancestor_sibling_group_rejections[parent_id] = (outcome.rejection_count, outcome.test_count)
    return outcome.rejection_count, outcome.test_count


__all__ = [
    "collect_eligible_sibling_groups_at_depth",
    "compute_child_depths",
    "compute_sibling_group_alpha",
    "find_root_nodes",
    "group_child_indices_by_parent",
    "initialize_tree_bh_arrays",
    "record_sibling_group_outcome",
    "run_bh_within_sibling_group",
]
