"""Tree-BH hierarchical FDR correction implementation."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .helpers import (
    collect_eligible_sibling_groups_at_depth,
    compute_child_depths,
    compute_sibling_group_alpha,
    find_root_nodes,
    group_child_indices_by_parent,
    initialize_tree_bh_arrays,
    record_sibling_group_outcome,
    run_bh_within_sibling_group,
)
from .models import ChildParentEdgeTreeBHResult


def apply_tree_bh_correction(
    tree: nx.DiGraph,
    p_values: np.ndarray,
    child_ids: list[str],
    alpha: float = 0.05,
    verbose: bool = False,
) -> ChildParentEdgeTreeBHResult:
    """Apply Tree-BH hierarchical FDR correction.

    The procedure works top-down through the tree. Within each reached sibling
    group, it applies BH at a sibling-group-specific level obtained by scaling
    the base alpha by ancestor rejection proportions.
    """
    hypothesis_count = len(p_values)
    (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
    ) = initialize_tree_bh_arrays(hypothesis_count)

    if hypothesis_count == 0:
        return ChildParentEdgeTreeBHResult(
            child_parent_edge_null_rejected_by_tree_bh=child_parent_edge_null_rejected_by_tree_bh,
            child_parent_edge_corrected_p_values_by_tree_bh=(
                child_parent_edge_corrected_p_values_by_tree_bh
            ),
            child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
            tree_bh_base_alpha_by_depth={},
            sibling_group_outcomes={},
        )

    child_depths = compute_child_depths(tree, child_ids)
    sibling_group_indices_by_parent = group_child_indices_by_parent(tree, child_ids)
    root_nodes = set(find_root_nodes(tree))
    eligible_parent_nodes: set[str] = root_nodes.copy()
    ancestor_sibling_group_rejections: dict[str, tuple[int, int]] = {}
    tree_bh_base_alpha_by_depth: dict[int, float] = {}
    sibling_group_outcomes = {}

    if verbose:
        print(
            f"Tree-BH: {hypothesis_count} edges, {len(set(child_depths))} depths, "
            f"{len(sibling_group_indices_by_parent)} sibling groups"
        )
        print(f"Roots: {root_nodes}")

    for depth in sorted(set(child_depths)):
        if verbose:
            print(f"\n--- Depth {depth} ---")

        sibling_groups_at_depth = collect_eligible_sibling_groups_at_depth(
            sibling_group_indices_by_parent,
            child_depths,
            depth,
            eligible_parent_nodes,
        )
        if not sibling_groups_at_depth:
            if verbose:
                print(f"  No sibling groups to test at depth {depth}")
            continue

        for parent_id, sibling_group_child_indices in sibling_groups_at_depth:
            sibling_group_alpha = compute_sibling_group_alpha(
                tree,
                parent_id,
                ancestor_sibling_group_rejections,
                alpha,
                root_nodes,
            )

            if verbose:
                print(
                    f"  Sibling group under {parent_id}: {len(sibling_group_child_indices)} children, "
                    f"alpha_adj = {sibling_group_alpha:.6f}"
                )

            sibling_group_p_values = p_values[sibling_group_child_indices]
            sibling_group_bh_result = run_bh_within_sibling_group(
                sibling_group_p_values,
                sibling_group_alpha,
            )
            if sibling_group_bh_result is None:
                continue

            (
                child_hypotheses_rejected_by_bh,
                child_hypothesis_corrected_p_values_by_bh,
            ) = sibling_group_bh_result
            rejection_count, test_count = record_sibling_group_outcome(
                parent_id=parent_id,
                depth=depth,
                sibling_group_child_indices=sibling_group_child_indices,
                sibling_group_p_values=sibling_group_p_values,
                child_hypotheses_rejected_by_bh=child_hypotheses_rejected_by_bh,
                child_hypothesis_corrected_p_values_by_bh=(
                    child_hypothesis_corrected_p_values_by_bh
                ),
                child_ids=child_ids,
                child_parent_edge_null_rejected_by_tree_bh=(
                    child_parent_edge_null_rejected_by_tree_bh
                ),
                child_parent_edge_corrected_p_values_by_tree_bh=(
                    child_parent_edge_corrected_p_values_by_tree_bh
                ),
                child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
                eligible_parent_nodes=eligible_parent_nodes,
                ancestor_sibling_group_rejections=ancestor_sibling_group_rejections,
                sibling_group_outcomes=sibling_group_outcomes,
                sibling_group_alpha=sibling_group_alpha,
            )

            if verbose:
                print(
                    f"    Rejected {rejection_count}/{test_count} "
                    f"(proportion: {rejection_count / test_count:.2%})"
                )

        tree_bh_base_alpha_by_depth[int(depth)] = alpha

    return ChildParentEdgeTreeBHResult(
        child_parent_edge_null_rejected_by_tree_bh=child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh=(
            child_parent_edge_corrected_p_values_by_tree_bh
        ),
        child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
        tree_bh_base_alpha_by_depth=tree_bh_base_alpha_by_depth,
        sibling_group_outcomes=sibling_group_outcomes,
    )


__all__ = ["apply_tree_bh_correction"]
