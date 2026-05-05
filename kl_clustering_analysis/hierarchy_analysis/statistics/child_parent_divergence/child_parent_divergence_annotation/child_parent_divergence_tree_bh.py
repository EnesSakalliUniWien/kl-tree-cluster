"""Tree-BH correction and stopping-edge recovery for child-parent divergence."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ...multiple_testing.tree_bh import ChildParentEdgeTreeBHResult, apply_tree_bh_correction


def apply_child_parent_divergence_tree_bh_correction(
    tree: nx.DiGraph,
    p_values_for_correction: np.ndarray,
    child_ids: list[str],
    edge_alpha: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ChildParentEdgeTreeBHResult | None,
]:
    """Apply Tree-BH correction to child-parent edge p-values."""
    tree_bh_result = apply_tree_bh_correction(
        tree,
        p_values_for_correction,
        child_ids,
        alpha=edge_alpha,
    )
    child_parent_edge_null_rejected_by_tree_bh = (
        tree_bh_result.child_parent_edge_null_rejected_by_tree_bh
    )
    child_parent_edge_corrected_p_values_by_tree_bh = (
        tree_bh_result.child_parent_edge_corrected_p_values_by_tree_bh.copy()
    )
    child_parent_edge_tested_by_tree_bh = np.asarray(
        tree_bh_result.child_parent_edge_tested_by_tree_bh,
        dtype=bool,
    )
    ancestor_blocked_mask = ~child_parent_edge_tested_by_tree_bh
    child_parent_edge_corrected_p_values_by_tree_bh = np.where(
        child_parent_edge_tested_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        np.nan,
    )

    return (
        child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh,
        child_parent_edge_tested_by_tree_bh,
        ancestor_blocked_mask,
        tree_bh_result,
    )


def attach_child_parent_stopping_edge_recovery_metadata(
    annotations_df: pd.DataFrame,
    *,
    tree: nx.DiGraph,
    child_ids: list[str],
    child_parent_edge_null_rejected_by_tree_bh: np.ndarray,
    child_parent_edge_tested_by_tree_bh: np.ndarray,
    child_parent_edge_corrected_p_values_by_tree_bh: np.ndarray,
    tree_bh_result: ChildParentEdgeTreeBHResult | None,
    ancestor_blocked_edge_flags: np.ndarray,
) -> None:
    """Attach stopping-edge recovery metadata for ancestor-blocked edges."""
    if int(np.sum(ancestor_blocked_edge_flags)) <= 0:
        return

    from ...multiple_testing.stopping_edge_recovery.serialization import (
        STOPPING_EDGE_INFO_ATTR_KEY,
        build_stopping_edge_attrs,
    )
    from ...multiple_testing.stopping_edge_recovery.signals import recover_signal_neighbors
    from ...multiple_testing.stopping_edge_recovery.stopping_edges import (
        recover_stopping_edge_info,
    )

    assert tree_bh_result is not None
    stopping_edge_info_by_child = recover_stopping_edge_info(tree, tree_bh_result, child_ids)

    nearest_signal_neighbor_by_child = recover_signal_neighbors(
        tree,
        child_ids,
        child_parent_edge_null_rejected_by_tree_bh=child_parent_edge_null_rejected_by_tree_bh,
        child_parent_edge_tested_by_tree_bh=child_parent_edge_tested_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh=child_parent_edge_corrected_p_values_by_tree_bh,
    )
    annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = build_stopping_edge_attrs(
        child_node_ids=child_ids,
        stopping_edge_info_by_child=stopping_edge_info_by_child,
        signal_neighbor_info_by_child=nearest_signal_neighbor_by_child,
    )


__all__ = [
    "apply_child_parent_divergence_tree_bh_correction",
    "attach_child_parent_stopping_edge_recovery_metadata",
]
