"""Tree-BH correction for child-parent divergence."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ...multiple_testing.tree_bh import apply_tree_bh_correction


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
    )


__all__ = [
    "apply_child_parent_divergence_tree_bh_correction",
]
