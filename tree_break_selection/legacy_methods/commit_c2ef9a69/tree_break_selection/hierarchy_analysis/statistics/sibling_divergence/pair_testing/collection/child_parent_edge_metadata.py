"""Child-parent edge metadata helpers used by sibling pair testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection.core_utils.data_utils import (
    extract_bool_column_dict,
)


def validate_child_parent_edge_annotation_requirements(
    annotations_dataframe: pd.DataFrame,
) -> None:
    """Validate that required child-parent edge statistics exist."""
    if "Child_Parent_Divergence_Significant" not in annotations_dataframe.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. Run child-parent test first."
        )
    if "Child_Parent_Divergence_P_Value_BH" not in annotations_dataframe.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_P_Value_BH' column. Run child-parent test first."
        )


def extract_child_parent_edge_significance_by_node(
    annotations_dataframe: pd.DataFrame,
) -> dict[str, bool]:
    """Return child-parent edge significance decisions keyed by node."""
    return extract_bool_column_dict(
        annotations_dataframe,
        "Child_Parent_Divergence_Significant",
    )


def extract_child_parent_edge_pvalues_by_node(
    annotations_dataframe: pd.DataFrame,
) -> dict[str, float]:
    """Return BH-corrected child-parent edge p-values keyed by node."""
    edge_p_value_series = annotations_dataframe["Child_Parent_Divergence_P_Value_BH"].astype(float)
    return {
        str(node_id): (
            float(edge_p_value_series[node_id])
            if np.isfinite(edge_p_value_series[node_id])
            else 1.0
        )
        for node_id in annotations_dataframe.index
    }


def extract_child_parent_edge_testing_status_by_node(
    annotations_dataframe: pd.DataFrame,
) -> tuple[dict[str, bool] | None, dict[str, bool] | None]:
    """Return tested and ancestor-blocked child-parent edge status maps."""
    child_parent_edge_tested_by_node: dict[str, bool] | None = None
    child_parent_edge_ancestor_blocked_by_node: dict[str, bool] | None = None
    if "Child_Parent_Divergence_Tested" in annotations_dataframe.columns:
        child_parent_edge_tested_by_node = extract_bool_column_dict(
            annotations_dataframe,
            "Child_Parent_Divergence_Tested",
        )
    if "Child_Parent_Divergence_Ancestor_Blocked" in annotations_dataframe.columns:
        child_parent_edge_ancestor_blocked_by_node = extract_bool_column_dict(
            annotations_dataframe,
            "Child_Parent_Divergence_Ancestor_Blocked",
        )
    return (
        child_parent_edge_tested_by_node,
        child_parent_edge_ancestor_blocked_by_node,
    )


def determine_whether_sibling_pair_is_gate2_blocked(
    left_child_id: str,
    right_child_id: str,
    *,
    child_parent_edge_tested_by_node: dict[str, bool] | None,
    child_parent_edge_ancestor_blocked_by_node: dict[str, bool] | None,
) -> bool:
    """Return whether a sibling pair is blocked by child-parent edge status."""
    left_edge_tested = (
        child_parent_edge_tested_by_node.get(left_child_id, True)
        if child_parent_edge_tested_by_node
        else True
    )
    right_edge_tested = (
        child_parent_edge_tested_by_node.get(right_child_id, True)
        if child_parent_edge_tested_by_node
        else True
    )

    left_edge_blocked = (
        child_parent_edge_ancestor_blocked_by_node.get(left_child_id, False)
        if child_parent_edge_ancestor_blocked_by_node
        else False
    )
    right_edge_blocked = (
        child_parent_edge_ancestor_blocked_by_node.get(right_child_id, False)
        if child_parent_edge_ancestor_blocked_by_node
        else False
    )

    return (
        (not left_edge_tested)
        or (not right_edge_tested)
        or left_edge_blocked
        or right_edge_blocked
    )


def determine_whether_sibling_pair_is_null_like(
    left_child_id: str,
    right_child_id: str,
    *,
    child_parent_edge_significance_by_node: dict[str, bool],
) -> bool:
    """Return whether the sibling pair is null-like under edge evidence."""
    return not (
        child_parent_edge_significance_by_node.get(left_child_id, False)
        or child_parent_edge_significance_by_node.get(right_child_id, False)
    )


def estimate_sibling_null_prior_from_child_parent_edges(
    left_child_id: str,
    right_child_id: str,
    *,
    child_parent_edge_pvalues_by_node: dict[str, float],
) -> float:
    """Estimate the sibling null prior from child-parent edge p-values."""
    return min(
        child_parent_edge_pvalues_by_node.get(left_child_id, 1.0),
        child_parent_edge_pvalues_by_node.get(right_child_id, 1.0),
    )


__all__ = [
    "determine_whether_sibling_pair_is_gate2_blocked",
    "determine_whether_sibling_pair_is_null_like",
    "estimate_sibling_null_prior_from_child_parent_edges",
    "extract_child_parent_edge_pvalues_by_node",
    "extract_child_parent_edge_significance_by_node",
    "extract_child_parent_edge_testing_status_by_node",
    "validate_child_parent_edge_annotation_requirements",
]
