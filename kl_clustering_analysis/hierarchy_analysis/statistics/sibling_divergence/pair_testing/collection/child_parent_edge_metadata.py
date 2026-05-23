"""Child-parent edge metadata helpers used by sibling pair testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    EDGE_GATE_COLUMNS,
)


def validate_child_parent_edge_annotation_requirements(
    annotations_dataframe: pd.DataFrame,
) -> None:
    """Validate that required child-parent edge statistics exist."""
    missing_columns = [
        column_name
        for column_name in EDGE_GATE_COLUMNS
        if column_name not in annotations_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing child-parent edge annotation columns. "
            "Run the canonical child-parent gate first. "
            f"Missing columns: {missing_columns!r}."
        )


def extract_child_parent_edge_significance_by_node(
    annotations_dataframe: pd.DataFrame,
) -> dict[object, bool]:
    """Return child-parent edge significance decisions keyed by node."""
    return extract_bool_column_dict(
        annotations_dataframe,
        "Child_Parent_Divergence_Significant",
        coerce_index_to_str=False,
    )


def extract_child_parent_edge_testing_status_by_node(
    annotations_dataframe: pd.DataFrame,
) -> tuple[dict[object, bool], dict[object, bool]]:
    """Return tested and ancestor-blocked child-parent edge status maps."""
    child_parent_edge_tested_by_node = extract_bool_column_dict(
        annotations_dataframe,
        "Child_Parent_Divergence_Tested",
        coerce_index_to_str=False,
    )
    child_parent_edge_ancestor_blocked_by_node = extract_bool_column_dict(
        annotations_dataframe,
        "Child_Parent_Divergence_Ancestor_Blocked",
        coerce_index_to_str=False,
    )
    return (
        child_parent_edge_tested_by_node,
        child_parent_edge_ancestor_blocked_by_node,
    )


def extract_child_parent_edge_p_values_by_node(
    annotations_dataframe: pd.DataFrame,
) -> dict[object, float]:
    """Return Tree-BH child-parent edge p-values keyed by node."""
    edge_p_values = annotations_dataframe["Child_Parent_Divergence_P_Value_BH"].astype(
        float
    )
    return {node_id: float(edge_p_values[node_id]) for node_id in annotations_dataframe.index}


def _require_child_node_value(
    values_by_node: dict[object, object],
    child_id: object,
    value_name: str,
) -> object:
    if child_id not in values_by_node:
        raise KeyError(f"Missing {value_name} for child node {child_id!r}.")
    return values_by_node[child_id]


def determine_whether_sibling_pair_is_gate2_blocked(
    left_child_id: object,
    right_child_id: object,
    *,
    child_parent_edge_tested_by_node: dict[object, bool],
    child_parent_edge_ancestor_blocked_by_node: dict[object, bool],
) -> bool:
    """Return whether a sibling pair is blocked by child-parent edge status."""
    left_edge_tested = bool(
        _require_child_node_value(
            child_parent_edge_tested_by_node,
            left_child_id,
            "Child_Parent_Divergence_Tested",
        )
    )
    right_edge_tested = bool(
        _require_child_node_value(
            child_parent_edge_tested_by_node,
            right_child_id,
            "Child_Parent_Divergence_Tested",
        )
    )

    left_edge_blocked = bool(
        _require_child_node_value(
            child_parent_edge_ancestor_blocked_by_node,
            left_child_id,
            "Child_Parent_Divergence_Ancestor_Blocked",
        )
    )
    right_edge_blocked = bool(
        _require_child_node_value(
            child_parent_edge_ancestor_blocked_by_node,
            right_child_id,
            "Child_Parent_Divergence_Ancestor_Blocked",
        )
    )

    return (
        (not left_edge_tested)
        or (not right_edge_tested)
        or left_edge_blocked
        or right_edge_blocked
    )


def determine_whether_sibling_pair_is_null_like(
    left_child_id: object,
    right_child_id: object,
    *,
    child_parent_edge_significance_by_node: dict[object, bool],
) -> bool:
    """Return whether the sibling pair is null-like under edge evidence."""
    return not (
        bool(
            _require_child_node_value(
                child_parent_edge_significance_by_node,
                left_child_id,
                "Child_Parent_Divergence_Significant",
            )
        )
        or bool(
            _require_child_node_value(
                child_parent_edge_significance_by_node,
                right_child_id,
                "Child_Parent_Divergence_Significant",
            )
        )
    )


def estimate_sibling_null_weight_from_child_parent_edges(
    left_child_id: object,
    right_child_id: object,
    *,
    child_parent_edge_p_values_by_node: dict[object, float],
    child_parent_edge_tested_by_node: dict[object, bool],
    child_parent_edge_ancestor_blocked_by_node: dict[object, bool],
) -> float:
    """Return the monotone empirical-null weight implied by both child edges."""

    def _edge_null_weight(child_id: object) -> float:
        p_value = float(
            _require_child_node_value(
                child_parent_edge_p_values_by_node,
                child_id,
                "Child_Parent_Divergence_P_Value_BH",
            )
        )
        if np.isfinite(p_value):
            if not 0.0 <= p_value <= 1.0:
                raise ValueError(
                    "Child-parent BH p-values must lie in [0, 1] for sibling "
                    f"null-weight estimation; child={child_id!r}, p_value={p_value!r}."
                )
            return p_value

        edge_tested = bool(
            _require_child_node_value(
                child_parent_edge_tested_by_node,
                child_id,
                "Child_Parent_Divergence_Tested",
            )
        )
        edge_blocked = bool(
            _require_child_node_value(
                child_parent_edge_ancestor_blocked_by_node,
                child_id,
                "Child_Parent_Divergence_Ancestor_Blocked",
            )
        )
        if edge_tested and not edge_blocked:
            raise ValueError(
                "Tested child-parent edges must provide a finite Tree-BH p-value "
                f"for sibling null-weight estimation; child={child_id!r}."
            )
        return 1.0

    left_weight = _edge_null_weight(left_child_id)
    right_weight = _edge_null_weight(right_child_id)
    return float(left_weight * right_weight)


__all__ = [
    "determine_whether_sibling_pair_is_gate2_blocked",
    "determine_whether_sibling_pair_is_null_like",
    "estimate_sibling_null_weight_from_child_parent_edges",
    "extract_child_parent_edge_p_values_by_node",
    "extract_child_parent_edge_significance_by_node",
    "extract_child_parent_edge_testing_status_by_node",
    "validate_child_parent_edge_annotation_requirements",
]
