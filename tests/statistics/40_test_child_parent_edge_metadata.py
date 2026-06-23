from __future__ import annotations

import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.child_parent_edge_metadata import (
    determine_whether_sibling_pair_is_edge_blocked,
    determine_whether_sibling_pair_is_null_like,
    estimate_sibling_null_weight_from_child_parent_edges,
    extract_child_parent_edge_significance_by_node,
    extract_child_parent_edge_testing_status_by_node,
)


def test_child_parent_edge_metadata_preserves_tree_node_id_identity() -> None:
    annotations = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, False],
            "Child_Parent_Divergence_P_Value_BH": [1.0, 0.03, 0.8],
            "Child_Parent_Divergence_Tested": [True, False, True],
            "Child_Parent_Divergence_Ancestor_Blocked": [False, True, False],
        },
        index=[0, 1, 2],
    )

    significance_by_node = extract_child_parent_edge_significance_by_node(annotations)
    tested_by_node, blocked_by_node = extract_child_parent_edge_testing_status_by_node(
        annotations
    )

    assert significance_by_node == {0: False, 1: True, 2: False}
    assert tested_by_node == {0: True, 1: False, 2: True}
    assert blocked_by_node == {0: False, 1: True, 2: False}

    assert (
        determine_whether_sibling_pair_is_edge_blocked(
            1,
            2,
            child_parent_edge_tested_by_node=tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=blocked_by_node,
        )
        is True
    )
    assert (
        determine_whether_sibling_pair_is_null_like(
            1,
            2,
            child_parent_edge_significance_by_node=significance_by_node,
        )
        is False
    )


def test_sibling_null_weight_uses_joint_child_edge_weights() -> None:
    edge_p_values = {"L": 0.25, "R": 0.8}
    edge_significant = {"L": False, "R": False}
    edge_tested = {"L": True, "R": True}
    edge_blocked = {"L": False, "R": False}

    weight = estimate_sibling_null_weight_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_p_values_by_node=edge_p_values,
        child_parent_edge_significance_by_node=edge_significant,
        child_parent_edge_tested_by_node=edge_tested,
        child_parent_edge_ancestor_blocked_by_node=edge_blocked,
    )

    assert weight == 0.25 * 0.8


def test_sibling_null_weight_keeps_small_significant_child_edge_evidence() -> None:
    weight = estimate_sibling_null_weight_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_p_values_by_node={"L": 0.01, "R": 0.8},
        child_parent_edge_significance_by_node={"L": True, "R": False},
        child_parent_edge_tested_by_node={"L": True, "R": True},
        child_parent_edge_ancestor_blocked_by_node={"L": False, "R": False},
    )

    assert weight == 0.01 * 0.8


def test_sibling_null_weight_maps_significant_zero_p_value_to_positive_underflow_bound() -> None:
    weight = estimate_sibling_null_weight_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_p_values_by_node={"L": 0.0, "R": 0.8},
        child_parent_edge_significance_by_node={"L": True, "R": False},
        child_parent_edge_tested_by_node={"L": True, "R": True},
        child_parent_edge_ancestor_blocked_by_node={"L": False, "R": False},
    )

    assert 0.0 < weight < 1e-160


def test_sibling_null_weight_joint_product_is_underflow_stable() -> None:
    weight = estimate_sibling_null_weight_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_p_values_by_node={"L": 0.0, "R": 0.0},
        child_parent_edge_significance_by_node={"L": True, "R": True},
        child_parent_edge_tested_by_node={"L": True, "R": True},
        child_parent_edge_ancestor_blocked_by_node={"L": False, "R": False},
    )

    assert weight > 0.0


def test_sibling_null_weight_treats_tree_bh_stopped_edges_as_ancestor_null_evidence() -> None:
    weight = estimate_sibling_null_weight_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_p_values_by_node={"L": float("nan"), "R": 0.8},
        child_parent_edge_significance_by_node={"L": False, "R": False},
        child_parent_edge_tested_by_node={"L": False, "R": True},
        child_parent_edge_ancestor_blocked_by_node={"L": True, "R": False},
    )

    assert weight == 0.8
