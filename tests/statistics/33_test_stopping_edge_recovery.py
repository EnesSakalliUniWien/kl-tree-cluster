from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from tree_break_selection.hierarchy_analysis.statistics.multiple_testing import (
    ChildParentEdgeTreeBHResult,
    TreeBHSiblingGroupOutcome,
    recover_signal_neighbors,
    recover_stopping_edge_info,
)


def _build_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "C"),
            ("A", "D"),
            ("C", "E"),
            ("C", "F"),
        ]
    )
    return tree


def _build_unbalanced_signal_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "C"),
            ("C", "D"),
            ("D", "E"),
            ("B", "F"),
        ]
    )
    return tree


def _build_multi_parent_graph() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "C"),
            ("B", "C"),
        ]
    )
    return tree


def test_recover_stopping_edge_info_walks_to_nearest_nonrejected_ancestor() -> None:
    tree = _build_tree()
    child_node_ids = ["A", "B", "C", "D", "E", "F"]
    tree_bh_result = ChildParentEdgeTreeBHResult(
        child_parent_edge_null_rejected_by_tree_bh=np.array(
            [True, True, False, True, False, False], dtype=bool
        ),
        child_parent_edge_corrected_p_values_by_tree_bh=np.array(
            [0.01, 0.02, 0.8, 0.03, 1.0, 1.0], dtype=float
        ),
        child_parent_edge_tested_by_tree_bh=np.array(
            [True, True, True, True, False, False], dtype=bool
        ),
        tree_bh_base_alpha_by_depth={1: 0.05, 2: 0.05},
        sibling_group_outcomes={
            "root": TreeBHSiblingGroupOutcome(
                depth=1,
                sibling_group_alpha=0.05,
                tested_child_ids=["A", "B"],
                raw_p_values=[0.01, 0.02],
                child_hypotheses_rejected_by_bh=[True, True],
            ),
            "A": TreeBHSiblingGroupOutcome(
                depth=2,
                sibling_group_alpha=0.05,
                tested_child_ids=["C", "D"],
                raw_p_values=[0.6, 0.01],
                child_hypotheses_rejected_by_bh=[False, True],
            ),
        },
    )

    stopping_edge_map = recover_stopping_edge_info(tree, tree_bh_result, child_node_ids)
    signal_map = recover_signal_neighbors(
        tree,
        child_node_ids,
        child_parent_edge_null_rejected_by_tree_bh=(
            tree_bh_result.child_parent_edge_null_rejected_by_tree_bh
        ),
        child_parent_edge_tested_by_tree_bh=tree_bh_result.child_parent_edge_tested_by_tree_bh,
        child_parent_edge_corrected_p_values_by_tree_bh=(
            tree_bh_result.child_parent_edge_corrected_p_values_by_tree_bh
        ),
    )

    assert stopping_edge_map["E"].stopping_child_node == "C"
    assert stopping_edge_map["F"].stopping_child_node == "C"
    assert stopping_edge_map["E"].stopping_edge_p_value == 0.8
    assert stopping_edge_map["E"].distance_to_stopping_edge == 1.0
    assert stopping_edge_map["E"].generations_above == 1

    assert signal_map["E"].signal_node_id == "A"
    assert signal_map["E"].signal_p_value == 0.01
    assert signal_map["E"].tree_distance_to_signal_node == 2.0


def test_recover_signal_neighbors_uses_tree_distance_not_depth_gap() -> None:
    tree = _build_unbalanced_signal_tree()
    child_node_ids = ["A", "B", "C", "D", "E", "F"]

    signal_map = recover_signal_neighbors(
        tree,
        child_node_ids,
        child_parent_edge_null_rejected_by_tree_bh=np.array(
            [False, False, True, False, False, True], dtype=bool
        ),
        child_parent_edge_tested_by_tree_bh=np.array(
            [True, True, True, True, False, True], dtype=bool
        ),
        child_parent_edge_corrected_p_values_by_tree_bh=np.array(
            [0.8, 0.9, 0.03, 0.7, np.nan, 0.04], dtype=float
        ),
    )

    assert signal_map["E"].signal_node_id == "C"
    assert signal_map["E"].signal_p_value == 0.03
    assert signal_map["E"].tree_distance_to_signal_node == 2.0


def test_recover_stopping_edge_info_rejects_multi_parent_graph() -> None:
    tree = _build_multi_parent_graph()
    child_node_ids = ["A", "B", "C"]
    tree_bh_result = ChildParentEdgeTreeBHResult(
        child_parent_edge_null_rejected_by_tree_bh=np.array([True, True, False], dtype=bool),
        child_parent_edge_corrected_p_values_by_tree_bh=np.array(
            [0.01, 0.02, 1.0], dtype=float
        ),
        child_parent_edge_tested_by_tree_bh=np.array([True, True, False], dtype=bool),
        tree_bh_base_alpha_by_depth={1: 0.05},
        sibling_group_outcomes={
            "root": TreeBHSiblingGroupOutcome(
                depth=1,
                sibling_group_alpha=0.05,
                tested_child_ids=["A", "B"],
                raw_p_values=[0.01, 0.02],
                child_hypotheses_rejected_by_bh=[True, True],
            ),
        },
    )

    with pytest.raises(ValueError, match="at most one parent per node"):
        recover_stopping_edge_info(tree, tree_bh_result, child_node_ids)
