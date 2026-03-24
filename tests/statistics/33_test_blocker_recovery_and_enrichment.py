from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing import (
    TreeBHFamilyOutcome,
    TreeBHResult,
    recover_blocker_metadata,
    recover_signal_neighbors,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.nearby_stable import (
    enrich_blocked_weights,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
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


def test_recover_blocker_metadata_walks_to_nearest_nonrejected_ancestor() -> None:
    tree = _build_tree()
    child_ids = ["A", "B", "C", "D", "E", "F"]
    tree_bh_result = TreeBHResult(
        reject=np.array([True, True, False, True, False, False], dtype=bool),
        adjusted_p_values=np.array([0.01, 0.02, 0.8, 0.03, 1.0, 1.0], dtype=float),
        tested_mask=np.array([True, True, True, True, False, False], dtype=bool),
        level_alpha_by_depth={1: 0.05, 2: 0.05},
        family_outcomes={
            "root": TreeBHFamilyOutcome(
                depth=1,
                adjusted_alpha=0.05,
                tested_child_ids=["A", "B"],
                raw_p_values=[0.01, 0.02],
                reject_mask=[True, True],
            ),
            "A": TreeBHFamilyOutcome(
                depth=2,
                adjusted_alpha=0.05,
                tested_child_ids=["C", "D"],
                raw_p_values=[0.6, 0.01],
                reject_mask=[False, True],
            ),
        },
    )

    blocker_map = recover_blocker_metadata(tree, tree_bh_result, child_ids)
    signal_map = recover_signal_neighbors(
        tree,
        child_ids,
        reject_mask=tree_bh_result.reject,
        tested_mask=tree_bh_result.tested_mask,
        corrected_p_values=tree_bh_result.adjusted_p_values,
        depths={"root": 0, "A": 1, "B": 1, "C": 2, "D": 2, "E": 3, "F": 3},
    )

    assert blocker_map["E"].blocker_node == "C"
    assert blocker_map["F"].blocker_node == "C"
    assert blocker_map["E"].blocker_p_value == 0.8
    assert blocker_map["E"].distance_to_blocker == 1.0
    assert blocker_map["E"].generations_above == 1

    assert signal_map["E"].sig_node == "A"
    assert signal_map["E"].sig_p_value == 0.01
    assert signal_map["E"].distance_to_sig == 2.0


def test_recover_signal_neighbors_uses_tree_distance_not_depth_gap() -> None:
    tree = _build_unbalanced_signal_tree()
    child_ids = ["A", "B", "C", "D", "E", "F"]

    signal_map = recover_signal_neighbors(
        tree,
        child_ids,
        reject_mask=np.array([False, False, True, False, False, True], dtype=bool),
        tested_mask=np.array([True, True, True, True, False, True], dtype=bool),
        corrected_p_values=np.array([0.8, 0.9, 0.03, 0.7, np.nan, 0.04], dtype=float),
        depths={"root": 0, "A": 1, "B": 1, "C": 2, "D": 3, "E": 4, "F": 2},
    )

    assert signal_map["E"].sig_node == "C"
    assert signal_map["E"].sig_p_value == 0.03
    assert signal_map["E"].distance_to_sig == 2.0


def test_recover_blocker_metadata_rejects_multi_parent_graph() -> None:
    tree = _build_multi_parent_graph()
    child_ids = ["A", "B", "C"]
    tree_bh_result = TreeBHResult(
        reject=np.array([True, True, False], dtype=bool),
        adjusted_p_values=np.array([0.01, 0.02, 1.0], dtype=float),
        tested_mask=np.array([True, True, False], dtype=bool),
        level_alpha_by_depth={1: 0.05},
        family_outcomes={
            "root": TreeBHFamilyOutcome(
                depth=1,
                adjusted_alpha=0.05,
                tested_child_ids=["A", "B"],
                raw_p_values=[0.01, 0.02],
                reject_mask=[True, True],
            ),
        },
    )

    with pytest.raises(ValueError, match="at most one parent per node"):
        recover_blocker_metadata(tree, tree_bh_result, child_ids)


def test_enrich_blocked_weights_replaces_legacy_weight_and_populates_audit_fields() -> None:
    tree = _build_tree()
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Tested": [True, True, True, True, False, False],
            "Child_Parent_Divergence_Ancestor_Blocked": [False, False, False, False, True, True],
            "Child_Parent_Divergence_Significant": [True, True, False, True, False, False],
            "Child_Parent_Divergence_P_Value_BH": [0.01, 0.02, 0.8, 0.03, np.nan, np.nan],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )
    annotations_df.attrs["_blocker_metadata"] = {
        "child_ids": ["A", "B", "C", "D", "E", "F"],
        "blocker_p_values": np.array([np.nan, np.nan, np.nan, np.nan, 0.8, 0.8]),
        "distances_to_blocker": np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 1.0]),
        "signal_p_values": np.array([np.nan, np.nan, np.nan, np.nan, 0.03, 0.03]),
        "distances_to_signal": np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 1.0]),
    }

    stable_record = SiblingPairRecord(
        parent="A",
        left="C",
        right="D",
        stat=2.0,
        degrees_of_freedom=2.0,
        p_value=0.5,
        branch_length_sum=0.1,
        n_parent=10,
        is_null_like=True,
        edge_weight=0.4,
        structural_dimension=2.0,
    )
    blocked_record = SiblingPairRecord(
        parent="C",
        left="E",
        right="F",
        stat=2.0,
        degrees_of_freedom=2.0,
        p_value=0.5,
        branch_length_sum=0.1,
        n_parent=8,
        is_null_like=True,
        is_gate2_blocked=True,
        edge_weight=1.0,
        structural_dimension=2.0,
    )

    records = enrich_blocked_weights([stable_record, blocked_record], tree, annotations_df)
    enriched_blocked = records[1]

    assert 0.0 <= enriched_blocked.edge_weight < 1.0
    assert enriched_blocked.nearby_stable_support is not None
    assert enriched_blocked.ancestor_support is not None
    assert enriched_blocked.blend_lambda is not None
