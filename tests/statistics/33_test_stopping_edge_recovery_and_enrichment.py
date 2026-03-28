from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing import (
    ChildParentEdgeTreeBHResult,
    TreeBHSiblingGroupOutcome,
    recover_signal_neighbors,
    recover_stopping_edge_info,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.stopping_edge_recovery.serialization import (
    STOPPING_EDGE_INFO_ATTR_KEY,
    build_stopping_edge_attrs,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.stopping_edge_recovery.models import (
    SignalNeighborInfo,
    StoppingEdgeInfo,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_null_prior_interpolation import (
    interpolate_sibling_null_priors,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_null_prior_interpolation.edge_metadata import (
    extract_edge_metadata,
    extract_stopping_edge_info,
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

    assert signal_map["E"].sig_node == "A"
    assert signal_map["E"].sig_p_value == 0.01
    assert signal_map["E"].distance_to_sig == 2.0


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

    assert signal_map["E"].sig_node == "C"
    assert signal_map["E"].sig_p_value == 0.03
    assert signal_map["E"].distance_to_sig == 2.0


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
    annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = build_stopping_edge_attrs(
        child_node_ids=["A", "B", "C", "D", "E", "F"],
        stopping_edge_info_by_child={
            "E": StoppingEdgeInfo(
                stopping_child_node="C",
                stopping_edge_p_value=0.8,
                distance_to_stopping_edge=1.0,
                generations_above=1,
            ),
            "F": StoppingEdgeInfo(
                stopping_child_node="C",
                stopping_edge_p_value=0.8,
                distance_to_stopping_edge=1.0,
                generations_above=1,
            ),
        },
        signal_neighbor_info_by_child={
            "E": SignalNeighborInfo(sig_node="D", sig_p_value=0.03, distance_to_sig=1.0),
            "F": SignalNeighborInfo(sig_node="D", sig_p_value=0.03, distance_to_sig=1.0),
        },
    )

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
        sibling_null_prior_from_edge_pvalue=0.4,
        sibling_scale=2.0,
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
        sibling_null_prior_from_edge_pvalue=1.0,
        sibling_scale=2.0,
    )

    records = interpolate_sibling_null_priors([stable_record, blocked_record], tree, annotations_df)
    enriched_blocked = records[1]

    assert 0.0 <= enriched_blocked.sibling_null_prior_from_edge_pvalue < 1.0
    assert enriched_blocked.smoothed_sibling_null_prior is not None
    assert enriched_blocked.ancestor_support is not None
    assert enriched_blocked.neighborhood_reliance is not None


def test_extract_stopping_edge_info_preserves_serialized_values() -> None:
    annotations_df = pd.DataFrame(index=["A", "B", "C"])
    annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = build_stopping_edge_attrs(
        child_node_ids=["A", "B", "C"],
        stopping_edge_info_by_child={
            "B": StoppingEdgeInfo(
                stopping_child_node="A",
                stopping_edge_p_value=0.25,
                distance_to_stopping_edge=2.0,
                generations_above=1,
            )
        },
        signal_neighbor_info_by_child={
            "B": SignalNeighborInfo(sig_node="C", sig_p_value=0.1, distance_to_sig=1.0)
        },
    )

    stopping_edge_info = extract_stopping_edge_info(annotations_df)

    assert stopping_edge_info is not None
    assert list(stopping_edge_info) == ["B"]
    assert stopping_edge_info["B"].stopping_edge_p_value == pytest.approx(0.25)
    assert stopping_edge_info["B"].distance_to_stopping_edge == pytest.approx(2.0)


def test_extract_edge_metadata_requires_child_parent_edge_columns() -> None:
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [True, False],
        },
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Missing required child-parent edge metadata columns",
    ):
        extract_edge_metadata(annotations_df)


def test_interpolate_sibling_null_priors_rejects_malformed_stopping_edge_attrs() -> None:
    tree = _build_tree()
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Tested": [True, True, True, True, False, False],
            "Child_Parent_Divergence_Significant": [True, True, False, True, False, False],
            "Child_Parent_Divergence_P_Value_BH": [0.01, 0.02, 0.8, 0.03, np.nan, np.nan],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )
    annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = {
        "child_node_ids": ["A", "B", "C", "D", "E", "F"],
        "stopping_edge_p_values": np.array([np.nan, np.nan, np.nan, np.nan, 0.8]),
        "distances_to_stopping_edge": np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 1.0]),
        "signal_p_values": np.array([np.nan, np.nan, np.nan, np.nan, 0.03, 0.03]),
        "distances_to_signal": np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 1.0]),
    }

    record = SiblingPairRecord(
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
        sibling_null_prior_from_edge_pvalue=1.0,
        sibling_scale=2.0,
    )

    with pytest.raises(ValueError, match="Malformed _stopping_edge_info attrs payload"):
        interpolate_sibling_null_priors([record], tree, annotations_df)


def test_interpolate_sibling_null_priors_handles_partial_child_stopping_edge_coverage() -> None:
    tree = _build_tree()
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Tested": [True, True, True, True, False, False],
            "Child_Parent_Divergence_Ancestor_Blocked": [False, False, False, False, True, True],
            "Child_Parent_Divergence_Significant": [True, True, False, True, False, False],
            "Child_Parent_Divergence_P_Value_BH": [0.01, 0.02, 0.8, 0.03, 0.61, 0.91],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )
    annotations_df.attrs[STOPPING_EDGE_INFO_ATTR_KEY] = build_stopping_edge_attrs(
        child_node_ids=["A", "B", "C", "D", "E", "F"],
        stopping_edge_info_by_child={
            "E": StoppingEdgeInfo(
                stopping_child_node="C",
                stopping_edge_p_value=0.8,
                distance_to_stopping_edge=1.0,
                generations_above=1,
            ),
        },
        signal_neighbor_info_by_child={
            "E": SignalNeighborInfo(sig_node="D", sig_p_value=0.03, distance_to_sig=1.0),
        },
    )

    record = SiblingPairRecord(
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
        sibling_null_prior_from_edge_pvalue=0.61,
        sibling_scale=2.0,
    )

    [enriched_record] = interpolate_sibling_null_priors([record], tree, annotations_df)

    assert 0.0 <= enriched_record.sibling_null_prior_from_edge_pvalue <= 1.0
    assert enriched_record.smoothed_sibling_null_prior is not None
    assert enriched_record.ancestor_support is not None
    assert enriched_record.neighborhood_reliance is not None
    # The pair-level prior remains finite even when one child has no stopping-edge context.
    assert enriched_record.sibling_null_prior_from_edge_pvalue <= 0.61
