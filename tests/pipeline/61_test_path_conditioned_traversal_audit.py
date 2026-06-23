"""Tests for path-conditioned traversal audit helpers."""

from __future__ import annotations

import json
import math

import pandas as pd
from benchmarks.diagnostics.calibration.branch_length_traversal_audit import (
    BranchLengthTraversalAuditConfig,
)
from benchmarks.diagnostics.calibration.path_conditioned_traversal_audit import (
    MIXED_BOUNDARY,
    PURE_BOUNDARY,
    TRUTH_COHERENT_SPLIT_PATH,
    PathConditionedTraversalAuditConfig,
    _path_nodes,
    add_descendant_covariates,
    add_path_covariates,
    add_truth_covariates,
    annotate_path_conditioned_tuples,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)


def _tuple_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "synthetic_case",
                "method": "tbs_internal_filter_branch_length_v1",
                "node_id": "root",
                "left_child": "A",
                "right_child": "B",
                "path": "root",
                "actual_visited": True,
                "actual_decision": "split",
                "edge_traversal_action": "continue",
                "sibling_gate_open": True,
                "left_edge_open": True,
                "right_edge_open": True,
                "left_branch_length": 1.0,
                "right_branch_length": 2.0,
                "depth": 0,
            },
            {
                "case_id": "synthetic_case",
                "method": "tbs_internal_filter_branch_length_v1",
                "node_id": "A",
                "left_child": "C",
                "right_child": "D",
                "path": "root > A",
                "actual_visited": True,
                "actual_decision": "pass_through",
                "edge_traversal_action": "continue",
                "sibling_gate_open": False,
                "left_edge_open": True,
                "right_edge_open": True,
                "left_branch_length": 0.25,
                "right_branch_length": 0.75,
                "depth": 1,
            },
            {
                "case_id": "synthetic_case",
                "method": "tbs_internal_filter_branch_length_v1",
                "node_id": "B",
                "left_child": "B1",
                "right_child": "B2",
                "path": "root > B",
                "actual_visited": True,
                "actual_decision": "boundary",
                "edge_traversal_action": "stop",
                "sibling_gate_open": False,
                "left_edge_open": False,
                "right_edge_open": False,
                "left_branch_length": 0.5,
                "right_branch_length": 0.5,
                "depth": 1,
            },
            {
                "case_id": "synthetic_case",
                "method": "tbs_internal_filter_branch_length_v1",
                "node_id": "C",
                "left_child": "C1",
                "right_child": "C2",
                "path": "root > A > C",
                "actual_visited": True,
                "actual_decision": "boundary",
                "edge_traversal_action": "stop",
                "sibling_gate_open": False,
                "left_edge_open": False,
                "right_edge_open": False,
                "left_branch_length": 0.1,
                "right_branch_length": 0.1,
                "depth": 2,
            },
            {
                "case_id": "synthetic_case",
                "method": "tbs_internal_filter_branch_length_v1",
                "node_id": "D",
                "left_child": "D1",
                "right_child": "D2",
                "path": "root > A > D",
                "actual_visited": True,
                "actual_decision": "split",
                "edge_traversal_action": "continue",
                "sibling_gate_open": True,
                "left_edge_open": True,
                "right_edge_open": True,
                "left_branch_length": 0.2,
                "right_branch_length": 0.2,
                "depth": 2,
            },
        ]
    )


def _descendant_leaf_sets() -> dict[str, set[str]]:
    return {
        "root": {"x1", "x2", "x3", "x4", "x5", "x6"},
        "A": {"x1", "x2", "x3", "x4"},
        "B": {"x5", "x6"},
        "C": {"x1", "x2"},
        "D": {"x3", "x4"},
        "B1": {"x5"},
        "B2": {"x6"},
        "C1": {"x1"},
        "C2": {"x2"},
        "D1": {"x3"},
        "D2": {"x4"},
    }


def _truth_by_leaf() -> dict[str, str]:
    return {
        "x1": "left",
        "x2": "left",
        "x3": "right",
        "x4": "alt",
        "x5": "outside",
        "x6": "outside",
    }


def test_path_nodes_normalizes_string_and_sequence_paths() -> None:
    assert _path_nodes("root > A > C") == ("root", "A", "C")
    assert _path_nodes(("root", " A ", "C")) == ("root", "A", "C")
    assert _path_nodes(None) == ()


def test_traversal_audit_configs_use_benchmark_alpha_contract(tmp_path) -> None:
    base_config = BranchLengthTraversalAuditConfig(output_dir=tmp_path / "base")
    path_config = PathConditionedTraversalAuditConfig(output_dir=tmp_path / "path")

    assert base_config.significance_level == DEFAULT_SIBLING_ALPHA
    assert base_config.edge_alpha == DEFAULT_EDGE_ALPHA
    assert path_config.significance_level == DEFAULT_SIBLING_ALPHA
    assert path_config.edge_alpha == DEFAULT_EDGE_ALPHA


def test_path_covariates_reconstruct_incoming_edges_and_ancestors() -> None:
    rows = add_path_covariates(_tuple_rows())
    c_row = rows.loc[rows["node_id"] == "C"].iloc[0]

    assert c_row["incoming_parent_id"] == "A"
    assert c_row["incoming_child_side"] == "left"
    assert c_row["incoming_branch_length"] == 0.25
    assert bool(c_row["incoming_edge_open"]) is True
    assert bool(c_row["incoming_parent_sibling_gate_open"]) is False
    assert c_row["incoming_parent_decision"] == "pass_through"
    assert c_row["ancestor_split_count"] == 1
    assert c_row["ancestor_pass_through_count"] == 1
    assert c_row["ancestor_sibling_closed_edge_open_count"] == 1
    assert bool(c_row["ancestor_chain_complete"]) is True
    assert c_row["missing_ancestor_count"] == 0
    assert c_row["min_incoming_branch_length_on_path"] == 0.25
    assert c_row["median_incoming_branch_length_on_path"] == 0.625


def test_descendant_covariates_count_prefix_reachable_rows() -> None:
    rows = add_descendant_covariates(_tuple_rows())
    root_row = rows.loc[rows["node_id"] == "root"].iloc[0]
    a_row = rows.loc[rows["node_id"] == "A"].iloc[0]

    assert root_row["descendant_edge_open_continue_count"] == 2
    assert root_row["descendant_sibling_open_count"] == 1
    assert root_row["descendant_live_split_count"] == 1
    assert root_row["descendant_live_boundary_count"] == 2
    assert a_row["descendant_edge_open_continue_count"] == 1
    assert a_row["descendant_live_split_count"] == 1
    assert a_row["descendant_live_boundary_count"] == 1


def test_truth_covariates_classify_boundaries_and_coherent_split_paths() -> None:
    rows = add_truth_covariates(
        _tuple_rows(),
        descendant_leaf_sets=_descendant_leaf_sets(),
        truth_by_leaf=_truth_by_leaf(),
    )

    root_row = rows.loc[rows["node_id"] == "root"].iloc[0]
    c_row = rows.loc[rows["node_id"] == "C"].iloc[0]
    d_row = rows.loc[rows["node_id"] == "D"].iloc[0]

    assert root_row["truth_audit_label"] == TRUTH_COHERENT_SPLIT_PATH
    assert bool(root_row["children_majority_truth_labels_differ"]) is True
    assert c_row["truth_audit_label"] == PURE_BOUNDARY
    assert c_row["tuple_descendant_purity"] == 1.0
    assert d_row["truth_audit_label"] == TRUTH_COHERENT_SPLIT_PATH
    assert json.loads(d_row["tuple_truth_counts_json"]) == {"alt": 1, "right": 1}


def test_annotate_path_conditioned_tuples_combines_audit_columns() -> None:
    rows = annotate_path_conditioned_tuples(
        _tuple_rows(),
        descendant_leaf_sets=_descendant_leaf_sets(),
        truth_by_leaf=_truth_by_leaf(),
    )
    a_row = rows.loc[rows["node_id"] == "A"].iloc[0]
    b_row = rows.loc[rows["node_id"] == "B"].iloc[0]

    assert a_row["schema_version"] == "path_conditioned_traversal_audit/v1"
    assert a_row["ancestor_split_count"] == 1
    assert a_row["descendant_truth_coherent_live_split_count"] == 1
    assert b_row["truth_audit_label"] == PURE_BOUNDARY
    assert math.isnan(b_row["min_incoming_branch_length_on_path"]) is False


def test_truth_covariates_mark_mixed_boundaries() -> None:
    rows = _tuple_rows()
    rows.loc[rows["node_id"] == "D", "actual_decision"] = "boundary"
    audited = add_truth_covariates(
        rows,
        descendant_leaf_sets=_descendant_leaf_sets(),
        truth_by_leaf=_truth_by_leaf(),
    )
    d_row = audited.loc[audited["node_id"] == "D"].iloc[0]

    assert d_row["truth_audit_label"] == MIXED_BOUNDARY
    assert d_row["tuple_truth_label_count"] == 2
