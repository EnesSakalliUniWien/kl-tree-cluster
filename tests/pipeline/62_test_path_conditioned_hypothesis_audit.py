"""Tests for path-conditioned hypothesis audit helpers."""

from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.path_conditioned_hypothesis_audit import (
    BRANCH_METHOD,
    CURRENT_METHOD,
    _delta_connection_class,
    _outcome_class,
    aggregate_tuple_burden,
    build_branch_current_table,
    build_method_case_table,
    summarize_burden_outcomes,
)


def _tuple_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": BRANCH_METHOD,
                "case_id": "case_loss",
                "actual_visited": True,
                "actual_decision": "pass_through",
                "ancestor_pass_through_count": 2,
                "ancestor_sibling_closed_edge_open_count": 2,
                "incoming_branch_length": 0.01,
                "depth": 4,
                "descendant_truth_coherent_live_split_count": 0,
                "truth_audit_label": "unresolved_tuple",
                "sibling_gate_open": False,
                "edge_traversal_action": "continue",
            },
            {
                "method": BRANCH_METHOD,
                "case_id": "case_loss",
                "actual_visited": True,
                "actual_decision": "boundary",
                "ancestor_pass_through_count": 2,
                "ancestor_sibling_closed_edge_open_count": 2,
                "incoming_branch_length": 0.02,
                "depth": 5,
                "descendant_truth_coherent_live_split_count": 0,
                "truth_audit_label": "pure_boundary",
                "sibling_gate_open": False,
                "edge_traversal_action": "continue",
            },
            {
                "method": CURRENT_METHOD,
                "case_id": "case_loss",
                "actual_visited": True,
                "actual_decision": "boundary",
                "ancestor_pass_through_count": 0,
                "ancestor_sibling_closed_edge_open_count": 0,
                "incoming_branch_length": 0.5,
                "depth": 1,
                "descendant_truth_coherent_live_split_count": 0,
                "truth_audit_label": "pure_boundary",
                "sibling_gate_open": False,
                "edge_traversal_action": "continue",
            },
            {
                "method": BRANCH_METHOD,
                "case_id": "case_gain",
                "actual_visited": True,
                "actual_decision": "boundary",
                "ancestor_pass_through_count": 0,
                "ancestor_sibling_closed_edge_open_count": 0,
                "incoming_branch_length": 0.8,
                "depth": 1,
                "descendant_truth_coherent_live_split_count": 0,
                "truth_audit_label": "pure_boundary",
                "sibling_gate_open": False,
                "edge_traversal_action": "stop",
            },
            {
                "method": CURRENT_METHOD,
                "case_id": "case_gain",
                "actual_visited": True,
                "actual_decision": "pass_through",
                "ancestor_pass_through_count": 0,
                "ancestor_sibling_closed_edge_open_count": 0,
                "incoming_branch_length": 0.7,
                "depth": 1,
                "descendant_truth_coherent_live_split_count": 0,
                "truth_audit_label": "unresolved_tuple",
                "sibling_gate_open": False,
                "edge_traversal_action": "continue",
            },
        ]
    )


def _case_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case_loss",
                "method": CURRENT_METHOD,
                "status": "ok",
                "ari": 1.0,
                "found_clusters": 4,
                "true_clusters": 4,
                "path_conditioned_pass_through_rows": 0,
                "stacked_pass_through_rows": 0,
                "mixed_boundary_rows": 0,
                "pure_boundary_rows": 1,
                "edge_open_closed_sibling_boundary_rows": 1,
                "median_pass_through_ancestor_pass_through_count": 0,
                "median_pass_through_incoming_branch_length": pd.NA,
                "pass_through_with_truth_coherent_descendant_split_rows": 0,
            },
            {
                "case_id": "case_loss",
                "method": BRANCH_METHOD,
                "status": "ok",
                "ari": 0.5,
                "found_clusters": 5,
                "true_clusters": 4,
                "path_conditioned_pass_through_rows": 1,
                "stacked_pass_through_rows": 1,
                "mixed_boundary_rows": 0,
                "pure_boundary_rows": 1,
                "edge_open_closed_sibling_boundary_rows": 1,
                "median_pass_through_ancestor_pass_through_count": 2,
                "median_pass_through_incoming_branch_length": 0.01,
                "pass_through_with_truth_coherent_descendant_split_rows": 0,
            },
            {
                "case_id": "case_gain",
                "method": CURRENT_METHOD,
                "status": "ok",
                "ari": 0.7,
                "found_clusters": 6,
                "true_clusters": 4,
                "path_conditioned_pass_through_rows": 1,
                "stacked_pass_through_rows": 0,
                "mixed_boundary_rows": 0,
                "pure_boundary_rows": 0,
                "edge_open_closed_sibling_boundary_rows": 0,
                "median_pass_through_ancestor_pass_through_count": 0,
                "median_pass_through_incoming_branch_length": 0.7,
                "pass_through_with_truth_coherent_descendant_split_rows": 0,
            },
            {
                "case_id": "case_gain",
                "method": BRANCH_METHOD,
                "status": "ok",
                "ari": 0.9,
                "found_clusters": 4,
                "true_clusters": 4,
                "path_conditioned_pass_through_rows": 0,
                "stacked_pass_through_rows": 0,
                "mixed_boundary_rows": 0,
                "pure_boundary_rows": 1,
                "edge_open_closed_sibling_boundary_rows": 0,
                "median_pass_through_ancestor_pass_through_count": pd.NA,
                "median_pass_through_incoming_branch_length": pd.NA,
                "pass_through_with_truth_coherent_descendant_split_rows": 0,
            },
        ]
    )


def _pairwise() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case_loss",
                "source_family": "synthetic",
                "feature_representation": "synthetic",
                CURRENT_METHOD: 1.0,
                BRANCH_METHOD: 0.5,
                f"{CURRENT_METHOD}_status": "ok",
                f"{BRANCH_METHOD}_status": "ok",
                f"{CURRENT_METHOD}_found_clusters": 4,
                f"{BRANCH_METHOD}_found_clusters": 5,
                "branch_minus_current_ari": -0.5,
                "branch_vs_current_relation": "branch_lower_ari",
            },
            {
                "case_id": "case_gain",
                "source_family": "synthetic",
                "feature_representation": "synthetic",
                CURRENT_METHOD: 0.7,
                BRANCH_METHOD: 0.9,
                f"{CURRENT_METHOD}_status": "ok",
                f"{BRANCH_METHOD}_status": "ok",
                f"{CURRENT_METHOD}_found_clusters": 6,
                f"{BRANCH_METHOD}_found_clusters": 4,
                "branch_minus_current_ari": 0.2,
                "branch_vs_current_relation": "branch_higher_ari",
            },
        ]
    )


def test_aggregate_tuple_burden_counts_stacked_pass_throughs() -> None:
    burden = aggregate_tuple_burden(_tuple_rows())
    branch_loss = burden[
        (burden["method"] == BRANCH_METHOD) & (burden["case_id"] == "case_loss")
    ].iloc[0]

    assert branch_loss["pass_through_rows_from_tuples"] == 1
    assert branch_loss["stacked_pass_through_rows_from_tuples"] == 1
    assert branch_loss["max_ancestor_pass_through_count"] == 2
    assert branch_loss["min_pass_through_incoming_branch_length"] == 0.01
    assert branch_loss["edge_open_closed_sibling_boundary_rows_from_tuples"] == 1


def test_outcome_class_keeps_support_and_path_burden_separate() -> None:
    assert (
        _outcome_class(
            current_status="ok",
            branch_status="skip",
            branch_minus_current_ari=pd.NA,
            branch_stacked_pass_through_rows=0,
            branch_pass_through_rows=0,
            branch_mixed_boundary_rows=0,
            epsilon=1e-9,
        )
        == "current_ok_branch_skip"
    )
    assert (
        _outcome_class(
            current_status="ok",
            branch_status="ok",
            branch_minus_current_ari=-0.2,
            branch_stacked_pass_through_rows=2,
            branch_pass_through_rows=2,
            branch_mixed_boundary_rows=0,
            epsilon=1e-9,
        )
        == "branch_loss_with_stacked_pass_through"
    )


def test_delta_connection_class_distinguishes_direction_and_missing_values() -> None:
    assert (
        _delta_connection_class(
            promotion_delta=0.2,
            path_delta=0.2,
            epsilon=1e-9,
        )
        == "delta_exact_match"
    )
    assert (
        _delta_connection_class(
            promotion_delta=0.2,
            path_delta=0.1,
            epsilon=1e-9,
        )
        == "delta_direction_consistent"
    )
    assert (
        _delta_connection_class(
            promotion_delta=0.2,
            path_delta=-0.1,
            epsilon=1e-9,
        )
        == "delta_direction_mismatch"
    )
    assert (
        _delta_connection_class(
            promotion_delta=0.2,
            path_delta=pd.NA,
            epsilon=1e-9,
        )
        == "delta_missing_in_one_source"
    )


def test_branch_current_table_classifies_gain_and_loss_rows() -> None:
    burden = aggregate_tuple_burden(_tuple_rows())
    method_case = build_method_case_table(
        case_summary=_case_summary(),
        tuple_burden=burden,
        pairwise=_pairwise(),
    )
    table = build_branch_current_table(
        method_case=method_case,
        pairwise=_pairwise(),
        epsilon=1e-9,
    )

    labels = dict(zip(table["case_id"], table["path_hypothesis_class"], strict=False))
    assert labels["case_loss"] == "branch_loss_with_stacked_pass_through"
    assert labels["case_gain"] == "branch_gain_without_pass_through_burden"

    summary = summarize_burden_outcomes(table)
    assert set(summary["path_hypothesis_class"]) == {
        "branch_gain_without_pass_through_burden",
        "branch_loss_with_stacked_pass_through",
    }
