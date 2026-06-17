from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.retained_pass_through_topology_likelihood_panel import (
    RetainedPassThroughTopologyLikelihoodConfig,
    build_retained_pass_through_topology_likelihood_rows,
    build_retained_pass_through_topology_match_rows,
    build_traversal_network_context_rows,
    run_retained_pass_through_topology_likelihood_panel,
    summarize_retained_pass_through_topology_likelihood,
)


def _candidate_rows() -> pd.DataFrame:
    base = {
        "left_traversal_state": "pass_through",
        "right_traversal_state": "boundary",
        "left_child_parent_edge_open": True,
        "left_sibling_open": False,
        "left_sibling_p_value": 0.003,
        "left_depth": 5,
        "left_n_descendant_leaves": 90,
        "left_method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
        "left_neighborhood_evidence_family": "traversal_only",
        "right_neighborhood_evidence_family": "traversal_only",
        "left_descendant_accepted_split_count": 1,
        "right_descendant_accepted_split_count": 0,
        "traversal_decision_agrees": False,
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "case_id": "signal_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "signal_pass",
                "left_balance_product": float("nan"),
                "left_outgoing_edge_norm_balance": float("nan"),
            },
            {
                **base,
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_pass",
                "left_depth": 4,
                "left_n_descendant_leaves": 120,
                "left_neighborhood_evidence_family": "old_and_current",
                "right_neighborhood_evidence_family": "old_and_current",
                "left_balance_product": 0.18,
                "left_outgoing_edge_norm_balance": 0.7,
            },
        ]
    )


def _selected_neighborhood_rows() -> pd.DataFrame:
    method_id = "fixed_coordinate_conditional_topology_diagnostic_v1"
    base = {
        "method_id": method_id,
        "replicate": 0,
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "case_id": "signal_case",
                "data_role": "signal",
                "node_id": "signal_root",
                "parent_id": "",
                "traversal_state": "boundary",
                "decision_class": "stable_boundary",
            },
            {
                **base,
                "case_id": "signal_case",
                "data_role": "signal",
                "node_id": "signal_pass",
                "parent_id": "signal_root",
                "traversal_state": "pass_through",
                "decision_class": "unstable_passthrough_zone",
            },
            {
                **base,
                "case_id": "signal_case",
                "data_role": "signal",
                "node_id": "signal_split",
                "parent_id": "signal_pass",
                "traversal_state": "split",
                "decision_class": "accepted_internal_split",
            },
            {
                **base,
                "case_id": "null_case",
                "data_role": "selected_null",
                "node_id": "null_root",
                "parent_id": "",
                "traversal_state": "boundary",
                "decision_class": "stable_boundary",
            },
            {
                **base,
                "case_id": "null_case",
                "data_role": "selected_null",
                "node_id": "null_pass",
                "parent_id": "null_root",
                "traversal_state": "pass_through",
                "decision_class": "unstable_passthrough_zone",
            },
            {
                **base,
                "case_id": "null_case",
                "data_role": "selected_null",
                "node_id": "null_split",
                "parent_id": "null_pass",
                "traversal_state": "split",
                "decision_class": "accepted_internal_split",
            },
        ]
    )


def test_retained_pass_through_rows_define_signal_and_controls() -> None:
    rows = build_retained_pass_through_topology_likelihood_rows(_candidate_rows())

    assert set(rows["likelihood_class"]) == {
        "signal_retained_pass_through",
        "selected_null_pass_through_control",
    }
    signal = rows.loc[rows["likelihood_class"].eq("signal_retained_pass_through")].iloc[
        0
    ]
    control = rows.loc[
        rows["likelihood_class"].eq("selected_null_pass_through_control")
    ].iloc[0]
    assert bool(signal["traversal_only_pair"]) is True
    assert int(signal["finite_topology_feature_count"]) == 0
    assert int(control["finite_topology_feature_count"]) == 2


def test_traversal_network_context_records_directed_neighborhood() -> None:
    context = build_traversal_network_context_rows(_selected_neighborhood_rows())
    rows = build_retained_pass_through_topology_likelihood_rows(
        _candidate_rows(),
        context,
    )
    matches = build_retained_pass_through_topology_match_rows(
        rows,
        max_depth_delta=4.0,
        max_log_descendant_leaves_delta=2.0,
        max_pass_through_context_delta=4.0,
        max_downstream_accepted_split_delta=6.0,
    )

    signal = rows.loc[rows["likelihood_class"].eq("signal_retained_pass_through")].iloc[
        0
    ]
    assert signal["distance_to_pass_through_context"] == 0.0
    assert signal["distance_to_downstream_accepted_split"] == 1.0
    assert signal["traversal_network_context_status"] == "network_context_observed"
    assert matches["abs_pass_through_context_delta"].iloc[0] == 0.0
    assert matches["abs_downstream_accepted_split_delta"].iloc[0] == 0.0
    assert matches["traversal_network_match_status"].iloc[0] == (
        "matched_traversal_network_context"
    )
    assert matches["tree_network_distance_status"].iloc[0] == (
        "cross_case_network_distance_unavailable"
    )


def test_retained_pass_through_matching_and_identifiability_fail_closed() -> None:
    rows = build_retained_pass_through_topology_likelihood_rows(_candidate_rows())
    matches = build_retained_pass_through_topology_match_rows(
        rows,
        max_depth_delta=4.0,
        max_log_descendant_leaves_delta=2.0,
        max_pass_through_context_delta=4.0,
        max_downstream_accepted_split_delta=6.0,
    )
    summary = summarize_retained_pass_through_topology_likelihood(
        rows,
        matches,
        min_matched_signal_count=1,
    )

    assert matches["match_status"].iloc[0] == "matched_selected_null_control"
    assert summary["signal_row_count"].iloc[0] == 1
    assert summary["selected_null_control_count"].iloc[0] == 1
    assert summary["matched_control_finite_topology_count"].iloc[0] == 1
    assert summary["likelihood_identifiability_status"].iloc[0] == (
        "signal_topology_likelihood_not_identifiable"
    )
    assert summary["production_action"].iloc[0] == (
        "fail_closed_until_likelihood_identifiable"
    )


def test_retained_pass_through_panel_writes_outputs(tmp_path) -> None:
    candidate_path = tmp_path / "candidate_rows.csv"
    selected_neighborhood_path = tmp_path / "selected_neighborhood_rows.csv"
    _candidate_rows().to_csv(candidate_path, index=False)
    _selected_neighborhood_rows().to_csv(selected_neighborhood_path, index=False)

    outputs = run_retained_pass_through_topology_likelihood_panel(
        RetainedPassThroughTopologyLikelihoodConfig(
            candidate_rows_path=candidate_path,
            output_dir=tmp_path / "out",
            selected_neighborhood_rows_path=selected_neighborhood_path,
            min_matched_signal_count=1,
        )
    )

    for path in outputs.values():
        assert path.exists(), path

    summary = pd.read_csv(outputs["summary"])
    assert summary["likelihood_identifiability_status"].iloc[0] == (
        "signal_topology_likelihood_not_identifiable"
    )
