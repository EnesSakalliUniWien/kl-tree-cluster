from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_distribution_panel import (
    SelectedNeighborhoodDistributionPanelConfig,
    build_selected_neighborhood_candidate_method_contrast_rows,
    build_selected_neighborhood_distribution_rows,
    run_selected_neighborhood_distribution_panel,
    summarize_selected_neighborhood_candidate_ambiguity,
    summarize_selected_neighborhood_candidate_law_targets,
    summarize_selected_neighborhood_candidate_local_features,
    summarize_selected_neighborhood_candidate_method_contrast,
    summarize_selected_neighborhood_case_coverage,
    summarize_selected_neighborhood_coverage,
    summarize_selected_neighborhood_distributions,
    summarize_selected_neighborhood_joint_distributions,
    summarize_selected_neighborhood_method_contract,
    summarize_selected_neighborhood_method_contrast,
    summarize_selected_neighborhood_method_readiness,
    summarize_selected_neighborhood_profile_config_contract,
    summarize_selected_neighborhood_retention_evidence,
    summarize_selected_neighborhood_retention_gaps,
    summarize_selected_neighborhood_stop_rule_comparison,
)


def _node_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
                "replicate": 0,
                "node_id": "split",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "split",
                "decision_class": "accepted_internal_split",
                "n_children": 2,
                "n_descendant_leaves": 12,
                "child_parent_edge_open": True,
                "sibling_open": True,
                "sibling_p_value": 0.001,
                "sibling_projection_dimension": 7,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": False,
            },
            {
                "case_id": "case",
                "data_role": "selected_null",
                "method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
                "replicate": 0,
                "node_id": "edge_closed",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 2,
                "n_descendant_leaves": 10,
                "child_parent_edge_open": False,
                "sibling_open": True,
                "sibling_p_value": 0.001,
                "sibling_projection_dimension": 9,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": False,
            },
            {
                "case_id": "case",
                "data_role": "selected_null",
                "method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
                "replicate": 1,
                "node_id": "pass",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "pass_through",
                "decision_class": "unstable_passthrough_zone",
                "n_children": 2,
                "n_descendant_leaves": 9,
                "child_parent_edge_open": True,
                "sibling_open": False,
                "sibling_p_value": 0.8,
                "sibling_projection_dimension": 11,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": True,
            },
            {
                "case_id": "case",
                "data_role": "selected_null",
                "method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
                "replicate": 2,
                "node_id": "guarded",
                "parent_id": "",
                "depth": 0,
                "traversal_decision": "boundary",
                "decision_class": "selected_root_blocked",
                "n_children": 2,
                "n_descendant_leaves": 20,
                "child_parent_edge_open": True,
                "sibling_open": False,
                "sibling_p_value": 0.5,
                "sibling_projection_dimension": 13,
                "root_stability_guard_blocked": True,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "root",
                "topology_pass_through_candidate": False,
            },
        ]
    )


def _paired_node_rows() -> pd.DataFrame:
    rows = _node_rows()
    other = rows.copy()
    other["method_id"] = "fixed_coordinate_global_passthrough_refined_v1"
    pass_mask = other["node_id"].eq("pass")
    other.loc[pass_mask, "traversal_decision"] = "boundary"
    other.loc[pass_mask, "decision_class"] = "stable_boundary"
    signal_mask = other["node_id"].eq("split")
    other.loc[signal_mask, "traversal_decision"] = "boundary"
    other.loc[signal_mask, "decision_class"] = "stable_boundary"
    return pd.concat([rows, other], ignore_index=True)


def _topology_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "split",
                "guard_truth_role": "truth_recovery",
                "topology_support_role": "",
                "topology_signal_role": "signal",
                "incoming_branch_balance": 0.48,
                "outgoing_balance": 0.49,
                "outgoing_edge_norm_balance": 0.97,
                "outgoing_fragment_risk_proxy_score": 0.55,
                "balance_product": 0.2352,
                "neighborhood_scale": 7.0,
                "distance_to_stopping_edge": 2.0,
            },
            {
                "case_id": "case",
                "data_role": "selected_null",
                "replicate": 1,
                "node_id": "pass",
                "guard_truth_role": "null_like",
                "topology_support_role": "strict_null",
                "topology_signal_role": "",
                "incoming_branch_balance": 0.38,
                "outgoing_balance": 0.28,
                "outgoing_edge_norm_balance": 0.51,
                "outgoing_fragment_risk_proxy_score": 1.20,
                "balance_product": 0.1064,
                "neighborhood_scale": 11.0,
                "distance_to_stopping_edge": 1.0,
            },
        ]
    )


def _law_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "split",
                "topology_neighborhood_tau_b": 1.5,
                "topology_neighborhood_tau_t": 2.5,
                "topology_neighborhood_tau_s": 0.5,
                "topology_neighborhood_h_k": 3.0,
                "topology_neighborhood_support_count": 4,
                "topology_neighborhood_selected_nonnull_excluded_count": 1,
                "topology_neighborhood_support_status": (
                    "topology_neighborhood_support_observed_diagnostic_only"
                ),
                "guarded_recovery_status": ("guarded_internal_recovery_candidate_diagnostic_only"),
                "recover_internal_split": True,
            },
            {
                "case_id": "case",
                "data_role": "selected_null",
                "replicate": 1,
                "node_id": "pass",
                "topology_neighborhood_tau_b": 0.5,
                "topology_neighborhood_tau_t": 0.7,
                "topology_neighborhood_tau_s": 0.2,
                "topology_neighborhood_h_k": 1.0,
                "topology_neighborhood_support_count": 2,
                "topology_neighborhood_selected_nonnull_excluded_count": 0,
                "topology_neighborhood_support_status": (
                    "topology_neighborhood_support_observed_diagnostic_only"
                ),
                "guarded_recovery_status": "root_or_null_guard_blocked",
                "recover_internal_split": False,
            },
        ]
    )


def test_selected_neighborhood_rows_classify_traversal_stop_reasons() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}

    assert by_node["split"]["traversal_state"] == "split"
    assert by_node["split"]["traversal_stop_reason"] == "accepted_split"
    assert by_node["split"]["neighborhood_evidence_family"] == "old_and_current"
    assert by_node["split"]["recover_internal_split"] is True
    assert by_node["edge_closed"]["traversal_stop_reason"] == "edge_closed"
    assert by_node["pass"]["traversal_state"] == "pass_through"
    assert by_node["pass"]["traversal_stop_reason"] == ("sibling_closed_descendant_split_available")
    assert by_node["guarded"]["traversal_stop_reason"] == "explicit_guard_blocked"


def test_selected_neighborhood_rows_preserve_parent_branch_lengths() -> None:
    node_decisions = _node_rows()
    node_decisions.loc[node_decisions["node_id"].eq("split"), "branch_length_to_parent"] = 0.25

    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=node_decisions,
    )
    split = rows.loc[rows["node_id"].eq("split")].iloc[0]

    assert split["branch_length_to_parent"] == 0.25


def test_selected_neighborhood_distribution_summaries_are_robust() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    summary = summarize_selected_neighborhood_distributions(rows)
    joint = summarize_selected_neighborhood_joint_distributions(rows)

    split_balance = summary.loc[
        summary["metric"].eq("balance_product") & summary["traversal_state"].eq("split")
    ].iloc[0]
    assert int(split_balance["finite_count"]) == 1
    assert float(split_balance["p50"]) == 0.2352
    assert "balance_product__outgoing_edge_norm_balance" in set(joint["metric_pair"])


def test_selected_neighborhood_coverage_summarizes_evidence_by_state() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    coverage = summarize_selected_neighborhood_coverage(rows)

    split = coverage.loc[
        coverage["traversal_state"].eq("split")
        & coverage["traversal_stop_reason"].eq("accepted_split")
    ].iloc[0]
    edge_closed = coverage.loc[
        coverage["traversal_state"].eq("boundary")
        & coverage["traversal_stop_reason"].eq("edge_closed")
    ].iloc[0]
    pass_through = coverage.loc[coverage["traversal_state"].eq("pass_through")].iloc[0]

    assert int(split["old_and_current_count"]) == 1
    assert int(split["recover_internal_split_count"]) == 1
    assert split["coverage_status"] == "coverage_complete_old_current"
    assert int(edge_closed["traversal_only_count"]) == 1
    assert edge_closed["coverage_status"] == "coverage_traversal_only"
    assert int(pass_through["old_and_current_count"]) == 1


def test_selected_neighborhood_case_coverage_preserves_case_id() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    case_coverage = summarize_selected_neighborhood_case_coverage(rows)

    split = case_coverage.loc[
        case_coverage["case_id"].eq("case") & case_coverage["traversal_state"].eq("split")
    ].iloc[0]

    assert int(split["old_and_current_count"]) == 1
    assert split["coverage_status"] == "coverage_complete_old_current"
    assert set(case_coverage["case_id"]) == {"case"}


def test_selected_neighborhood_method_contrast_pairs_same_nodes() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    contrast = summarize_selected_neighborhood_method_contrast(rows)

    selected_null = contrast.loc[contrast["data_role"].eq("selected_null")].iloc[0]

    assert int(selected_null["paired_node_count"]) == 3
    assert int(selected_null["traversal_decision_agreement_count"]) == 2
    assert int(selected_null["left_pass_through_count"]) == 1
    assert int(selected_null["right_pass_through_count"]) == 0
    assert selected_null["contrast_status"] == "method_decisions_diverge"


def test_selected_neighborhood_candidate_method_contrast_filters_background() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    contrast = summarize_selected_neighborhood_candidate_method_contrast(rows)

    selected_null = contrast.loc[contrast["data_role"].eq("selected_null")].iloc[0]

    assert int(selected_null["paired_node_count"]) == 2
    assert int(selected_null["traversal_decision_agreement_count"]) == 1
    assert int(selected_null["left_pass_through_count"]) == 1
    assert int(selected_null["right_pass_through_count"]) == 0
    assert int(selected_null["left_explicit_guard_blocked_count"]) == 1


def test_selected_neighborhood_candidate_method_contrast_rows_explain_candidates() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    by_node = {row["node_id"]: row for _, row in candidate_rows.iterrows()}

    assert set(by_node) == {"guarded", "pass", "split"}
    assert by_node["pass"]["candidate_contrast_status"] == ("candidate_decision_diverges")
    assert "left_pass_through" in by_node["pass"]["candidate_reason"]
    assert by_node["guarded"]["candidate_contrast_status"] == ("candidate_decision_agrees")
    assert "left_guard_blocked" in by_node["guarded"]["candidate_reason"]
    assert "left_old_and_current" in by_node["split"]["candidate_reason"]
    assert float(by_node["pass"]["left_sibling_p_value"]) == 0.8
    assert bool(by_node["pass"]["left_child_parent_edge_open"]) is True


def test_selected_neighborhood_candidate_ambiguity_summarizes_buckets() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    ambiguity = summarize_selected_neighborhood_candidate_ambiguity(candidate_rows)

    buckets = set(ambiguity["ambiguity_bucket"])
    selected_null = ambiguity.loc[
        ambiguity["ambiguity_bucket"].eq("conservative_selected_null_suppression")
    ].iloc[0]
    signal = ambiguity.loc[
        ambiguity["ambiguity_bucket"].eq("possible_signal_over_suppression")
    ].iloc[0]

    assert "conservative_selected_null_suppression" in buckets
    assert "possible_signal_over_suppression" in buckets
    assert int(selected_null["decision_divergence_count"]) == 1
    assert int(signal["decision_divergence_count"]) == 1
    assert signal["ambiguity_status"] == "candidate_ambiguity_requires_law"


def test_selected_neighborhood_candidate_local_features_summarize_geometry() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    local = summarize_selected_neighborhood_candidate_local_features(candidate_rows)

    signal_left = local.loc[
        local["ambiguity_bucket"].eq("possible_signal_over_suppression")
        & local["profile_side"].eq("left")
    ].iloc[0]
    selected_null_left = local.loc[
        local["ambiguity_bucket"].eq("conservative_selected_null_suppression")
        & local["profile_side"].eq("left")
    ].iloc[0]

    assert int(signal_left["edge_open_count"]) == 1
    assert int(signal_left["sibling_open_count"]) == 1
    assert float(signal_left["median_sibling_p_value"]) == 0.001
    assert int(selected_null_left["pass_through_candidate_count"]) == 1


def test_selected_neighborhood_candidate_law_targets_fail_closed_when_missing_law() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    ambiguity = summarize_selected_neighborhood_candidate_ambiguity(candidate_rows)
    local = summarize_selected_neighborhood_candidate_local_features(candidate_rows)
    targets = summarize_selected_neighborhood_candidate_law_targets(ambiguity, local)

    signal = targets.loc[targets["ambiguity_bucket"].eq("possible_signal_over_suppression")].iloc[0]
    selected_null = targets.loc[
        targets["ambiguity_bucket"].eq("conservative_selected_null_suppression")
    ].iloc[0]

    assert signal["law_target"] == ("derive_traversal_only_pass_through_retention_law")
    assert signal["production_action"] == "fail_closed_until_law_validated"
    assert selected_null["production_action"] == "retain_fail_closed_refined_guard"


def test_selected_neighborhood_candidate_rows_count_descendant_outcomes() -> None:
    left_method = "fixed_coordinate_conditional_topology_diagnostic_v1"
    right_method = "fixed_coordinate_global_passthrough_refined_v1"
    node_decisions = pd.DataFrame.from_records(
        [
            {
                "case_id": "desc",
                "data_role": "signal",
                "method_id": left_method,
                "replicate": 0,
                "node_id": "pass",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "pass_through",
                "decision_class": "unstable_passthrough_zone",
                "n_children": 2,
                "n_descendant_leaves": 12,
                "child_parent_edge_open": True,
                "sibling_open": False,
                "sibling_p_value": 0.004,
                "sibling_projection_dimension": 5,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": True,
            },
            {
                "case_id": "desc",
                "data_role": "signal",
                "method_id": left_method,
                "replicate": 0,
                "node_id": "deep_split",
                "parent_id": "pass",
                "depth": 2,
                "traversal_decision": "split",
                "decision_class": "accepted_internal_split",
                "n_children": 2,
                "n_descendant_leaves": 6,
                "child_parent_edge_open": True,
                "sibling_open": True,
                "sibling_p_value": 0.001,
                "sibling_projection_dimension": 4,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": False,
            },
            {
                "case_id": "desc",
                "data_role": "signal",
                "method_id": right_method,
                "replicate": 0,
                "node_id": "pass",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 2,
                "n_descendant_leaves": 12,
                "child_parent_edge_open": True,
                "sibling_open": False,
                "sibling_p_value": 0.004,
                "sibling_projection_dimension": 5,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": False,
            },
            {
                "case_id": "desc",
                "data_role": "signal",
                "method_id": right_method,
                "replicate": 0,
                "node_id": "deep_split",
                "parent_id": "pass",
                "depth": 2,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 2,
                "n_descendant_leaves": 6,
                "child_parent_edge_open": True,
                "sibling_open": False,
                "sibling_p_value": 0.2,
                "sibling_projection_dimension": 4,
                "root_stability_guard_blocked": False,
                "root_selective_guard_blocked": False,
                "selected_family_guard_blocked": False,
                "topology_incidence_role": "internal",
                "topology_pass_through_candidate": False,
            },
        ]
    )
    rows = build_selected_neighborhood_distribution_rows(node_decisions=node_decisions)
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    ambiguity = summarize_selected_neighborhood_candidate_ambiguity(candidate_rows)
    local = summarize_selected_neighborhood_candidate_local_features(candidate_rows)
    targets = summarize_selected_neighborhood_candidate_law_targets(ambiguity, local)

    pass_row = candidate_rows.loc[candidate_rows["node_id"].eq("pass")].iloc[0]
    target = targets.loc[targets["ambiguity_bucket"].eq("possible_signal_over_suppression")].iloc[0]
    comparison = summarize_selected_neighborhood_stop_rule_comparison(candidate_rows)
    retention = summarize_selected_neighborhood_retention_evidence(candidate_rows)
    gaps = summarize_selected_neighborhood_retention_gaps(retention)
    contract = summarize_selected_neighborhood_method_contract(gaps)
    profile_contract = summarize_selected_neighborhood_profile_config_contract(candidate_rows)
    readiness = summarize_selected_neighborhood_method_readiness(
        method_contract=contract,
        profile_config_contract=profile_contract,
    )
    stop_rule = comparison.loc[
        comparison["ambiguity_bucket"].eq("possible_signal_over_suppression")
    ].iloc[0]
    retention_row = retention.loc[
        retention["stop_rule_pattern"].eq("left_pass_through_downstream_split_right_stops")
    ].iloc[0]

    assert int(pass_row["left_descendant_accepted_split_count"]) == 1
    assert int(pass_row["right_descendant_accepted_split_count"]) == 0
    assert int(target["left_descendant_accepted_split_count"]) >= 1
    assert target["evidence_gap"] == ("missing_topology_evidence_for_downstream_splits")
    assert stop_rule["stop_rule_pattern"] == ("left_pass_through_downstream_split_right_stops")
    assert "Joined topology-neighborhood evidence is absent" in str(
        stop_rule["stop_rule_interpretation"]
    )
    assert retention_row["retention_evidence_status"] == (
        "retention_evidence_signal_unresolved_no_topology"
    )
    assert int(retention_row["traversal_only_pair_count"]) == 1
    signal_gap = gaps.loc[
        gaps["retention_evidence_status"].eq("retention_evidence_signal_unresolved_no_topology")
    ].iloc[0]
    assert signal_gap["missing_evidence"] == (
        "missing_structural_topology_coverage_at_signal_pass_through_rows"
    )
    assert signal_gap["production_action"] == ("fail_closed_until_topology_law_validated")
    by_component = {row["contract_component"]: row for _, row in contract.iterrows()}
    assert by_component["base_traversal_skeleton"]["shared_by_methods"] is True
    assert by_component["signal_pass_through_retention"]["production_action"] == (
        "fail_closed_until_topology_law_validated"
    )
    by_field = {row["profile_field"]: row for _, row in profile_contract.iterrows()}
    assert by_field["sibling_gate_method"]["values_match"] is True
    assert (
        by_field["root_selective_permutation_guard_scope"]["profile_contract_status"]
        == "refined_selected_family_guard_difference"
    )
    readiness_by_scope = {row["readiness_scope"]: row for _, row in readiness.iterrows()}
    assert (
        readiness_by_scope["selected_null_pass_through_control"]["readiness_status"]
        == "ready_as_fail_closed_guard_candidate"
    )
    assert (
        readiness_by_scope["signal_pass_through_retention"]["readiness_status"]
        == "not_ready_missing_topology_likelihood"
    )


def test_selected_neighborhood_stop_rule_comparison_labels_downstream_walk() -> None:
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=_paired_node_rows(),
        topology_rows=_topology_rows(),
        conditional_law_rows=_law_rows(),
    )
    candidate_rows = build_selected_neighborhood_candidate_method_contrast_rows(rows)
    comparison = summarize_selected_neighborhood_stop_rule_comparison(candidate_rows)

    assert "stop_rule_pattern" in comparison.columns
    assert set(comparison["ambiguity_bucket"]) >= {
        "conservative_selected_null_suppression",
        "possible_signal_over_suppression",
    }


def test_selected_neighborhood_distribution_panel_writes_outputs(tmp_path) -> None:
    node_path = tmp_path / "nodes.csv"
    topology_path = tmp_path / "topology.csv"
    law_path = tmp_path / "law.csv"
    _node_rows().to_csv(node_path, index=False)
    _topology_rows().to_csv(topology_path, index=False)
    _law_rows().to_csv(law_path, index=False)

    outputs = run_selected_neighborhood_distribution_panel(
        SelectedNeighborhoodDistributionPanelConfig(
            node_decisions_path=node_path,
            output_dir=tmp_path / "out",
            topology_rows_path=topology_path,
            conditional_law_rows_path=law_path,
        )
    )

    for path in outputs.values():
        assert path.exists(), path

    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    coverage = pd.read_csv(outputs["coverage_summary"])
    case_coverage = pd.read_csv(outputs["case_coverage_summary"])
    method_contrast = pd.read_csv(outputs["method_contrast_summary"])
    candidate_method_contrast = pd.read_csv(outputs["candidate_method_contrast_summary"])
    candidate_method_contrast_rows = pd.read_csv(outputs["candidate_method_contrast_rows"])
    candidate_ambiguity = pd.read_csv(outputs["candidate_ambiguity_summary"])
    candidate_local_features = pd.read_csv(outputs["candidate_local_feature_summary"])
    candidate_law_targets = pd.read_csv(outputs["candidate_law_target_summary"])
    stop_rule_comparison = pd.read_csv(outputs["stop_rule_comparison_summary"])
    retention_evidence = pd.read_csv(outputs["retention_evidence_summary"])
    retention_gaps = pd.read_csv(outputs["retention_gap_summary"])
    method_contract = pd.read_csv(outputs["method_contract_summary"])
    profile_contract = pd.read_csv(outputs["profile_config_contract_summary"])
    method_readiness = pd.read_csv(outputs["method_readiness_summary"])
    assert rows.shape[0] == 4
    assert "old_and_current" in set(rows["neighborhood_evidence_family"])
    assert not summary.empty
    assert "coverage_status" in coverage.columns
    assert "case_id" in case_coverage.columns
    assert "contrast_status" in method_contrast.columns
    assert "contrast_status" in candidate_method_contrast.columns
    assert "candidate_contrast_status" in candidate_method_contrast_rows.columns
    assert "ambiguity_bucket" in candidate_ambiguity.columns
    assert "median_sibling_p_value" in candidate_local_features.columns
    assert "law_target" in candidate_law_targets.columns
    assert "stop_rule_pattern" in stop_rule_comparison.columns
    assert "retention_evidence_status" in retention_evidence.columns
    assert "missing_evidence" in retention_gaps.columns
    assert "contract_component" in method_contract.columns
    assert "profile_contract_status" in profile_contract.columns
    assert "readiness_status" in method_readiness.columns
