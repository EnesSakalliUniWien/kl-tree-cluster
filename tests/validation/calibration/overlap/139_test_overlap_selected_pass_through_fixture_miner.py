from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_selected_pass_through_fixture_miner import (
    OverlapSelectedPassThroughFixtureMinerConfig,
    build_overlap_selected_pass_through_case_rows,
    build_overlap_selected_pass_through_node_rows,
    build_selected_pass_through_truth_context_rows,
    build_structural_topology_context_rows,
    run_overlap_selected_pass_through_fixture_miner,
    summarize_overlap_selected_pass_through_support,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)


def _candidate_rows() -> pd.DataFrame:
    base = {
        "left_method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
        "right_method_id": "fixed_coordinate_global_passthrough_refined_v1",
        "left_traversal_state": "pass_through",
        "right_traversal_state": "boundary",
        "left_child_parent_edge_open": True,
        "left_sibling_open": False,
        "left_sibling_p_value": 0.004,
        "left_depth": 3,
        "left_n_descendant_leaves": 80,
        "left_neighborhood_evidence_family": "traversal_only",
        "right_neighborhood_evidence_family": "traversal_only",
        "right_explicit_guard_blocked": False,
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
                "left_neighborhood_evidence_family": "old_and_current",
                "right_neighborhood_evidence_family": "old_and_current",
                "left_balance_product": 0.18,
                "left_outgoing_edge_norm_balance": 0.72,
            },
        ]
    )


def _selected_neighborhood_rows() -> pd.DataFrame:
    method_id = "fixed_coordinate_conditional_topology_diagnostic_v1"
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "signal_pass",
                "parent_id": "",
                "n_descendant_leaves": 80,
                "traversal_state": "pass_through",
                "decision_class": "unstable_passthrough_zone",
                "guard_truth_role": "",
                "topology_support_role": "",
                "topology_signal_role": "",
                "support_status": "",
                "guarded_recovery_status": "",
            },
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "signal_split",
                "parent_id": "signal_pass",
                "n_descendant_leaves": 40,
                "traversal_state": "split",
                "decision_class": "accepted_internal_split",
                "guard_truth_role": "truth_recovery",
                "topology_support_role": "",
                "topology_signal_role": "signal",
                "support_status": "support_observed_diagnostic_only",
                "guarded_recovery_status": "",
            },
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "null_pass",
                "parent_id": "",
                "n_descendant_leaves": 90,
                "traversal_state": "pass_through",
                "decision_class": "unstable_passthrough_zone",
                "guard_truth_role": "null_like",
                "topology_support_role": "strict_null",
                "topology_signal_role": "",
                "support_status": "support_observed_diagnostic_only",
                "guarded_recovery_status": "root_or_null_guard_blocked",
            },
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "null_split",
                "parent_id": "null_pass",
                "n_descendant_leaves": 45,
                "traversal_state": "split",
                "decision_class": "accepted_internal_split",
                "guard_truth_role": "null_like",
                "topology_support_role": "strict_null",
                "topology_signal_role": "",
                "support_status": "support_observed_diagnostic_only",
                "guarded_recovery_status": "root_or_null_guard_blocked",
            },
        ]
    )


def _truth_context_rows(role: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "method_id": "fixed_coordinate_conditional_topology_diagnostic_v1",
                "replicate": 0,
                "node_id": "signal_pass",
                "data_seed": 123,
                "truth_geometry_role": role,
                "truth_context_status": "truth_context_observed",
                "truth_node_sample_count": 8,
                "truth_node_cluster_count": 2,
                "truth_node_majority_fraction": 0.5,
                "truth_downstream_split_node_id": "signal_split",
                "truth_downstream_split_distance": 1.0,
                "truth_downstream_split_ari": 1.0,
                "truth_downstream_child_mean_purity": 1.0,
                "truth_downstream_child_majority_distinct": True,
            }
        ]
    )


def test_selected_pass_through_miner_classifies_signal_and_control_rows() -> None:
    rows = build_overlap_selected_pass_through_node_rows(
        _candidate_rows(),
        _selected_neighborhood_rows(),
    )

    assert set(rows["fixture_role"]) == {
        "signal_pass_through_candidate_unresolved",
        "selected_null_pass_through_control",
    }
    signal = rows.loc[rows["data_role"].eq("signal")].iloc[0]
    control = rows.loc[rows["data_role"].eq("selected_null")].iloc[0]
    assert signal["topology_feature_status"] == "topology_features_missing"
    assert control["topology_feature_status"] == "topology_features_observed"
    assert signal["distance_to_pass_through_context"] == 0.0
    assert signal["distance_to_downstream_accepted_split"] == 1.0


def test_truth_context_promotes_recovery_and_fragment_fixture_roles() -> None:
    recovery = build_overlap_selected_pass_through_node_rows(
        _candidate_rows(),
        _selected_neighborhood_rows(),
        _truth_context_rows("truth_recovery_pass_through_positive"),
    )
    fragment = build_overlap_selected_pass_through_node_rows(
        _candidate_rows(),
        _selected_neighborhood_rows(),
        _truth_context_rows("fragment_false_pass_through"),
    )

    assert (
        recovery.loc[recovery["data_role"].eq("signal"), "fixture_role"].iloc[0]
        == "truth_recovery_pass_through_positive"
    )
    assert (
        fragment.loc[fragment["data_role"].eq("signal"), "fixture_role"].iloc[0]
        == "fragment_false_pass_through"
    )


def test_truth_context_builder_uses_sample_paths_for_downstream_recovery() -> None:
    case = _select_cases(suite="binary", case_names=["overlap_unbal_4c_small"])[0]
    (
        case_id,
        source_family,
        feature_representation,
        n_samples,
        n_features,
        n_categories,
    ) = _case_contract(case)
    seed = 20260615
    data, _feature_space, labels, _true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role="signal",
        seed=seed,
    )
    by_label: dict[int, list[str]] = {}
    for sample_id, label in zip(data.index.astype(str), labels.astype(int)):
        by_label.setdefault(int(label), []).append(str(sample_id))
    left_samples = by_label[0][:4]
    right_samples = by_label[1][:4]
    method_id = "fixed_coordinate_conditional_topology_diagnostic_v1"
    node_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "root",
                "parent_id": "",
                "depth": 0,
                "decision_class": "stable_boundary",
            },
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "pass",
                "parent_id": "root",
                "depth": 1,
                "decision_class": "unstable_passthrough_zone",
            },
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "split",
                "parent_id": "pass",
                "depth": 2,
                "decision_class": "accepted_internal_split",
            },
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "left",
                "parent_id": "split",
                "depth": 3,
                "decision_class": "leaf_fragment",
            },
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "node_id": "right",
                "parent_id": "split",
                "depth": 3,
                "decision_class": "leaf_fragment",
            },
        ]
    )
    gene_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "sample_id": sample_id,
                "path_node_ids": f"root;pass;split;left;{sample_id}",
            }
            for sample_id in left_samples
        ]
        + [
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "sample_id": sample_id,
                "path_node_ids": f"root;pass;split;right;{sample_id}",
            }
            for sample_id in right_samples
        ]
    )
    traversal_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": method_id,
                "replicate": 0,
                "data_seed": seed,
            }
        ]
    )

    truth = build_selected_pass_through_truth_context_rows(
        node_rows,
        gene_rows,
        traversal_rows,
        suite="binary",
    )
    pass_row = truth[truth["node_id"].eq("pass")].iloc[0]

    assert pass_row["truth_geometry_role"] == "truth_recovery_pass_through_positive"
    assert pass_row["truth_downstream_split_node_id"] == "split"
    assert pass_row["truth_downstream_child_majority_distinct"]


def test_structural_topology_context_uses_incoming_and_outgoing_tree_balance() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "root",
                "parent_id": "",
                "n_descendant_leaves": 100,
            },
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "pass",
                "parent_id": "root",
                "n_descendant_leaves": 70,
            },
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "sibling",
                "parent_id": "root",
                "n_descendant_leaves": 30,
            },
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "left",
                "parent_id": "pass",
                "n_descendant_leaves": 20,
            },
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "right",
                "parent_id": "pass",
                "n_descendant_leaves": 50,
            },
        ]
    )

    context = build_structural_topology_context_rows(rows)
    pass_row = context.loc[context["node_id"].eq("pass")].iloc[0]

    assert pass_row["structural_incoming_branch_balance"] == 0.3
    assert pass_row["structural_outgoing_balance"] == 20 / 70
    assert pass_row["structural_balance_product"] == 0.3 * (20 / 70)
    assert pass_row["structural_topology_context_status"] == ("structural_topology_observed")


def test_selected_pass_through_support_summary_fails_closed_without_signal_topology() -> None:
    rows = build_overlap_selected_pass_through_node_rows(
        _candidate_rows(),
        _selected_neighborhood_rows(),
    )
    cases = build_overlap_selected_pass_through_case_rows(rows)
    summary = summarize_overlap_selected_pass_through_support(
        rows,
        cases,
        min_signal_candidate_count=1,
        min_selected_null_control_count=1,
        min_finite_topology_per_side=1,
    )

    assert int(summary["signal_candidate_count"].iloc[0]) == 1
    assert int(summary["selected_null_control_count"].iloc[0]) == 1
    assert summary["fixture_support_status"].iloc[0] == "signal_topology_support_missing"
    assert summary["next_required_step"].iloc[0] == (
        "compute_topology_features_for_signal_retained_pass_through_rows"
    )


def test_selected_pass_through_fixture_miner_writes_outputs(tmp_path) -> None:
    candidate_path = tmp_path / "candidate_rows.csv"
    selected_path = tmp_path / "selected_neighborhood_rows.csv"
    _candidate_rows().to_csv(candidate_path, index=False)
    _selected_neighborhood_rows().to_csv(selected_path, index=False)

    outputs = run_overlap_selected_pass_through_fixture_miner(
        OverlapSelectedPassThroughFixtureMinerConfig(
            candidate_rows_path=candidate_path,
            selected_neighborhood_rows_path=selected_path,
            output_dir=tmp_path / "out",
            min_signal_candidate_count=1,
            min_selected_null_control_count=1,
            min_finite_topology_per_side=1,
        )
    )

    for path in outputs.values():
        assert path.exists(), path
    summary = pd.read_csv(outputs["support_summary"])
    assert summary["production_action"].iloc[0] == ("fail_closed_until_fixture_support_observed")
