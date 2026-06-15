from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_income_outcome_junction_law import (
    OverlapIncomeOutcomeJunctionLawConfig,
    build_income_outcome_junction_rows,
    run_overlap_income_outcome_junction_law,
    summarize_income_outcome_junction_rows,
)


def _structural_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "P",
                "parent_id": "",
                "depth": 0,
                "n_parent": 100,
                "n_left": 50,
                "n_right": 50,
                "barycentric_balance": 0.5,
                "sibling_p_value": 0.1,
                "homogeneity_gain_min": 0.002,
                "subspace_consensus_jaccard_topk": 0.2,
                "structural_sibling_status": "weak_homogeneity_gain",
                "structural_change_mode": "weak_or_mixed_structural_change",
            },
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "C",
                "parent_id": "P",
                "depth": 1,
                "n_parent": 50,
                "n_left": 25,
                "n_right": 25,
                "barycentric_balance": 0.5,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.01,
                "subspace_consensus_jaccard_topk": 0.4,
                "structural_sibling_status": "weak_homogeneity_gain",
                "structural_change_mode": "weak_or_mixed_structural_change",
            },
            {
                "case_id": "case_a",
                "data_role": "null",
                "replicate": 0,
                "node_id": "N",
                "parent_id": "P",
                "depth": 1,
                "n_parent": 50,
                "n_left": 25,
                "n_right": 25,
                "barycentric_balance": 0.5,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.01,
                "subspace_consensus_jaccard_topk": 0.4,
                "structural_sibling_status": "weak_homogeneity_gain",
                "structural_change_mode": "weak_or_mixed_structural_change",
            },
        ]
    )


def _decision_zone_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "P",
                "continuous_context_min_margin": -0.01,
                "structural_decision_zone": "nonaccepted_or_leaf",
            },
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "C",
                "continuous_context_min_margin": -0.004,
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
            },
            {
                "case_id": "case_a",
                "data_role": "null",
                "replicate": 0,
                "node_id": "N",
                "continuous_context_min_margin": -0.002,
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
            },
        ]
    )


def _gap_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "C",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.004,
                "subspace_consensus_jaccard_topk": 0.4,
                "size_balance": 0.5,
                "edge_norm_balance": 0.8,
                "fragment_risk_proxy_score": 0.7,
                "context_margin_pass": False,
                "soft_structure_pass": True,
                "default_internal_node_candidate": False,
                "conditional_bayesian_gap_status": (
                    "context_negative_but_soft_structure_supported"
                ),
            },
            {
                "case_id": "case_a",
                "data_role": "null",
                "replicate": 0,
                "node_id": "N",
                "guard_truth_role": "null_like",
                "truth_geometry_mode": "null_like",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.002,
                "subspace_consensus_jaccard_topk": 0.4,
                "size_balance": 0.5,
                "edge_norm_balance": 0.8,
                "fragment_risk_proxy_score": 0.7,
                "context_margin_pass": False,
                "soft_structure_pass": True,
                "default_internal_node_candidate": False,
                "conditional_bayesian_gap_status": "negative_blocked_by_context",
            },
        ]
    )


def test_income_outcome_rows_keep_incoming_and_outgoing_roles_separate() -> None:
    rows = build_income_outcome_junction_rows(
        structural_rows=_structural_rows(),
        decision_zone_rows=_decision_zone_rows(),
        transfer_gap_rows=_gap_rows(),
    )

    truth = rows[rows["guard_truth_role"].eq("truth_recovery")].iloc[0]
    assert truth["incidence_signature"] == "income1_outcome2_internal_junction"
    assert truth["incoming_parent_structural_decision_zone"] == "nonaccepted_or_leaf"
    assert not bool(truth["incoming_parent_context_pass"])
    assert not bool(truth["outgoing_context_pass"])
    assert bool(truth["outgoing_soft_structure_pass"])
    assert truth["income_outcome_transition_status"] == (
        "truth_recovery_income_outcome_context_transition_required"
    )


def test_income_outcome_summary_reports_transition_law_required() -> None:
    rows = build_income_outcome_junction_rows(
        structural_rows=_structural_rows(),
        decision_zone_rows=_decision_zone_rows(),
        transfer_gap_rows=_gap_rows(),
    )
    summary = summarize_income_outcome_junction_rows(rows)

    assert summary["diagnostic_status"].iloc[0] == (
        "income_outcome_transition_law_required"
    )
    assert int(summary["truth_transition_required_count"].iloc[0]) == 1
    assert int(summary["negative_default_candidate_count"].iloc[0]) == 0
    assert int(summary["negative_outgoing_context_blocked_count"].iloc[0]) == 1


def test_run_income_outcome_junction_law_writes_outputs(tmp_path) -> None:
    structural_path = tmp_path / "structural.csv"
    decision_zone_path = tmp_path / "decision_zone.csv"
    gap_path = tmp_path / "gap.csv"
    _structural_rows().to_csv(structural_path, index=False)
    _decision_zone_rows().to_csv(decision_zone_path, index=False)
    _gap_rows().to_csv(gap_path, index=False)

    outputs = run_overlap_income_outcome_junction_law(
        OverlapIncomeOutcomeJunctionLawConfig(
            structural_rows_path=structural_path,
            decision_zone_rows_path=decision_zone_path,
            transfer_gap_rows_path=gap_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
