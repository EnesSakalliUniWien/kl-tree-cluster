from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_bayesian_incidence_mode_law import (
    OverlapBayesianIncidenceModeLawConfig,
    build_bayesian_incidence_mode_rows,
    run_overlap_bayesian_incidence_mode_law,
    summarize_bayesian_incidence_mode_rows,
)


def _branch_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "truth_local",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T1",
                "branch_incidence_geometry_status": "coordinate_income_outcome_branch_mismatch",
                "incoming_family_outgoing_jaccard_topk": 0.0,
                "incoming_family_outgoing_abs_cosine": 0.05,
                "incoming_edge_outgoing_jaccard_topk": 0.0,
                "incoming_edge_outgoing_abs_cosine": 0.05,
            },
            {
                "case_id": "truth_emergent",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T2",
                "branch_incidence_geometry_status": "coordinate_income_outcome_branch_mismatch",
                "incoming_family_outgoing_jaccard_topk": 0.0,
                "incoming_family_outgoing_abs_cosine": 0.05,
                "incoming_edge_outgoing_jaccard_topk": 0.0,
                "incoming_edge_outgoing_abs_cosine": 0.05,
            },
            {
                "case_id": "null_emergent",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "N1",
                "branch_incidence_geometry_status": "coordinate_income_outcome_branch_mismatch",
                "incoming_family_outgoing_jaccard_topk": 0.0,
                "incoming_family_outgoing_abs_cosine": 0.05,
                "incoming_edge_outgoing_jaccard_topk": 0.0,
                "incoming_edge_outgoing_abs_cosine": 0.05,
            },
        ]
    )


def _gap_rows() -> pd.DataFrame:
    base = {
        "selected_family_log_bayes_factor_lower": 8.0,
        "subspace_consensus_jaccard_topk": 0.4,
        "size_balance": 0.45,
        "edge_norm_balance": 0.8,
        "fragment_risk_proxy_score": 0.7,
        "soft_structure_pass": True,
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "case_id": "truth_local",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T1",
                "guard_truth_role": "truth_recovery",
                "conditional_bayesian_gap_status": (
                    "recovered_by_default_internal_node_likelihood"
                ),
                "continuous_context_min_margin": 0.002,
                "context_margin_pass": True,
                "default_internal_node_candidate": True,
            },
            {
                **base,
                "case_id": "truth_emergent",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T2",
                "guard_truth_role": "truth_recovery",
                "conditional_bayesian_gap_status": (
                    "context_negative_but_soft_structure_supported"
                ),
                "continuous_context_min_margin": -0.004,
                "context_margin_pass": False,
                "default_internal_node_candidate": False,
            },
            {
                **base,
                "case_id": "null_emergent",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "N1",
                "guard_truth_role": "null_like",
                "conditional_bayesian_gap_status": "negative_blocked_by_context",
                "continuous_context_min_margin": -0.004,
                "context_margin_pass": False,
                "default_internal_node_candidate": False,
            },
        ]
    )


def test_incidence_mode_law_keeps_context_negative_emergent_fail_closed() -> None:
    rows = build_bayesian_incidence_mode_rows(
        branch_rows=_branch_rows(),
        transfer_gap_rows=_gap_rows(),
    )

    statuses = dict(zip(rows["case_id"], rows["bayesian_incidence_mode_status"]))
    assert statuses["truth_local"] == "local_outcome_mode_candidate"
    assert statuses["truth_emergent"] == "context_negative_emergent_mode_ambiguous"
    assert statuses["null_emergent"] == "context_negative_emergent_mode_ambiguous"


def test_incidence_mode_law_uses_metric_family_alignment_when_available() -> None:
    branch = _branch_rows().iloc[[1]].copy()
    branch["metric_family_alignment_score"] = 0.60
    gap = _gap_rows().iloc[[1]].copy()

    rows = build_bayesian_incidence_mode_rows(
        branch_rows=branch,
        transfer_gap_rows=gap,
    )

    assert rows["bayesian_incidence_mode_status"].iloc[0] == ("continuation_mode_candidate")


def test_incidence_mode_summary_reports_unidentified_emergent_mode() -> None:
    rows = build_bayesian_incidence_mode_rows(
        branch_rows=_branch_rows(),
        transfer_gap_rows=_gap_rows(),
    )
    summary = summarize_bayesian_incidence_mode_rows(rows)

    assert summary["diagnostic_status"].iloc[0] == ("context_negative_emergent_mode_not_identified")
    assert int(summary["local_outcome_truth_count"].iloc[0]) == 1
    assert int(summary["local_outcome_negative_count"].iloc[0]) == 0
    assert int(summary["context_negative_emergent_truth_count"].iloc[0]) == 1
    assert int(summary["context_negative_emergent_negative_count"].iloc[0]) == 1


def test_run_bayesian_incidence_mode_law_writes_outputs(tmp_path) -> None:
    branch_path = tmp_path / "branch.csv"
    gap_path = tmp_path / "gap.csv"
    _branch_rows().to_csv(branch_path, index=False)
    _gap_rows().to_csv(gap_path, index=False)

    outputs = run_overlap_bayesian_incidence_mode_law(
        OverlapBayesianIncidenceModeLawConfig(
            branch_rows_path=branch_path,
            transfer_gap_rows_path=gap_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
