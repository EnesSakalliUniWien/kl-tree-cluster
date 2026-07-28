from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_internal_node_transfer_gap_audit import (
    OverlapInternalNodeTransferGapAuditConfig,
    build_internal_node_transfer_gap_rows,
    run_overlap_internal_node_transfer_gap_audit,
    summarize_transfer_gap_rows,
)


def _likelihood_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "truth_recovered",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "N1",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": 0.002,
                "subspace_consensus_jaccard_topk": 0.4,
                "size_balance": 0.45,
                "edge_norm_balance": 0.7,
                "fragment_risk_proxy_score": 0.8,
            },
            {
                "case_id": "truth_context_gap",
                "data_role": "signal",
                "replicate": 1,
                "node_id": "N2",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.004,
                "subspace_consensus_jaccard_topk": 0.4,
                "size_balance": 0.45,
                "edge_norm_balance": 0.7,
                "fragment_risk_proxy_score": 0.8,
            },
            {
                "case_id": "truth_soft_blocked",
                "data_role": "signal",
                "replicate": 2,
                "node_id": "N3",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "partial_truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": 0.002,
                "subspace_consensus_jaccard_topk": 0.05,
                "size_balance": 0.45,
                "edge_norm_balance": 0.7,
                "fragment_risk_proxy_score": 0.8,
            },
            {
                "case_id": "null_context_blocked",
                "data_role": "null",
                "replicate": 3,
                "node_id": "N4",
                "guard_truth_role": "null_like",
                "truth_geometry_mode": "null_like",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.001,
                "subspace_consensus_jaccard_topk": 0.4,
                "size_balance": 0.45,
                "edge_norm_balance": 0.7,
                "fragment_risk_proxy_score": 0.8,
            },
        ]
    )


def _transfer_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "rule_id": "ctx=-0.002|sub=0.1|size=0.3|edge=0.45|frag=1.25|bf=3",
                "test_negative_candidate_count": 2,
            },
            {
                "rule_id": "ctx=0|sub=0.1|size=0.3|edge=0.45|frag=1.25|bf=3",
                "test_negative_candidate_count": 0,
            },
        ]
    )


def test_gap_rows_mark_context_negative_soft_supported_truth() -> None:
    rows = build_internal_node_transfer_gap_rows(_likelihood_rows())

    context_gap = rows[rows["case_id"].eq("truth_context_gap")].iloc[0]
    assert context_gap["conditional_bayesian_gap_status"] == (
        "context_negative_but_soft_structure_supported"
    )
    assert context_gap["blocking_components"] == "local_context_margin"
    assert bool(context_gap["soft_structure_pass"])


def test_summary_requires_higher_order_conditional_law_when_relaxed_context_leaks() -> None:
    rows = build_internal_node_transfer_gap_rows(
        _likelihood_rows()[lambda frame: ~frame["case_id"].eq("truth_soft_blocked")]
    )
    summary = summarize_transfer_gap_rows(rows, _transfer_rows())

    assert summary["diagnostic_status"].iloc[0] == (
        "context_exception_requires_higher_order_conditional_law"
    )
    assert int(summary["truth_recovered_by_default_count"].iloc[0]) == 1
    assert int(summary["truth_context_negative_soft_supported_count"].iloc[0]) == 1
    assert int(summary["relaxed_context_leakage_rule_count"].iloc[0]) == 1
    assert int(summary["nonnegative_context_leakage_rule_count"].iloc[0]) == 0


def test_run_transfer_gap_audit_writes_outputs(tmp_path) -> None:
    likelihood_path = tmp_path / "likelihood.csv"
    transfer_path = tmp_path / "transfer.csv"
    _likelihood_rows().to_csv(likelihood_path, index=False)
    _transfer_rows().to_csv(transfer_path, index=False)

    outputs = run_overlap_internal_node_transfer_gap_audit(
        OverlapInternalNodeTransferGapAuditConfig(
            likelihood_rows_path=likelihood_path,
            transfer_rows_path=transfer_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
