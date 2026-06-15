from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_residual_family_recovery import (
    OverlapResidualFamilyRecoveryConfig,
    build_residual_family_recovery,
    build_residual_family_rows,
    run_overlap_residual_family_recovery,
)


def _policy_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_a",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "null_like",
            },
            {
                "case_id": "recovery_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery_a",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "truth_recovery",
            },
            {
                "case_id": "fragment_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment_blocked",
                "diagnostic_traversal_action": "weak_fragment_guard_blocked",
                "guard_truth_role": "fragment_like",
            },
            {
                "case_id": "wrong_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "wrong_a",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "diffuse_or_wrong",
            },
        ]
    )


def _zone_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_a",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.002,
                "subspace_consensus_jaccard_topk": 0.20,
                "continuous_homogeneity_threshold": 0.010,
                "depth": 1,
                "n_parent": 200,
                "barycentric_balance": 0.25,
            },
            {
                "case_id": "recovery_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery_a",
                "sibling_p_value": 0.0001,
                "homogeneity_gain_min": 0.012,
                "subspace_consensus_jaccard_topk": 0.50,
                "continuous_homogeneity_threshold": 0.009,
                "depth": 3,
                "n_parent": 100,
                "barycentric_balance": 0.45,
            },
            {
                "case_id": "fragment_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment_blocked",
                "sibling_p_value": 0.00001,
                "homogeneity_gain_min": 0.000,
                "subspace_consensus_jaccard_topk": 0.30,
                "continuous_homogeneity_threshold": 0.012,
                "depth": 1,
                "n_parent": 200,
                "barycentric_balance": 0.10,
            },
            {
                "case_id": "wrong_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "wrong_a",
                "sibling_p_value": 0.0002,
                "homogeneity_gain_min": 0.008,
                "subspace_consensus_jaccard_topk": 0.25,
                "continuous_homogeneity_threshold": 0.010,
                "depth": 1,
                "n_parent": 180,
                "barycentric_balance": 0.35,
            },
        ]
    )


def _guard_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_a",
                "fragment_risk_proxy_score": 0.7,
                "balanced_recovery_proxy_score": 1.2,
                "size_balance": 0.30,
                "edge_norm_balance": 0.50,
            },
            {
                "case_id": "recovery_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery_a",
                "fragment_risk_proxy_score": 0.5,
                "balanced_recovery_proxy_score": 2.1,
                "size_balance": 0.45,
                "edge_norm_balance": 0.80,
            },
            {
                "case_id": "fragment_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment_blocked",
                "fragment_risk_proxy_score": 1.8,
                "balanced_recovery_proxy_score": 0.5,
                "size_balance": 0.10,
                "edge_norm_balance": 0.30,
            },
            {
                "case_id": "wrong_case",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "wrong_a",
                "fragment_risk_proxy_score": 0.6,
                "balanced_recovery_proxy_score": 1.0,
                "size_balance": 0.35,
                "edge_norm_balance": 0.60,
            },
        ]
    )


def test_residual_family_rows_exclude_fragment_guard_blocked_rows() -> None:
    rows = build_residual_family_rows(_policy_rows(), _zone_rows(), _guard_rows())

    assert set(rows["case_id"]) == {"null_case", "recovery_case", "wrong_case"}
    roles = dict(zip(rows["case_id"], rows["residual_family_truth_role"], strict=True))
    assert roles == {
        "null_case": "residual_null_like_family",
        "recovery_case": "residual_truth_recovery_family",
        "wrong_case": "residual_nonrecovery_family",
    }


def test_residual_family_metric_summary_reports_auc() -> None:
    rows = build_residual_family_rows(_policy_rows(), _zone_rows(), _guard_rows())
    summary, scan = build_residual_family_recovery(
        rows,
        metrics=("residual_max_balanced_recovery_proxy_score",),
    )

    assert not summary.empty
    assert not scan.empty
    vs_nonrecovery = summary[
        summary["comparison"].eq("recovery_vs_nonrecovery")
    ].iloc[0]
    assert vs_nonrecovery["best_direction"] == "greater_equal"
    assert float(vs_nonrecovery["best_auc"]) == 1.0


def test_run_residual_family_recovery_writes_outputs(tmp_path) -> None:
    policy_path = tmp_path / "policy.csv"
    zone_path = tmp_path / "zones.csv"
    guard_path = tmp_path / "guards.csv"
    _policy_rows().to_csv(policy_path, index=False)
    _zone_rows().to_csv(zone_path, index=False)
    _guard_rows().to_csv(guard_path, index=False)

    outputs = run_overlap_residual_family_recovery(
        OverlapResidualFamilyRecoveryConfig(
            policy_rows_path=policy_path,
            decision_zone_rows_path=zone_path,
            fragment_guard_rows_path=guard_path,
            output_dir=tmp_path / "out",
            metrics=("residual_max_balanced_recovery_proxy_score",),
        )
    )

    for path in outputs.values():
        assert path.exists()
