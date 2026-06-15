from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_internal_node_bayesian_likelihood_probe import (
    OverlapInternalNodeBayesianLikelihoodProbeConfig,
    build_internal_node_likelihood_rows,
    run_overlap_internal_node_bayesian_likelihood_probe,
    summarize_internal_node_likelihood_families,
    summarize_internal_node_likelihood_rows,
)


def _policy_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "good",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "partial_truth_recovery",
            },
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "bad_neighbor",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "diffuse_or_wrong",
                "truth_geometry_mode": "diffuse_truth_mismatch",
            },
            {
                "case_id": "null_family",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_node",
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "guard_truth_role": "null_like",
                "truth_geometry_mode": "nan",
            },
        ]
    )


def _zone_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "good",
                "sibling_p_value": 1e-8,
                "homogeneity_gain_min": 0.012,
                "continuous_context_min_margin": 0.001,
                "subspace_consensus_jaccard_topk": 0.20,
            },
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "bad_neighbor",
                "sibling_p_value": 1e-10,
                "homogeneity_gain_min": 0.006,
                "continuous_context_min_margin": -0.01,
                "subspace_consensus_jaccard_topk": 0.30,
            },
            {
                "case_id": "null_family",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_node",
                "sibling_p_value": 1e-6,
                "homogeneity_gain_min": 0.004,
                "continuous_context_min_margin": -0.02,
                "subspace_consensus_jaccard_topk": 0.40,
            },
        ]
    )


def _guard_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "good",
                "fragment_risk_proxy_score": 1.20,
                "size_balance": 0.34,
                "edge_norm_balance": 0.50,
            },
            {
                "case_id": "recovery_family",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "bad_neighbor",
                "fragment_risk_proxy_score": 1.00,
                "size_balance": 0.40,
                "edge_norm_balance": 0.70,
            },
            {
                "case_id": "null_family",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_node",
                "fragment_risk_proxy_score": 0.70,
                "size_balance": 0.40,
                "edge_norm_balance": 0.80,
            },
        ]
    )


def test_internal_node_likelihood_accepts_row_not_whole_family() -> None:
    rows = build_internal_node_likelihood_rows(
        _policy_rows(),
        _zone_rows(),
        _guard_rows(),
    )
    by_node = rows.set_index("node_id")

    assert bool(by_node.loc["good", "row_overlap_coherent_candidate"])
    assert by_node.loc["bad_neighbor", "row_overlap_likelihood_status"] == (
        "row_overlap_context_blocked"
    )
    assert by_node.loc["null_node", "row_overlap_likelihood_status"] == (
        "row_overlap_context_blocked"
    )


def test_internal_node_likelihood_summaries() -> None:
    rows = build_internal_node_likelihood_rows(
        _policy_rows(),
        _zone_rows(),
        _guard_rows(),
    )
    summary = summarize_internal_node_likelihood_rows(rows)
    families = summarize_internal_node_likelihood_families(rows)

    recovery = summary[summary["guard_truth_role"].eq("truth_recovery")].iloc[0]
    assert int(recovery["candidate_count"]) == 1
    family = families[families["case_id"].eq("recovery_family")].iloc[0]
    assert family["family_overlap_likelihood_status"] == (
        "family_truth_recovery_internal_node_candidate"
    )


def test_internal_node_likelihood_runner_writes_outputs(tmp_path) -> None:
    policy_path = tmp_path / "policy.csv"
    zone_path = tmp_path / "zones.csv"
    guard_path = tmp_path / "guards.csv"
    _policy_rows().to_csv(policy_path, index=False)
    _zone_rows().to_csv(zone_path, index=False)
    _guard_rows().to_csv(guard_path, index=False)

    outputs = run_overlap_internal_node_bayesian_likelihood_probe(
        OverlapInternalNodeBayesianLikelihoodProbeConfig(
            policy_rows_path=policy_path,
            decision_zone_rows_path=zone_path,
            fragment_guard_rows_path=guard_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
