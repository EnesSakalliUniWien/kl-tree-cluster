from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_diagnostic_traversal_policy import (
    OverlapDiagnosticTraversalPolicyConfig,
    assign_diagnostic_traversal_policy,
    run_overlap_diagnostic_traversal_policy,
    summarize_diagnostic_traversal_policy,
)


def _zone_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_signal",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "stable",
                "decision_class": "accepted_internal_split",
                "structural_decision_zone": "stable_structural_accept",
                "structural_truth_role": "signal_truth_aligned",
                "continuous_context_min_margin": 0.02,
            },
            {
                "case_id": "case_signal",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "decision_class": "accepted_internal_split",
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "structural_truth_role": "signal_truth_misaligned",
                "continuous_context_min_margin": -0.01,
            },
            {
                "case_id": "case_signal",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "decision_class": "accepted_internal_split",
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "structural_truth_role": "signal_truth_aligned",
                "continuous_context_min_margin": -0.005,
            },
            {
                "case_id": "case_null",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "leaf",
                "decision_class": "leaf_fragment",
                "structural_decision_zone": "nonaccepted_or_leaf",
                "structural_truth_role": "null_like",
                "continuous_context_min_margin": -1.0,
            },
        ]
    )


def _guard_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_signal",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "guard_truth_role": "fragment_like",
                "truth_geometry_mode": "one_sided_pure_fragment",
                "fragment_risk_proxy_score": 1.8,
            },
            {
                "case_id": "case_signal",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "guard_truth_role": "truth_recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
                "fragment_risk_proxy_score": 0.8,
            },
        ]
    )


def test_policy_assigns_stable_blocked_and_unstable_actions() -> None:
    policy = assign_diagnostic_traversal_policy(
        _zone_rows(),
        _guard_rows(),
        fragment_guard_threshold=1.25,
    )

    by_node = dict(zip(policy["node_id"], policy["diagnostic_traversal_action"], strict=True))
    assert by_node == {
        "stable": "stable_region_accept",
        "fragment": "weak_fragment_guard_blocked",
        "recovery": "weak_unstable_multiscale_zone",
        "leaf": "nonaccepted_or_leaf",
    }


def test_policy_summary_counts_guard_roles() -> None:
    policy = assign_diagnostic_traversal_policy(
        _zone_rows(),
        _guard_rows(),
        fragment_guard_threshold=1.25,
    )
    summary = summarize_diagnostic_traversal_policy(policy)

    blocked = summary[
        summary["diagnostic_traversal_action"].eq("weak_fragment_guard_blocked")
    ].iloc[0]
    unstable = summary[
        summary["diagnostic_traversal_action"].eq("weak_unstable_multiscale_zone")
    ].iloc[0]

    assert int(blocked["fragment_like_count"]) == 1
    assert int(blocked["truth_recovery_count"]) == 0
    assert int(unstable["truth_recovery_count"]) == 1


def test_run_policy_writes_outputs(tmp_path) -> None:
    zone_path = tmp_path / "zones.csv"
    guard_path = tmp_path / "guards.csv"
    _zone_rows().to_csv(zone_path, index=False)
    _guard_rows().to_csv(guard_path, index=False)

    outputs = run_overlap_diagnostic_traversal_policy(
        OverlapDiagnosticTraversalPolicyConfig(
            decision_zone_rows_path=zone_path,
            fragment_guard_rows_path=guard_path,
            output_dir=tmp_path / "out",
            fragment_guard_threshold=1.25,
        )
    )

    for path in outputs.values():
        assert path.exists()
