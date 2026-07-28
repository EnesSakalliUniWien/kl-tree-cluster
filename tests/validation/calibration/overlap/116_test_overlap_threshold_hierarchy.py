from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_threshold_hierarchy import (
    OverlapThresholdHierarchyConfig,
    build_overlap_threshold_hierarchy,
    run_overlap_threshold_hierarchy,
)


def _decision_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "structural_decision_zone": "stable_structural_accept",
                "signal_truth_aligned_count": 3,
                "null_count": 0,
                "signal_truth_misaligned_count": 0,
            },
            {
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "signal_truth_aligned_count": 2,
                "null_count": 4,
                "signal_truth_misaligned_count": 1,
            },
        ]
    )


def _policy_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "diagnostic_traversal_action": "weak_fragment_guard_blocked",
                "fragment_like_count": 2,
                "truth_recovery_count": 0,
            },
            {
                "diagnostic_traversal_action": "weak_unstable_multiscale_zone",
                "fragment_like_count": 1,
                "truth_recovery_count": 2,
            },
        ]
    )


def _fragment_scan() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "metric": "fragment_risk_proxy_score",
                "guard_status": "fragment_guard_candidate",
                "guard_score": 1.0,
                "threshold": 1.25,
            }
        ]
    )


def _residual_thresholds() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "threshold_name": "null_evidence_threshold",
                "metric": "residual_neg_log10_min_sibling_p_value",
                "threshold": 8.0,
                "truth_recovery_pass_count": 2,
                "truth_recovery_family_count": 2,
                "truth_recovery_retention": 1.0,
                "negative_pass_count": 0,
                "negative_family_count": 4,
                "negative_selection_rate": 0.0,
            },
            {
                "threshold_name": "nonrecovery_structural_threshold",
                "metric": "residual_min_fragment_risk_proxy_score",
                "threshold": 0.75,
                "truth_recovery_pass_count": 1,
                "truth_recovery_family_count": 2,
                "truth_recovery_retention": 0.5,
                "negative_pass_count": 0,
                "negative_family_count": 1,
                "negative_selection_rate": 0.0,
            },
            {
                "threshold_name": "full_negative_structural_threshold",
                "metric": "residual_min_fragment_risk_proxy_score",
                "threshold": 1.1,
                "truth_recovery_pass_count": 1,
                "truth_recovery_family_count": 2,
                "truth_recovery_retention": 0.5,
                "negative_pass_count": 0,
                "negative_family_count": 5,
                "negative_selection_rate": 0.0,
            },
        ]
    )


def _eligibility_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "residual_recovery_eligibility_status": ("residual_null_evidence_only_unresolved"),
                "truth_recovery_family_count": 1,
                "nonrecovery_family_count": 1,
            },
            {
                "residual_recovery_eligibility_status": (
                    "residual_strict_structural_recovery_candidate"
                ),
                "truth_recovery_family_count": 1,
                "nonrecovery_family_count": 0,
            },
        ]
    )


def test_threshold_hierarchy_builds_ordered_stage_table() -> None:
    hierarchy = build_overlap_threshold_hierarchy(
        _decision_summary(),
        _policy_summary(),
        _fragment_scan(),
        _residual_thresholds(),
        _eligibility_summary(),
    )

    assert list(hierarchy["stage_order"]) == [1, 2, 3, 4, 5, 6]
    fragment = hierarchy[hierarchy["stage_name"].eq("weak_fragment_risk_block")].iloc[0]
    assert float(fragment["threshold"]) == 1.25
    assert int(fragment["target_count"]) == 2
    assert int(fragment["negative_count"]) == 0


def test_run_threshold_hierarchy_writes_outputs(tmp_path) -> None:
    decision_path = tmp_path / "decision.csv"
    policy_path = tmp_path / "policy.csv"
    fragment_path = tmp_path / "fragment.csv"
    thresholds_path = tmp_path / "thresholds.csv"
    eligibility_path = tmp_path / "eligibility.csv"
    _decision_summary().to_csv(decision_path, index=False)
    _policy_summary().to_csv(policy_path, index=False)
    _fragment_scan().to_csv(fragment_path, index=False)
    _residual_thresholds().to_csv(thresholds_path, index=False)
    _eligibility_summary().to_csv(eligibility_path, index=False)

    outputs = run_overlap_threshold_hierarchy(
        OverlapThresholdHierarchyConfig(
            decision_zone_summary_path=decision_path,
            policy_summary_path=policy_path,
            fragment_guard_scan_path=fragment_path,
            residual_thresholds_path=thresholds_path,
            residual_eligibility_summary_path=eligibility_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
