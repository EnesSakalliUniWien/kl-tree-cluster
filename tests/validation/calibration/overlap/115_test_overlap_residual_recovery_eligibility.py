from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_residual_recovery_eligibility import (
    OverlapResidualRecoveryEligibilityConfig,
    assign_residual_recovery_eligibility,
    run_overlap_residual_recovery_eligibility,
    summarize_residual_recovery_eligibility,
    summarize_residual_recovery_thresholds,
)


def _family_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null",
                "data_role": "selected_null",
                "replicate": 0,
                "residual_family_truth_role": "residual_null_like_family",
                "residual_family_size": 1,
                "residual_neg_log10_min_sibling_p_value": 5.0,
                "residual_min_fragment_risk_proxy_score": 0.40,
            },
            {
                "case_id": "wrong",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_nonrecovery_family",
                "residual_family_size": 1,
                "residual_neg_log10_min_sibling_p_value": 9.0,
                "residual_min_fragment_risk_proxy_score": 0.60,
            },
            {
                "case_id": "weak_recovery",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 1,
                "residual_neg_log10_min_sibling_p_value": 8.0,
                "residual_min_fragment_risk_proxy_score": 0.50,
            },
            {
                "case_id": "candidate",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 1,
                "residual_neg_log10_min_sibling_p_value": 10.0,
                "residual_min_fragment_risk_proxy_score": 0.70,
            },
        ]
    )


def test_residual_recovery_eligibility_assigns_gate_statuses() -> None:
    rows = assign_residual_recovery_eligibility(_family_rows())

    by_case = dict(
        zip(
            rows["case_id"],
            rows["residual_recovery_eligibility_status"],
            strict=True,
        )
    )
    assert by_case == {
        "null": "residual_no_selected_family_null_evidence",
        "wrong": "residual_null_evidence_only_unresolved",
        "weak_recovery": "residual_null_evidence_only_unresolved",
        "candidate": "residual_strict_structural_recovery_candidate",
    }


def test_residual_recovery_threshold_summary_counts_retention() -> None:
    rows = assign_residual_recovery_eligibility(_family_rows())
    thresholds = summarize_residual_recovery_thresholds(rows)

    null_threshold = thresholds[thresholds["threshold_name"].eq("null_evidence_threshold")].iloc[0]
    nonrecovery_threshold = thresholds[
        thresholds["threshold_name"].eq("nonrecovery_structural_threshold")
    ].iloc[0]

    assert int(null_threshold["truth_recovery_pass_count"]) == 2
    assert int(null_threshold["negative_pass_count"]) == 0
    assert int(nonrecovery_threshold["truth_recovery_pass_count"]) == 1
    assert int(nonrecovery_threshold["negative_pass_count"]) == 0


def test_residual_recovery_eligibility_summary_and_outputs(tmp_path) -> None:
    rows = assign_residual_recovery_eligibility(_family_rows())
    summary = summarize_residual_recovery_eligibility(rows)

    assert set(summary["residual_recovery_eligibility_status"]) == {
        "residual_no_selected_family_null_evidence",
        "residual_null_evidence_only_unresolved",
        "residual_strict_structural_recovery_candidate",
    }

    family_path = tmp_path / "families.csv"
    _family_rows().to_csv(family_path, index=False)
    outputs = run_overlap_residual_recovery_eligibility(
        OverlapResidualRecoveryEligibilityConfig(
            residual_family_rows_path=family_path,
            output_dir=tmp_path / "out",
        )
    )
    for path in outputs.values():
        assert path.exists()
