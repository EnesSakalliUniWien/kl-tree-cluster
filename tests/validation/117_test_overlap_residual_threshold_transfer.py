from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_residual_threshold_transfer import (
    OverlapResidualThresholdTransferConfig,
    build_residual_threshold_transfer_rows,
    run_overlap_residual_threshold_transfer,
    summarize_residual_threshold_transfer,
)


def _family_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "a",
                "replicate": 0,
                "residual_family_truth_role": "residual_null_like_family",
                "residual_neg_log10_min_sibling_p_value": 5.0,
                "residual_min_fragment_risk_proxy_score": 0.30,
            },
            {
                "case_id": "b",
                "replicate": 0,
                "residual_family_truth_role": "residual_nonrecovery_family",
                "residual_neg_log10_min_sibling_p_value": 9.0,
                "residual_min_fragment_risk_proxy_score": 0.60,
            },
            {
                "case_id": "c",
                "replicate": 1,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_neg_log10_min_sibling_p_value": 8.0,
                "residual_min_fragment_risk_proxy_score": 0.50,
            },
            {
                "case_id": "d",
                "replicate": 1,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_neg_log10_min_sibling_p_value": 10.0,
                "residual_min_fragment_risk_proxy_score": 0.70,
            },
        ]
    )


def test_transfer_rows_evaluate_leave_one_splits() -> None:
    rows = build_residual_threshold_transfer_rows(
        _family_rows(),
        split_columns=("case_id",),
    )

    assert set(rows["threshold_name"]) == {
        "null_evidence_threshold",
        "nonrecovery_structural_threshold",
        "full_negative_structural_threshold",
    }
    null_holdout = rows[
        rows["threshold_name"].eq("null_evidence_threshold")
        & rows["holdout_value"].eq("a")
    ].iloc[0]
    assert int(null_holdout["test_null_like_count"]) == 1
    assert int(null_holdout["test_null_like_pass_count"]) == 0


def test_transfer_summary_reports_retention_and_leakage() -> None:
    rows = build_residual_threshold_transfer_rows(
        _family_rows(),
        split_columns=("case_id",),
    )
    summary = summarize_residual_threshold_transfer(rows)

    null_summary = summary[
        summary["threshold_name"].eq("null_evidence_threshold")
    ].iloc[0]
    structural_summary = summary[
        summary["threshold_name"].eq("nonrecovery_structural_threshold")
    ].iloc[0]

    assert int(null_summary["truth_recovery_test_total"]) == 2
    assert int(null_summary["null_like_test_pass_total"]) == 0
    assert int(structural_summary["truth_recovery_test_pass_total"]) == 1


def test_run_transfer_writes_outputs(tmp_path) -> None:
    family_path = tmp_path / "families.csv"
    _family_rows().to_csv(family_path, index=False)

    outputs = run_overlap_residual_threshold_transfer(
        OverlapResidualThresholdTransferConfig(
            residual_family_rows_path=family_path,
            output_dir=tmp_path / "out",
            split_columns=("case_id",),
        )
    )

    for path in outputs.values():
        assert path.exists()
