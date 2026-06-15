from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_threshold_stability_contract import (
    OverlapThresholdStabilityContractConfig,
    build_threshold_stability_contract,
    run_overlap_threshold_stability_contract,
)


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "stage_order": 1,
                "stage_name": "continuous_structural_stable_accept",
                "selection_unit": "row",
                "metric": "continuous_context_min_margin",
                "threshold": 0.0,
                "target_rate": 0.7,
                "negative_rate": 0.0,
            },
            {
                "stage_order": 2,
                "stage_name": "weak_fragment_risk_block",
                "selection_unit": "row",
                "metric": "fragment_risk_proxy_score",
                "threshold": 1.25,
                "target_rate": 0.8,
                "negative_rate": 0.0,
            },
            {
                "stage_order": 3,
                "stage_name": "residual_selected_family_null_evidence",
                "selection_unit": "selected_family",
                "metric": "residual_neg_log10_min_sibling_p_value",
                "threshold": 8.0,
                "target_rate": 1.0,
                "negative_rate": 0.0,
            },
            {
                "stage_order": 6,
                "stage_name": "residual_null_evidence_only_unresolved",
                "selection_unit": "selected_family",
                "metric": "p-value evidence without structural recovery evidence",
                "threshold": pd.NA,
                "target_rate": 0.4,
                "negative_rate": 0.3,
            },
        ]
    )


def _transfer_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "threshold_name": "null_evidence_threshold",
                "truth_recovery_retention": 1.0,
                "null_like_leakage_rate": 0.1,
                "nonrecovery_leakage_rate": 0.0,
                "transfer_status": "transfer_leakage",
            }
        ]
    )


def test_threshold_stability_contract_marks_nontransferable_and_law_required() -> None:
    rows, summary = build_threshold_stability_contract(_hierarchy(), _transfer_summary())

    by_stage = dict(
        zip(rows["stage_name"], rows["threshold_stability_status"], strict=True)
    )
    assert by_stage["continuous_structural_stable_accept"] == (
        "stable_reporting_candidate"
    )
    assert by_stage["weak_fragment_risk_block"] == "diagnostic_only_guard_candidate"
    assert by_stage["residual_selected_family_null_evidence"] == (
        "nontransferable_focused_cutpoint"
    )
    assert by_stage["residual_null_evidence_only_unresolved"] == (
        "selected_family_law_required"
    )
    assert summary.iloc[0]["production_promotion_status"] == (
        "fail_closed_selected_family_law_required"
    )


def test_run_threshold_stability_contract_writes_outputs(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.csv"
    transfer_path = tmp_path / "transfer.csv"
    _hierarchy().to_csv(hierarchy_path, index=False)
    _transfer_summary().to_csv(transfer_path, index=False)

    outputs = run_overlap_threshold_stability_contract(
        OverlapThresholdStabilityContractConfig(
            threshold_hierarchy_path=hierarchy_path,
            transfer_summary_path=transfer_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
