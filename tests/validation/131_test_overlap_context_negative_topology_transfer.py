from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_context_negative_topology_transfer import (
    OverlapContextNegativeTopologyTransferConfig,
    build_topology_transfer_rows,
    run_overlap_context_negative_topology_transfer,
    summarize_topology_transfer_rows,
)


def _rows_with_two_truth_cases() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "replicate": 0,
                "guard_truth_role": "truth_recovery",
                "balance_product": 0.90,
            },
            {
                "case_id": "case_b",
                "replicate": 0,
                "guard_truth_role": "truth_recovery",
                "balance_product": 0.85,
            },
            {
                "case_id": "case_c",
                "replicate": 0,
                "guard_truth_role": "null_like",
                "balance_product": 0.40,
            },
            {
                "case_id": "case_d",
                "replicate": 0,
                "guard_truth_role": "diffuse_or_wrong",
                "balance_product": 0.40,
            },
        ]
    )


def _rows_with_single_truth_case() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "truth_case",
                "replicate": 0,
                "guard_truth_role": "truth_recovery",
                "balance_product": 0.90,
            },
            {
                "case_id": "negative_case",
                "replicate": 0,
                "guard_truth_role": "null_like",
                "balance_product": 0.40,
            },
            {
                "case_id": "negative_case_2",
                "replicate": 0,
                "guard_truth_role": "diffuse_or_wrong",
                "balance_product": 0.40,
            },
        ]
    )


def test_topology_transfer_validates_when_holdout_truth_has_train_support() -> None:
    transfer = build_topology_transfer_rows(
        _rows_with_two_truth_cases(),
        metrics=("balance_product",),
        split_columns=("case_id",),
    )
    summary = summarize_topology_transfer_rows(transfer)

    assert summary["diagnostic_status"].iloc[0] == (
        "transfer_validated_zero_leakage_full_truth_retention"
    )
    assert int(summary["positive_holdout_with_rule_count"].iloc[0]) == 2
    assert float(summary["test_truth_retention"].iloc[0]) == 1.0


def test_topology_transfer_reports_missing_holdout_truth_support() -> None:
    transfer = build_topology_transfer_rows(
        _rows_with_single_truth_case(),
        metrics=("balance_product",),
        split_columns=("case_id",),
    )
    summary = summarize_topology_transfer_rows(transfer)

    assert summary["diagnostic_status"].iloc[0] == (
        "transfer_unvalidated_no_truth_holdout_support"
    )
    assert int(summary["positive_holdout_count"].iloc[0]) == 1
    assert int(summary["positive_holdout_with_rule_count"].iloc[0]) == 0


def test_run_topology_transfer_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "topology.csv"
    _rows_with_two_truth_cases().to_csv(rows_path, index=False)

    outputs = run_overlap_context_negative_topology_transfer(
        OverlapContextNegativeTopologyTransferConfig(
            topology_rows_path=rows_path,
            output_dir=tmp_path / "out",
            metrics=("balance_product",),
            split_columns=("case_id",),
        )
    )

    for path in outputs.values():
        assert path.exists()
