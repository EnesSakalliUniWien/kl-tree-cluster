from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.focused_overlap_balance_product_benchmark import (
    FocusedOverlapBalanceProductBenchmarkConfig,
    build_focused_overlap_positive_rows,
    run_focused_overlap_balance_product_benchmark,
    summarize_focused_overlap_metrics,
)


def test_focused_overlap_rows_have_multiple_positive_support_cases() -> None:
    rows = build_focused_overlap_positive_rows()

    truth = rows["guard_truth_role"].astype(str).eq("truth_recovery")
    assert int(truth.sum()) >= 3
    assert set(rows.loc[truth, "topology_signal_role"]) == {"signal"}
    assert set(rows.loc[~truth, "topology_signal_role"]) == {""}
    assert "selected_nonnull" in set(rows["topology_support_role"])
    assert int(rows["parent_id"].astype(str).str.len().gt(0).sum()) == rows.shape[0]


def test_balance_product_generalizes_where_outgoing_balance_does_not() -> None:
    rows = build_focused_overlap_positive_rows()
    summary = summarize_focused_overlap_metrics(rows)
    by_metric = {row["metric"]: row for _, row in summary.iterrows()}

    assert by_metric["balance_product"]["zero_negative_status"] == (
        "zero_negative_separates_all_truth"
    )
    assert by_metric["balance_product"]["truth_count"] >= 3
    assert by_metric["balance_product"]["zero_negative_value_margin"] > 0.0
    assert by_metric["outgoing_balance"]["zero_negative_status"] != (
        "zero_negative_separates_all_truth"
    )


def test_focused_overlap_benchmark_writes_outputs(tmp_path) -> None:
    outputs = run_focused_overlap_balance_product_benchmark(
        FocusedOverlapBalanceProductBenchmarkConfig(output_dir=tmp_path)
    )

    for path in outputs.values():
        assert path.exists(), path

    rows = pd.read_csv(outputs["rows"])
    metric_summary = pd.read_csv(outputs["metric_summary"])
    law_summary = pd.read_csv(outputs["conditional_law_summary"])
    assert rows["guard_truth_role"].eq("truth_recovery").sum() >= 3
    assert (
        metric_summary.loc[
            metric_summary["metric"].eq("balance_product"),
            "zero_negative_status",
        ].iloc[0]
        == "zero_negative_separates_all_truth"
    )
    assert law_summary["production_status"].iloc[0] in {
        "diagnostic_only_candidate",
        "diagnostic_only_support_insufficient_fail_closed",
    }
