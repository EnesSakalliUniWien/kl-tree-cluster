"""Tests for benchmark failure-report timestamps."""

import pandas as pd
from benchmarks.diagnostics.failure.debug_trace import diagnose_benchmark_failures


def test_failure_report_records_generated_timestamp(tmp_path):
    results_path = tmp_path / "results.csv"
    audit_dir = tmp_path / "audit"
    output_path = tmp_path / "failure_report.md"
    audit_dir.mkdir()
    pd.DataFrame(
        [
            {
                "test_case": 1,
                "case_id": "case_a",
                "method": "tbs",
                "ari": 0.0,
                "found_clusters": 1,
                "true_clusters": 3,
            }
        ]
    ).to_csv(results_path, index=False)

    diagnose_benchmark_failures(
        str(results_path),
        str(audit_dir),
        str(output_path),
        generated_at="2026-06-24T20:12:00+02:00",
    )

    report = output_path.read_text(encoding="utf-8")
    assert "Generated at: 2026-06-24T20:12:00+02:00" in report
    assert "| case_a | 0.000 | 1 / 3 | MISSING | Audit log not found |" in report
