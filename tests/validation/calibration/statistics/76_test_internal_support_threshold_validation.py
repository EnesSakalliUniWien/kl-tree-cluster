from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.statistics.internal_support_threshold_validation import (
    STUDY_ROLE,
    evaluate_internal_support_threshold_rows,
    run_internal_support_threshold_validation,
    summarize_internal_support_threshold_rows,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context_id": ["null_supported", "null_sparse", "signal_supported"],
            "n_supported_records": [80, 8, 45],
            "n_family_supported_records": [40, 4, 20],
            "n_stopped_or_null_records": [60, 4, 30],
            "family_effective_sample_size": [35.0, 3.0, 18.0],
            "local_effective_sample_size": [20.0, 2.5, 12.0],
            "local_max_weight_share": [0.08, 0.6, 0.12],
            "leave_one_record_max_delta_log_c": [0.02, 0.5, 0.05],
            "n_selected_nonnull_positive_weight_records": [0, 1, 3],
            "is_null_context": [True, True, False],
            "is_signal_context": [False, False, True],
            "split_rejected_at_alpha": [False, True, True],
        }
    )


def test_evaluate_internal_support_threshold_rows_scores_profiles() -> None:
    rows = evaluate_internal_support_threshold_rows(_panel())

    assert rows["study_role"].eq(STUDY_ROLE).all()
    assert set(rows["threshold_profile_id"]) == {
        "permissive",
        "current_default",
        "strict",
    }
    default = rows[
        rows["threshold_profile_id"].eq("current_default") & rows["context_id"].eq("null_sparse")
    ].iloc[0]
    assert default["threshold_status"] == "below_internal_support_thresholds"
    assert "supported_records_below_threshold" in default["failure_reasons"]


def test_summarize_internal_support_threshold_rows_uses_outcome_labels() -> None:
    rows = evaluate_internal_support_threshold_rows(_panel())
    summary = summarize_internal_support_threshold_rows(rows)

    default = summary[summary["threshold_profile_id"].eq("current_default")].iloc[0]
    assert default["outcome_status"] == "has_mixed_null_signal_outcomes"
    assert default["n_admissible_contexts"] == 2
    assert default["null_false_split_rate_among_admissible"] == 0.0
    assert default["signal_retention_rate_among_admissible"] == 1.0
    assert default["max_selected_nonnull_leakage_count"] == 3


def test_run_internal_support_threshold_validation_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    output_dir = tmp_path / "out"
    _panel().to_csv(panel_path, index=False)

    outputs = run_internal_support_threshold_validation(
        panel_path=panel_path,
        output_dir=output_dir,
    )

    assert set(outputs) == {"decisions", "summary", "manifest"}
    assert (output_dir / "internal_support_threshold_decisions.csv").exists()
    assert (output_dir / "internal_support_threshold_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
