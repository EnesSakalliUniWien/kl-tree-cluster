from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.sibling_null_calibration_panel import (
    STUDY_ROLE,
    evaluate_sibling_null_calibration_rows,
    run_sibling_null_calibration_panel,
    summarize_sibling_null_calibration_rows,
)


def _role_rows(
    role: str,
    *,
    n_rows: int,
    n_rejected: int,
    external_rule_admissible: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n_rows):
        rejected = index < n_rejected
        p_value = 0.01 if rejected else 0.60
        rows.append(
            {
                "case_id": role,
                "replicate_id": index // 10,
                "parent_id": f"p{index}",
                "sibling_context_role": role,
                "sibling_p_value": p_value,
                "sibling_adjusted_p_value": p_value,
                "internal_support_admissible": role != "selected_nonnull_only",
                "external_rule_admissible": external_rule_admissible,
                "sibling_projection_dimension": 2,
            }
        )
    return rows


def _panel() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            *_role_rows("strict_null", n_rows=100, n_rejected=4),
            *_role_rows("stopped_edge_null", n_rows=100, n_rejected=18),
            *_role_rows("selected_nonnull_only", n_rows=100, n_rejected=82),
            *_role_rows(
                "external_selected_tail_context",
                n_rows=100,
                n_rejected=50,
                external_rule_admissible=False,
            ),
        ]
    )


def test_sibling_null_panel_scores_context_roles() -> None:
    rows = evaluate_sibling_null_calibration_rows(_panel(), alpha=0.05)
    summary = summarize_sibling_null_calibration_rows(
        rows,
        alpha=0.05,
        tolerance=0.02,
        min_rows=30,
    )

    assert rows["study_role"].eq(STUDY_ROLE).all()

    strict = summary[summary["sibling_context_role"].eq("strict_null")].iloc[0]
    stopped = summary[summary["sibling_context_role"].eq("stopped_edge_null")].iloc[0]
    selected = summary[
        summary["sibling_context_role"].eq("selected_nonnull_only")
    ].iloc[0]
    external = summary[
        summary["sibling_context_role"].eq("external_selected_tail_context")
    ].iloc[0]

    assert strict["sibling_rejection_rate"] == pytest.approx(0.04)
    assert strict["sibling_calibration_status"] == "within_nominal_tolerance"
    assert stopped["sibling_rejection_rate"] == pytest.approx(0.18)
    assert stopped["sibling_calibration_status"] == "above_nominal_tolerance"
    assert selected["sibling_calibration_status"] == (
        "selected_nonnull_retention_descriptive"
    )
    assert external["sibling_calibration_status"] == "external_selected_tail_fail_closed"


def test_sibling_null_panel_reports_external_candidates() -> None:
    panel = pd.DataFrame.from_records(
        _role_rows(
            "external_selected_tail_context",
            n_rows=40,
            n_rejected=20,
            external_rule_admissible=True,
        )
    )
    rows = evaluate_sibling_null_calibration_rows(panel)
    summary = summarize_sibling_null_calibration_rows(rows, min_rows=30)
    external = summary[
        summary["sibling_context_role"].eq("external_selected_tail_context")
    ].iloc[0]

    assert external["external_rule_admissible_rate"] == pytest.approx(1.0)
    assert external["sibling_calibration_status"] == (
        "external_selected_tail_candidate_descriptive"
    )


def test_run_sibling_null_calibration_panel_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "sibling_panel.csv"
    output_dir = tmp_path / "out"
    _panel().to_csv(panel_path, index=False)

    outputs = run_sibling_null_calibration_panel(
        panel_path=panel_path,
        output_dir=output_dir,
        alpha=0.05,
        tolerance=0.02,
        min_rows=30,
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert (output_dir / "sibling_null_calibration_rows.csv").exists()
    assert (output_dir / "sibling_null_calibration_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
