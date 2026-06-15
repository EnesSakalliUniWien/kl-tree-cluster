from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.edge_null_calibration_panel import (
    STUDY_ROLE,
    evaluate_edge_null_calibration_rows,
    run_edge_null_calibration_panel,
    summarize_edge_null_calibration_rows,
)


def _role_rows(
    role: str,
    *,
    n_rows: int,
    n_rejected: int,
    case_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n_rows):
        rejected = index < n_rejected
        p_value = 0.01 if rejected else 0.50
        rows.append(
            {
                "case_id": case_id,
                "replicate_id": index // 10,
                "parent_id": f"p{index}",
                "child_id": f"c{index}",
                "edge_context_role": role,
                "edge_p_value": p_value,
                "edge_bh_p_value": p_value,
                "edge_path_open": rejected,
                "edge_projection_dimension": 2,
            }
        )
    return rows


def _panel() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            *_role_rows(
                "fixed_tree_null",
                n_rows=100,
                n_rejected=5,
                case_id="fixed_null",
            ),
            *_role_rows(
                "selected_tree_null",
                n_rows=100,
                n_rejected=22,
                case_id="selected_null",
            ),
            *_role_rows(
                "selected_tree_signal",
                n_rows=100,
                n_rejected=76,
                case_id="selected_signal",
            ),
        ]
    )


def test_edge_null_panel_scores_context_roles() -> None:
    rows = evaluate_edge_null_calibration_rows(_panel(), alpha=0.05)
    summary = summarize_edge_null_calibration_rows(
        rows,
        alpha=0.05,
        tolerance=0.02,
        min_rows=30,
    )

    assert rows["study_role"].eq(STUDY_ROLE).all()
    assert rows["edge_id"].str.contains("->").all()

    fixed = summary[summary["edge_context_role"].eq("fixed_tree_null")].iloc[0]
    selected_null = summary[
        summary["edge_context_role"].eq("selected_tree_null")
    ].iloc[0]
    selected_signal = summary[
        summary["edge_context_role"].eq("selected_tree_signal")
    ].iloc[0]

    assert fixed["edge_rejection_rate"] == pytest.approx(0.05)
    assert fixed["edge_calibration_status"] == "within_nominal_tolerance"
    assert selected_null["edge_rejection_rate"] == pytest.approx(0.22)
    assert selected_null["edge_calibration_status"] == "above_nominal_tolerance"
    assert selected_signal["edge_rejection_rate"] == pytest.approx(0.76)
    assert selected_signal["edge_calibration_status"] == "signal_retention_descriptive"


def test_edge_null_panel_rejects_unknown_roles() -> None:
    panel = _panel()
    panel.loc[0, "edge_context_role"] = "pooled_null"

    with pytest.raises(ValueError, match="unknown roles"):
        evaluate_edge_null_calibration_rows(panel)


def test_run_edge_null_calibration_panel_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "edge_panel.csv"
    output_dir = tmp_path / "out"
    _panel().to_csv(panel_path, index=False)

    outputs = run_edge_null_calibration_panel(
        panel_path=panel_path,
        output_dir=output_dir,
        alpha=0.05,
        tolerance=0.02,
        min_rows=30,
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert (output_dir / "edge_null_calibration_rows.csv").exists()
    assert (output_dir / "edge_null_calibration_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
