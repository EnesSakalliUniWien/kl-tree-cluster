from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.traversal_guard_validation_panel import (
    STUDY_ROLE,
    evaluate_traversal_guard_rows,
    run_traversal_guard_validation_panel,
    summarize_traversal_guard_rows,
)


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(30):
        rows.append(
            {
                "case_id": "pure",
                "parent_id": f"p{index}",
                "traversal_context_role": "pure_fragment",
                "action_budget_proxy_capped": 0.95,
                "geometry_angle_to_leading_axis_deg": 80.0,
                "geometry_independent_radius_fraction": 0.95,
            }
        )
    for index in range(70):
        rows.append(
            {
                "case_id": "signal",
                "parent_id": f"s{index}",
                "traversal_context_role": "mixed_signal"
                if index % 2 == 0
                else "true_signal",
                "action_budget_proxy_capped": 0.40,
                "geometry_angle_to_leading_axis_deg": 30.0,
                "geometry_independent_radius_fraction": 0.40,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_traversal_guard_panel_scores_candidate_guard() -> None:
    rows = evaluate_traversal_guard_rows(_panel(), mixed_context_cost_ratio=2.0)
    summary = summarize_traversal_guard_rows(
        rows,
        min_flagged=10,
        min_pure_precision=0.7,
        max_signal_flag_rate=0.1,
    )

    assert rows["study_role"].eq(STUDY_ROLE).all()
    candidate = summary[
        summary["guard_id"].eq("action_ge_0.9__angle_ge_75__ind_ge_0.85")
    ].iloc[0]

    assert candidate["n_flagged"] == 30
    assert candidate["pure_fragment_precision"] == 1.0
    assert candidate["signal_context_flag_rate"] == 0.0
    assert candidate["guard_validation_status"] == "guard_validation_candidate"


def test_traversal_guard_panel_keeps_low_precision_guard_diagnostic_only() -> None:
    panel = _panel()
    panel.loc[
        panel["traversal_context_role"].ne("pure_fragment"),
        "action_budget_proxy_capped",
    ] = 0.95
    panel.loc[
        panel["traversal_context_role"].ne("pure_fragment"),
        "geometry_angle_to_leading_axis_deg",
    ] = 80.0
    panel.loc[
        panel["traversal_context_role"].ne("pure_fragment"),
        "geometry_independent_radius_fraction",
    ] = 0.95

    rows = evaluate_traversal_guard_rows(panel, mixed_context_cost_ratio=2.0)
    summary = summarize_traversal_guard_rows(rows)
    broad = summary[
        summary["guard_id"].eq("action_ge_0.9__angle_ge_75__ind_ge_0.85")
    ].iloc[0]

    assert broad["n_flagged"] == 100
    assert broad["guard_validation_status"] == "diagnostic_only_guard"


def test_run_traversal_guard_validation_panel_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "traversal_panel.csv"
    output_dir = tmp_path / "out"
    _panel().to_csv(panel_path, index=False)

    outputs = run_traversal_guard_validation_panel(
        panel_path=panel_path,
        output_dir=output_dir,
        mixed_context_cost_ratio=2.0,
        min_flagged=10,
        min_pure_precision=0.7,
        max_signal_flag_rate=0.1,
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert (output_dir / "traversal_guard_validation_rows.csv").exists()
    assert (output_dir / "traversal_guard_validation_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
