from __future__ import annotations

import json

import pandas as pd
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_deformed_external_law_target_panel as panel,
)


def _gap_row(
    *,
    case_id: str = "target",
    status: str = "same_stratum_support_missing",
    dominant: str = "selected_ratio_action",
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "target_root_tail_stratum_key": "component|tie|action|edge|B|H",
        "target_tie_fraction": 0.8,
        "target_action_log1p": 6.0,
        "target_edge_log1p": 7.0,
        "target_s_h_u_excess_log": 0.5,
        "target_bandwidth_topology_status": "bandwidth_root_reopen_observed",
        "nearest_support_tie_gap": 0.1,
        "nearest_support_action_gap": 1.5,
        "nearest_support_edge_gap": 0.25,
        "nearest_support_s_h_u_excess_log": 0.1,
        "required_s_h_u_gap_to_nearest_support": 0.4,
        "support_gap_status": status,
        "required_s_h_u_lift_multiplier": 1.5,
        "dominant_conditioning_gap": dominant,
    }


def test_external_law_target_records_conditional_tilt_moments() -> None:
    rows = panel.build_root_selected_deformed_external_law_target_rows(
        pd.DataFrame.from_records([_gap_row()])
    )

    row = rows.iloc[0]
    target = json.loads(row["target_moment_vector_json"])
    nearest = json.loads(row["nearest_support_moment_vector_json"])
    gap = json.loads(row["required_moment_gap_json"])

    assert row["conditional_law_family"] == ("selected_root_conditional_exponential_tilt")
    assert row["required_tilt_axes"] == "A,S_Hu"
    assert target == {"A": 6.0, "E": 7.0, "S_Hu": 0.5, "T": 0.8}
    assert nearest == {"A": None, "E": None, "S_Hu": 0.1, "T": None}
    assert gap == {"A": 1.5, "E": 0.25, "S_Hu": 0.4, "T": 0.1}
    assert row["admissibility_status"] == (
        "external_law_required_fail_closed_until_support_generated"
    )


def test_external_law_target_summary_counts_existing_support() -> None:
    rows = panel.build_root_selected_deformed_external_law_target_rows(
        pd.DataFrame.from_records(
            [
                _gap_row(case_id="missing", dominant="edge_action"),
                _gap_row(
                    case_id="supported",
                    status="exact_tail_support_available",
                    dominant="selected_ratio_action",
                ),
            ]
        )
    )
    summary = panel.summarize_root_selected_deformed_external_law_target_rows(rows).iloc[0]

    assert summary["target_count"] == 2
    assert summary["external_law_required_count"] == 1
    assert summary["existing_support_count"] == 1
    assert summary["edge_axis_count"] == 1
    assert summary["selected_ratio_axis_count"] == 0
    assert summary["summary_status"] == "external_law_targets_required"
