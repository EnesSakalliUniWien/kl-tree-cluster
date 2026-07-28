from __future__ import annotations

import json

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_conditional_tilt_feasibility_panel as panel,
)


def _target_row(
    *,
    s_h_u: float,
    required_axes: str = "A,S_Hu",
) -> dict[str, object]:
    return {
        "target_case_id": "target",
        "target_status": "same_stratum_support_missing",
        "target_root_tail_stratum_key": (
            "discrete_tie_rank_region|tie_mid_0_70_0_85|"
            "action_log_mid_5_7|action_log_mid_5_7|"
            "bandwidth_root_reopen_observed|deformed_mp_edge_measured_support_side"
        ),
        "required_tilt_axes": required_axes,
        "target_moment_vector_json": json.dumps({"T": 0.8, "A": 6.0, "E": 6.0, "S_Hu": s_h_u}),
    }


def _support_row(
    *,
    case_id: str,
    action_ratio: float,
    edge_margin: float,
    bandwidth: str = "bandwidth_root_reopen_observed",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": action_ratio,
        "root_edge_path_statistic_margin": edge_margin,
        "root_bandwidth_reopen_band": bandwidth,
    }


def _support_deformed(case_id: str, s_h_u: float) -> dict[str, object]:
    return {"case_id": case_id, "s_root_deformed_excess_log": s_h_u}


def _rows_for(s_h_u: float) -> pd.DataFrame:
    return panel.build_root_selected_conditional_tilt_feasibility_rows(
        external_law_target_rows=pd.DataFrame.from_records([_target_row(s_h_u=s_h_u)]),
        joined_feasibility_rows=pd.DataFrame.from_records(
            [
                _support_row(case_id="low", action_ratio=99.0, edge_margin=99.0),
                _support_row(case_id="high", action_ratio=999.0, edge_margin=999.0),
            ]
        ),
        deformed_support_rows=pd.DataFrame.from_records(
            [_support_deformed("low", 0.0), _support_deformed("high", 0.0)]
        ),
    )


def test_required_axes_fail_when_s_h_u_outside_support_hull() -> None:
    rows = _rows_for(s_h_u=0.5)
    required = rows.loc[rows["checked_moment_axes"].eq("A,S_Hu")].iloc[0]

    residual = json.loads(required["moment_residual_vector_json"])

    assert required["support_pool_scope"] == "same_B_Hu"
    assert not bool(required["moment_range_contains_target"])
    assert not bool(required["convex_hull_moment_feasible"])
    assert required["tilt_feasibility_status"] == (
        "target_outside_current_support_hull_fail_closed"
    )
    assert residual["S_Hu"] == pytest.approx(0.5)


def test_required_axes_feasible_on_current_support_boundary() -> None:
    rows = _rows_for(s_h_u=0.0)
    required = rows.loc[rows["checked_moment_axes"].eq("A,S_Hu")].iloc[0]

    assert bool(required["moment_range_contains_target"])
    assert bool(required["convex_hull_moment_feasible"])
    assert required["tilt_feasibility_status"] in {
        "finite_conditional_tilt_feasible_diagnostic_only",
        "boundary_hull_match_requires_infinite_or_degenerate_tilt",
    }
    assert required["effective_support_size"] > 0.0


def test_summary_counts_required_axes_outside_hull() -> None:
    rows = _rows_for(s_h_u=0.5)
    summary = panel.summarize_root_selected_conditional_tilt_feasibility_rows(rows).iloc[0]

    assert summary["row_count"] == 3
    assert summary["required_axes_feasible_count"] == 0
    assert summary["required_axes_outside_hull_count"] == 1
    assert summary["summary_status"] == "conditional_tilt_support_hull_incomplete"
