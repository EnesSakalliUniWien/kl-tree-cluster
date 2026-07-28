from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_binary_resolution_panel as panel,
)


def _tail(
    *,
    case_id: str,
    t_rank: float,
    action: float,
    edge: float,
    tail_status: str,
    s_h_u: float = 0.0,
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "t_selected_tie_rank_fraction": t_rank,
        "a_selected_ratio_action_log1p": action,
        "e_edge_margin_action_log1p": edge,
        "b_bandwidth_topology_status": "bandwidth_root_reopen_observed",
        "h_u_population_law_status": "deformed_mp_edge_measured_support_side",
        "s_root_deformed_excess_log": s_h_u,
        "root_tail_inference_status": tail_status,
    }


def _equation(case_id: str, status: str) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "external_law_equation_status": status,
    }


def test_binary_resolution_uses_existing_tail_support() -> None:
    rows = panel.build_root_selected_binary_resolution_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="supported",
                    t_rank=0.8,
                    action=4.0,
                    edge=4.5,
                    tail_status="calibrated_selected_root_spectral_tail_available",
                )
            ]
        ),
        external_law_equation_rows=pd.DataFrame.from_records(
            [
                _equation(
                    "supported",
                    "existing_tail_support_defer_to_tail_panel",
                )
            ]
        ),
    )

    row = rows.iloc[0]
    assert row["root_binary_resolution_band"] == "weak_binary_resolution"
    assert row["root_binary_resolution_inference_status"] == ("binary_root_tail_calibrated")
    assert row["method_action"] == "use_existing_conservative_root_tail"


def test_binary_resolution_fails_closed_for_missing_spectral_law() -> None:
    rows = panel.build_root_selected_binary_resolution_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="hard",
                    t_rank=0.9,
                    action=7.0,
                    edge=8.0,
                    s_h_u=0.8,
                    tail_status=("fail_closed_selected_root_spectral_tail_support_missing"),
                )
            ]
        ),
        external_law_equation_rows=pd.DataFrame.from_records(
            [
                _equation(
                    "hard",
                    "requires_new_same_stratum_nonzero_s_h_u_support",
                )
            ]
        ),
    )

    row = rows.iloc[0]
    assert row["root_binary_resolution_band"] == "strong_binary_resolution"
    assert row["root_binary_resolution_inference_status"] == (
        "binary_root_selected_resolution_requires_new_spectral_law"
    )
    assert row["method_action"] == ("fail_closed_generate_same_geometry_nonzero_s_h_u_support")


def test_binary_resolution_summary_counts_underpowered_geometry_match() -> None:
    rows = panel.build_root_selected_binary_resolution_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="underpowered",
                    t_rank=0.75,
                    action=4.5,
                    edge=5.5,
                    tail_status=("fail_closed_selected_root_spectral_tail_support_missing"),
                )
            ]
        ),
        external_law_equation_rows=pd.DataFrame.from_records(
            [
                _equation(
                    "underpowered",
                    "moment_equation_feasible_but_still_diagnostic_only",
                )
            ]
        ),
    )
    summary = panel.summarize_root_selected_binary_resolution_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["root_binary_resolution_inference_status"] == (
        "binary_root_geometry_match_underpowered_tail"
    )
    assert summary["moment_only_reweighting_possible_count"] == 1
    assert summary["fail_closed_count"] == 1
    assert summary["summary_status"] == "selected_root_binary_resolution_law_incomplete"
