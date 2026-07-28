from __future__ import annotations

import math

import pandas as pd
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_action_conditioning_ladder_panel as panel,
)


def _target() -> dict[str, object]:
    return {
        "case_id": "target",
        "data_role": "observed_target",
        "calibration_role": "observed_target_not_null_support",
        "proposal_family": "observed_target",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": 100.0,
        "root_edge_path_statistic_margin": 200.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 4.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
    }


def _support(*, selected_ratio: float = 1000.0) -> dict[str, object]:
    return {
        "case_id": "support",
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_probe",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.81,
        "root_sibling_selected_ratio": selected_ratio,
        "root_edge_path_statistic_margin": 250.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
    }


def test_action_relaxed_level_finds_support_without_production_p_value() -> None:
    rows = panel.build_root_selected_action_conditioning_ladder_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target(), _support()]),
    )

    exact = rows.loc[rows["ladder_level"].eq("exact_T_A_E_B_H")].iloc[0]
    relaxed = rows.loc[rows["ladder_level"].eq("relax_A_keep_T_E_B_H")].iloc[0]

    assert exact["support_count"] == 0
    assert math.isnan(exact["conservative_spectral_tail_p_value"])
    assert relaxed["support_count"] == 1
    assert math.isnan(relaxed["conservative_spectral_tail_p_value"])
    assert relaxed["p_value_status"] == "relaxed_conditioning_not_calibration"
    assert relaxed["production_inference_status"] == ("fail_closed_action_ladder_diagnostic_only")
    assert relaxed["conditioning_gap_interpretation"] == (
        "selected_ratio_action_is_minimal_missing_coordinate"
    )
    assert "legacy_full_selected_null_legacy_false_split" not in rows.columns


def test_ladder_summary_counts_action_only_gap() -> None:
    rows = panel.build_root_selected_action_conditioning_ladder_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target(), _support()]),
    )

    summary = panel.summarize_root_selected_action_conditioning_ladder_rows(rows).iloc[0]

    assert summary["target_count"] == 1
    assert summary["exact_supported_target_count"] == 0
    assert summary["action_relaxed_supported_target_count"] == 1
    assert summary["action_only_gap_target_count"] == 1
    assert summary["fail_closed_target_count"] == 1
    assert summary["summary_status"] == "selected_ratio_action_conditioning_gap_localized"
