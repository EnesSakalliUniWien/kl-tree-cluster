from __future__ import annotations

import math

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_selected_spectral_tail_nearest_support_panel as panel,
)


def _target(case_id: str = "target") -> dict[str, object]:
    return {
        "case_id": case_id,
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


def _support(case_id: str = "support") -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_probe",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.9,
        "root_sibling_selected_ratio": 1000.0,
        "root_edge_path_statistic_margin": 180.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_bandwidth_reopen_band": "bandwidth_no_root_reopen",
    }


def _tail_row(*, support_count: int = 0) -> dict[str, object]:
    return {
        "target_case_id": "target",
        "selected_null_support_count": support_count,
        "root_tail_inference_status": (
            "calibrated_selected_root_spectral_tail_available"
            if support_count
            else "fail_closed_selected_root_spectral_tail_support_missing"
        ),
        "spectral_tail_p_value_status": (
            "importance_weighted_external_selected_null_tail"
            if support_count
            else "selected_root_spectral_tail_support_missing"
        ),
        "conservative_spectral_tail_p_value": 0.5 if support_count else math.nan,
    }


def test_nearest_support_is_diagnostic_and_production_fails_closed() -> None:
    rows = panel.build_root_selected_spectral_tail_nearest_support_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target(), _support()]),
        root_tail_rows=pd.DataFrame.from_records([_tail_row(support_count=0)]),
    )

    row = rows.iloc[0]
    assert row["nearest_support_case_id"] == "support"
    assert row["nearest_support_status"] == "nearest_support_diagnostic_only"
    assert row["production_inference_status"] == "fail_closed_nearest_support_only"
    assert not row["nearest_support_root_tail_stratum_match"]
    assert "legacy_full_selected_null_legacy_false_split" not in rows.columns


def test_summary_counts_exact_support_separately_from_nearest_support() -> None:
    rows = panel.build_root_selected_spectral_tail_nearest_support_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target(), _support()]),
        root_tail_rows=pd.DataFrame.from_records([_tail_row(support_count=1)]),
    )

    summary = panel.summarize_root_selected_spectral_tail_nearest_support_rows(rows).iloc[0]

    assert summary["target_count"] == 1
    assert summary["exact_support_target_count"] == 1
    assert summary["nearest_support_available_count"] == 1
    assert summary["fail_closed_nearest_only_count"] == 0
    assert summary["summary_status"] == "all_targets_have_exact_tail_support"
