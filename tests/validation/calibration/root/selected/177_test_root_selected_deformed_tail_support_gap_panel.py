from __future__ import annotations

import math

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_deformed_tail_support_gap_panel as panel,
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
        "root_edge_path_statistic_margin": 1000.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 4.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
    }


def _support(
    *,
    case_id: str = "support",
    selected_ratio: float = 1000.0,
    edge_margin: float = 1000.0,
    bandwidth: str = "bandwidth_no_root_reopen",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_probe",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": selected_ratio,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_bandwidth_reopen_band": bandwidth,
    }


def _tail_row(*, support_count: int = 0, s_h_u: float = 0.7) -> dict[str, object]:
    return {
        "target_case_id": "target",
        "s_root_deformed_excess_log": s_h_u,
        "s_root_identity_excess_log": math.log(4.0),
        "selected_null_support_count": support_count,
        "root_tail_inference_status": (
            "calibrated_selected_root_spectral_tail_available"
            if support_count
            else "fail_closed_selected_root_spectral_tail_support_missing"
        ),
    }


def _support_deformed(case_id: str = "support", s_h_u: float = 0.0) -> dict[str, object]:
    return {
        "case_id": case_id,
        "s_root_deformed_excess_log": s_h_u,
        "s_root_identity_excess_log": math.log(2.0),
    }


def test_deformed_tail_gap_reports_missing_same_stratum_support() -> None:
    rows = panel.build_root_selected_deformed_tail_support_gap_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [_target(), _support(selected_ratio=1000.0)]
        ),
        deformed_tail_rows=pd.DataFrame.from_records([_tail_row(support_count=0)]),
        deformed_support_rows=pd.DataFrame.from_records([_support_deformed(s_h_u=0.1)]),
    )

    row = rows.iloc[0]
    assert row["support_gap_status"] == "same_stratum_support_missing"
    assert row["production_inference_status"] == ("fail_closed_same_stratum_support_missing")
    assert row["dominant_conditioning_gap"] in {
        "selected_ratio_action",
        "bandwidth_topology",
    }
    assert row["required_s_h_u_gap_to_nearest_support"] == pytest.approx(0.6)
    assert row["required_s_h_u_lift_multiplier"] == pytest.approx(math.exp(0.6))


def test_deformed_tail_gap_defers_when_exact_support_exists() -> None:
    rows = panel.build_root_selected_deformed_tail_support_gap_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [
                _target(),
                _support(
                    selected_ratio=100.0,
                    bandwidth="bandwidth_root_reopen_observed",
                ),
            ]
        ),
        deformed_tail_rows=pd.DataFrame.from_records([_tail_row(support_count=1)]),
        deformed_support_rows=pd.DataFrame.from_records([_support_deformed(s_h_u=0.8)]),
    )
    summary = panel.summarize_root_selected_deformed_tail_support_gap_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["support_gap_status"] == "exact_tail_support_available"
    assert row["same_stratum_support_count"] == 1
    assert row["same_stratum_s_h_u_exceedance_count"] == 1
    assert row["production_inference_status"] == "defer_to_deformed_tail_panel"
    assert summary["exact_support_target_count"] == 1
    assert summary["fail_closed_target_count"] == 0
