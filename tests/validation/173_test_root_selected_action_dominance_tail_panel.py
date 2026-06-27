from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_selected_action_dominance_tail_panel as panel,
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


def _action_dominating_support() -> dict[str, object]:
    return {
        "case_id": "support",
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_probe",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.81,
        "root_sibling_selected_ratio": 1000.0,
        "root_edge_path_statistic_margin": 250.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
    }


def test_action_dominating_support_does_not_become_production_p_value() -> None:
    rows = panel.build_root_selected_action_dominance_tail_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [_target(), _action_dominating_support()]
        ),
    )

    row = rows.iloc[0]
    assert row["exact_support_count"] == 0
    assert row["action_dominating_support_count"] == 1
    assert row["action_dominating_spectral_exceedance_count"] == 0
    assert row["action_dominating_conservative_tail_p_value"] == 0.5
    assert row["action_dominating_p_value_status"] == (
        "diagnostic_one_sided_action_monotonicity_required"
    )
    assert row["production_inference_status"] == (
        "fail_closed_action_dominance_diagnostic_only"
    )
    assert row["next_mathematical_step"] == (
        "prove_or_reject_one_sided_action_spectral_tail_monotonicity"
    )
    assert "legacy_full_selected_null_legacy_false_split" not in rows.columns


def test_action_dominance_summary_counts_new_diagnostic_support() -> None:
    rows = panel.build_root_selected_action_dominance_tail_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [_target(), _action_dominating_support()]
        ),
    )

    summary = panel.summarize_root_selected_action_dominance_tail_rows(rows).iloc[0]

    assert summary["target_count"] == 1
    assert summary["exact_supported_target_count"] == 0
    assert summary["action_dominating_supported_target_count"] == 1
    assert summary["action_dominating_no_spectral_exceedance_count"] == 1
    assert summary["fail_closed_target_count"] == 1
    assert summary["summary_status"] == (
        "one_sided_action_dominance_diagnostic_support_available"
    )
