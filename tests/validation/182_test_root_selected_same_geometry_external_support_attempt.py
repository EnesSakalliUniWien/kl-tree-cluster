from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_selected_same_geometry_external_support_attempt as panel,
)


def _equation(case_id: str, status: str) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "external_law_equation_status": status,
        "next_generator_requirement": (
            "generate_same_T_A_E_B_Hu_roots_with_nonzero_deformed_spectral_excess"
        ),
    }


def _tail(case_id: str, support_count: int = 0) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "case_id": case_id,
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": 100.0,
        "root_edge_path_statistic_margin": 200.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
        "root_tail_stratum_key": (
            "discrete_tie_rank_region|tie_mid_0_70_0_85|"
            "action_log_low_lt_5|action_log_mid_5_7|"
            "bandwidth_root_reopen_observed|deformed_mp_edge_measured_support_side"
        ),
        "s_root_spectral_excess_log": 0.2,
        "s_root_deformed_excess_log": 0.2,
        "selected_null_support_count": support_count,
        "selected_null_exceedance_count": 0,
        "selected_null_importance_effective_sample_size": 0.0,
        "conservative_spectral_tail_p_value": float("nan"),
        "root_tail_inference_status": (
            "calibrated_selected_root_spectral_tail_available"
            if support_count
            else "fail_closed_selected_root_spectral_tail_support_missing"
        ),
    }


def _support(case_id: str, target_id: str = "target") -> dict[str, object]:
    return {
        "case_id": case_id,
        "conditioning_target_case_id": target_id,
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_correlated_two_factor_external_null",
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": 100.0,
        "root_edge_path_statistic_margin": 200.0,
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
    }


def test_selects_only_external_law_targets() -> None:
    rows = pd.DataFrame.from_records(
        [
            _equation(
                "hard",
                "requires_new_same_stratum_nonzero_s_h_u_support",
            ),
            {
                "target_case_id": "done",
                "external_law_equation_status": (
                    "existing_tail_support_defer_to_tail_panel"
                ),
                "next_generator_requirement": "use_existing_tail_panel",
            },
        ]
    )

    assert panel._select_external_law_target_case_ids(rows) == ("hard",)
    assert panel._select_external_law_target_case_ids(
        rows,
        explicit_target_case_ids=("manual",),
    ) == ("manual",)


def test_h_u_missing_external_rows_are_not_tail_support() -> None:
    joined = pd.DataFrame.from_records(
        [
            {"case_id": "target", "data_role": "observed_target"},
            _support("missing_hu"),
            _support("computed_hu"),
        ]
    )
    deformed_support = pd.DataFrame.from_records(
        [
            {
                "case_id": "computed_hu",
                "support_deformed_mp_edge_status": (
                    "support_deformed_mp_edge_computed_diagnostic_only"
                ),
            },
            {
                "case_id": "missing_hu",
                "support_deformed_mp_edge_status": (
                    "support_deformed_mp_edge_missing_inputs"
                ),
            },
        ]
    )

    admissible = panel._tail_admissible_joined_rows(
        joined_rows=joined,
        deformed_support_rows=deformed_support,
        h_u_population_law_status=panel.DEFAULT_H_U_STATUS,
    )

    missing = admissible.loc[admissible["case_id"].eq("missing_hu")].iloc[0]
    computed = admissible.loc[admissible["case_id"].eq("computed_hu")].iloc[0]
    assert str(missing["data_role"]).endswith("_h_u_missing_not_tail_support")
    assert str(missing["calibration_role"]).endswith("_h_u_missing_not_tail_support")
    assert computed["data_role"] == "external_selected_null"
    assert computed["calibration_role"] == "external_null_support"


def test_attempt_rows_count_new_same_geometry_nonzero_support() -> None:
    tail_rows = pd.DataFrame.from_records([_tail("target", support_count=1)])
    joined_rows = pd.DataFrame.from_records([_tail("target"), _support("new_support")])
    deformed_support = pd.DataFrame.from_records(
        [
            {
                "case_id": "new_support",
                "s_root_deformed_excess_log": 0.3,
                "support_deformed_mp_edge_status": (
                    "support_deformed_mp_edge_computed_diagnostic_only"
                ),
            }
        ]
    )
    rows = panel.build_same_geometry_external_support_attempt_rows(
        external_law_equation_rows=pd.DataFrame.from_records(
            [
                _equation(
                    "target",
                    "requires_new_same_stratum_nonzero_s_h_u_support",
                )
            ]
        ),
        generated_rows=pd.DataFrame.from_records([_support("new_support")]),
        target_conditioning_rows=pd.DataFrame.from_records(
            [
                {
                    "target_case_id": "target",
                    "pre_topology_stratum_hit_count": 1,
                }
            ]
        ),
        replay_rows=pd.DataFrame.from_records(
            [{"case_id": "new_support", "run_status": "ok"}]
        ),
        joined_rows=joined_rows,
        deformed_rows=pd.DataFrame(),
        deformed_support_rows=deformed_support,
        tail_rows=tail_rows,
        target_case_ids=("target",),
    )
    row = rows.iloc[0]
    summary = panel.summarize_same_geometry_external_support_attempt_rows(rows).iloc[0]

    assert row["generated_candidate_count"] == 1
    assert row["pre_topology_stratum_hit_count"] == 1
    assert row["replay_completed_candidate_count"] == 1
    assert row["new_same_tail_positive_s_h_u_support_count"] == 1
    assert row["new_same_tail_exceedance_count"] == 1
    assert row["attempt_status"] == "new_same_geometry_tail_exceedance_support_found"
    assert summary["new_tail_exceedance_supported_target_count"] == 1
    assert summary["summary_status"] == (
        "new_same_geometry_tail_exceedance_support_observed"
    )
