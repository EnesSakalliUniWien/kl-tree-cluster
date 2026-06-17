from __future__ import annotations

import math

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_selected_h_u_observability_panel as panel,
)


def _target(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": "target",
        "data_role": "observed_target",
        "calibration_role": "observed_target_not_null_support",
        "proposal_family": "observed_target",
        "root_selected_eigenvalue_over_mp_upper_bound": 4.0,
        "root_selected_eigenvalue_mass_fraction": 0.75,
        "root_raw_mp_signal_count": 2,
        "root_mp_threshold_rows": 32,
    }
    row.update(overrides)
    return row


def _requirement(
    *,
    status: str = "substantial_population_law_multiplier_required",
    multiplier: float = 2.5,
    production_status: str = (
        "fail_closed_population_law_requirement_diagnostic_only"
    ),
) -> dict[str, object]:
    return {
        "target_case_id": "target",
        "required_population_law_status": status,
        "required_mp_edge_multiplier": multiplier,
        "production_inference_status": production_status,
        "legacy_comparison_interpretation": "legacy_comparison_neutral_or_missing",
    }


def test_h_u_observability_blocks_compressed_root_spectrum() -> None:
    rows = panel.build_root_selected_h_u_observability_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target()]),
        population_law_requirement_rows=pd.DataFrame.from_records([_requirement()]),
    )
    summary = panel.summarize_root_selected_h_u_observability_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["has_root_top_eigenvalue_ratio"]
    assert row["has_root_mp_threshold_rows"]
    assert not row["has_root_eigenvalue_spectrum"]
    assert not row["has_root_active_feature_count"]
    assert row["h_u_observability_status"] == (
        "h_u_estimation_blocked_missing_capture"
    )
    assert row["missing_h_u_fields"] == (
        "root_eigenvalue_spectrum;root_active_feature_count;root_deformed_mp_edge"
    )
    assert summary["h_u_estimable_target_count"] == 0
    assert summary["missing_spectrum_target_count"] == 1


def test_h_u_observability_accepts_captured_bulk_spectrum_inputs() -> None:
    rows = panel.build_root_selected_h_u_observability_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [
                _target(
                    root_principal_component_eigenvalues_json="[5.0, 2.0, 1.0, 0.5]",
                    root_active_feature_count=4,
                )
            ]
        ),
        population_law_requirement_rows=pd.DataFrame.from_records([_requirement()]),
    )

    row = rows.iloc[0]
    assert row["has_root_eigenvalue_spectrum"]
    assert row["root_eigenvalue_spectrum_count"] == 4
    assert row["has_root_active_feature_count"]
    assert row["h_u_observability_status"] == (
        "h_u_estimation_inputs_available_deformed_edge_missing"
    )
    assert row["next_mathematical_step"] == (
        "compute_deformed_mp_edge_from_captured_root_bulk_spectrum"
    )


def test_h_u_observability_marks_missing_support_before_estimation() -> None:
    rows = panel.build_root_selected_h_u_observability_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [
                _target(
                    root_principal_component_eigenvalues_json="[5.0, 2.0, 1.0]",
                    root_active_feature_count=3,
                )
            ]
        ),
        population_law_requirement_rows=pd.DataFrame.from_records(
            [
                _requirement(
                    status="population_law_requirement_missing_support",
                    multiplier=math.nan,
                )
            ]
        ),
    )

    assert rows.iloc[0]["h_u_observability_status"] == (
        "support_missing_before_h_u_estimation"
    )
    assert rows.iloc[0]["next_mathematical_step"] == (
        "generate_support_or_external_law_before_tail_calibration"
    )


def test_h_u_observability_defers_to_exact_tail_support() -> None:
    rows = panel.build_root_selected_h_u_observability_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target()]),
        population_law_requirement_rows=pd.DataFrame.from_records(
            [
                _requirement(
                    status="substantial_population_law_multiplier_required",
                    multiplier=2.9,
                    production_status="exact_support_available_defer_to_root_tail_panel",
                )
            ]
        ),
    )

    assert rows.iloc[0]["h_u_observability_status"] == (
        "exact_tail_support_available_h_u_optional"
    )
    assert rows.iloc[0]["next_mathematical_step"] == "use_exact_root_tail_panel"
