from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_deformed_mp_edge_panel as panel,
)


def _target(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": "target",
        "data_role": "observed_target",
        "calibration_role": "observed_target_not_null_support",
        "proposal_family": "observed_target",
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_mp_upper_bound": 2.25,
        "root_mp_threshold_rows": 40,
        "root_active_feature_count": 10,
        "root_raw_mp_signal_count": 0,
        "root_full_component_eigenvalues_json": "[1.0, 1.0, 1.0, 1.0, 1.0]",
    }
    row.update(overrides)
    return row


def _h_u(
    *,
    status: str = "h_u_estimation_inputs_available_deformed_edge_missing",
    production_status: str = "fail_closed_population_law_requirement_diagnostic_only",
) -> dict[str, object]:
    return {
        "target_case_id": "target",
        "h_u_observability_status": status,
        "production_inference_status": production_status,
    }


def test_deformed_mp_upper_edge_matches_identity_mp_limit() -> None:
    edge = panel.deformed_mp_upper_edge(
        np.ones(20, dtype=float),
        aspect_ratio=0.25,
    )

    assert edge == pytest.approx(2.25, rel=1e-8)


def test_deformed_mp_edge_rows_compute_adjusted_root_excess() -> None:
    rows = panel.build_root_selected_deformed_mp_edge_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target()]),
        h_u_observability_rows=pd.DataFrame.from_records([_h_u()]),
    )
    summary = panel.summarize_root_selected_deformed_mp_edge_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["deformed_mp_edge_status"] == ("deformed_mp_edge_computed_diagnostic_only")
    assert row["deformed_mp_upper_edge"] == pytest.approx(2.25, rel=1e-8)
    assert row["selected_root_eigenvalue"] == pytest.approx(4.5)
    assert row["selected_root_deformed_ratio"] == pytest.approx(2.0, rel=1e-8)
    assert row["s_root_deformed_excess_log"] == pytest.approx(math.log(2.0))
    assert row["production_inference_status"] == ("fail_closed_deformed_mp_edge_diagnostic_only")
    assert summary["deformed_edge_computed_count"] == 1


def test_deformed_mp_edge_rows_respect_exact_tail_support() -> None:
    rows = panel.build_root_selected_deformed_mp_edge_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target()]),
        h_u_observability_rows=pd.DataFrame.from_records(
            [
                _h_u(
                    status="exact_tail_support_available_h_u_optional",
                    production_status="exact_support_available_defer_to_root_tail_panel",
                )
            ]
        ),
    )

    assert rows.iloc[0]["deformed_mp_edge_status"] == ("exact_tail_support_h_u_optional")
    assert rows.iloc[0]["production_inference_status"] == (
        "exact_support_available_defer_to_root_tail_panel"
    )


def test_deformed_mp_edge_rows_mark_missing_bulk_inputs() -> None:
    rows = panel.build_root_selected_deformed_mp_edge_rows(
        joined_feasibility_rows=pd.DataFrame.from_records(
            [
                _target(
                    root_full_component_eigenvalues_json="[1.0]",
                    root_active_feature_count=math.nan,
                )
            ]
        ),
        h_u_observability_rows=pd.DataFrame.from_records([_h_u()]),
    )

    assert rows.iloc[0]["deformed_mp_edge_status"] == ("deformed_mp_edge_missing_inputs")


def test_support_deformed_mp_edge_rows_compute_generated_root_excess() -> None:
    support = {
        **_target(
            case_id="support",
            data_role="diagnostic_proposal",
            calibration_role="diagnostic_proposal_not_null_support",
            proposal_family="conditioned_coherent_rank_one_spike_proposal",
        ),
        "conditioning_target_case_id": "target",
    }

    rows = panel.build_root_support_deformed_mp_edge_rows(
        joined_feasibility_rows=pd.DataFrame.from_records([_target(), support]),
    )

    row = rows.iloc[0]
    assert row["case_id"] == "support"
    assert row["conditioning_target_case_id"] == "target"
    assert row["support_deformed_mp_edge_status"] == (
        "support_deformed_mp_edge_computed_diagnostic_only"
    )
    assert row["selected_root_deformed_ratio"] == pytest.approx(2.0, rel=1e-8)
    assert row["s_root_deformed_excess_log"] == pytest.approx(math.log(2.0))
