from __future__ import annotations

import json

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_external_law_equation_panel as panel,
)


def _target(
    *,
    case_id: str = "target",
    s_h_u: float = 0.5,
    required_axes: str = "A,S_Hu",
    target_status: str = "same_stratum_support_missing",
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "target_status": target_status,
        "target_root_tail_stratum_key": "component|tie|action|edge|B|H",
        "target_moment_vector_json": json.dumps({"T": 0.8, "A": 6.0, "E": 7.0, "S_Hu": s_h_u}),
        "required_tilt_axes": required_axes,
        "conditioning_event": "R_root_selected AND stratum=component|tie|action|edge|B|H",
    }


def _tilt(
    *,
    case_id: str = "target",
    axes: str = "A,S_Hu",
    feasible: bool = False,
    support_max_s_h_u: float = 0.0,
) -> dict[str, object]:
    residual_s = 0.0 if feasible else 0.5
    return {
        "target_case_id": case_id,
        "support_pool_scope": "same_B_Hu",
        "checked_moment_axes": axes,
        "support_max_moment_vector_json": json.dumps(
            {"T": 0.9, "A": 7.0, "E": 8.0, "S_Hu": support_max_s_h_u}
        ),
        "moment_residual_vector_json": json.dumps({"A": 0.0, "S_Hu": residual_s}),
        "convex_hull_moment_feasible": feasible,
        "tilt_feasibility_status": (
            "finite_conditional_tilt_feasible_diagnostic_only"
            if feasible
            else "target_outside_current_support_hull_fail_closed"
        ),
    }


def _gap(case_id: str = "target") -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "required_external_law": (
            "generate_same_T_A_E_B_Hu_roots_with_nonzero_deformed_spectral_excess"
        ),
    }


def _edge(case_id: str = "target") -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "selected_root_deformed_ratio": 1.8,
        "deformed_mp_upper_edge": 3.0,
        "selected_root_eigenvalue": 5.4,
    }


def test_external_law_equation_requires_new_nonzero_s_h_u_support() -> None:
    rows = panel.build_root_selected_external_law_equation_rows(
        conditional_tilt_rows=pd.DataFrame.from_records(
            [
                _tilt(axes="T,A,E,S_Hu", feasible=False),
                _tilt(axes="A,S_Hu", feasible=False),
            ]
        ),
        external_law_target_rows=pd.DataFrame.from_records([_target(s_h_u=0.5)]),
        support_gap_rows=pd.DataFrame.from_records([_gap()]),
        deformed_edge_rows=pd.DataFrame.from_records([_edge()]),
        target_alpha=0.01,
    )

    row = rows.iloc[0]

    assert row["external_law_equation_status"] == (
        "requires_new_same_stratum_nonzero_s_h_u_support"
    )
    assert row["required_s_h_u_gap_to_support_hull"] == pytest.approx(0.5)
    assert row["required_deformed_ratio_lower_bound"] == pytest.approx(1.6487212707)
    assert row["minimum_support_count_for_alpha_resolution"] == 99
    assert "P_Q(S_Hu >= target_S_Hu" in row["tail_support_equation"]


def test_external_law_equation_marks_moment_only_feasible_rows() -> None:
    rows = panel.build_root_selected_external_law_equation_rows(
        conditional_tilt_rows=pd.DataFrame.from_records(
            [
                _tilt(axes="T,A,E,S_Hu", feasible=False),
                _tilt(axes="E,S_Hu", feasible=True, support_max_s_h_u=0.0),
            ]
        ),
        external_law_target_rows=pd.DataFrame.from_records(
            [_target(s_h_u=0.0, required_axes="E,S_Hu")]
        ),
        support_gap_rows=pd.DataFrame.from_records([_gap()]),
        deformed_edge_rows=pd.DataFrame.from_records([_edge()]),
        target_alpha=0.05,
    )
    summary = panel.summarize_root_selected_external_law_equation_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["external_law_equation_status"] == (
        "moment_equation_feasible_but_still_diagnostic_only"
    )
    assert row["required_s_h_u_gap_to_support_hull"] == pytest.approx(0.0)
    assert row["minimum_support_count_for_alpha_resolution"] == 19
    assert summary["moment_only_reweighting_possible_count"] == 1
    assert summary["new_spectral_support_required_count"] == 0


def test_external_law_equation_defers_existing_tail_support() -> None:
    rows = panel.build_root_selected_external_law_equation_rows(
        conditional_tilt_rows=pd.DataFrame.from_records([_tilt(axes="T,A,E,S_Hu", feasible=False)]),
        external_law_target_rows=pd.DataFrame.from_records(
            [
                _target(
                    s_h_u=0.7,
                    required_axes="none",
                    target_status="exact_tail_support_available",
                )
            ]
        ),
        support_gap_rows=pd.DataFrame.from_records([_gap()]),
        deformed_edge_rows=pd.DataFrame.from_records([_edge()]),
        target_alpha=0.01,
    )
    summary = panel.summarize_root_selected_external_law_equation_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["external_law_equation_status"] == ("existing_tail_support_defer_to_tail_panel")
    assert summary["existing_tail_support_count"] == 1
    assert summary["new_spectral_support_required_count"] == 0
