from __future__ import annotations

import math

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_selected_population_law_requirement_panel as panel,
)


def _action_row(
    *,
    case_id: str = "target",
    target_s: float = 2.0,
    support_s: float = 1.0,
    action_support: int = 1,
    exact_support: int = 0,
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "target_s_root_spectral_excess_log": target_s,
        "exact_support_count": exact_support,
        "action_dominating_support_count": action_support,
        "best_action_dominating_support_s_root_log": support_s,
        "legacy_comparison_interpretation": "legacy_comparison_neutral_or_missing",
    }


def test_population_law_requirement_uses_exp_spectral_gap() -> None:
    rows = panel.build_root_selected_population_law_requirement_rows(
        action_dominance_rows=pd.DataFrame.from_records([_action_row()])
    )

    row = rows.iloc[0]
    assert row["support_reference_status"] == "action_dominating_support"
    assert row["spectral_excess_gap_to_support_max"] == 1.0
    assert row["required_mp_edge_multiplier"] == math.exp(1.0)
    assert row["required_population_law_status"] == (
        "substantial_population_law_multiplier_required"
    )
    assert row["production_inference_status"] == (
        "fail_closed_population_law_requirement_diagnostic_only"
    )


def test_population_law_requirement_marks_missing_support() -> None:
    rows = panel.build_root_selected_population_law_requirement_rows(
        action_dominance_rows=pd.DataFrame.from_records(
            [
                _action_row(
                    support_s=math.nan,
                    action_support=0,
                    exact_support=0,
                )
            ]
        )
    )
    summary = panel.summarize_root_selected_population_law_requirement_rows(rows).iloc[0]

    assert rows.iloc[0]["required_population_law_status"] == (
        "population_law_requirement_missing_support"
    )
    assert summary["missing_support_count"] == 1
    assert summary["requirement_available_count"] == 0


def test_exact_support_without_action_support_needs_no_population_law_requirement() -> None:
    rows = panel.build_root_selected_population_law_requirement_rows(
        action_dominance_rows=pd.DataFrame.from_records(
            [
                _action_row(
                    support_s=math.nan,
                    action_support=0,
                    exact_support=1,
                )
            ]
        )
    )
    summary = panel.summarize_root_selected_population_law_requirement_rows(rows).iloc[0]

    assert rows.iloc[0]["required_mp_edge_multiplier"] == 1.0
    assert rows.iloc[0]["required_population_law_status"] == (
        "exact_tail_support_available_no_population_law_requirement"
    )
    assert summary["missing_support_count"] == 0
    assert summary["exact_no_requirement_count"] == 1
