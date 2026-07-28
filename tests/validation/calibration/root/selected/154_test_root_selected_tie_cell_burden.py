from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import root_selected_tie_cell_burden as panel


def _merge_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "root_region_role": "root_child_construction",
                "root_child_side": "left",
                "tied_minimum_pair_count": 1,
                "selected_pair_tied_for_minimum": False,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 10,
                "selected_tie_rank_lexicographic": 1,
                "selected_tie_rank_fraction": 1.0,
            },
            {
                "case_id": "case_a",
                "root_region_role": "root_child_construction",
                "root_child_side": "right",
                "tied_minimum_pair_count": 2,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 10,
                "selected_tie_rank_lexicographic": 2,
                "selected_tie_rank_fraction": 1.0,
            },
            {
                "case_id": "case_a",
                "root_region_role": "root_child_construction",
                "root_child_side": "right",
                "tied_minimum_pair_count": 4,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.1,
                "candidate_pair_count": 20,
                "selected_tie_rank_lexicographic": 1,
                "selected_tie_rank_fraction": 0.25,
            },
            {
                "case_id": "case_a",
                "root_region_role": "root_final_merge",
                "root_child_side": "root",
                "tied_minimum_pair_count": 1,
                "selected_pair_tied_for_minimum": False,
                "merge_margin_to_nearest_competitor": math.nan,
                "candidate_pair_count": 1,
                "selected_tie_rank_lexicographic": 1,
                "selected_tie_rank_fraction": 1.0,
            },
            {
                "case_id": "case_b",
                "root_region_role": "root_child_construction",
                "root_child_side": "left",
                "tied_minimum_pair_count": 3,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 30,
                "selected_tie_rank_lexicographic": 1,
                "selected_tie_rank_fraction": 1.0 / 3.0,
            },
            {
                "case_id": "case_b",
                "root_region_role": "root_child_construction",
                "root_child_side": "right",
                "tied_minimum_pair_count": 3,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 30,
                "selected_tie_rank_lexicographic": 3,
                "selected_tie_rank_fraction": 1.0,
            },
            {
                "case_id": "case_c",
                "root_region_role": "root_child_construction",
                "root_child_side": "left",
                "tied_minimum_pair_count": 5,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 50,
                "selected_tie_rank_lexicographic": 4,
                "selected_tie_rank_fraction": 0.8,
            },
            {
                "case_id": "case_c",
                "root_region_role": "root_child_construction",
                "root_child_side": "right",
                "tied_minimum_pair_count": 6,
                "selected_pair_tied_for_minimum": True,
                "merge_margin_to_nearest_competitor": 0.0,
                "candidate_pair_count": 50,
                "selected_tie_rank_lexicographic": 5,
                "selected_tie_rank_fraction": 5.0 / 6.0,
            },
        ]
    )


def _root_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "root_selected_region_law_status": ("discrete_tie_cell_geometry_required"),
                "root_sibling_selected_ratio": 2.0,
                "root_child_balance": 0.4,
            },
            {
                "case_id": "case_b",
                "root_selected_region_law_status": ("discrete_tie_cell_geometry_required"),
                "root_sibling_selected_ratio": 4.0,
                "root_child_balance": 0.5,
            },
            {
                "case_id": "case_c",
                "root_selected_region_law_status": ("discrete_tie_cell_geometry_required"),
                "root_sibling_selected_ratio": 8.0,
                "root_child_balance": 0.45,
            },
        ]
    )


def test_tie_cell_burden_sums_log_multiplicity() -> None:
    rows = panel.build_root_selected_tie_cell_burden_rows(
        root_summary=_root_summary(),
        merge_margins=_merge_rows(),
    )
    case_a = rows[rows["case_id"].eq("case_a")].iloc[0]

    assert case_a["root_child_construction_merge_count"] == 3
    assert case_a["root_tie_step_count"] == 2
    assert case_a["root_tie_step_fraction"] == pytest.approx(2.0 / 3.0)
    assert case_a["root_near_zero_margin_count"] == 2
    assert case_a["root_tie_cell_log_burden"] == pytest.approx(math.log(8.0))
    assert case_a["root_tie_cell_bits_burden"] == pytest.approx(3.0)
    assert case_a["root_tie_cell_geometric_mean_multiplicity"] == pytest.approx(2.0)
    assert case_a["root_tie_cell_max_multiplicity"] == 4.0
    assert case_a["root_tie_rank_log_burden"] == pytest.approx(math.log(2.0))
    assert case_a["root_tie_rank_mean_fraction"] == pytest.approx(0.75)
    assert case_a["root_tie_rank_median_fraction"] == pytest.approx(1.0)
    assert case_a["root_tie_rank_max_fraction"] == pytest.approx(1.0)
    assert case_a["root_left_tie_cell_log_burden"] == pytest.approx(0.0)
    assert case_a["root_right_tie_cell_log_burden"] == pytest.approx(math.log(8.0))
    assert case_a["root_tie_cell_side_log_burden_asymmetry"] == pytest.approx(1.0)
    assert case_a["root_tie_cell_status"] == ("discrete_tie_burden_observed_diagnostic_only")


def test_tie_cell_relationships_are_reported() -> None:
    rows = panel.build_root_selected_tie_cell_burden_rows(
        root_summary=_root_summary(),
        merge_margins=_merge_rows(),
    )
    relationships = panel.summarize_tie_cell_relationships(rows)

    assert set(relationships["covariate"]) == set(panel.RELATIONSHIP_COVARIATES)
    evaluated = relationships[relationships["covariate"].eq("root_tie_cell_log_burden")].iloc[0]
    assert evaluated["relationship_status"] == "evaluated"
    assert evaluated["valid_pair_count"] == 3


def test_tie_cell_burden_writes_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    margins_path = tmp_path / "margins.csv"
    _root_summary().to_csv(summary_path, index=False)
    _merge_rows().to_csv(margins_path, index=False)

    outputs = panel.run_root_selected_tie_cell_burden(
        panel.RootSelectedTieCellBurdenConfig(
            output_dir=tmp_path / "out",
            root_selected_region_summary_path=summary_path,
            root_selected_region_merge_margins_path=margins_path,
        )
    )

    assert set(outputs) == {"rows", "relationships", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["relationships"].exists()
    assert outputs["manifest"].exists()
