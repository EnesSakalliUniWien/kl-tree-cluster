from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.same_z_functional_decomposition import (
    classify_scaling_row,
    summarize_same_z_rows,
)


def test_classify_scaling_row_identifies_projection_coordinate_gap() -> None:
    scaling_class, note = classify_scaling_row(
        edge_projected_delta=0.2,
        projected_fixed_coordinate_delta=8.0,
        fixed_global_fixed_coordinate_delta=1.0,
    )

    assert scaling_class == "selected_projection_energy_vs_coordinate_bh"
    assert "same scale" in note


def test_classify_scaling_row_identifies_l2_coordinate_gap() -> None:
    scaling_class, _note = classify_scaling_row(
        edge_projected_delta=2.0,
        projected_fixed_coordinate_delta=0.5,
        fixed_global_fixed_coordinate_delta=5.0,
    )

    assert scaling_class == "l2_aggregation_vs_coordinate_bh"


def test_summarize_same_z_rows_reports_median_scaling_gaps() -> None:
    rows = pd.DataFrame(
        {
            "case_id": ["case_a", "case_a"],
            "data_role": ["selected_null", "selected_null"],
            "method_id": ["fixed", "fixed"],
            "replicate": [0, 0],
            "max_edge_neglog10_bh": [10.0, 12.0],
            "projected_sibling_neglog10": [9.5, 11.5],
            "fixed_coordinate_neglog10": [0.5, 1.5],
            "fixed_global_neglog10": [8.0, 10.0],
            "edge_projected_sibling_neglog10_delta": [0.5, 0.5],
            "projected_fixed_coordinate_neglog10_delta": [9.0, 10.0],
            "fixed_global_fixed_coordinate_neglog10_delta": [7.5, 8.5],
            "scaling_class": [
                "selected_projection_energy_vs_coordinate_bh",
                "selected_projection_energy_vs_coordinate_bh",
            ],
        }
    )

    summary = summarize_same_z_rows(rows)

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["median_max_edge_neglog10_bh"] == 11.0
    assert row["median_projected_fixed_coordinate_delta"] == 9.5
    assert row["dominant_scaling_class"] == "selected_projection_energy_vs_coordinate_bh"
