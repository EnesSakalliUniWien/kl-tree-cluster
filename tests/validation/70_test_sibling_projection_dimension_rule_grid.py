from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.spectral.sibling_projection_dimension_rule_grid import (
    RULE_IDS,
    STUDY_ROLE,
    add_projection_dimension_rule_columns,
    evaluate_projection_dimension_rule_grid,
    run_projection_dimension_rule_grid,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sibling_projection_dimension": [1, 2, 2, 1],
            "parent_test_projection_dimension": [2, 2, 3, 4],
            "raw_mp_signal_count": [0, 1, 3, 2],
        }
    )


def test_add_projection_dimension_rule_columns() -> None:
    table = add_projection_dimension_rule_columns(_records())

    assert set(RULE_IDS) <= set(table.columns)
    assert table["raw_mp_parent_signal_count_floor1"].tolist() == [1, 1, 3, 2]
    assert table["raw_mp_parent_signal_count_floor2"].tolist() == [2, 2, 3, 2]


def test_evaluate_projection_dimension_rule_grid_reports_distribution() -> None:
    summary = evaluate_projection_dimension_rule_grid(_records())

    assert set(summary["rule_id"]) == set(RULE_IDS)
    assert summary["study_role"].eq(STUDY_ROLE).all()
    current = summary[summary["rule_id"].eq("current_edge_derived_rule")].iloc[0]
    assert current["frequency_k1"] == 0.5
    assert current["frequency_k2"] == 0.5
    raw = summary[summary["rule_id"].eq("raw_mp_parent_signal_count")].iloc[0]
    assert raw["frequency_k0"] == 0.25
    assert raw["differs_from_current_fraction"] > 0.0


def test_run_projection_dimension_rule_grid_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        records_path = Path(tmpdir) / "records.csv"
        output_dir = Path(tmpdir) / "out"
        _records().to_csv(records_path, index=False)

        outputs = run_projection_dimension_rule_grid(
            records_path=records_path,
            output_dir=output_dir,
        )

        assert set(outputs) == {"summary", "manifest"}
        assert (output_dir / "sibling_projection_dimension_rule_grid.csv").exists()
        assert (output_dir / "manifest.json").exists()
