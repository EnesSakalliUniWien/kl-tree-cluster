from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.sibling.nulls.sibling_null_weight_rule_validation import (
    STUDY_ROLE,
    WEIGHT_RULE_IDS,
    add_sibling_null_weight_rule_columns,
    evaluate_sibling_null_weight_rules,
    run_sibling_null_weight_rule_validation,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left_edge_bh_p_value": [0.9, 0.2, 0.01, 0.5],
            "right_edge_bh_p_value": [0.8, 0.1, 0.02, 0.4],
            "selected_hierarchy_ratio": [1.0, 4.0, 20.0, 2.0],
            "is_null_like": [True, False, False, True],
            "is_edge_blocked": [False, False, False, False],
        }
    )


def test_add_sibling_null_weight_rule_columns() -> None:
    table = add_sibling_null_weight_rule_columns(_records())

    assert set(WEIGHT_RULE_IDS) <= set(table.columns)
    assert table.loc[0, "current_product_bh_p"] == pytest.approx(0.72)
    assert table.loc[0, "hard_null_indicator_0_05"] == 1.0
    assert table.loc[2, "hard_null_indicator_0_05"] == 0.0


def test_evaluate_sibling_null_weight_rules_reports_weight_concentration() -> None:
    summary = evaluate_sibling_null_weight_rules(_records())

    assert set(summary["weight_rule_id"]) == set(WEIGHT_RULE_IDS)
    assert summary["study_role"].eq(STUDY_ROLE).all()
    assert summary["support_status"].eq("has_internal_support_labels").all()
    current = summary[summary["weight_rule_id"].eq("current_product_bh_p")].iloc[0]
    assert current["n_positive_weight_records"] == 4
    assert float(current["effective_sample_size"]) > 1.0
    assert float(current["max_weight_share"]) < 1.0
    assert current["n_supported_records_by_label"] == 2
    assert current["n_selected_nonnull_records_by_label"] == 2
    assert current["n_selected_nonnull_positive_weight_records"] == 2
    assert float(current["selected_nonnull_weight_share"]) > 0.0
    assert float(current["selected_nonnull_weight_share"]) < 1.0


def test_evaluate_sibling_null_weight_rules_marks_missing_support_labels() -> None:
    records = _records().drop(columns=["is_null_like", "is_edge_blocked"])

    summary = evaluate_sibling_null_weight_rules(records)

    assert summary["support_status"].eq("support_labels_unavailable").all()
    assert summary["selected_nonnull_weight_share"].isna().all()


def test_run_sibling_null_weight_rule_validation_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        records_path = Path(tmpdir) / "records.csv"
        output_dir = Path(tmpdir) / "out"
        _records().to_csv(records_path, index=False)

        outputs = run_sibling_null_weight_rule_validation(
            records_path=records_path,
            output_dir=output_dir,
        )

        assert set(outputs) == {"summary", "manifest"}
        assert (output_dir / "sibling_null_weight_rule_summary.csv").exists()
        assert (output_dir / "manifest.json").exists()
