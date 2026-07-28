from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.edge.selected_edge_sibling_null_equation import (
    CONTEXT_COLUMNS,
    STUDY_ROLE,
    evaluate_selected_edge_sibling_null_equation,
    prepare_selected_edge_sibling_equation_records,
    run_selected_edge_sibling_null_equation,
    summarize_selected_edge_sibling_null_equation,
)


def _records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(40):
        rows.append(
            {
                "record_id": f"null_{index}",
                "sibling_test_statistic": 1.0 + 0.05 * index,
                "left_edge_p_value": 0.001,
                "right_edge_p_value": 0.002,
                "left_child_sample_size": 20,
                "right_child_sample_size": 20,
                "sibling_projection_dimension": 2,
                "feature_family": "bernoulli",
                "edge_path_open": True,
                "is_null_context": True,
                "is_signal_context": False,
            }
        )
    rows.append(
        {
            "record_id": "signal_supported",
            "sibling_test_statistic": 4.0,
            "left_edge_p_value": 0.001,
            "right_edge_p_value": 0.002,
            "left_child_sample_size": 20,
            "right_child_sample_size": 20,
            "sibling_projection_dimension": 2,
            "feature_family": "bernoulli",
            "edge_path_open": True,
            "is_null_context": False,
            "is_signal_context": True,
        }
    )
    rows.append(
        {
            "record_id": "signal_unsupported",
            "sibling_test_statistic": 4.0,
            "left_edge_p_value": 0.80,
            "right_edge_p_value": 0.90,
            "left_child_sample_size": 19,
            "right_child_sample_size": 21,
            "sibling_projection_dimension": 1,
            "feature_family": "categorical",
            "edge_path_open": False,
            "is_null_context": False,
            "is_signal_context": True,
        }
    )
    return pd.DataFrame.from_records(rows)


def test_prepare_selected_edge_sibling_records_adds_barycentric_context() -> None:
    prepared = prepare_selected_edge_sibling_equation_records(_records())
    supported = prepared[prepared["record_id"].eq("signal_supported")].iloc[0]

    assert set(CONTEXT_COLUMNS).issubset(prepared.columns)
    assert supported["left_barycentric_weight"] == pytest.approx(0.5)
    assert supported["barycentric_balance"] == pytest.approx(0.5)
    assert supported["log_barycentric_leverage"] == pytest.approx(0.0)
    assert supported["sampling_variance_scale"] == pytest.approx(0.1)
    assert supported["edge_action_bin"] == "edge_action_2_4"
    assert supported["barycentric_balance_bin"] == "balance_0.4_0.5"


def test_selected_edge_sibling_equation_returns_matched_empirical_p_values() -> None:
    rows, contexts = evaluate_selected_edge_sibling_null_equation(
        _records(),
        min_null_records=30,
    )
    summary = summarize_selected_edge_sibling_null_equation(rows)

    supported = rows[rows["record_id"].eq("signal_supported")].iloc[0]
    unsupported = rows[rows["record_id"].eq("signal_unsupported")].iloc[0]

    assert contexts.shape[0] == 1
    assert supported["selected_edge_sibling_status"] == "conditional_empirical_p_value"
    assert supported["n_matched_null_records"] == 40
    assert supported["selected_edge_sibling_p_value"] == pytest.approx(1.0 / 41.0)
    assert unsupported["selected_edge_sibling_status"] == ("undefined_no_matched_null_context")
    assert pd.isna(unsupported["selected_edge_sibling_p_value"])
    assert summary["study_role"].eq(STUDY_ROLE).all()


def test_selected_edge_sibling_equation_fails_closed_when_support_is_sparse() -> None:
    rows, contexts = evaluate_selected_edge_sibling_null_equation(
        _records(),
        min_null_records=50,
    )
    supported = rows[rows["record_id"].eq("signal_supported")].iloc[0]

    assert contexts.iloc[0]["context_support_status"] == "insufficient_null_support"
    assert supported["selected_edge_sibling_status"] == "insufficient_null_support"
    assert pd.isna(supported["selected_edge_sibling_p_value"])


def test_run_selected_edge_sibling_null_equation_writes_outputs(tmp_path: Path) -> None:
    records_path = tmp_path / "records.csv"
    output_dir = tmp_path / "out"
    _records().to_csv(records_path, index=False)

    outputs = run_selected_edge_sibling_null_equation(
        records_path=records_path,
        output_dir=output_dir,
        min_null_records=30,
    )

    assert set(outputs) == {"rows", "contexts", "summary", "manifest"}
    assert (output_dir / "selected_edge_sibling_null_equation_rows.csv").exists()
    assert (output_dir / "selected_edge_sibling_null_equation_contexts.csv").exists()
    assert (output_dir / "selected_edge_sibling_null_equation_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
