from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected_tail_parent_size_balance_stability import (
    STUDY_ROLE,
    evaluate_parent_size_balance_stability,
    prepare_parent_size_balance_records,
    run_parent_size_balance_stability_diagnostic,
)


def _stable_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parent_size_bin in ("small_0_0.25", "medium_0.25_0.5"):
        for index in range(50):
            ratio = 10.0 if index >= 45 else 1.0
            rows.append(
                {
                    "source_family": "gaussian_blobs",
                    "feature_family": "bernoulli",
                    "parent_size_bin": parent_size_bin,
                    "sibling_projection_dimension": 1,
                    "negative_log10_min_child_edge_bh_p_value": 8.5,
                    "parent_sample_size": 20,
                    "left_child_sample_size": 10,
                    "right_child_sample_size": 10,
                    "selected_hierarchy_ratio": ratio,
                    "selected_hierarchy_simulation_id": f"{parent_size_bin}:{index}",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_prepare_parent_size_balance_records_adds_predeclared_bins() -> None:
    table = prepare_parent_size_balance_records(_stable_records())

    assert table["barycentric_balance"].eq(0.5).all()
    assert table["barycentric_balance_bin"].eq("balance_0.4_0.5").all()
    assert table["edge_action_bin"].eq("edge_action_ge8").all()


def test_prepare_parent_size_balance_records_rejects_inconsistent_child_sizes() -> None:
    records = _stable_records()
    records.loc[0, "right_child_sample_size"] = 11

    with pytest.raises(ValueError, match="left_child_sample_size"):
        prepare_parent_size_balance_records(records)


def test_parent_size_balance_stability_can_mark_external_candidate() -> None:
    contexts, folds, summary = evaluate_parent_size_balance_stability(
        _stable_records(),
        alpha=0.1,
        min_train_simulations=10,
        min_train_records=10,
        min_test_records=10,
        required_min_matching_simulations=20,
        required_min_matched_records=20,
        max_relative_c_simulation_se=0.5,
        max_exceedance_standard_error=0.1,
        max_parent_size_abs_error=0.02,
    )

    assert contexts.shape[0] == 1
    row = contexts.iloc[0]
    assert row["stability_decision"] == "parent_size_balance_external_candidate"
    assert bool(row["support_contract_met"])
    assert bool(row["c_hat_precision_met"])
    assert bool(row["parent_size_abs_error_met"])
    assert folds["fold_status"].eq("ok").all()
    assert summary["study_role"].eq(STUDY_ROLE).all()


def test_parent_size_balance_stability_marks_support_failure() -> None:
    contexts, _folds, summary = evaluate_parent_size_balance_stability(
        _stable_records(),
        alpha=0.1,
        min_train_simulations=10,
        min_train_records=10,
        min_test_records=10,
        required_min_matching_simulations=200,
        required_min_matched_records=200,
        max_relative_c_simulation_se=0.5,
        max_exceedance_standard_error=0.1,
        max_parent_size_abs_error=0.02,
    )

    assert contexts.iloc[0]["stability_decision"] == "undefined_support_failure"
    assert "support_contract_failed" in contexts.iloc[0]["stability_failure_reasons"]
    assert "undefined_support_failure" in set(summary["stability_decision"])


def test_run_parent_size_balance_stability_diagnostic_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        records_path = Path(tmpdir) / "records.csv"
        output_dir = Path(tmpdir) / "out"
        _stable_records().to_csv(records_path, index=False)

        outputs = run_parent_size_balance_stability_diagnostic(
            records_path=records_path,
            output_dir=output_dir,
            alpha=0.1,
            min_train_simulations=10,
            min_train_records=10,
            min_test_records=10,
            required_min_matching_simulations=20,
            required_min_matched_records=20,
            max_relative_c_simulation_se=0.5,
            max_exceedance_standard_error=0.1,
            max_parent_size_abs_error=0.02,
        )

        assert set(outputs) == {"contexts", "parent_folds", "summary", "manifest"}
        assert (output_dir / "parent_size_balance_contexts.csv").exists()
        assert (output_dir / "parent_size_balance_parent_folds.csv").exists()
        assert (output_dir / "parent_size_balance_summary.csv").exists()
        assert (output_dir / "manifest.json").exists()
