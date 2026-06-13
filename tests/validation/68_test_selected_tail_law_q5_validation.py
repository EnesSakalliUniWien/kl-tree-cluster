from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected_tail_law_q5_validation import (
    STUDY_ROLE,
    evaluate_q5_selected_tail_law,
    prepare_q5_selected_tail_records,
    run_q5_selected_tail_law_validation,
    summarize_q5_selected_tail_law,
)


def _records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_families = ("bernoulli", "categorical")
    parent_bins = ("small_0_0.25", "medium_0.25_0.5", "root_0.75_1")
    for index in range(180):
        feature_family = feature_families[index % len(feature_families)]
        parent_size_bin = parent_bins[index % len(parent_bins)]
        parent_sample_size = 20 + (index % 30)
        left_child_sample_size = 5 + (index % max(parent_sample_size - 10, 1))
        left_child_sample_size = min(left_child_sample_size, parent_sample_size - 1)
        right_child_sample_size = parent_sample_size - left_child_sample_size
        feature_dimension = 50 + (index % 7)
        projection_dimension = 1 + ((index // 2) % 2)
        edge_action = 2.0 + 0.05 * index
        eigen_ratio = 1.0 + 0.02 * index
        spectral_mass = 0.2 + 0.6 * ((index % 17) / 16.0)
        effective_rank = 1.0 + (index % 9) / 4.0
        barycentric_balance = min(left_child_sample_size, right_child_sample_size) / (
            left_child_sample_size + right_child_sample_size
        )
        leverage = np.log(
            max(left_child_sample_size, right_child_sample_size)
            / min(left_child_sample_size, right_child_sample_size)
        )
        family_offset = 0.35 if feature_family == "categorical" else 0.0
        parent_offset = 0.2 if parent_size_bin == "root_0.75_1" else 0.0
        log_ratio = (
            0.25 * edge_action
            + 0.15 * np.log(parent_sample_size)
            + 0.1 * np.log(feature_dimension / parent_sample_size)
            + 0.1 * projection_dimension
            + family_offset
            + parent_offset
            + 0.2 * barycentric_balance
            + 0.05 * leverage
            + 0.5 * np.log(eigen_ratio)
            + 0.3 * spectral_mass
            + 0.08 * effective_rank
            + 0.03 * np.sin(index)
        )
        rows.append(
            {
                "case_id": f"case_{index % 4}",
                "feature_family": feature_family,
                "parent_size_bin": parent_size_bin,
                "replicate_index": index,
                "selected_hierarchy_simulation_id": f"sim_{index}",
                "log_selected_hierarchy_ratio": log_ratio,
                "negative_log10_min_child_edge_bh_p_value": edge_action,
                "feature_dimension": feature_dimension,
                "parent_sample_size": parent_sample_size,
                "left_child_sample_size": left_child_sample_size,
                "right_child_sample_size": right_child_sample_size,
                "sibling_projection_dimension": projection_dimension,
                "selected_eigenvalue_over_mp_upper_bound": eigen_ratio,
                "selected_eigenvalue_mass_fraction": spectral_mass,
                "eigenvalue_effective_rank": effective_rank,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_prepare_q5_selected_tail_records_adds_predeclared_predictors() -> None:
    table = prepare_q5_selected_tail_records(_records())

    assert "edge_action" in table.columns
    assert "log_parent_sample_size" in table.columns
    assert "log_feature_parent_aspect_ratio" in table.columns
    assert "barycentric_balance" in table.columns
    assert "log_barycentric_leverage" in table.columns
    assert "log_sampling_variance_scale" in table.columns
    assert "log_selected_eigenvalue_over_mp_upper_bound" in table.columns
    assert any(column.startswith("feature_family__") for column in table.columns)
    assert any(column.startswith("parent_size_bin__") for column in table.columns)


def test_prepare_q5_selected_tail_records_rejects_inconsistent_child_sizes() -> None:
    records = _records()
    records.loc[0, "right_child_sample_size"] += 1

    with pytest.raises(ValueError, match="left_child_sample_size"):
        prepare_q5_selected_tail_records(records)


def test_q5_selected_tail_law_reports_holdout_tail_metrics() -> None:
    validation = evaluate_q5_selected_tail_law(
        _records(),
        alpha=0.1,
        tail_quantile=0.8,
        n_replicate_folds=3,
        min_train_rows_per_predictor=3,
        min_test_rows=5,
    )

    assert validation.shape[0] == 24
    assert validation["study_role"].eq(STUDY_ROLE).all()
    assert {
        "replicate_modulo",
        "leave_one_case_out",
        "leave_one_feature_family_out",
        "leave_one_parent_size_bin_out",
    } == set(validation["split_strategy"])

    full_replicate = validation[
        validation["model_id"].eq("q5_full_selected_tail_law")
        & validation["split_strategy"].eq("replicate_modulo")
    ].iloc[0]
    assert full_replicate["model_status"] == "diagnostic_holdout_replicate_modulo"
    assert float(full_replicate["holdout_tail_auc_from_linear_score"]) >= 0.5
    assert float(full_replicate["residual_tail_exceedance_rate"]) >= 0.0
    assert float(full_replicate["residual_tail_exceedance_absolute_error"]) >= 0.0

    summary = summarize_q5_selected_tail_law(validation)
    assert set(summary["model_id"]) == {
        "q5_barycentric_context",
        "q5_barycentric_edge_spectral",
        "q5_barycentric_full_selected_tail_law",
        "q5_edge_spectral_only",
        "q5_full_selected_tail_law",
        "q5_without_spectral_geometry",
    }
    assert summary["study_role"].eq(STUDY_ROLE).all()


def test_run_q5_selected_tail_law_validation_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        records_path = Path(tmpdir) / "records.csv"
        output_dir = Path(tmpdir) / "out"
        _records().to_csv(records_path, index=False)

        outputs = run_q5_selected_tail_law_validation(
            records_path=records_path,
            output_dir=output_dir,
            alpha=0.1,
            tail_quantile=0.8,
            n_replicate_folds=3,
            min_train_rows_per_predictor=3,
            min_test_rows=5,
        )

        assert set(outputs) == {"validation", "summary", "manifest"}
        assert (output_dir / "q5_selected_tail_law_validation.csv").exists()
        assert (output_dir / "q5_selected_tail_law_summary.csv").exists()
        assert (output_dir / "manifest.json").exists()
