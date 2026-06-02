from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected_hierarchy_geometry_covariates import (
    STUDY_ROLE,
    evaluate_candidate_equation_holdout,
    evaluate_candidate_equations,
    evaluate_covariate_block_models,
    evaluate_covariate_relationships,
    run_selected_hierarchy_geometry_covariate_study,
    summarize_selected_geometry_by_case,
)


def _geometry_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(20):
        ratio = 1.0 + 0.2 * index
        cos2 = 0.05 + 0.04 * index
        rows.append(
            {
                "case_id": "case",
                "replicate_index": index // 2,
                "feature_dimension": 40,
                "parent_sample_size": 20,
                "left_child_sample_size": 10,
                "right_child_sample_size": 10,
                "selected_hierarchy_ratio": ratio,
                "log_selected_hierarchy_ratio": float(np.log(ratio)),
                "parent_depth": index % 4,
                "parent_size_fraction": 1.0 - index / 40.0,
                "child_balance": 0.5 - index / 100.0,
                "child_size_ratio": 1.0 + index / 10.0,
                "branch_length_sum": 1.0 + index / 20.0,
                "branch_length_asymmetry": index / 50.0,
                "left_edge_raw_p_value": 0.001 + index / 1000.0,
                "right_edge_raw_p_value": 0.002 + index / 1000.0,
                "min_child_edge_raw_p_value": 0.001 + index / 1000.0,
                "max_child_edge_raw_p_value": 0.002 + index / 1000.0,
                "left_edge_bh_p_value": 0.01 + index / 1000.0,
                "right_edge_bh_p_value": 0.02 + index / 1000.0,
                "min_child_edge_bh_p_value": 0.01 + index / 1000.0,
                "max_child_edge_bh_p_value": 0.02 + index / 1000.0,
                "negative_log10_min_child_edge_bh_p_value": float(
                    -np.log10(0.01 + index / 1000.0)
                ),
                "raw_mp_signal_count": 2 + index % 3,
                "parent_test_projection_dimension": 2 + index % 3,
                "sibling_projection_dimension": 2 + index % 3,
                "effective_independent_rows": 20 + index,
                "mp_threshold_rows": 20 + index,
                "eigenvalue_effective_rank": 1.5 + index / 30.0,
                "top_eigenvalue_share": 0.8 - index / 100.0,
                "selected_eigenvalue_mass_fraction": 0.5 + index / 80.0,
                "eigengap_at_sibling_projection_dimension": 1.1 + index / 20.0,
                "selected_eigenvalue_over_mp_upper_bound": 0.7 + index / 50.0,
                "selected_subspace_cos2": cos2,
                "selected_subspace_sin2": 1.0 - cos2,
                "selected_subspace_tan2": (1.0 - cos2) / cos2,
                "top_component_cos2": 0.01 + (index % 5) / 30.0,
                "max_component_cos2": 0.02 + (index % 7) / 25.0,
                "component_cos2_entropy": 0.2 + ((index * 7) % 11) / 40.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _multi_case_geometry_records() -> pd.DataFrame:
    first = _geometry_records()
    second = _geometry_records().copy()
    second["case_id"] = "case_b"
    second["replicate_index"] = second["replicate_index"] + 10
    second["selected_hierarchy_ratio"] = second["selected_hierarchy_ratio"] * 1.25
    second["log_selected_hierarchy_ratio"] = np.log(second["selected_hierarchy_ratio"])
    return pd.concat([first, second], ignore_index=True)


def test_relationships_report_descriptive_correlations() -> None:
    records = _geometry_records()

    relationships = evaluate_covariate_relationships(records)

    row = relationships[
        relationships["covariate"].eq("selected_subspace_cos2")
    ].iloc[0]
    assert row["relationship_status"] == "descriptive_unadjusted"
    assert row["covariate_block"] == "angular"
    assert row["n_pairs"] == 20
    assert float(row["spearman_rho"]) > 0.9


def test_block_models_report_in_sample_descriptive_fit() -> None:
    records = _geometry_records()

    models = evaluate_covariate_block_models(records)

    angular = models[models["covariate_block"].eq("angular")].iloc[0]
    assert angular["model_status"] == "descriptive_in_sample"
    assert float(angular["r_squared"]) >= 0.0
    assert angular["study_role"] == STUDY_ROLE


def test_candidate_equations_score_mean_and_tail_behavior() -> None:
    records = _geometry_records()

    equations = evaluate_candidate_equations(records)

    edge = equations[equations["equation_id"].eq("edge_action")].iloc[0]
    assert edge["equation_status"] == "descriptive_in_sample"
    assert float(edge["mean_log_ratio_r_squared"]) >= 0.0
    assert 0.0 <= float(edge["tail_auc_from_linear_score"]) <= 1.0
    assert float(edge["tail_event_rate"]) > 0.0


def test_candidate_equations_report_replicate_and_case_holdout() -> None:
    records = _multi_case_geometry_records()

    holdout = evaluate_candidate_equation_holdout(records, n_replicate_folds=2)

    assert set(holdout["split_strategy"]) == {
        "replicate_modulo",
        "leave_one_case_out",
    }
    edge = holdout[
        holdout["equation_id"].eq("edge_action")
        & holdout["split_strategy"].eq("replicate_modulo")
    ].iloc[0]
    assert edge["equation_status"] == "descriptive_holdout_replicate_modulo"
    assert float(edge["n_test_rows"]) > 0
    assert 0.0 <= float(edge["holdout_tail_auc_from_linear_score"]) <= 1.0


def test_case_summary_uses_selected_ratio_and_geometry_fields() -> None:
    summary = summarize_selected_geometry_by_case(_geometry_records())

    row = summary.iloc[0]
    assert row["case_id"] == "case"
    assert row["n_records"] == 20
    assert row["selected_hierarchy_ratio_q95"] > row["selected_hierarchy_ratio_median"]
    assert row["selected_subspace_cos2_mean"] > 0.0


def test_smoke_run_writes_geometry_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        outputs = run_selected_hierarchy_geometry_covariate_study(
            case_names=["gauss_null_large"],
            output_dir=output_dir,
            n_replicates=2,
            seed=11,
        )

        assert "selected_geometry_records" not in outputs
        assert (output_dir / "case_summary.csv").exists()
        assert (output_dir / "geometry_summary_by_case.csv").exists()
        assert (output_dir / "covariate_relationships.csv").exists()
        assert (output_dir / "covariate_block_models.csv").exists()
        assert (output_dir / "candidate_equations.csv").exists()
        assert (output_dir / "candidate_equation_holdout.csv").exists()
        assert (output_dir / "manifest.json").exists()


def test_smoke_run_can_write_optional_row_level_geometry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        outputs = run_selected_hierarchy_geometry_covariate_study(
            case_names=["gauss_null_large"],
            output_dir=output_dir,
            n_replicates=2,
            seed=11,
            write_selected_records=True,
        )

        assert "selected_geometry_records" in outputs
        assert (output_dir / "selected_geometry_records.csv").exists()


def test_relationships_require_declared_predictor_columns() -> None:
    with pytest.raises(KeyError, match="Missing predictor column"):
        evaluate_covariate_relationships(
            pd.DataFrame({"log_selected_hierarchy_ratio": [1.0, 2.0]}),
            covariates=(("angular", "selected_subspace_cos2"),),
        )
