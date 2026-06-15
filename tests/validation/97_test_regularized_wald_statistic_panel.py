from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.diagnostics.calibration.regularized_wald_statistic_panel import (
    RegularizedWaldStatisticConfig,
    build_regularized_wald_production_components,
    compute_regularized_sibling_wald_variant,
    regularize_feature_distribution,
    run_regularized_wald_statistic_panel,
    summarize_regularized_wald_rows,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)


def test_jeffreys_bernoulli_smoothing_moves_boundary_probabilities_inside() -> None:
    feature_space = bernoulli_feature_space_from_columns(("a", "b", "c"))

    smoothed = regularize_feature_distribution(
        np.array([0.0, 1.0, 0.5]),
        sample_size=4,
        feature_space=feature_space,
        smoothing_rule="jeffreys",
    )

    assert smoothed.tolist() == pytest.approx([0.1, 0.9, 0.5])
    assert np.all(smoothed > 0.0)
    assert np.all(smoothed < 1.0)


def _three_category_feature_space() -> FeatureSpace:
    return FeatureSpace(
        column_names=("F0_c0", "F0_c1", "F0_c2"),
        blocks=(
            FeatureBlock(
                name="F0",
                family="categorical",
                column_indices=(0, 1, 2),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=2,
            ),
        ),
    )


def test_dirichlet_categorical_smoothing_preserves_simplex() -> None:
    feature_space = _three_category_feature_space()

    smoothed = regularize_feature_distribution(
        np.array([1.0, 0.0, 0.0]),
        sample_size=5,
        feature_space=feature_space,
        smoothing_rule="dirichlet_1",
    )

    assert smoothed.tolist() == pytest.approx([0.75, 0.125, 0.125])
    assert smoothed.sum() == pytest.approx(1.0)
    assert np.all(smoothed > 0.0)


def test_regularized_sibling_wald_variant_reduces_boundary_instability() -> None:
    feature_space = bernoulli_feature_space_from_columns(("a", "b"))
    projection = np.eye(2)
    eigenvalues = np.array([1.0, 1.0])

    plugin = compute_regularized_sibling_wald_variant(
        left_distribution=np.array([1.0, 0.0]),
        right_distribution=np.array([0.0, 0.0]),
        left_sample_size=2,
        right_sample_size=2,
        parent_projection=projection,
        parent_eigenvalues=eigenvalues,
        projection_dimension=2,
        feature_space=feature_space,
        smoothing_rule="plugin",
        root_distribution=np.array([0.5, 0.5]),
    )
    jeffreys = compute_regularized_sibling_wald_variant(
        left_distribution=np.array([1.0, 0.0]),
        right_distribution=np.array([0.0, 0.0]),
        left_sample_size=2,
        right_sample_size=2,
        parent_projection=projection,
        parent_eigenvalues=eigenvalues,
        projection_dimension=2,
        feature_space=feature_space,
        smoothing_rule="jeffreys",
        root_distribution=np.array([0.5, 0.5]),
    )

    assert plugin["boundary_status"] == "boundary_unstable"
    assert jeffreys["boundary_status"] == "interior"
    assert np.isfinite(jeffreys["test_statistic"])
    assert 0.0 <= jeffreys["p_value"] <= 1.0


def test_regularized_wald_summary_marks_fixed_tree_candidate_diagnostic_only() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "mode": "fixed_tree",
                "source_family": "binary_template",
                "smoothing_rule": "jeffreys",
                "sibling_df": 2.0,
                "df_bin": "df_1_2",
                "boundary_status": "interior",
                "test_statistic": 1.0,
                "p_value": 0.50,
                "tail_reject_at_0.05": False,
            },
            {
                "case_id": "unit",
                "mode": "fixed_tree",
                "source_family": "binary_template",
                "smoothing_rule": "jeffreys",
                "sibling_df": 2.0,
                "df_bin": "df_1_2",
                "boundary_status": "interior",
                "test_statistic": 2.0,
                "p_value": 0.10,
                "tail_reject_at_0.05": False,
            },
        ]
    )

    summary = summarize_regularized_wald_rows(rows, min_rows=2)
    components = build_regularized_wald_production_components(summary)
    contract_rows = evaluate_production_admissibility_components(components)
    contract = summarize_production_admissibility_contracts(contract_rows)

    assert summary.iloc[0]["regularized_wald_status"] == "regularized_fixed_tree_candidate"
    assert contract.iloc[0]["production_decision"] == "diagnostic_only"


def test_run_regularized_wald_statistic_panel_writes_outputs(tmp_path: Path) -> None:
    outputs = run_regularized_wald_statistic_panel(
        RegularizedWaldStatisticConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            modes=("fixed_tree",),
            edge_alphas=(0.001,),
            sibling_alpha=0.01,
            replicates=1,
            base_seed=20260613,
        )
    )

    assert set(outputs) == {
        "rows",
        "summary",
        "production_components",
        "production_summary",
        "manifest",
    }
    rows = pd.read_csv(tmp_path / "regularized_wald_statistic_rows.csv")
    summary = pd.read_csv(tmp_path / "regularized_wald_statistic_summary.csv")
    production_summary = pd.read_csv(tmp_path / "production_admissibility_summary.csv")

    assert {"plugin", "jeffreys", "dirichlet_1", "root_shrink_0.1"} <= set(
        rows["smoothing_rule"]
    )
    assert not summary.empty
    assert production_summary["production_decision"].isin(
        {"diagnostic_only", "fail_closed_undefined", "production_admissible"}
    ).all()
