from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.statistics.differential_statistic_validity_panel import (
    DifferentialStatisticValidityConfig,
    _projection_status,
    build_differential_validity_production_components,
    compute_fisher_geometry_summary,
    finite_difference_projected_quadratic_derivative,
    projected_quadratic_directional_derivative,
    run_differential_statistic_validity_panel,
)
from benchmarks.diagnostics.calibration.traversal.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from tree_break_selection.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)


def test_fisher_geometry_detects_bernoulli_boundary() -> None:
    feature_space = bernoulli_feature_space_from_columns(("a", "b", "c"))

    interior = compute_fisher_geometry_summary(
        np.array([0.5, 0.4, 0.6]),
        feature_space,
    )
    boundary = compute_fisher_geometry_summary(
        np.array([0.5, 0.0, 1.0]),
        feature_space,
    )

    assert interior["fisher_boundary_status"] == "interior"
    assert interior["fisher_variance_floor"] > 0.0
    assert boundary["fisher_boundary_status"] == "boundary_unstable"
    assert boundary["fisher_condition_number"] == pytest.approx(np.inf)


def test_fisher_geometry_detects_categorical_boundary() -> None:
    feature_space = FeatureSpace(
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

    interior = compute_fisher_geometry_summary(np.array([0.3, 0.3, 0.4]), feature_space)
    boundary = compute_fisher_geometry_summary(np.array([1.0, 0.0, 0.0]), feature_space)

    assert interior["fisher_boundary_status"] == "interior"
    assert boundary["fisher_boundary_status"] == "boundary_unstable"


def test_projected_quadratic_finite_difference_matches_analytic_derivative() -> None:
    z = np.array([1.0, -2.0, 0.5])
    projection = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    direction = np.array([0.25, -0.5, 0.75])

    analytic = projected_quadratic_directional_derivative(z, projection, direction)
    finite_difference = finite_difference_projected_quadratic_derivative(
        z,
        projection,
        direction,
        epsilon=1e-6,
    )

    assert finite_difference == pytest.approx(analytic, rel=1e-6, abs=1e-6)


def test_projection_status_marks_small_eigengap_as_unstable() -> None:
    status = _projection_status(
        eigengap=1e-12,
        instability_score=1e9,
        recomputed_delta_norm=0.0,
    )

    assert status == "projection_unstable_small_gap"


def test_fixed_subspace_candidate_remains_diagnostic_only() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "mode": "fixed_tree",
                "source_family": "binary_template",
                "df_bin": "df_1_2",
                "statistic_validity_status": "fixed_subspace_candidate",
            }
        ]
    )

    components = build_differential_validity_production_components(summary)
    rows = evaluate_production_admissibility_components(components)
    contract = summarize_production_admissibility_contracts(rows)

    assert contract.iloc[0]["production_decision"] == "diagnostic_only"


def test_run_differential_statistic_validity_panel_writes_outputs(tmp_path: Path) -> None:
    outputs = run_differential_statistic_validity_panel(
        DifferentialStatisticValidityConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            modes=("selected_tree",),
            edge_alphas=(0.001,),
            sibling_alpha=0.01,
            replicates=1,
            base_seed=20260613,
            finite_diff_directions=2,
            epsilon_scale=1e-4,
        )
    )

    assert set(outputs) == {
        "rows",
        "summary",
        "production_components",
        "production_summary",
        "manifest",
    }
    rows = pd.read_csv(tmp_path / "differential_statistic_validity_rows.csv")
    production_summary = pd.read_csv(tmp_path / "production_admissibility_summary.csv")

    assert not rows.empty
    assert rows["selection_derivative_status"].eq("nonsmooth_hamming_selection").all()
    assert "nonsmooth_selection_geometry" in set(rows["statistic_validity_status"])
    assert production_summary.iloc[0]["production_decision"] == "fail_closed_undefined"
