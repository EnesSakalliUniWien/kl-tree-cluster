from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.math_trace.barycentric_action import (
    action_budget,
    barycentric_contrast_residuals,
    z_identity_residuals,
)
from benchmarks.diagnostics.math_trace.path_conditioned_barycentric_action import (
    run_path_conditioned_barycentric_action_diagnostic,
)


def test_barycentric_contrast_residuals_are_zero_for_exact_parent() -> None:
    left = np.array([1.0, 3.0, 5.0])
    right = np.array([5.0, 1.0, 1.0])
    parent = (2 * left + 6 * right) / 8

    residuals = barycentric_contrast_residuals(
        theta_parent=parent,
        theta_left=left,
        theta_right=right,
        left_size=2,
        right_size=6,
    )

    assert residuals["status"] == "ok"
    assert residuals["parent_barycenter_residual_norm"] < 1e-12
    assert residuals["left_contrast_residual_norm"] < 1e-12
    assert residuals["right_contrast_residual_norm"] < 1e-12
    assert math.isclose(residuals["beta"], 0.25)


def test_z_identity_residuals_compare_signed_vectors() -> None:
    sibling = np.array([1.0, -2.0, 0.5])
    result = z_identity_residuals(
        z_edge_left=sibling + 1e-10,
        z_edge_right=-sibling - 2e-10,
        z_sibling=sibling,
    )

    assert result["status"] == "ok"
    assert result["left_z_identity_residual_norm"] < 1e-8
    assert result["right_z_identity_residual_norm"] < 1e-8


def test_action_budget_matches_parallel_axis_split_energy() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ]
    )
    left_indices = np.array([0, 1])
    right_indices = np.array([2, 3])

    result = action_budget(
        coords=coords,
        left_indices=left_indices,
        right_indices=right_indices,
    )

    assert result["status"] == "ok"
    assert math.isclose(result["split_action"], 16.0)
    assert math.isclose(result["parent_inertia"], 20.0)
    assert math.isclose(result["action_fraction"], 0.8)


def test_cached_kak_trace_writes_reports(tmp_path: Path) -> None:
    geometry = pd.DataFrame(
        {
            "weighting": ["binary", "binary", "tfidf", "tfidf"],
            "block_name": ["b1", "b1", "b2", "b2"],
            "parent_size": [4, 6, 5, 7],
            "balance_fraction": [0.5, 0.33, 0.4, 0.5],
            "sibling_separation_to_parent_radius_ratio": [2.0, 1.5, 1.0, 2.5],
            "sibling_separation_to_child_radius_ratio": [1.8, 1.2, 0.8, 2.0],
            "parent_centroid_norm": [1.0, 2.0, 1.5, 2.5],
            "parent_centroid_angle_to_leading_axis_deg": [80.0, 70.0, 20.0, 80.0],
            "parent_centroid_independent_fraction": [0.95, 0.9, 0.2, 0.95],
            "abs_sibling_common_axis_mean_delta": [0.01, 0.08, 0.02, 0.10],
            "parent_dominant_cluster_fraction": [1.0, 0.5, 1.0, 0.4],
            "parent_cluster_count": [1, 2, 1, 3],
        }
    )
    validation = pd.DataFrame(
        {
            "model_id": ["radius_angle_action", "size_balance_only"],
            "holdout_run_id": ["binary__b1", "binary__b1"],
            "n_train": [2, 2],
            "n_test": [2, 2],
            "test_pure_rate": [0.5, 0.5],
            "auc_for_pure_fragment": [0.75, 0.5],
            "study_role": [
                "diagnostic_kak_traversal_fragmentation_not_calibration",
                "diagnostic_kak_traversal_fragmentation_not_calibration",
            ],
        }
    )
    full_results = pd.DataFrame(
        {
            "case_id": ["case_a", "case_b"],
            "method": ["kl", "kl"],
            "status": ["ok", "skip"],
            "ari": [0.9, 0.0],
            "error": ["", "Cannot fit sibling inflation model"],
        }
    )
    geometry_path = tmp_path / "geometry.csv"
    validation_path = tmp_path / "validation.csv"
    full_path = tmp_path / "full.csv"
    output_dir = tmp_path / "out"
    geometry.to_csv(geometry_path, index=False)
    validation.to_csv(validation_path, index=False)
    full_results.to_csv(full_path, index=False)

    summary = run_path_conditioned_barycentric_action_diagnostic(
        output_dir=output_dir,
        kak_internal_geometry_csv=geometry_path,
        kak_fragmentation_validation_csv=validation_path,
        benchmark_comparison_csv=full_path,
    )

    assert summary["schema_version"] == "path_conditioned_barycentric_action/v1"
    assert summary["n_internal_geometry_rows"] == 4
    assert (output_dir / "kak_radius_angle_action_summary.csv").exists()
    assert (output_dir / "kak_radius_angle_action_annotations.csv").exists()
    assert (output_dir / "action_budget_guard_panel.csv").exists()
    assert (output_dir / "action_budget_guard_utility_curve.csv").exists()
    assert (output_dir / "missing_equation_candidate_panel.csv").exists()
    assert (output_dir / "path_conditioned_barycentric_action_report.md").exists()

    annotations = pd.read_csv(output_dir / "kak_radius_angle_action_annotations.csv")
    guard_panel = pd.read_csv(output_dir / "action_budget_guard_panel.csv")
    guard_utility = pd.read_csv(output_dir / "action_budget_guard_utility_curve.csv")
    assert {"is_null_context", "is_signal_context", "action_budget_proxy"}.issubset(
        annotations.columns
    )
    assert annotations["is_null_context"].any()
    assert annotations["is_signal_context"].any()
    assert {
        "n_pure_flagged",
        "n_mixed_flagged",
        "pure_fragment_flag_rate",
        "mixed_context_flag_rate",
    }.issubset(guard_panel.columns)
    assert {
        "mixed_context_cost_ratio",
        "net_utility",
        "net_utility_per_row",
        "break_even_mixed_context_cost_ratio",
    }.issubset(guard_utility.columns)
    assert guard_utility["utility_positive"].any()
