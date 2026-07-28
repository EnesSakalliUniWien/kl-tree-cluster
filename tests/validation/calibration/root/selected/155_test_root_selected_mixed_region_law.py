from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import root_selected_mixed_region_law as panel


def _root_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_discrete",
                "root_selected_region_law_status": ("discrete_tie_cell_geometry_required"),
                "root_sibling_selected_ratio": 4.0,
                "root_child_balance": 0.4,
                "root_child_construction_merge_count": 10,
                "root_child_min_merge_margin": 0.0,
                "root_child_near_active_merge_count": 6,
                "root_child_tied_minimum_merge_count": 6,
                "root_child_discrete_tie_cell_count": 6,
                "root_child_smooth_constraint_count": 0,
                "root_edge_path_radial_distance": 3.0,
                "root_edge_path_statistic_margin": 15.0,
                "root_edge_extra_parent_projection_energy": 2.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 1.5,
                "root_selected_eigenvalue_mass_fraction": 0.7,
                "root_raw_mp_signal_count": 2,
                "root_mp_threshold_rows": 10,
                "root_active_feature_count": 6,
                "root_full_eigenvalue_count": 6,
                "root_full_component_eigenvalues_json": "[4.0, 2.0, 1.0]",
                "root_projected_eigenvalues_json": "[4.0, 2.0]",
                "root_mp_upper_bound": 2.0,
            },
            {
                "case_id": "case_smooth",
                "root_selected_region_law_status": ("smooth_first_order_signed_distance_defined"),
                "root_sibling_selected_ratio": 2.0,
                "root_child_balance": 0.5,
                "root_child_construction_merge_count": 5,
                "root_child_min_merge_margin": 0.2,
                "root_child_near_active_merge_count": 0,
                "root_child_tied_minimum_merge_count": 0,
                "root_child_discrete_tie_cell_count": 0,
                "root_child_smooth_constraint_count": 5,
                "root_edge_path_radial_distance": 1.0,
                "root_edge_path_statistic_margin": 3.0,
                "root_edge_extra_parent_projection_energy": 0.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 0.5,
                "root_selected_eigenvalue_mass_fraction": 0.2,
                "root_raw_mp_signal_count": 0,
                "root_mp_threshold_rows": 5,
                "root_active_feature_count": 3,
                "root_full_eigenvalue_count": 3,
                "root_full_component_eigenvalues_json": "[1.0, 0.5, 0.25]",
                "root_projected_eigenvalues_json": "[1.0]",
                "root_mp_upper_bound": 1.5,
            },
            {
                "case_id": "case_mixed",
                "root_selected_region_law_status": ("mixed_smooth_and_nonsmooth_geometry"),
                "root_sibling_selected_ratio": 8.0,
                "root_child_balance": 0.3,
                "root_child_construction_merge_count": 8,
                "root_child_min_merge_margin": 0.0,
                "root_child_near_active_merge_count": 3,
                "root_child_tied_minimum_merge_count": 2,
                "root_child_discrete_tie_cell_count": 2,
                "root_child_smooth_constraint_count": 4,
                "root_edge_path_radial_distance": 4.0,
                "root_edge_path_statistic_margin": 31.0,
                "root_edge_extra_parent_projection_energy": 3.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
                "root_selected_eigenvalue_mass_fraction": 0.8,
                "root_raw_mp_signal_count": 3,
                "root_mp_threshold_rows": 8,
                "root_active_feature_count": 5,
                "root_full_eigenvalue_count": 5,
                "root_full_component_eigenvalues_json": "[5.0, 3.0, 1.0]",
                "root_projected_eigenvalues_json": "[5.0, 3.0]",
                "root_mp_upper_bound": 2.5,
            },
        ]
    )


def _tie_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_discrete",
                "root_tie_step_count": 4,
                "root_tie_step_fraction": 0.4,
                "root_tie_cell_log_burden": math.log(16.0),
                "root_tie_cell_mean_log_multiplicity": math.log(2.0),
                "root_tie_cell_geometric_mean_multiplicity": 2.0,
                "root_tie_rank_log_burden": math.log(4.0),
                "root_tie_rank_mean_fraction": 0.5,
                "root_tie_rank_median_fraction": 0.5,
            },
            {
                "case_id": "case_smooth",
                "root_tie_step_count": 0,
                "root_tie_step_fraction": 0.0,
                "root_tie_cell_log_burden": 0.0,
                "root_tie_cell_mean_log_multiplicity": 0.0,
                "root_tie_cell_geometric_mean_multiplicity": 1.0,
                "root_tie_rank_log_burden": 0.0,
                "root_tie_rank_mean_fraction": math.nan,
                "root_tie_rank_median_fraction": math.nan,
            },
            {
                "case_id": "case_mixed",
                "root_tie_step_count": 2,
                "root_tie_step_fraction": 0.25,
                "root_tie_cell_log_burden": math.log(8.0),
                "root_tie_cell_mean_log_multiplicity": math.log(2.0),
                "root_tie_cell_geometric_mean_multiplicity": 2.0,
                "root_tie_rank_log_burden": math.log(2.0),
                "root_tie_rank_mean_fraction": 0.75,
                "root_tie_rank_median_fraction": 0.75,
            },
        ]
    )


def _frontier_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_discrete",
                "data_role": "selected_null",
                "candidate_scope": "root_non_direct",
                "bandwidth_reference_reopens": True,
                "bandwidth_reference_direct_positive_reopens": True,
                "root_structural_proxy_pass": True,
                "hybrid_strict_support": False,
                "interpolation_best_case_required_tau_s_for_alpha": 12.0,
            },
            {
                "case_id": "case_discrete",
                "data_role": "signal",
                "candidate_scope": "root_non_direct",
                "bandwidth_reference_reopens": False,
                "bandwidth_reference_direct_positive_reopens": False,
                "root_structural_proxy_pass": True,
                "hybrid_strict_support": False,
                "interpolation_best_case_required_tau_s_for_alpha": 20.0,
            },
            {
                "case_id": "case_mixed",
                "data_role": "signal",
                "candidate_scope": "root_non_direct",
                "bandwidth_reference_reopens": False,
                "bandwidth_reference_direct_positive_reopens": False,
                "root_structural_proxy_pass": False,
                "hybrid_strict_support": False,
                "interpolation_best_case_required_tau_s_for_alpha": 30.0,
            },
        ]
    )


def test_mixed_law_classifies_discrete_root_and_bandwidth_reopen() -> None:
    rows = panel.build_root_selected_mixed_region_law_rows(
        root_summary=_root_summary(),
        tie_cell_burden_rows=_tie_rows(),
        topology_frontier_rows=_frontier_rows(),
    )
    row = rows[rows["case_id"].eq("case_discrete")].iloc[0]

    assert row["root_mixed_region_component"] == "discrete_tie_rank_region"
    assert row["root_calibration_status"] == ("blocked_until_discrete_tie_rank_null_calibrated")
    assert row["root_continuous_margin_status"] == ("active_or_numerically_tied_margin")
    assert row["root_tie_rank_status"] == "selected_tie_rank_coordinate_observed"
    assert row["root_tie_rank_to_tie_burden_fraction"] == pytest.approx(0.5)
    assert row["root_tie_rank_shortfall_log_burden"] == pytest.approx(math.log(4.0))
    assert row["root_bandwidth_reopen_count"] == 1
    assert row["root_frontier_data_roles"] == "selected_null,signal"
    assert row["root_bandwidth_locality_status"] == ("bandwidth_reopens_without_root_law_support")
    assert row["root_mixed_law_inference_status"] == (
        "fail_closed_discrete_root_law_missing_bandwidth_descriptive"
    )
    assert row["root_rank_fraction_edge_margin_product"] == pytest.approx(0.5 * math.log1p(15.0))
    assert row["root_rank_fraction_spectral_product"] == pytest.approx(0.75)
    assert row["root_active_feature_count"] == 6
    assert row["root_full_eigenvalue_count"] == 6
    assert row["root_full_component_eigenvalues_json"] == "[4.0, 2.0, 1.0]"
    assert row["root_projected_eigenvalues_json"] == "[4.0, 2.0]"
    assert row["root_mp_upper_bound"] == pytest.approx(2.0)


def test_mixed_law_handles_smooth_and_mixed_regions() -> None:
    rows = panel.build_root_selected_mixed_region_law_rows(
        root_summary=_root_summary(),
        tie_cell_burden_rows=_tie_rows(),
        topology_frontier_rows=_frontier_rows(),
    )
    smooth = rows[rows["case_id"].eq("case_smooth")].iloc[0]
    mixed = rows[rows["case_id"].eq("case_mixed")].iloc[0]

    assert smooth["root_mixed_region_component"] == "smooth_margin_region"
    assert smooth["root_calibration_status"] == ("smooth_first_order_candidate_diagnostic_only")
    assert smooth["root_continuous_margin_status"] == "positive_margin_observed"
    assert smooth["root_discrete_tie_status"] == "no_discrete_tie_cell_observed"
    assert smooth["root_mixed_law_inference_status"] == (
        "smooth_margin_diagnostic_only_no_production_p_value"
    )

    assert mixed["root_mixed_region_component"] == ("mixed_smooth_and_discrete_region")
    assert mixed["root_calibration_status"] == ("blocked_until_discrete_tie_rank_null_calibrated")
    assert mixed["root_bandwidth_locality_status"] == ("bandwidth_does_not_reopen_root_rows")


def test_mixed_law_counts_direct_measurable_root_frontier_rows() -> None:
    frontier = pd.DataFrame.from_records(
        [
            {
                "case_id": "case_smooth",
                "data_role": "diagnostic_proposal",
                "candidate_scope": "direct_measurable",
                "parent_id": "",
                "depth": 0,
                "bandwidth_reference_reopens": False,
                "bandwidth_reference_direct_positive_reopens": False,
                "root_structural_proxy_pass": False,
                "hybrid_strict_support": False,
                "interpolation_best_case_required_tau_s_for_alpha": 40.0,
            }
        ]
    )

    rows = panel.build_root_selected_mixed_region_law_rows(
        root_summary=_root_summary(),
        tie_cell_burden_rows=_tie_rows(),
        topology_frontier_rows=frontier,
    )
    smooth = rows[rows["case_id"].eq("case_smooth")].iloc[0]

    assert smooth["root_frontier_row_count"] == 1
    assert smooth["root_frontier_data_roles"] == "diagnostic_proposal"
    assert smooth["root_bandwidth_reopen_count"] == 0
    assert smooth["root_bandwidth_locality_status"] == ("bandwidth_does_not_reopen_root_rows")


def test_mixed_law_relationships_and_summary_are_reported() -> None:
    rows = panel.build_root_selected_mixed_region_law_rows(
        root_summary=_root_summary(),
        tie_cell_burden_rows=_tie_rows(),
        topology_frontier_rows=_frontier_rows(),
    )
    relationships = panel.summarize_mixed_region_relationships(rows)
    summary = panel.summarize_mixed_region_law_rows(rows)

    assert set(relationships["covariate"]) == set(panel.RELATIONSHIP_COVARIATES)
    evaluated = relationships[
        relationships["covariate"].eq("root_edge_path_statistic_margin")
    ].iloc[0]
    assert evaluated["relationship_status"] == "evaluated"
    assert evaluated["valid_pair_count"] == 3
    assert set(summary["root_mixed_region_component"]) == {
        "discrete_tie_rank_region",
        "mixed_smooth_and_discrete_region",
        "smooth_margin_region",
    }


def test_mixed_law_writes_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    tie_path = tmp_path / "tie.csv"
    frontier_path = tmp_path / "frontier.csv"
    _root_summary().to_csv(summary_path, index=False)
    _tie_rows().to_csv(tie_path, index=False)
    _frontier_rows().to_csv(frontier_path, index=False)

    outputs = panel.run_root_selected_mixed_region_law(
        panel.RootSelectedMixedRegionLawConfig(
            output_dir=tmp_path / "out",
            root_selected_region_summary_path=summary_path,
            tie_cell_burden_rows_path=tie_path,
            topology_frontier_rows_path=frontier_path,
        )
    )

    assert set(outputs) == {"rows", "relationships", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["relationships"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
