from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_kernel_spectral_tail_law_panel as panel,
)


def _joined_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "target_supported",
                "data_role": "observed_target",
                "calibration_role": "observed_target_not_null_support",
                "proposal_family": "observed_target",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 100.0,
                "root_tie_rank_median_fraction": 0.8,
                "root_edge_path_statistic_margin": 120.0,
                "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
                "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
                "root_active_feature_count": 40.0,
                "root_effective_independent_rows": 100.0,
                "root_mp_upper_bound": 2.0,
                "root_child_balance": 0.42,
                "root_child_construction_merge_count": 100,
                "root_child_tied_minimum_merge_count": 60,
                "root_child_discrete_tie_cell_count": 60,
                "root_child_smooth_constraint_count": 0,
            },
            {
                "case_id": "target_unsupported",
                "data_role": "observed_target",
                "calibration_role": "observed_target_not_null_support",
                "proposal_family": "observed_target",
                "root_mixed_region_component": "different_region",
                "root_sibling_selected_ratio": 100.0,
                "root_tie_rank_median_fraction": 0.8,
                "root_edge_path_statistic_margin": 120.0,
                "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
                "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
                "root_active_feature_count": 40.0,
                "root_effective_independent_rows": 100.0,
                "root_mp_upper_bound": 2.0,
                "root_child_balance": 0.42,
                "root_child_construction_merge_count": 100,
                "root_child_tied_minimum_merge_count": 60,
                "root_child_discrete_tie_cell_count": 60,
                "root_child_smooth_constraint_count": 0,
            },
            {
                "case_id": "support_high",
                "data_role": "external_selected_null",
                "calibration_role": "external_null_support",
                "proposal_family": "external",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 110.0,
                "root_tie_rank_median_fraction": 0.82,
                "root_edge_path_statistic_margin": 125.0,
                "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
                "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
                "root_active_feature_count": 42.0,
                "root_effective_independent_rows": 105.0,
                "root_mp_upper_bound": 2.1,
                "importance_log_weight": 0.0,
                "root_child_balance": 0.43,
                "root_child_construction_merge_count": 100,
                "root_child_tied_minimum_merge_count": 61,
                "root_child_discrete_tie_cell_count": 61,
                "root_child_smooth_constraint_count": 0,
            },
            {
                "case_id": "support_low",
                "data_role": "external_selected_null",
                "calibration_role": "external_null_support",
                "proposal_family": "external",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 80.0,
                "root_tie_rank_median_fraction": 0.78,
                "root_edge_path_statistic_margin": 115.0,
                "root_bandwidth_reopen_band": "bandwidth_no_root_reopen",
                "root_selected_eigenvalue_over_mp_upper_bound": 1.1,
                "root_active_feature_count": 38.0,
                "root_effective_independent_rows": 95.0,
                "root_mp_upper_bound": 1.9,
                "importance_log_weight": 0.0,
                "root_child_balance": 0.10,
                "root_child_construction_merge_count": 200,
                "root_child_tied_minimum_merge_count": 20,
                "root_child_discrete_tie_cell_count": 20,
                "root_child_smooth_constraint_count": 0,
            },
            {
                "case_id": "diagnostic_neighbor",
                "data_role": "diagnostic_proposal",
                "calibration_role": "diagnostic_proposal_not_null_support",
                "proposal_family": "diagnostic",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 100.0,
                "root_tie_rank_median_fraction": 0.8,
                "root_edge_path_statistic_margin": 120.0,
                "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
                "root_selected_eigenvalue_over_mp_upper_bound": 5.0,
                "root_active_feature_count": 40.0,
                "root_effective_independent_rows": 100.0,
                "root_mp_upper_bound": 2.0,
                "root_child_balance": 0.42,
                "root_child_construction_merge_count": 100,
                "root_child_tied_minimum_merge_count": 60,
                "root_child_discrete_tie_cell_count": 60,
                "root_child_smooth_constraint_count": 0,
            },
        ]
    )


def _strict_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "target_case_id": "target_supported",
                "selected_null_support_count": 0,
                "conservative_spectral_tail_p_value": float("nan"),
                "root_tail_inference_status": (
                    "fail_closed_selected_root_spectral_tail_support_missing"
                ),
            },
            {
                "target_case_id": "target_unsupported",
                "selected_null_support_count": 0,
                "conservative_spectral_tail_p_value": float("nan"),
                "root_tail_inference_status": (
                    "fail_closed_selected_root_spectral_tail_support_missing"
                ),
            },
        ]
    )


def _deformed_targets() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "target_case_id": "target_supported",
                "s_root_deformed_excess_log": 0.5,
            },
            {
                "target_case_id": "target_unsupported",
                "s_root_deformed_excess_log": 0.5,
            },
        ]
    )


def _deformed_support() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {"case_id": "support_high", "s_root_deformed_excess_log": 0.7},
            {"case_id": "support_low", "s_root_deformed_excess_log": 0.0},
        ]
    )


def _observed_root_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "target_supported",
                "root_child_balance": 0.42,
                "root_child_construction_merge_count": 100,
                "root_child_tied_minimum_merge_count": 60,
                "root_child_discrete_tie_cell_count": 60,
                "root_child_smooth_constraint_count": 0,
            }
        ]
    )


def test_kernel_spectral_tail_adds_support_without_using_diagnostic_neighbors(
    tmp_path: Path,
) -> None:
    config = panel.RootSelectedKernelSpectralTailLawConfig(
        output_dir=tmp_path,
        min_kernel_effective_sample_size=1.0,
        min_topology_effective_sample_size=1.0,
        max_topology_weight_share=1.0,
    )
    rows = panel.build_kernel_spectral_tail_rows(
        joined_feasibility_rows=_joined_rows(),
        strict_tail_rows=_strict_rows(),
        deformed_mp_edge_rows=_deformed_targets(),
        deformed_mp_edge_support_rows=_deformed_support(),
        config=config,
    )

    by_case = rows.set_index("target_case_id")
    supported = by_case.loc["target_supported"]
    assert supported["kernel_candidate_decision"] == ("candidate_tail_available_diagnostic_only")
    assert supported["admissible_kernel_support_count"] == 2
    assert supported["non_support_neighbor_excluded_count"] == 1
    assert supported["kernel_positive_s_h_u_support_count"] == 1
    assert supported["kernel_conservative_tail_p_value"] < 1.0
    assert supported["topology_kernel_candidate_decision"] == (
        "topology_candidate_tail_available_diagnostic_only"
    )
    assert supported["topology_admissible_support_count"] == 1
    assert supported["topology_positive_s_h_u_support_count"] == 1
    assert supported["topology_nearest_support_case_id"] == "support_high"

    unsupported = by_case.loc["target_unsupported"]
    assert unsupported["kernel_candidate_decision"] == ("fail_closed_kernel_weight_support_missing")
    assert unsupported["topology_kernel_candidate_decision"] == (
        "fail_closed_topology_support_missing"
    )


def test_kernel_spectral_tail_summary_and_writes_outputs(tmp_path: Path) -> None:
    joined_path = tmp_path / "joined.csv"
    strict_path = tmp_path / "strict.csv"
    deformed_path = tmp_path / "deformed.csv"
    support_path = tmp_path / "support.csv"
    _joined_rows().to_csv(joined_path, index=False)
    _strict_rows().to_csv(strict_path, index=False)
    _deformed_targets().to_csv(deformed_path, index=False)
    _deformed_support().to_csv(support_path, index=False)

    outputs = panel.run_kernel_spectral_tail_law_panel(
        panel.RootSelectedKernelSpectralTailLawConfig(
            output_dir=tmp_path / "out",
            joined_feasibility_rows_path=joined_path,
            strict_tail_rows_path=strict_path,
            deformed_mp_edge_rows_path=deformed_path,
            deformed_mp_edge_support_rows_path=support_path,
            min_topology_effective_sample_size=1.0,
            max_topology_weight_share=1.0,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert all(path.exists() for path in outputs.values())
    summary = pd.read_csv(outputs["summary"]).iloc[0]
    assert summary["row_count"] == 2
    assert summary["strict_fail_closed_kernel_available_count"] == 1
    assert summary["topology_strict_fail_closed_kernel_available_count"] == 1
    assert summary["summary_status"] == ("topology_kernel_adds_candidate_support_diagnostic_only")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["generated_by"] == panel.GENERATED_BY


def test_positive_target_without_nonzero_support_fails_closed(tmp_path: Path) -> None:
    joined = _joined_rows()
    support = _deformed_support()
    support["s_root_deformed_excess_log"] = 0.0
    config = panel.RootSelectedKernelSpectralTailLawConfig(
        output_dir=tmp_path,
        min_topology_effective_sample_size=1.0,
        max_topology_weight_share=1.0,
    )
    rows = panel.build_kernel_spectral_tail_rows(
        joined_feasibility_rows=joined,
        strict_tail_rows=_strict_rows(),
        deformed_mp_edge_rows=_deformed_targets(),
        deformed_mp_edge_support_rows=support,
        config=config,
    )
    supported = rows.set_index("target_case_id").loc["target_supported"]
    assert supported["kernel_candidate_decision"] == (
        "fail_closed_kernel_nonzero_s_h_u_support_missing"
    )
    assert supported["topology_kernel_candidate_decision"] == (
        "fail_closed_topology_nonzero_s_h_u_support_missing"
    )
    assert pd.isna(supported["kernel_conservative_tail_p_value"])


def test_observed_root_summary_enriches_missing_target_topology(
    tmp_path: Path,
) -> None:
    joined = _joined_rows()
    topology_columns = [
        "root_child_balance",
        "root_child_construction_merge_count",
        "root_child_tied_minimum_merge_count",
        "root_child_discrete_tie_cell_count",
        "root_child_smooth_constraint_count",
    ]
    target_mask = joined["case_id"].eq("target_supported")
    joined.loc[target_mask, topology_columns] = float("nan")

    rows = panel.build_kernel_spectral_tail_rows(
        joined_feasibility_rows=joined,
        strict_tail_rows=_strict_rows(),
        deformed_mp_edge_rows=_deformed_targets(),
        deformed_mp_edge_support_rows=_deformed_support(),
        observed_root_summary_rows=_observed_root_summary(),
        config=panel.RootSelectedKernelSpectralTailLawConfig(
            output_dir=tmp_path,
            min_topology_effective_sample_size=1.0,
            max_topology_weight_share=1.0,
        ),
    )

    supported = rows.set_index("target_case_id").loc["target_supported"]
    assert supported["topology_kernel_candidate_decision"] == (
        "topology_candidate_tail_available_diagnostic_only"
    )
    assert "missing" not in supported["target_root_topology_signature"]
