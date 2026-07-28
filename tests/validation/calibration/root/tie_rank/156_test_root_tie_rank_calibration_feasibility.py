from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_calibration_feasibility as panel,
)


def _mixed_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_a",
                "data_role": "selected_null",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 5.0,
                "root_tie_rank_median_fraction": 0.80,
                "root_edge_path_statistic_margin": 1500.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.0,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "null_b",
                "data_role": "selected_null",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 7.0,
                "root_tie_rank_median_fraction": 0.82,
                "root_edge_path_statistic_margin": 1200.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.5,
                "root_bandwidth_reopen_count": 2,
            },
            {
                "case_id": "null_c",
                "data_role": "selected_null",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 9.0,
                "root_tie_rank_median_fraction": 0.85,
                "root_edge_path_statistic_margin": 1500.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.8,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "signal_a",
                "data_role": "signal",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 8.0,
                "root_tie_rank_median_fraction": 0.83,
                "root_edge_path_statistic_margin": 1300.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.2,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "unlabeled_a",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 3.0,
                "root_tie_rank_median_fraction": 0.95,
                "root_edge_path_statistic_margin": 80.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 1.5,
                "root_bandwidth_reopen_count": 0,
            },
        ]
    )


def test_required_null_counts_use_plus_one_resolution_and_tail_precision() -> None:
    assert panel.alpha_resolution_required_null_count(0.01) == 99
    assert (
        panel.tail_precision_required_null_count(
            target_alpha=0.01,
            relative_se_target=0.25,
        )
        == 1584
    )


def test_feasibility_rows_count_only_selected_null_support() -> None:
    rows = panel.build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=_mixed_rows(),
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    signal = rows[rows["case_id"].eq("signal_a")].iloc[0]
    unlabeled = rows[rows["case_id"].eq("unlabeled_a")].iloc[0]

    assert signal["stratum_calibration_null_support_count"] == 3
    assert signal["stratum_calibration_null_exceedance_count"] == 1
    assert signal["empirical_conservative_tail_p_value"] == pytest.approx(0.5)
    assert signal["calibration_feasibility_status"] == ("tail_precision_ready_diagnostic_only")

    assert unlabeled["calibration_role"] == "unlabeled_diagnostic_not_null_support"
    assert unlabeled["stratum_calibration_null_support_count"] == 0
    assert pd.isna(unlabeled["empirical_conservative_tail_p_value"])
    assert unlabeled["calibration_feasibility_status"] == ("external_null_support_missing")


def test_feasibility_strata_and_summary_report_missing_support() -> None:
    rows = panel.build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=_mixed_rows(),
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    strata = panel.summarize_root_tie_rank_calibration_strata(rows)
    summary = panel.summarize_root_tie_rank_calibration_feasibility(rows, strata)

    assert strata.shape[0] == 2
    missing = strata[strata["stratum_feasibility_status"].eq("external_null_support_missing")].iloc[
        0
    ]
    assert missing["additional_null_count_for_alpha_resolution"] == 3
    assert missing["additional_null_count_for_tail_precision"] == 3

    overall = summary.iloc[0]
    assert overall["row_count"] == 5
    assert overall["stratum_count"] == 2
    assert overall["calibration_null_support_count"] == 3
    assert overall["alpha_resolution_ready_stratum_count"] == 1
    assert overall["tail_precision_ready_stratum_count"] == 1
    assert overall["summary_status"] == "partial_alpha_resolution_support"


def test_feasibility_summary_distinguishes_subresolution_null_support() -> None:
    rows = panel.build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=_mixed_rows().iloc[:2],
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    strata = panel.summarize_root_tie_rank_calibration_strata(rows)
    summary = panel.summarize_root_tie_rank_calibration_feasibility(rows, strata)

    assert summary.iloc[0]["calibration_null_support_count"] == 2
    assert summary.iloc[0]["alpha_resolution_ready_stratum_count"] == 0
    assert summary.iloc[0]["summary_status"] == (
        "calibration_null_support_observed_below_alpha_resolution"
    )


def test_unjoined_bandwidth_frontier_is_missing_not_no_reopen() -> None:
    mixed = pd.DataFrame.from_records(
        [
            {
                "case_id": "proposal_unmeasured",
                "data_role": "diagnostic_proposal",
                "calibration_role": "diagnostic_proposal_not_null_support",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 12.0,
                "root_tie_rank_median_fraction": 0.84,
                "root_edge_path_statistic_margin": 1600.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.8,
                "root_bandwidth_reopen_count": 0,
                "root_bandwidth_locality_status": "topology_frontier_not_joined",
            }
        ]
    )
    rows = panel.build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=mixed,
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    row = rows.iloc[0]

    assert row["root_bandwidth_reopen_band"] == "bandwidth_reopen_missing"
    assert "bandwidth=bandwidth_reopen_missing" in row["root_conditioning_stratum_key"]
    assert pd.isna(row["root_bandwidth_reopen_count"])


def test_feasibility_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "mixed.csv"
    _mixed_rows().to_csv(input_path, index=False)

    outputs = panel.run_root_tie_rank_calibration_feasibility(
        panel.RootTieRankCalibrationFeasibilityConfig(
            output_dir=tmp_path / "out",
            mixed_region_rows_path=input_path,
            target_alpha=0.25,
            relative_se_target=1.0,
        )
    )

    assert set(outputs) == {"rows", "strata", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["strata"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
