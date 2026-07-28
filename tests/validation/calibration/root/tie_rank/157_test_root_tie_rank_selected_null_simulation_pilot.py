from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_selected_null_simulation_pilot as panel,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_calibration_feasibility import (
    build_root_tie_rank_calibration_feasibility_rows,
)


def _mixed_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "target",
                "data_role": "observed_target",
                "calibration_role": "observed_target_not_null_support",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 10.0,
                "root_tie_rank_median_fraction": 0.82,
                "root_edge_path_statistic_margin": 1500.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.0,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "null_low",
                "data_role": "selected_null",
                "calibration_role": "selected_null_candidate_support",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 5.0,
                "root_tie_rank_median_fraction": 0.81,
                "root_edge_path_statistic_margin": 1200.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.5,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "null_high",
                "data_role": "selected_null",
                "calibration_role": "selected_null_candidate_support",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 12.0,
                "root_tie_rank_median_fraction": 0.84,
                "root_edge_path_statistic_margin": 1600.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.8,
                "root_bandwidth_reopen_count": 2,
            },
        ]
    )


def test_binary_null_probability_matches_sparse_template_marginal_mean() -> None:
    assert panel.binary_null_probability_for_case(
        {"n_clusters": 4, "feature_sparsity": 0.05}
    ) == pytest.approx(0.275)
    assert panel.binary_null_probability_for_case(
        {"n_clusters": 6, "feature_sparsity": 0.05}
    ) == pytest.approx(0.2)
    assert panel.binary_null_probability_for_case({}) == pytest.approx(0.5)
    assert panel.binary_null_probability_for_case(
        {"null_feature_probability": 0.01}
    ) == pytest.approx(0.01)


def test_target_support_counts_null_rows_in_matching_conditioning_stratum() -> None:
    feasibility = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=_mixed_rows(),
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    support = panel.build_target_support_rows(
        observed_case_ids=("target",),
        combined_feasibility_rows=feasibility,
    )
    target = support.iloc[0]

    assert target["stratum_calibration_null_support_count"] == 2
    assert target["stratum_calibration_null_exceedance_count"] == 1
    assert target["empirical_conservative_tail_p_value"] == pytest.approx(2.0 / 3.0)
    assert target["calibration_feasibility_status"] == (
        "insufficient_null_support_for_alpha_resolution"
    )


def test_simulation_summary_reports_target_support_gap() -> None:
    feasibility = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=_mixed_rows(),
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    support = panel.build_target_support_rows(
        observed_case_ids=("target",),
        combined_feasibility_rows=feasibility,
    )
    null_mixed = pd.DataFrame.from_records(
        [
            {"case_id": "null_low", "root_conditioning_stratum_key": "same"},
            {"case_id": "null_high", "root_conditioning_stratum_key": "same"},
        ]
    )
    summary = panel.summarize_selected_null_simulation(
        base_case_count=1,
        replicates_per_case=2,
        root_rows=pd.DataFrame.from_records([{"case_id": "null_low"}, {"case_id": "null_high"}]),
        failures=pd.DataFrame(),
        target_support=support,
        null_mixed_rows=null_mixed,
    ).iloc[0]

    assert summary["attempted_null_replicates"] == 2
    assert summary["successful_null_replicates"] == 2
    assert summary["target_strata_with_null_support_count"] == 1
    assert summary["target_strata_alpha_resolution_ready_count"] == 0
    assert summary["summary_status"] == "target_null_support_observed_below_alpha_resolution"


def test_simulation_runner_writes_outputs_with_stubbed_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate(
        _config: panel.RootTieRankSelectedNullSimulationConfig,
    ) -> dict[str, pd.DataFrame]:
        frame = pd.DataFrame.from_records([{"case_id": "x"}])
        summary = pd.DataFrame.from_records(
            [
                {
                    "schema_version": panel.SCHEMA_VERSION,
                    "study_role": panel.STUDY_ROLE,
                    "base_case_count": 1,
                    "replicates_per_case": 1,
                    "attempted_null_replicates": 1,
                    "successful_null_replicates": 1,
                    "failed_null_replicates": 0,
                    "observed_target_case_count": 1,
                    "observed_target_stratum_count": 1,
                    "null_generated_stratum_count": 1,
                    "target_strata_with_null_support_count": 0,
                    "target_strata_alpha_resolution_ready_count": 0,
                    "target_strata_tail_precision_ready_count": 0,
                    "missing_alpha_resolution_null_count_total": 99,
                    "missing_tail_precision_null_count_total": 1584,
                    "summary_status": "no_observed_target_stratum_support_yet",
                }
            ]
        )
        return {
            "root_rows": frame,
            "merge_margins": frame,
            "tie_rows": frame,
            "mixed_rows": frame,
            "combined_feasibility_rows": frame,
            "combined_feasibility_strata": frame,
            "combined_feasibility_summary": frame,
            "target_support": frame,
            "summary": summary,
            "failures": pd.DataFrame(),
        }

    monkeypatch.setattr(
        panel,
        "evaluate_root_tie_rank_selected_null_simulation",
        fake_evaluate,
    )
    outputs = panel.run_root_tie_rank_selected_null_simulation(
        panel.RootTieRankSelectedNullSimulationConfig(output_dir=tmp_path / "out")
    )

    assert outputs["root_rows"].exists()
    assert outputs["target_support"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
