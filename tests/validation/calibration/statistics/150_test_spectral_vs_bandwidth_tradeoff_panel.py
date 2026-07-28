from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.statistics import (
    spectral_vs_bandwidth_tradeoff_panel as panel,
)


def _selected_family_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for replicate in (0, 1):
        records.extend(
            [
                {
                    "case_id": "null_case",
                    "data_role": "selected_null",
                    "method_id": panel.BASELINE_PROFILE,
                    "replicate": replicate,
                    "ari": 0.0,
                    "false_split": True,
                },
                {
                    "case_id": "null_case",
                    "data_role": "selected_null",
                    "method_id": panel.CANDIDATE_PROFILE,
                    "replicate": replicate,
                    "ari": 1.0,
                    "false_split": replicate == 1,
                },
                {
                    "case_id": "signal_case",
                    "data_role": "signal",
                    "method_id": panel.BASELINE_PROFILE,
                    "replicate": replicate,
                    "ari": 1.0,
                    "false_split": False,
                },
                {
                    "case_id": "signal_case",
                    "data_role": "signal",
                    "method_id": panel.CANDIDATE_PROFILE,
                    "replicate": replicate,
                    "ari": 1.0 if replicate == 0 else 0.2,
                    "false_split": False,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _bandwidth_summary() -> pd.DataFrame:
    rows = [
        {
            "data_role": "selected_null",
            "method_id": panel.BANDWIDTH_METHOD,
            "behavior_label": "selected_null_false_signal_suppressed",
            "direct_significant_count": 4,
            "interpolated_significant_count": 0,
            "signal_miss_count": 0,
            "signal_catch_count": 0,
            "median_best_case_required_tau_s_for_alpha": 14.0,
        },
        {
            "data_role": "signal",
            "method_id": panel.BANDWIDTH_METHOD,
            "behavior_label": "signal_not_caught_by_interpolation",
            "direct_significant_count": 5,
            "interpolated_significant_count": 0,
            "signal_miss_count": 5,
            "signal_catch_count": 0,
            "median_best_case_required_tau_s_for_alpha": 22.0,
        },
    ]
    return pd.DataFrame.from_records(rows)


def _tau_s_sensitivity() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "data_role": "selected_null",
                "method_id": panel.BANDWIDTH_METHOD,
                "tau_s_threshold": 20.0,
                "direct_significant_count": 4,
                "best_case_significant_count": 3,
                "best_case_significant_fraction": 0.75,
            },
            {
                "data_role": "signal",
                "method_id": panel.BANDWIDTH_METHOD,
                "tau_s_threshold": 20.0,
                "direct_significant_count": 5,
                "best_case_significant_count": 2,
                "best_case_significant_fraction": 0.40,
            },
        ]
    )


def _old_stack_behavior() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "method": "prior_full_julia_kl_stack_20260614",
                "n_samples": 10,
                "n_clusters": 8,
                "n_singleton_clusters": 6,
            },
            {
                "method": "current_conditional_topology_diagnostic_v1",
                "n_samples": 10,
                "n_clusters": 5,
                "n_singleton_clusters": 2,
            },
        ]
    )


def _old_stack_overlap() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "adjusted_rand_index": 0.1,
                "normalized_mutual_information": 0.9,
            }
        ]
    )


def _promotion_components() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "component_id": "selected_family_signal_retention",
                "component_status": "fails",
                "failure_reason": "signal regression",
            },
            {
                "component_id": "selected_family_null_false_split_reduction",
                "component_status": "passes",
                "failure_reason": "",
            },
        ]
    )


def _promotion_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "promotion_decision": "diagnostic_only_not_promoted",
                "blocking_component_ids": "selected_family_signal_retention",
            }
        ]
    )


def _write_inputs(tmp_path: Path) -> panel.SpectralVsBandwidthTradeoffConfig:
    paths = {
        "selected_family_rows_path": tmp_path / "selected_family.csv",
        "promotion_components_path": tmp_path / "promotion_components.csv",
        "promotion_summary_path": tmp_path / "promotion_summary.csv",
        "bandwidth_summary_path": tmp_path / "bandwidth_summary.csv",
        "bandwidth_tau_sensitivity_path": tmp_path / "tau_s.csv",
        "old_stack_behavior_path": tmp_path / "old_behavior.csv",
        "old_stack_overlap_path": tmp_path / "old_overlap.csv",
    }
    _selected_family_rows().to_csv(paths["selected_family_rows_path"], index=False)
    _promotion_components().to_csv(paths["promotion_components_path"], index=False)
    _promotion_summary().to_csv(paths["promotion_summary_path"], index=False)
    _bandwidth_summary().to_csv(paths["bandwidth_summary_path"], index=False)
    _tau_s_sensitivity().to_csv(paths["bandwidth_tau_sensitivity_path"], index=False)
    _old_stack_behavior().to_csv(paths["old_stack_behavior_path"], index=False)
    _old_stack_overlap().to_csv(paths["old_stack_overlap_path"], index=False)
    return panel.SpectralVsBandwidthTradeoffConfig(
        output_dir=tmp_path / "out",
        **paths,
    )


def test_tradeoff_summary_separates_spectral_and_bandwidth_blockers(
    tmp_path: Path,
) -> None:
    config = _write_inputs(tmp_path)
    rows, summary = panel.evaluate_spectral_vs_bandwidth_tradeoff(config)

    assert not rows.empty
    one = summary.iloc[0]
    assert one["comparison_status"] == "hybrid_needed_diagnostic_only"
    assert one["spectral_null_false_split_reduction"] == 1
    assert one["spectral_signal_regression_count"] == 1
    assert one["bandwidth_reference_tau_s_selected_null_reopened_fraction"] == 0.75
    assert one["bandwidth_reference_tau_s_signal_recovered_fraction"] == 0.40
    assert one["old_full_julia_prior_cluster_count"] == 8


def test_tradeoff_panel_writes_outputs(tmp_path: Path) -> None:
    config = _write_inputs(tmp_path)

    outputs = panel.run_spectral_vs_bandwidth_tradeoff_panel(config)

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    summary = pd.read_csv(outputs["summary"])
    assert summary.loc[0, "spectral_promotion_decision"] == ("diagnostic_only_not_promoted")
