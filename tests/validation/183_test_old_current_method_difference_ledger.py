from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration import old_current_method_difference_ledger as panel


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return path


def _legacy_summary(path: Path, *, null_legacy_false: int, signal_delta: float) -> Path:
    return _write_csv(
        path,
        [
            {
                "data_role": "selected_null",
                "current_false_split_count": 0,
                "legacy_false_split_count": null_legacy_false,
                "legacy_regression_count": null_legacy_false,
                "legacy_improvement_count": 0,
                "mean_delta_ari_legacy_minus_current": -0.5,
            },
            {
                "data_role": "signal",
                "current_false_split_count": 0,
                "legacy_false_split_count": 0,
                "legacy_regression_count": 0,
                "legacy_improvement_count": 1,
                "mean_delta_ari_legacy_minus_current": signal_delta,
            },
        ],
    )


def _config(tmp_path: Path) -> panel.OldCurrentMethodDifferenceLedgerConfig:
    stack_contract = _write_csv(
        tmp_path / "contract.csv",
        [
            {
                "surface": "null_prior_bandwidths",
                "old_c2ef": "old topology bandwidth smoother",
                "current": "removed from production, diagnostic only",
                "assessment": "use support-gated evidence",
            },
            {
                "surface": "support_contract",
                "old_c2ef": "neutral fallback",
                "current": "fail closed when sparse",
                "assessment": "current is safer",
            },
        ],
    )
    internal_spectral = _write_csv(
        tmp_path / "internal.csv",
        [
            {
                "data_role": "selected_null",
                "mean_delta_raw_mp_signal_count_sum": 11.0,
            },
            {
                "data_role": "signal",
                "mean_delta_raw_mp_signal_count_sum": 13.0,
            },
        ],
    )
    spectral = _write_csv(
        tmp_path / "spectral.csv",
        [
            {
                "bandwidth_default_selected_null_direct_significant_count": 4,
                "bandwidth_default_selected_null_interpolated_significant_count": 0,
                "bandwidth_default_signal_direct_significant_count": 5,
                "bandwidth_default_signal_interpolated_significant_count": 0,
                "bandwidth_default_signal_miss_count": 5,
                "bandwidth_reference_tau_s_selected_null_reopened_fraction": 0.75,
                "bandwidth_reference_tau_s_signal_recovered_fraction": 0.4,
                "spectral_null_false_split_reduction": 3,
                "spectral_signal_regression_count": 1,
                "spectral_min_delta_ari": -0.2,
                "spectral_blocking_component_ids": "signal_retention",
                "old_full_julia_prior_cluster_count": 8,
                "current_full_julia_cluster_count": 5,
                "old_current_full_julia_adjusted_rand_index": 0.1,
                "old_current_full_julia_normalized_mutual_information": 0.9,
            }
        ],
    )
    kernel_spectral = _write_csv(
        tmp_path / "kernel_spectral.csv",
        [
            {
                "kernel_available_count": 2,
                "strict_fail_closed_kernel_available_count": 1,
                "kernel_nonzero_support_target_count": 0,
                "legacy_selected_null_false_split_count": 2,
                "legacy_signal_improvement_count": 1,
            }
        ],
    )
    return panel.OldCurrentMethodDifferenceLedgerConfig(
        output_dir=tmp_path / "out",
        stack_contract_path=stack_contract,
        stack_behavior_path=tmp_path / "missing_behavior.csv",
        stack_overlap_path=tmp_path / "missing_overlap.csv",
        legacy_summary_path=_legacy_summary(
            tmp_path / "legacy.csv",
            null_legacy_false=1,
            signal_delta=0.1,
        ),
        legacy_root_tail_summary_path=_legacy_summary(
            tmp_path / "root_tail.csv",
            null_legacy_false=2,
            signal_delta=0.02,
        ),
        legacy_internal_spectral_summary_path=internal_spectral,
        spectral_bandwidth_summary_path=spectral,
        kernel_spectral_summary_path=kernel_spectral,
    )


def test_difference_ledger_tracks_component_and_metric_rows(tmp_path: Path) -> None:
    rows = panel.build_ledger_rows(_config(tmp_path))

    assert set(rows["difference_id"]) >= {
        "contract_null_prior_bandwidths",
        "contract_support_contract",
        "legacy_c2ef9a69_compact_cases",
        "legacy_c2ef9a69_root_tail_overlap",
        "legacy_internal_barycenter_spectral",
        "old_bandwidth_default_interpolation",
        "tau_s_widening_sensitivity",
        "strict_spectral_transport",
        "full_julia_fragmentation_gap",
        "candidate_kernel_spectral_root_tail_law",
    }

    root = rows.set_index("difference_id").loc["legacy_c2ef9a69_root_tail_overlap"]
    assert root["primary_metric_id"] == "selected_null_legacy_false_split_delta"
    assert root["primary_metric_value"] == 2.0
    assert root["decision"] == "retain_as_power_source_and_safety_warning"

    bandwidth = rows.set_index("difference_id").loc["old_bandwidth_default_interpolation"]
    assert bandwidth["decision"] == "conservative_smoother_not_rescue_rule"

    candidate = rows.set_index("difference_id").loc[
        "candidate_kernel_spectral_root_tail_law"
    ]
    assert candidate["primary_metric_value"] == 2.0
    assert candidate["decision"] == "not_promotable_positive_tail_support_missing"


def test_difference_ledger_writes_outputs_and_summary(tmp_path: Path) -> None:
    outputs = panel.run_old_current_method_difference_ledger(_config(tmp_path))

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert all(path.exists() for path in outputs.values())

    summary = pd.read_csv(outputs["summary"]).iloc[0]
    assert summary["old_safety_regression_count"] == 2
    assert summary["old_power_positive_count"] == 2
    assert summary["summary_status"] == "ledger_ready_diagnostic_only"

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["generated_by"] == panel.GENERATED_BY
