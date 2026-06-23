"""Build a compact ledger of real old-vs-current TBS method differences.

The ledger does not run clustering. It reads already-produced comparison
artifacts and normalizes them into component-level rows so method changes can
be tracked without re-reading every panel.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "old_current_method_difference_ledger/v1"
STUDY_ROLE = "diagnostic_old_current_method_difference_ledger"
GENERATED_BY = "benchmarks.diagnostics.calibration.old_current_method_difference_ledger"

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_STACK_ROOT = Path(
    "raw/assets/benchmark-results/old_vs_current_method_stack_20260615"
    "/stack_contract_comparison"
)

ROWS_OUTPUT = "old_current_method_difference_ledger_rows.csv"
SUMMARY_OUTPUT = "old_current_method_difference_ledger_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "difference_id",
    "component",
    "old_method_behavior",
    "current_method_behavior",
    "difference_status",
    "measurement_family",
    "measurement_source_path",
    "selected_null_effect",
    "signal_effect",
    "fragmentation_effect",
    "support_calibration_effect",
    "primary_metric_id",
    "primary_metric_value",
    "secondary_metric_id",
    "secondary_metric_value",
    "decision",
    "next_tracking_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "component_count",
    "diagnostic_retained_count",
    "replaced_or_removed_count",
    "added_current_guard_count",
    "old_power_positive_count",
    "old_safety_regression_count",
    "current_power_loss_count",
    "summary_status",
)

CONTRACT_STATUSES = {
    "edge_gate_alpha": "formalized_current_contract",
    "sibling_statistic_subspace": "replaced",
    "null_prior_bandwidths": "removed_and_reintroduced_diagnostic",
    "calibration_weighting": "replaced",
    "local_calibration_kernel": "replaced",
    "support_contract": "added_current_guard",
    "heuristic_root_guards": "added_current_guard",
    "selected_family_guard": "added_current_guard",
    "traversal_law": "shared_formalized",
    "fresh_julia_old_rerun_status": "unresolved_runtime_gap",
}


@dataclass(frozen=True)
class OldCurrentMethodDifferenceLedgerConfig:
    """Input/output paths for the method-difference ledger."""

    output_dir: Path
    stack_contract_path: Path = DEFAULT_STACK_ROOT / "method_stack_contract_comparison.csv"
    stack_behavior_path: Path = DEFAULT_STACK_ROOT / "method_stack_behavior_summary.csv"
    stack_overlap_path: Path = DEFAULT_STACK_ROOT / "method_stack_pairwise_overlap.csv"
    legacy_summary_path: Path = (
        DEFAULT_RESULT_ROOT
        / "legacy_c2ef9a69_method_comparison_panel"
        / "legacy_c2ef9a69_method_comparison_summary.csv"
    )
    legacy_root_tail_summary_path: Path = (
        DEFAULT_RESULT_ROOT
        / "legacy_c2ef9a69_root_tail_overlap_comparison_20260617"
        / "legacy_c2ef9a69_method_comparison_summary.csv"
    )
    legacy_internal_spectral_summary_path: Path = (
        DEFAULT_RESULT_ROOT
        / "legacy_internal_spectral_comparison_panel"
        / "legacy_internal_spectral_comparison_summary.csv"
    )
    spectral_bandwidth_summary_path: Path = (
        DEFAULT_RESULT_ROOT
        / "spectral_vs_bandwidth_tradeoff_panel"
        / "spectral_vs_bandwidth_tradeoff_summary.csv"
    )
    kernel_spectral_summary_path: Path = (
        DEFAULT_RESULT_ROOT
        / "root_selected_kernel_spectral_tail_law_20260617"
        / "root_selected_kernel_spectral_tail_law_summary.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stack-contract-path", type=Path, default=None)
    parser.add_argument("--stack-behavior-path", type=Path, default=None)
    parser.add_argument("--stack-overlap-path", type=Path, default=None)
    parser.add_argument("--legacy-summary-path", type=Path, default=None)
    parser.add_argument("--legacy-root-tail-summary-path", type=Path, default=None)
    parser.add_argument("--legacy-internal-spectral-summary-path", type=Path, default=None)
    parser.add_argument("--spectral-bandwidth-summary-path", type=Path, default=None)
    parser.add_argument("--kernel-spectral-summary-path", type=Path, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> OldCurrentMethodDifferenceLedgerConfig:
    defaults = OldCurrentMethodDifferenceLedgerConfig(output_dir=args.output_dir)
    overrides = {
        name: getattr(args, name)
        for name in (
            "stack_contract_path",
            "stack_behavior_path",
            "stack_overlap_path",
            "legacy_summary_path",
            "legacy_root_tail_summary_path",
            "legacy_internal_spectral_summary_path",
            "spectral_bandwidth_summary_path",
            "kernel_spectral_summary_path",
        )
        if getattr(args, name) is not None
    }
    return OldCurrentMethodDifferenceLedgerConfig(
        output_dir=args.output_dir,
        **{field: overrides.get(field, getattr(defaults, field)) for field in overrides},
    )


def _json_default(value: object) -> object:
    if isinstance(value, OldCurrentMethodDifferenceLedgerConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _finite_int(value: object) -> int:
    numeric = _finite_float(value)
    return int(numeric) if math.isfinite(numeric) else 0


def _string(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return str(value)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _base_row(**values: object) -> dict[str, object]:
    row = {column: "" for column in ROW_COLUMNS}
    row["schema_version"] = SCHEMA_VERSION
    row["study_role"] = STUDY_ROLE
    row.update(values)
    return row


def _summary_by_role(frame: pd.DataFrame | None) -> dict[str, pd.Series]:
    if frame is None or "data_role" not in frame.columns:
        return {}
    return {str(row["data_role"]): row for _, row in frame.iterrows()}


def _legacy_comparison_row(
    *,
    difference_id: str,
    component: str,
    summary_path: Path,
    summary: pd.DataFrame | None,
    next_step: str,
) -> dict[str, object] | None:
    by_role = _summary_by_role(summary)
    if not by_role:
        return None
    selected = by_role.get("selected_null", pd.Series(dtype=object))
    signal = by_role.get("signal", pd.Series(dtype=object))
    current_false = _finite_int(selected.get("current_false_split_count"))
    legacy_false = _finite_int(selected.get("legacy_false_split_count"))
    false_delta = legacy_false - current_false
    signal_delta = _finite_float(signal.get("mean_delta_ari_legacy_minus_current"))
    improvements = _finite_int(signal.get("legacy_improvement_count"))
    regressions = _finite_int(signal.get("legacy_regression_count"))

    return _base_row(
        difference_id=difference_id,
        component=component,
        old_method_behavior="full legacy c2ef9a69 package makes more permissive decisions",
        current_method_behavior="current tbs profile skips or fail-closes when support is missing",
        difference_status="legacy_stronger_but_less_controlled",
        measurement_family="paired_legacy_current_clustering",
        measurement_source_path=str(summary_path),
        selected_null_effect=(
            f"legacy_false_split_delta={false_delta}; "
            f"current={current_false}; legacy={legacy_false}"
        ),
        signal_effect=(
            f"mean_delta_ari_legacy_minus_current={signal_delta:.6g}; "
            f"legacy_improvements={improvements}; legacy_regressions={regressions}"
        ),
        fragmentation_effect=(
            "tracked by delta clusters, singleton clusters, and largest-cluster fraction"
        ),
        support_calibration_effect="legacy decisions are diagnostic-only, not calibrated support",
        primary_metric_id="selected_null_legacy_false_split_delta",
        primary_metric_value=float(false_delta),
        secondary_metric_id="signal_mean_delta_ari_legacy_minus_current",
        secondary_metric_value=signal_delta,
        decision="retain_as_power_source_and_safety_warning",
        next_tracking_step=next_step,
    )


def _contract_rows(config: OldCurrentMethodDifferenceLedgerConfig) -> list[dict[str, object]]:
    frame = _read_csv_if_exists(config.stack_contract_path)
    if frame is None:
        return []
    rows: list[dict[str, object]] = []
    for _, source in frame.iterrows():
        surface = _string(source.get("surface"), "unknown_surface")
        rows.append(
            _base_row(
                difference_id=f"contract_{surface}",
                component=surface,
                old_method_behavior=_string(source.get("old_c2ef")),
                current_method_behavior=_string(source.get("current")),
                difference_status=CONTRACT_STATUSES.get(surface, "documented_difference"),
                measurement_family="method_stack_contract",
                measurement_source_path=str(config.stack_contract_path),
                selected_null_effect="contractual_difference_only",
                signal_effect="contractual_difference_only",
                fragmentation_effect="contractual_difference_only",
                support_calibration_effect=_string(source.get("assessment")),
                primary_metric_id="contract_row_present",
                primary_metric_value=1.0,
                decision=_contract_decision(surface),
                next_tracking_step=_contract_next_step(surface),
            )
        )
    return rows


def _contract_decision(surface: str) -> str:
    if surface in {"null_prior_bandwidths", "local_calibration_kernel"}:
        return "reintroduce_only_as_support_gated_diagnostic_weight"
    if surface in {"calibration_weighting", "support_contract"}:
        return "keep_current_fail_closed_contract"
    if surface == "sibling_statistic_subspace":
        return "track_power_loss_from_removed_adaptive_subspace"
    if surface in {"heuristic_root_guards", "selected_family_guard"}:
        return "keep_as_guard_until_joint_selected_law_exists"
    if surface == "traversal_law":
        return "not_the_primary_difference"
    return "track_as_documented_method_difference"


def _contract_next_step(surface: str) -> str:
    if surface == "null_prior_bandwidths":
        return "measure kernel-weighted conditional support instead of p-value rescue"
    if surface == "sibling_statistic_subspace":
        return "compare adaptive parent-PCA power against fixed-coordinate safety"
    if surface == "fresh_julia_old_rerun_status":
        return "cache or vectorize tree-distance neighborhoods before full reruns"
    return "keep row in difference ledger and update after new benchmark runs"


def _internal_spectral_row(
    config: OldCurrentMethodDifferenceLedgerConfig,
) -> dict[str, object] | None:
    summary = _read_csv_if_exists(config.legacy_internal_spectral_summary_path)
    by_role = _summary_by_role(summary)
    if not by_role:
        return None
    selected = by_role.get("selected_null", pd.Series(dtype=object))
    signal = by_role.get("signal", pd.Series(dtype=object))
    selected_delta = _finite_float(selected.get("mean_delta_raw_mp_signal_count_sum"))
    signal_delta = _finite_float(signal.get("mean_delta_raw_mp_signal_count_sum"))
    return _base_row(
        difference_id="legacy_internal_barycenter_spectral",
        component="internal_node_spectral_distributions",
        old_method_behavior="internal/coarse barycenter rows perturb spectral MP counts",
        current_method_behavior="current spectral target uses descendant leaf rows and explicit support law",
        difference_status="diagnostic_retained_not_production",
        measurement_family="legacy_internal_spectral_comparison",
        measurement_source_path=str(config.legacy_internal_spectral_summary_path),
        selected_null_effect="no partition change on compact comparison rows",
        signal_effect="no paired ARI change despite large MP-count shifts",
        fragmentation_effect="partition ARI remains one on completed compact rows",
        support_calibration_effect="internal rows are tree filters, not independent MP samples",
        primary_metric_id="selected_null_delta_raw_mp_signal_count_sum",
        primary_metric_value=selected_delta,
        secondary_metric_id="signal_delta_raw_mp_signal_count_sum",
        secondary_metric_value=signal_delta,
        decision="retain_as_spectral_filter_diagnostic",
        next_tracking_step="condition eigenvalue spikes by matched eigenspace and selected geometry",
    )


def _spectral_bandwidth_rows(
    config: OldCurrentMethodDifferenceLedgerConfig,
) -> list[dict[str, object]]:
    summary = _read_csv_if_exists(config.spectral_bandwidth_summary_path)
    if summary is None or summary.empty:
        return []
    row = summary.iloc[0]
    rows = [
        _base_row(
            difference_id="old_bandwidth_default_interpolation",
            component="kernel_bandwidth_interpolation",
            old_method_behavior="bandwidth interpolation smooths direct selected-neighborhood p-like values",
            current_method_behavior="current method does not use old bandwidth as a production p-value",
            difference_status="removed_and_reintroduced_diagnostic",
            measurement_family="spectral_vs_bandwidth_tradeoff",
            measurement_source_path=str(config.spectral_bandwidth_summary_path),
            selected_null_effect=(
                "direct selected-null positives suppressed from "
                f"{_finite_float(row.get('bandwidth_default_selected_null_direct_significant_count')):.0f} "
                "to "
                f"{_finite_float(row.get('bandwidth_default_selected_null_interpolated_significant_count')):.0f}"
            ),
            signal_effect=(
                "direct signal positives caught after interpolation="
                f"{_finite_float(row.get('bandwidth_default_signal_interpolated_significant_count')):.0f}; "
                "misses="
                f"{_finite_float(row.get('bandwidth_default_signal_miss_count')):.0f}"
            ),
            fragmentation_effect="not a clustering profile in current tree",
            support_calibration_effect="use as locality evidence only",
            primary_metric_id="bandwidth_default_selected_null_interpolated_significant_count",
            primary_metric_value=_finite_float(
                row.get("bandwidth_default_selected_null_interpolated_significant_count")
            ),
            secondary_metric_id="bandwidth_default_signal_interpolated_significant_count",
            secondary_metric_value=_finite_float(
                row.get("bandwidth_default_signal_interpolated_significant_count")
            ),
            decision="conservative_smoother_not_rescue_rule",
            next_tracking_step="turn kernel smoothing into admissible support weights",
        ),
        _base_row(
            difference_id="tau_s_widening_sensitivity",
            component="signal_neighborhood_bandwidth_tau_s",
            old_method_behavior="wider signal-neighborhood bandwidth can reopen smoothed p-like rows",
            current_method_behavior="current method keeps tau_s reopening diagnostic-only",
            difference_status="unsafe_as_single_knob",
            measurement_family="spectral_vs_bandwidth_tradeoff",
            measurement_source_path=str(config.spectral_bandwidth_summary_path),
            selected_null_effect=(
                "selected_null_reopened_fraction="
                f"{_finite_float(row.get('bandwidth_reference_tau_s_selected_null_reopened_fraction')):.6g}"
            ),
            signal_effect=(
                "signal_recovered_fraction="
                f"{_finite_float(row.get('bandwidth_reference_tau_s_signal_recovered_fraction')):.6g}"
            ),
            fragmentation_effect="not measured as production clustering",
            support_calibration_effect="widening tau_s alone reopens null faster than signal",
            primary_metric_id="tau_s_selected_null_reopened_fraction",
            primary_metric_value=_finite_float(
                row.get("bandwidth_reference_tau_s_selected_null_reopened_fraction")
            ),
            secondary_metric_id="tau_s_signal_recovered_fraction",
            secondary_metric_value=_finite_float(
                row.get("bandwidth_reference_tau_s_signal_recovered_fraction")
            ),
            decision="do_not_promote_tau_s_only",
            next_tracking_step="condition tau_s weights on root geometry and spectral law",
        ),
        _base_row(
            difference_id="strict_spectral_transport",
            component="mp_spectral_transport_guard",
            old_method_behavior="old method did not require strict measured MP-mode transport",
            current_method_behavior="strict spectral transport blocks many selected-null splits",
            difference_status="added_current_guard_not_promoted",
            measurement_family="spectral_vs_bandwidth_tradeoff",
            measurement_source_path=str(config.spectral_bandwidth_summary_path),
            selected_null_effect=(
                "false_split_reduction="
                f"{_finite_float(row.get('spectral_null_false_split_reduction')):.0f}"
            ),
            signal_effect=(
                "signal_regression_count="
                f"{_finite_float(row.get('spectral_signal_regression_count')):.0f}; "
                "min_delta_ari="
                f"{_finite_float(row.get('spectral_min_delta_ari')):.6g}"
            ),
            fragmentation_effect="blocks oversplitting but may over-suppress signal",
            support_calibration_effect=_string(row.get("spectral_blocking_component_ids")),
            primary_metric_id="spectral_null_false_split_reduction",
            primary_metric_value=_finite_float(row.get("spectral_null_false_split_reduction")),
            secondary_metric_id="spectral_signal_regression_count",
            secondary_metric_value=_finite_float(row.get("spectral_signal_regression_count")),
            decision="guard_candidate_not_standalone_method",
            next_tracking_step="combine with kernel-weighted support rather than hard blocking only",
        ),
        _base_row(
            difference_id="full_julia_fragmentation_gap",
            component="full_julia_flat_clustering_behavior",
            old_method_behavior="prior full TBS stack produced more clusters and singletons",
            current_method_behavior="current conditional-topology diagnostic produced fewer clusters",
            difference_status="fragmentation_symptom_tracked",
            measurement_family="spectral_vs_bandwidth_tradeoff",
            measurement_source_path=str(config.spectral_bandwidth_summary_path),
            selected_null_effect="not a selected-null panel",
            signal_effect="full-Julia cluster overlap is descriptive",
            fragmentation_effect=(
                "old_clusters="
                f"{_finite_float(row.get('old_full_julia_prior_cluster_count')):.0f}; "
                "current_clusters="
                f"{_finite_float(row.get('current_full_julia_cluster_count')):.0f}; "
                "ARI="
                f"{_finite_float(row.get('old_current_full_julia_adjusted_rand_index')):.6g}; "
                "NMI="
                f"{_finite_float(row.get('old_current_full_julia_normalized_mutual_information')):.6g}"
            ),
            support_calibration_effect="fragmentation is outcome evidence, not an inference penalty",
            primary_metric_id="old_full_julia_prior_cluster_count",
            primary_metric_value=_finite_float(row.get("old_full_julia_prior_cluster_count")),
            secondary_metric_id="current_full_julia_cluster_count",
            secondary_metric_value=_finite_float(row.get("current_full_julia_cluster_count")),
            decision="track_as_downstream_symptom",
            next_tracking_step="measure cluster-size distribution after any candidate law change",
        ),
    ]
    return rows


def _kernel_spectral_candidate_row(
    config: OldCurrentMethodDifferenceLedgerConfig,
) -> dict[str, object] | None:
    summary = _read_csv_if_exists(config.kernel_spectral_summary_path)
    if summary is None or summary.empty:
        return None
    row = summary.iloc[0]
    kernel_available = _finite_int(row.get("kernel_available_count"))
    strict_fail_closed_available = _finite_int(
        row.get("strict_fail_closed_kernel_available_count")
    )
    nonzero_support = _finite_int(row.get("kernel_nonzero_support_target_count"))
    legacy_false = _finite_int(row.get("legacy_selected_null_false_split_count"))
    legacy_signal = _finite_int(row.get("legacy_signal_improvement_count"))
    decision = (
        "not_promotable_positive_tail_support_missing"
        if nonzero_support == 0
        else "candidate_ready_for_selected_null_false_split_check"
    )
    return _base_row(
        difference_id="candidate_kernel_spectral_root_tail_law",
        component="support_gated_kernel_spectral_root_tail",
        old_method_behavior="old kernel smoothing could act as a p-like rescue",
        current_method_behavior="current exact root-tail law fails closed when same-stratum support is missing",
        difference_status="candidate_diagnostic_measured",
        measurement_family="root_selected_kernel_spectral_tail_law",
        measurement_source_path=str(config.kernel_spectral_summary_path),
        selected_null_effect=(
            f"legacy_selected_null_false_split_count={legacy_false}; "
            "candidate remains diagnostic and does not change clustering"
        ),
        signal_effect=(
            f"kernel_available_count={kernel_available}; "
            f"strict_fail_closed_kernel_available_count={strict_fail_closed_available}; "
            f"legacy_signal_improvement_count={legacy_signal}"
        ),
        fragmentation_effect="not yet a runtime clustering profile",
        support_calibration_effect=(
            f"kernel_nonzero_support_target_count={nonzero_support}; "
            "requires nonzero S_Hu support before rescue"
        ),
        primary_metric_id="kernel_available_count",
        primary_metric_value=float(kernel_available),
        secondary_metric_id="strict_fail_closed_kernel_available_count",
        secondary_metric_value=float(strict_fail_closed_available),
        decision=decision,
        next_tracking_step=(
            "generate or reweight admissible nonzero S_Hu support in kernel neighborhoods"
        ),
    )


def build_ledger_rows(config: OldCurrentMethodDifferenceLedgerConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_contract_rows(config))
    for optional_row in (
        _legacy_comparison_row(
            difference_id="legacy_c2ef9a69_compact_cases",
            component="full_legacy_method_package",
            summary_path=config.legacy_summary_path,
            summary=_read_csv_if_exists(config.legacy_summary_path),
            next_step="compare candidate law against both current and full legacy on compact cases",
        ),
        _legacy_comparison_row(
            difference_id="legacy_c2ef9a69_root_tail_overlap",
            component="root_tail_overlap_cases",
            summary_path=config.legacy_root_tail_summary_path,
            summary=_read_csv_if_exists(config.legacy_root_tail_summary_path),
            next_step="recover old signal power only under selected-root kernel-spectral support",
        ),
        _internal_spectral_row(config),
        _kernel_spectral_candidate_row(config),
    ):
        if optional_row is not None:
            rows.append(optional_row)
    rows.extend(_spectral_bandwidth_rows(config))
    return pd.DataFrame.from_records(rows, columns=ROW_COLUMNS)


def summarize_ledger(rows: pd.DataFrame) -> pd.DataFrame:
    statuses = rows["difference_status"].astype(str) if not rows.empty else pd.Series(dtype=str)
    decisions = rows["decision"].astype(str) if not rows.empty else pd.Series(dtype=str)
    selected_effects = (
        rows["selected_null_effect"].astype(str) if not rows.empty else pd.Series(dtype=str)
    )
    signal_effects = rows["signal_effect"].astype(str) if not rows.empty else pd.Series(dtype=str)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "row_count": int(len(rows)),
        "component_count": int(rows["component"].nunique()) if not rows.empty else 0,
        "diagnostic_retained_count": int(statuses.str.contains("diagnostic").sum()),
        "replaced_or_removed_count": int(
            statuses.str.contains("replaced|removed", regex=True).sum()
        ),
        "added_current_guard_count": int(statuses.str.contains("added_current_guard").sum()),
        "old_power_positive_count": int(
            signal_effects.str.contains("legacy_improvements=[1-9]", regex=True).sum()
        ),
        "old_safety_regression_count": int(
            selected_effects.str.contains("legacy_false_split_delta=[1-9]", regex=True).sum()
        ),
        "current_power_loss_count": int(decisions.str.contains("not_promoted|not_rescue").sum()),
        "summary_status": "ledger_ready_diagnostic_only",
    }
    return pd.DataFrame.from_records([summary], columns=SUMMARY_COLUMNS)


def run_old_current_method_difference_ledger(
    config: OldCurrentMethodDifferenceLedgerConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_ledger_rows(config)
    summary = summarize_ledger(rows)

    rows_path = config.output_dir / ROWS_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT

    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "outputs": {
            "rows": rows_path,
            "summary": summary_path,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return {"rows": rows_path, "summary": summary_path, "manifest": manifest_path}


def main() -> None:
    outputs = run_old_current_method_difference_ledger(config_from_args(parse_args()))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
