"""Synthesize the overlap traversal threshold hierarchy.

This report composes the existing overlap diagnostics into one ordered table:
continuous structural acceptance, fragment-risk blocking, residual
selected-family null evidence, and residual structural recovery evidence. It
is diagnostic-only and does not define a production calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_threshold_hierarchy_not_calibration"
SCHEMA_VERSION = "overlap_threshold_hierarchy/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_threshold_hierarchy"

HIERARCHY_COLUMNS = (
    "schema_version",
    "study_role",
    "stage_order",
    "stage_name",
    "selection_unit",
    "threshold_role",
    "metric",
    "threshold",
    "direction",
    "target_event",
    "target_count",
    "target_total",
    "target_rate",
    "negative_event",
    "negative_count",
    "negative_total",
    "negative_rate",
    "diagnostic_status",
    "interpretation",
)


@dataclass(frozen=True)
class OverlapThresholdHierarchyConfig:
    """Runtime contract for overlap threshold hierarchy synthesis."""

    decision_zone_summary_path: Path
    policy_summary_path: Path
    fragment_guard_scan_path: Path
    residual_thresholds_path: Path
    residual_eligibility_summary_path: Path
    output_dir: Path

    @property
    def hierarchy_path(self) -> Path:
        return self.output_dir / "overlap_threshold_hierarchy.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _row_by_value(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame[frame[column].astype(str).eq(str(value))]
    if rows.empty:
        raise ValueError(f"Missing row where {column} == {value!r}.")
    return rows.iloc[0]


def _float(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def _int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def build_overlap_threshold_hierarchy(
    decision_zone_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    fragment_guard_scan: pd.DataFrame,
    residual_thresholds: pd.DataFrame,
    residual_eligibility_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build an ordered overlap threshold hierarchy from diagnostic outputs."""
    records: list[dict[str, object]] = []

    stable = _row_by_value(
        decision_zone_summary,
        "structural_decision_zone",
        "stable_structural_accept",
    )
    weak = _row_by_value(
        decision_zone_summary,
        "structural_decision_zone",
        "unstable_weak_homogeneity_zone",
    )
    accepted_truth_aligned_total = _int(stable["signal_truth_aligned_count"]) + _int(
        weak["signal_truth_aligned_count"]
    )
    accepted_negative_total = (
        _int(stable["null_count"])
        + _int(stable["signal_truth_misaligned_count"])
        + _int(weak["null_count"])
        + _int(weak["signal_truth_misaligned_count"])
    )
    stable_target = _int(stable["signal_truth_aligned_count"])
    stable_negative = _int(stable["null_count"]) + _int(stable["signal_truth_misaligned_count"])
    records.append(
        {
            "stage_order": 1,
            "stage_name": "continuous_structural_stable_accept",
            "selection_unit": "row",
            "threshold_role": "stable structural accept",
            "metric": "continuous_context_min_margin and same-subspace status",
            "threshold": 0.0,
            "direction": "greater_equal",
            "target_event": "truth_aligned_signal_accepted_rows",
            "target_count": stable_target,
            "target_total": accepted_truth_aligned_total,
            "target_rate": _rate(stable_target, accepted_truth_aligned_total),
            "negative_event": "selected_null_or_truth_misaligned_accepted_rows",
            "negative_count": stable_negative,
            "negative_total": accepted_negative_total,
            "negative_rate": _rate(stable_negative, accepted_negative_total),
            "diagnostic_status": "clean_stable_region_in_focused_run",
            "interpretation": (
                "Continuous structural margins identify a clean stable region, "
                "but leave weak truth-aligned rows unresolved."
            ),
        }
    )

    blocked = _row_by_value(
        policy_summary,
        "diagnostic_traversal_action",
        "weak_fragment_guard_blocked",
    )
    fragment_scan = fragment_guard_scan[
        fragment_guard_scan["metric"].astype(str).eq("fragment_risk_proxy_score")
        & fragment_guard_scan["guard_status"].astype(str).eq("fragment_guard_candidate")
    ].copy()
    if fragment_scan.empty:
        raise ValueError("Missing fragment_risk_proxy_score candidate threshold.")
    fragment_threshold = _float(
        fragment_scan.sort_values("guard_score", ascending=False).iloc[0]["threshold"]
    )
    unstable = _row_by_value(
        policy_summary,
        "diagnostic_traversal_action",
        "weak_unstable_multiscale_zone",
    )
    fragment_blocked = _int(blocked["fragment_like_count"])
    fragment_total = fragment_blocked + _int(unstable["fragment_like_count"])
    recovery_blocked = _int(blocked["truth_recovery_count"])
    recovery_total = recovery_blocked + _int(unstable["truth_recovery_count"])
    records.append(
        {
            "stage_order": 2,
            "stage_name": "weak_fragment_risk_block",
            "selection_unit": "row",
            "threshold_role": "one-sided fragment guard",
            "metric": "fragment_risk_proxy_score",
            "threshold": fragment_threshold,
            "direction": "greater_equal",
            "target_event": "fragment_like_weak_rows_blocked",
            "target_count": fragment_blocked,
            "target_total": fragment_total,
            "target_rate": _rate(fragment_blocked, fragment_total),
            "negative_event": "truth_recovery_weak_rows_blocked",
            "negative_count": recovery_blocked,
            "negative_total": recovery_total,
            "negative_rate": _rate(recovery_blocked, recovery_total),
            "diagnostic_status": "useful_fragment_block_diagnostic",
            "interpretation": (
                "Fragment-risk blocking removes most one-sided fragment rows "
                "without blocking recovery rows in this focused run."
            ),
        }
    )

    threshold_rows = {str(row["threshold_name"]): row for _, row in residual_thresholds.iterrows()}
    threshold_specs = (
        (
            3,
            "residual_selected_family_null_evidence",
            "selected-family null evidence",
            "null_evidence_threshold",
            "truth_recovery_families_passing_null_evidence",
            "selected_null_families_passing_null_evidence",
            "p-value evidence separates residual recovery from selected null.",
            "null_filter_passes",
        ),
        (
            4,
            "residual_nonrecovery_structural_evidence",
            "structural recovery evidence",
            "nonrecovery_structural_threshold",
            "truth_recovery_families_passing_nonrecovery_structural_gate",
            "nonrecovery_families_passing_structural_gate",
            "Structural evidence controls residual non-recovery signal but loses recovery power.",
            "partial_recovery_retention",
        ),
        (
            5,
            "residual_strict_all_negative_structural_evidence",
            "strict structural recovery evidence",
            "full_negative_structural_threshold",
            "truth_recovery_families_passing_strict_structural_gate",
            "selected_null_or_nonrecovery_families_passing_strict_gate",
            "Strict all-negative structural evidence is very conservative.",
            "strict_diagnostic_only",
        ),
    )
    for (
        stage_order,
        stage_name,
        threshold_role,
        threshold_name,
        target_event,
        negative_event,
        interpretation,
        status,
    ) in threshold_specs:
        row = threshold_rows.get(threshold_name)
        if row is None:
            raise ValueError(f"Missing residual threshold row: {threshold_name!r}.")
        records.append(
            {
                "stage_order": stage_order,
                "stage_name": stage_name,
                "selection_unit": "selected_family",
                "threshold_role": threshold_role,
                "metric": str(row["metric"]),
                "threshold": _float(row["threshold"]),
                "direction": "greater_equal",
                "target_event": target_event,
                "target_count": _int(row["truth_recovery_pass_count"]),
                "target_total": _int(row["truth_recovery_family_count"]),
                "target_rate": _float(row["truth_recovery_retention"]),
                "negative_event": negative_event,
                "negative_count": _int(row["negative_pass_count"]),
                "negative_total": _int(row["negative_family_count"]),
                "negative_rate": _float(row["negative_selection_rate"]),
                "diagnostic_status": status,
                "interpretation": interpretation,
            }
        )

    unresolved = _row_by_value(
        residual_eligibility_summary,
        "residual_recovery_eligibility_status",
        "residual_null_evidence_only_unresolved",
    )
    records.append(
        {
            "stage_order": 6,
            "stage_name": "residual_null_evidence_only_unresolved",
            "selection_unit": "selected_family",
            "threshold_role": "unresolved selected-family mixture",
            "metric": "p-value evidence without structural recovery evidence",
            "threshold": math.nan,
            "direction": "not_applicable",
            "target_event": "truth_recovery_families_still_unresolved",
            "target_count": _int(unresolved["truth_recovery_family_count"]),
            "target_total": int(residual_eligibility_summary["truth_recovery_family_count"].sum()),
            "target_rate": _rate(
                _int(unresolved["truth_recovery_family_count"]),
                int(residual_eligibility_summary["truth_recovery_family_count"].sum()),
            ),
            "negative_event": "nonrecovery_families_with_null_evidence_only",
            "negative_count": _int(unresolved["nonrecovery_family_count"]),
            "negative_total": int(residual_eligibility_summary["nonrecovery_family_count"].sum()),
            "negative_rate": _rate(
                _int(unresolved["nonrecovery_family_count"]),
                int(residual_eligibility_summary["nonrecovery_family_count"].sum()),
            ),
            "diagnostic_status": "remaining_selected_family_law_blocker",
            "interpretation": (
                "Families with selected-family null evidence but insufficient "
                "structural recovery evidence remain the main traversal blocker."
            ),
        }
    )

    output = pd.DataFrame.from_records(records)
    output.insert(0, "schema_version", SCHEMA_VERSION)
    output.insert(1, "study_role", STUDY_ROLE)
    return output.loc[:, HIERARCHY_COLUMNS]


def run_overlap_threshold_hierarchy(
    config: OverlapThresholdHierarchyConfig,
) -> dict[str, Path]:
    """Run threshold hierarchy synthesis and write outputs."""
    decision_summary = pd.read_csv(config.decision_zone_summary_path)
    policy_summary = pd.read_csv(config.policy_summary_path)
    fragment_guard_scan = pd.read_csv(config.fragment_guard_scan_path)
    residual_thresholds = pd.read_csv(config.residual_thresholds_path)
    residual_eligibility_summary = pd.read_csv(config.residual_eligibility_summary_path)
    hierarchy = build_overlap_threshold_hierarchy(
        decision_summary,
        policy_summary,
        fragment_guard_scan,
        residual_thresholds,
        residual_eligibility_summary,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy.to_csv(config.hierarchy_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "decision_zone_summary_path": str(config.decision_zone_summary_path),
        "policy_summary_path": str(config.policy_summary_path),
        "fragment_guard_scan_path": str(config.fragment_guard_scan_path),
        "residual_thresholds_path": str(config.residual_thresholds_path),
        "residual_eligibility_summary_path": str(config.residual_eligibility_summary_path),
        "outputs": {"hierarchy": str(config.hierarchy_path)},
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"hierarchy": config.hierarchy_path, "manifest": config.manifest_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-zone-summary-path", required=True, type=Path)
    parser.add_argument("--policy-summary-path", required=True, type=Path)
    parser.add_argument("--fragment-guard-scan-path", required=True, type=Path)
    parser.add_argument("--residual-thresholds-path", required=True, type=Path)
    parser.add_argument("--residual-eligibility-summary-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_threshold_hierarchy(
        OverlapThresholdHierarchyConfig(
            decision_zone_summary_path=args.decision_zone_summary_path,
            policy_summary_path=args.policy_summary_path,
            fragment_guard_scan_path=args.fragment_guard_scan_path,
            residual_thresholds_path=args.residual_thresholds_path,
            residual_eligibility_summary_path=args.residual_eligibility_summary_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
