"""Stability contract for overlap traversal thresholds.

This diagnostic classifies the threshold hierarchy into stable reporting,
diagnostic-only, non-transferable, and law-required stages. It consumes the
ordered threshold hierarchy and held-out transfer summaries; it does not change
production behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_threshold_stability_contract_not_calibration"
SCHEMA_VERSION = "overlap_threshold_stability_contract/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_threshold_stability_contract"

STAGE_STATUS_COLUMNS = (
    "schema_version",
    "study_role",
    "stage_order",
    "stage_name",
    "selection_unit",
    "metric",
    "threshold",
    "threshold_stability_status",
    "recommended_use",
    "requires_selected_family_law",
    "transfer_checked",
    "transfer_leakage_detected",
    "transfer_recovery_loss_detected",
    "focused_target_rate",
    "focused_negative_rate",
    "transfer_truth_recovery_retention",
    "transfer_null_like_leakage_rate",
    "transfer_nonrecovery_leakage_rate",
    "notes",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "contract_id",
    "stage_count",
    "stable_reporting_count",
    "diagnostic_only_count",
    "nontransferable_count",
    "law_required_count",
    "production_promotion_status",
    "blocking_stage_names",
)


@dataclass(frozen=True)
class OverlapThresholdStabilityContractConfig:
    """Runtime contract for threshold stability classification."""

    threshold_hierarchy_path: Path
    transfer_summary_path: Path
    output_dir: Path

    @property
    def stage_status_path(self) -> Path:
        return self.output_dir / "overlap_threshold_stability_contract_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_threshold_stability_contract_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_inputs(hierarchy: pd.DataFrame, transfer_summary: pd.DataFrame) -> None:
    hierarchy_missing = sorted(
        {
            "stage_order",
            "stage_name",
            "selection_unit",
            "metric",
            "threshold",
            "target_rate",
            "negative_rate",
        }
        - set(hierarchy.columns)
    )
    if hierarchy_missing:
        raise ValueError(f"Threshold hierarchy is missing columns: {hierarchy_missing!r}")
    transfer_missing = sorted(
        {
            "threshold_name",
            "truth_recovery_retention",
            "null_like_leakage_rate",
            "nonrecovery_leakage_rate",
            "transfer_status",
        }
        - set(transfer_summary.columns)
    )
    if transfer_missing:
        raise ValueError(f"Threshold transfer summary is missing columns: {transfer_missing!r}")


def _transfer_name_for_stage(stage_name: str) -> str | None:
    mapping = {
        "residual_selected_family_null_evidence": "null_evidence_threshold",
        "residual_nonrecovery_structural_evidence": "nonrecovery_structural_threshold",
        "residual_strict_all_negative_structural_evidence": ("full_negative_structural_threshold"),
    }
    return mapping.get(stage_name)


def _transfer_metrics(
    transfer_summary: pd.DataFrame,
    threshold_name: str,
) -> dict[str, object]:
    rows = transfer_summary[
        transfer_summary["threshold_name"].astype(str).eq(str(threshold_name))
    ].copy()
    if rows.empty:
        return {
            "transfer_checked": False,
            "transfer_leakage_detected": False,
            "transfer_recovery_loss_detected": False,
            "transfer_truth_recovery_retention": pd.NA,
            "transfer_null_like_leakage_rate": pd.NA,
            "transfer_nonrecovery_leakage_rate": pd.NA,
        }
    truth_retention = pd.to_numeric(
        rows["truth_recovery_retention"],
        errors="coerce",
    ).min()
    null_leak = pd.to_numeric(rows["null_like_leakage_rate"], errors="coerce").max()
    nonrecovery_leak = pd.to_numeric(
        rows["nonrecovery_leakage_rate"],
        errors="coerce",
    ).max()
    return {
        "transfer_checked": True,
        "transfer_leakage_detected": bool((null_leak > 0.0) or (nonrecovery_leak > 0.0)),
        "transfer_recovery_loss_detected": bool(truth_retention < 1.0),
        "transfer_truth_recovery_retention": float(truth_retention),
        "transfer_null_like_leakage_rate": float(null_leak),
        "transfer_nonrecovery_leakage_rate": float(nonrecovery_leak),
    }


def _classify_stage(
    stage: pd.Series,
    transfer: dict[str, object],
) -> tuple[str, str, bool, str]:
    stage_name = str(stage["stage_name"])
    target_rate = float(stage["target_rate"]) if pd.notna(stage["target_rate"]) else 0.0
    negative_rate = float(stage["negative_rate"]) if pd.notna(stage["negative_rate"]) else 0.0
    transfer_checked = bool(transfer["transfer_checked"])
    leakage = bool(transfer["transfer_leakage_detected"])
    recovery_loss = bool(transfer["transfer_recovery_loss_detected"])

    if stage_name == "continuous_structural_stable_accept":
        if negative_rate == 0.0 and target_rate > 0.0:
            return (
                "stable_reporting_candidate",
                "report_stable_regions_only",
                False,
                "Clean focused stable region; not a full traversal calibration.",
            )
    if stage_name == "weak_fragment_risk_block":
        if negative_rate == 0.0 and target_rate > 0.0:
            return (
                "diagnostic_only_guard_candidate",
                "diagnostic_fragment_guard_experiment",
                False,
                "Focused fragment guard is useful but transfer was not established here.",
            )
    if stage_name == "residual_null_evidence_only_unresolved":
        return (
            "selected_family_law_required",
            "keep_unstable_multiscale",
            True,
            "P-value evidence without structural recovery remains unresolved.",
        )
    if transfer_checked and (leakage or recovery_loss):
        return (
            "nontransferable_focused_cutpoint",
            "do_not_promote_threshold",
            True,
            "Held-out transfer shows leakage or recovery loss.",
        )
    if transfer_checked:
        return (
            "transfer_stable_diagnostic_candidate",
            "diagnostic_only_pending_broader_validation",
            False,
            "No transfer leakage detected in checked splits.",
        )
    return (
        "diagnostic_only_unchecked",
        "diagnostic_only",
        False,
        "No transfer evidence available.",
    )


def build_threshold_stability_contract(
    threshold_hierarchy: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build threshold stability rows and summary."""
    _validate_inputs(threshold_hierarchy, transfer_summary)
    rows: list[dict[str, object]] = []
    for _, stage in threshold_hierarchy.sort_values("stage_order").iterrows():
        threshold_name = _transfer_name_for_stage(str(stage["stage_name"]))
        transfer = (
            _transfer_metrics(transfer_summary, threshold_name)
            if threshold_name is not None
            else {
                "transfer_checked": False,
                "transfer_leakage_detected": False,
                "transfer_recovery_loss_detected": False,
                "transfer_truth_recovery_retention": pd.NA,
                "transfer_null_like_leakage_rate": pd.NA,
                "transfer_nonrecovery_leakage_rate": pd.NA,
            }
        )
        status, recommended_use, requires_law, notes = _classify_stage(stage, transfer)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "stage_order": int(stage["stage_order"]),
                "stage_name": str(stage["stage_name"]),
                "selection_unit": str(stage["selection_unit"]),
                "metric": str(stage["metric"]),
                "threshold": stage["threshold"],
                "threshold_stability_status": status,
                "recommended_use": recommended_use,
                "requires_selected_family_law": bool(requires_law),
                "transfer_checked": bool(transfer["transfer_checked"]),
                "transfer_leakage_detected": bool(transfer["transfer_leakage_detected"]),
                "transfer_recovery_loss_detected": bool(
                    transfer["transfer_recovery_loss_detected"]
                ),
                "focused_target_rate": stage["target_rate"],
                "focused_negative_rate": stage["negative_rate"],
                "transfer_truth_recovery_retention": transfer["transfer_truth_recovery_retention"],
                "transfer_null_like_leakage_rate": transfer["transfer_null_like_leakage_rate"],
                "transfer_nonrecovery_leakage_rate": transfer["transfer_nonrecovery_leakage_rate"],
                "notes": notes,
            }
        )
    stage_rows = pd.DataFrame.from_records(rows, columns=STAGE_STATUS_COLUMNS)
    nontransferable = stage_rows["threshold_stability_status"].eq(
        "nontransferable_focused_cutpoint"
    )
    law_required = stage_rows["requires_selected_family_law"].astype(bool)
    diagnostic_only = stage_rows["threshold_stability_status"].str.contains(
        "diagnostic_only",
        regex=False,
    )
    stable = stage_rows["threshold_stability_status"].eq("stable_reporting_candidate")
    if bool(nontransferable.any()) or bool(law_required.any()):
        production_status = "fail_closed_selected_family_law_required"
    elif bool(diagnostic_only.any()):
        production_status = "diagnostic_only"
    else:
        production_status = "production_candidate_requires_external_validation"
    summary = pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "contract_id": "overlap_threshold_stability",
                "stage_count": int(stage_rows.shape[0]),
                "stable_reporting_count": int(stable.sum()),
                "diagnostic_only_count": int(diagnostic_only.sum()),
                "nontransferable_count": int(nontransferable.sum()),
                "law_required_count": int(law_required.sum()),
                "production_promotion_status": production_status,
                "blocking_stage_names": ";".join(
                    stage_rows.loc[
                        nontransferable | law_required,
                        "stage_name",
                    ].astype(str)
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return stage_rows, summary


def run_overlap_threshold_stability_contract(
    config: OverlapThresholdStabilityContractConfig,
) -> dict[str, Path]:
    """Run threshold stability contract and write outputs."""
    hierarchy = pd.read_csv(config.threshold_hierarchy_path)
    transfer_summary = pd.read_csv(config.transfer_summary_path)
    stage_rows, summary = build_threshold_stability_contract(hierarchy, transfer_summary)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows.to_csv(config.stage_status_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "threshold_hierarchy_path": str(config.threshold_hierarchy_path),
        "transfer_summary_path": str(config.transfer_summary_path),
        "outputs": {
            "stage_status": str(config.stage_status_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "stage_status": config.stage_status_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-hierarchy-path", required=True, type=Path)
    parser.add_argument("--transfer-summary-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_threshold_stability_contract(
        OverlapThresholdStabilityContractConfig(
            threshold_hierarchy_path=args.threshold_hierarchy_path,
            transfer_summary_path=args.transfer_summary_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
