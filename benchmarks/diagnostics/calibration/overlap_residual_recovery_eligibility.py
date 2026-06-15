"""Residual recovery eligibility diagnostics for overlap selected families.

This diagnostic turns residual-family threshold evidence into explicit gates:
selected-family null evidence, structural recovery evidence against
non-recovery signal, and stricter structural evidence against all negatives.
It is not a production rule; it exists to make the remaining threshold conflict
visible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_residual_recovery_eligibility_not_calibration"
SCHEMA_VERSION = "overlap_residual_recovery_eligibility/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_residual_recovery_eligibility"

P_VALUE_EVIDENCE_METRIC = "residual_neg_log10_min_sibling_p_value"
STRUCTURAL_EVIDENCE_METRIC = "residual_min_fragment_risk_proxy_score"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "residual_family_size",
    "p_value_evidence_metric",
    "p_value_evidence_value",
    "null_evidence_threshold",
    "null_evidence_pass",
    "structural_evidence_metric",
    "structural_evidence_value",
    "nonrecovery_structural_threshold",
    "nonrecovery_structural_pass",
    "full_negative_structural_threshold",
    "full_negative_structural_pass",
    "residual_recovery_eligibility_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "residual_recovery_eligibility_status",
    "family_count",
    "null_like_family_count",
    "truth_recovery_family_count",
    "nonrecovery_family_count",
    "median_p_value_evidence",
    "median_structural_evidence",
)

THRESHOLD_COLUMNS = (
    "schema_version",
    "study_role",
    "threshold_name",
    "metric",
    "threshold",
    "negative_role",
    "negative_family_count",
    "truth_recovery_family_count",
    "truth_recovery_pass_count",
    "truth_recovery_retention",
    "negative_pass_count",
    "negative_selection_rate",
)


@dataclass(frozen=True)
class OverlapResidualRecoveryEligibilityConfig:
    """Runtime contract for residual recovery eligibility diagnostics."""

    residual_family_rows_path: Path
    output_dir: Path
    p_value_evidence_metric: str = P_VALUE_EVIDENCE_METRIC
    structural_evidence_metric: str = STRUCTURAL_EVIDENCE_METRIC

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_residual_recovery_eligibility_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_residual_recovery_eligibility_summary.csv"

    @property
    def thresholds_path(self) -> Path:
        return self.output_dir / "overlap_residual_recovery_thresholds.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_rows(
    family_rows: pd.DataFrame,
    *,
    p_value_metric: str,
    structural_metric: str,
) -> None:
    missing = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "residual_family_truth_role",
            "residual_family_size",
            p_value_metric,
            structural_metric,
        }
        - set(family_rows.columns)
    )
    if missing:
        raise ValueError(f"Residual family rows are missing columns: {missing!r}")


def _next_after_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(np.nextafter(float(numeric.max()), math.inf))


def _thresholds(
    family_rows: pd.DataFrame,
    *,
    p_value_metric: str,
    structural_metric: str,
) -> dict[str, float]:
    roles = family_rows["residual_family_truth_role"].astype(str)
    null_rows = family_rows[roles.eq("residual_null_like_family")]
    nonrecovery_rows = family_rows[roles.eq("residual_nonrecovery_family")]
    all_negative_rows = family_rows[
        roles.isin({"residual_null_like_family", "residual_nonrecovery_family"})
    ]
    return {
        "null_evidence_threshold": _next_after_max(null_rows[p_value_metric]),
        "nonrecovery_structural_threshold": _next_after_max(
            nonrecovery_rows[structural_metric]
        ),
        "full_negative_structural_threshold": _next_after_max(
            all_negative_rows[structural_metric]
        ),
    }


def assign_residual_recovery_eligibility(
    family_rows: pd.DataFrame,
    *,
    p_value_evidence_metric: str = P_VALUE_EVIDENCE_METRIC,
    structural_evidence_metric: str = STRUCTURAL_EVIDENCE_METRIC,
) -> pd.DataFrame:
    """Assign residual recovery eligibility statuses."""
    _validate_rows(
        family_rows,
        p_value_metric=str(p_value_evidence_metric),
        structural_metric=str(structural_evidence_metric),
    )
    thresholds = _thresholds(
        family_rows,
        p_value_metric=str(p_value_evidence_metric),
        structural_metric=str(structural_evidence_metric),
    )
    p_values = pd.to_numeric(family_rows[p_value_evidence_metric], errors="coerce")
    structural = pd.to_numeric(
        family_rows[structural_evidence_metric],
        errors="coerce",
    )
    null_pass = p_values.ge(thresholds["null_evidence_threshold"])
    nonrecovery_pass = structural.ge(thresholds["nonrecovery_structural_threshold"])
    full_negative_pass = structural.ge(thresholds["full_negative_structural_threshold"])
    statuses = pd.Series(
        "residual_no_selected_family_null_evidence",
        index=family_rows.index,
        dtype=object,
    )
    statuses.loc[null_pass] = "residual_null_evidence_only_unresolved"
    statuses.loc[null_pass & nonrecovery_pass] = (
        "residual_structural_recovery_candidate_nonrecovery_controlled"
    )
    statuses.loc[null_pass & full_negative_pass] = (
        "residual_strict_structural_recovery_candidate"
    )
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": family_rows["case_id"].astype(str),
            "data_role": family_rows["data_role"].astype(str),
            "replicate": family_rows["replicate"].astype(int),
            "residual_family_truth_role": family_rows[
                "residual_family_truth_role"
            ].astype(str),
            "residual_family_size": family_rows["residual_family_size"].astype(int),
            "p_value_evidence_metric": str(p_value_evidence_metric),
            "p_value_evidence_value": p_values,
            "null_evidence_threshold": thresholds["null_evidence_threshold"],
            "null_evidence_pass": null_pass.astype(bool),
            "structural_evidence_metric": str(structural_evidence_metric),
            "structural_evidence_value": structural,
            "nonrecovery_structural_threshold": thresholds[
                "nonrecovery_structural_threshold"
            ],
            "nonrecovery_structural_pass": nonrecovery_pass.astype(bool),
            "full_negative_structural_threshold": thresholds[
                "full_negative_structural_threshold"
            ],
            "full_negative_structural_pass": full_negative_pass.astype(bool),
            "residual_recovery_eligibility_status": statuses,
        },
        columns=ROW_COLUMNS,
    )


def summarize_residual_recovery_eligibility(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual recovery eligibility statuses."""
    records: list[dict[str, object]] = []
    for status, group in rows.groupby("residual_recovery_eligibility_status", sort=True):
        roles = group["residual_family_truth_role"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "residual_recovery_eligibility_status": str(status),
                "family_count": int(group.shape[0]),
                "null_like_family_count": int(
                    roles.eq("residual_null_like_family").sum()
                ),
                "truth_recovery_family_count": int(
                    roles.eq("residual_truth_recovery_family").sum()
                ),
                "nonrecovery_family_count": int(
                    roles.eq("residual_nonrecovery_family").sum()
                ),
                "median_p_value_evidence": _median(group["p_value_evidence_value"]),
                "median_structural_evidence": _median(group["structural_evidence_value"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def summarize_residual_recovery_thresholds(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold retention and leakage."""
    truth = rows["residual_family_truth_role"].astype(str).eq(
        "residual_truth_recovery_family"
    )
    records: list[dict[str, object]] = []
    specs = (
        (
            "null_evidence_threshold",
            "p_value_evidence_metric",
            "null_evidence_pass",
            "residual_null_like_family",
        ),
        (
            "nonrecovery_structural_threshold",
            "structural_evidence_metric",
            "nonrecovery_structural_pass",
            "residual_nonrecovery_family",
        ),
        (
            "full_negative_structural_threshold",
            "structural_evidence_metric",
            "full_negative_structural_pass",
            "residual_null_or_nonrecovery_family",
        ),
    )
    for threshold_name, metric_column, pass_column, negative_role in specs:
        if negative_role == "residual_null_or_nonrecovery_family":
            negative = rows["residual_family_truth_role"].astype(str).isin(
                {"residual_null_like_family", "residual_nonrecovery_family"}
            )
        else:
            negative = rows["residual_family_truth_role"].astype(str).eq(negative_role)
        threshold = float(rows[threshold_name].iloc[0]) if not rows.empty else math.nan
        pass_mask = rows[pass_column].astype(bool)
        truth_total = int(truth.sum())
        negative_total = int(negative.sum())
        truth_pass = int((pass_mask & truth).sum())
        negative_pass = int((pass_mask & negative).sum())
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "threshold_name": threshold_name,
                "metric": str(rows[metric_column].iloc[0]) if not rows.empty else "",
                "threshold": threshold,
                "negative_role": negative_role,
                "negative_family_count": negative_total,
                "truth_recovery_family_count": truth_total,
                "truth_recovery_pass_count": truth_pass,
                "truth_recovery_retention": (
                    float(truth_pass / truth_total) if truth_total else math.nan
                ),
                "negative_pass_count": negative_pass,
                "negative_selection_rate": (
                    float(negative_pass / negative_total) if negative_total else math.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=THRESHOLD_COLUMNS)


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.median())


def run_overlap_residual_recovery_eligibility(
    config: OverlapResidualRecoveryEligibilityConfig,
) -> dict[str, Path]:
    """Run residual recovery eligibility diagnostics and write outputs."""
    family_rows = pd.read_csv(config.residual_family_rows_path)
    rows = assign_residual_recovery_eligibility(
        family_rows,
        p_value_evidence_metric=str(config.p_value_evidence_metric),
        structural_evidence_metric=str(config.structural_evidence_metric),
    )
    summary = summarize_residual_recovery_eligibility(rows)
    thresholds = summarize_residual_recovery_thresholds(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    thresholds.to_csv(config.thresholds_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "residual_family_rows_path": str(config.residual_family_rows_path),
        "p_value_evidence_metric": str(config.p_value_evidence_metric),
        "structural_evidence_metric": str(config.structural_evidence_metric),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "thresholds": str(config.thresholds_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "thresholds": config.thresholds_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-family-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--p-value-evidence-metric", default=P_VALUE_EVIDENCE_METRIC)
    parser.add_argument("--structural-evidence-metric", default=STRUCTURAL_EVIDENCE_METRIC)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_residual_recovery_eligibility(
        OverlapResidualRecoveryEligibilityConfig(
            residual_family_rows_path=args.residual_family_rows_path,
            output_dir=args.output_dir,
            p_value_evidence_metric=str(args.p_value_evidence_metric),
            structural_evidence_metric=str(args.structural_evidence_metric),
        )
    )


if __name__ == "__main__":
    main()
