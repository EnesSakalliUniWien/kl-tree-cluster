"""Transfer diagnostics for residual overlap recovery thresholds.

The residual eligibility thresholds are exact max-negative thresholds on a
small focused panel. This diagnostic recomputes them on leave-one-case and
leave-one-replicate training splits, then evaluates held-out retention and
leakage. It is diagnostic-only and does not define a production rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_residual_threshold_transfer_not_calibration"
SCHEMA_VERSION = "overlap_residual_threshold_transfer/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_residual_threshold_transfer"

P_VALUE_EVIDENCE_METRIC = "residual_neg_log10_min_sibling_p_value"
STRUCTURAL_EVIDENCE_METRIC = "residual_min_fragment_risk_proxy_score"

THRESHOLD_SPECS = (
    (
        "null_evidence_threshold",
        P_VALUE_EVIDENCE_METRIC,
        ("residual_null_like_family",),
    ),
    (
        "nonrecovery_structural_threshold",
        STRUCTURAL_EVIDENCE_METRIC,
        ("residual_nonrecovery_family",),
    ),
    (
        "full_negative_structural_threshold",
        STRUCTURAL_EVIDENCE_METRIC,
        ("residual_null_like_family", "residual_nonrecovery_family"),
    ),
)

TRANSFER_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "holdout_value",
    "threshold_name",
    "metric",
    "train_threshold",
    "train_negative_roles",
    "train_negative_count",
    "train_truth_recovery_count",
    "test_family_count",
    "test_truth_recovery_count",
    "test_truth_recovery_pass_count",
    "test_truth_recovery_retention",
    "test_null_like_count",
    "test_null_like_pass_count",
    "test_null_like_leakage_rate",
    "test_nonrecovery_count",
    "test_nonrecovery_pass_count",
    "test_nonrecovery_leakage_rate",
    "transfer_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "threshold_name",
    "metric",
    "split_count",
    "defined_threshold_split_count",
    "truth_recovery_test_total",
    "truth_recovery_test_pass_total",
    "truth_recovery_retention",
    "null_like_test_total",
    "null_like_test_pass_total",
    "null_like_leakage_rate",
    "nonrecovery_test_total",
    "nonrecovery_test_pass_total",
    "nonrecovery_leakage_rate",
    "min_train_threshold",
    "median_train_threshold",
    "max_train_threshold",
    "transfer_status",
)


@dataclass(frozen=True)
class OverlapResidualThresholdTransferConfig:
    """Runtime contract for residual threshold transfer diagnostics."""

    residual_family_rows_path: Path
    output_dir: Path
    split_columns: tuple[str, ...] = ("case_id", "replicate")

    @property
    def transfer_rows_path(self) -> Path:
        return self.output_dir / "overlap_residual_threshold_transfer_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_residual_threshold_transfer_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_rows(rows: pd.DataFrame, split_columns: Sequence[str]) -> None:
    required = {
        "residual_family_truth_role",
        P_VALUE_EVIDENCE_METRIC,
        STRUCTURAL_EVIDENCE_METRIC,
        *split_columns,
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Residual family rows are missing columns: {missing!r}")


def _next_after_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(np.nextafter(float(numeric.max()), math.inf))


def _split_values(rows: pd.DataFrame, split_column: str) -> Iterable[Any]:
    values = rows[split_column].dropna().unique()
    return sorted(values.tolist(), key=str)


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def _transfer_status(
    *,
    threshold_defined: bool,
    truth_total: int,
    truth_pass: int,
    null_pass: int,
    nonrecovery_pass: int,
) -> str:
    if not threshold_defined:
        return "transfer_threshold_undefined"
    if null_pass or nonrecovery_pass:
        return "transfer_leakage"
    if truth_total and truth_pass == truth_total:
        return "transfer_full_recovery_retention"
    if truth_pass:
        return "transfer_partial_recovery_retention"
    if truth_total:
        return "transfer_no_recovery_retention"
    return "transfer_no_truth_recovery_in_holdout"


def build_residual_threshold_transfer_rows(
    family_rows: pd.DataFrame,
    *,
    split_columns: Sequence[str] = ("case_id", "replicate"),
) -> pd.DataFrame:
    """Build leave-one-split transfer rows for residual thresholds."""
    _validate_rows(family_rows, split_columns)
    records: list[dict[str, object]] = []
    for split_column in split_columns:
        for holdout_value in _split_values(family_rows, split_column):
            test_mask = family_rows[split_column].eq(holdout_value)
            train = family_rows.loc[~test_mask].copy()
            test = family_rows.loc[test_mask].copy()
            train_roles = train["residual_family_truth_role"].astype(str)
            test_roles = test["residual_family_truth_role"].astype(str)
            for threshold_name, metric, negative_roles in THRESHOLD_SPECS:
                train_negative = train_roles.isin(set(negative_roles))
                threshold = _next_after_max(train.loc[train_negative, metric])
                threshold_defined = not math.isnan(threshold)
                values = pd.to_numeric(test[metric], errors="coerce")
                passed = values.ge(threshold) if threshold_defined else values.eq(
                    math.inf
                )
                truth = test_roles.eq("residual_truth_recovery_family")
                null_like = test_roles.eq("residual_null_like_family")
                nonrecovery = test_roles.eq("residual_nonrecovery_family")
                truth_total = int(truth.sum())
                truth_pass = int((passed & truth).sum())
                null_total = int(null_like.sum())
                null_pass = int((passed & null_like).sum())
                nonrecovery_total = int(nonrecovery.sum())
                nonrecovery_pass = int((passed & nonrecovery).sum())
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "split_kind": f"leave_one_{split_column}",
                        "holdout_value": str(holdout_value),
                        "threshold_name": threshold_name,
                        "metric": metric,
                        "train_threshold": threshold,
                        "train_negative_roles": "|".join(negative_roles),
                        "train_negative_count": int(train_negative.sum()),
                        "train_truth_recovery_count": int(
                            train_roles.eq("residual_truth_recovery_family").sum()
                        ),
                        "test_family_count": int(test.shape[0]),
                        "test_truth_recovery_count": truth_total,
                        "test_truth_recovery_pass_count": truth_pass,
                        "test_truth_recovery_retention": _safe_rate(
                            truth_pass,
                            truth_total,
                        ),
                        "test_null_like_count": null_total,
                        "test_null_like_pass_count": null_pass,
                        "test_null_like_leakage_rate": _safe_rate(
                            null_pass,
                            null_total,
                        ),
                        "test_nonrecovery_count": nonrecovery_total,
                        "test_nonrecovery_pass_count": nonrecovery_pass,
                        "test_nonrecovery_leakage_rate": _safe_rate(
                            nonrecovery_pass,
                            nonrecovery_total,
                        ),
                        "transfer_status": _transfer_status(
                            threshold_defined=threshold_defined,
                            truth_total=truth_total,
                            truth_pass=truth_pass,
                            null_pass=null_pass,
                            nonrecovery_pass=nonrecovery_pass,
                        ),
                    }
                )
    return pd.DataFrame.from_records(records, columns=TRANSFER_COLUMNS)


def summarize_residual_threshold_transfer(transfer_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold transfer rows by split kind and threshold."""
    records: list[dict[str, object]] = []
    group_cols = ["split_kind", "threshold_name", "metric"]
    for (split_kind, threshold_name, metric), group in transfer_rows.groupby(
        group_cols,
        sort=True,
    ):
        truth_total = int(group["test_truth_recovery_count"].sum())
        truth_pass = int(group["test_truth_recovery_pass_count"].sum())
        null_total = int(group["test_null_like_count"].sum())
        null_pass = int(group["test_null_like_pass_count"].sum())
        nonrecovery_total = int(group["test_nonrecovery_count"].sum())
        nonrecovery_pass = int(group["test_nonrecovery_pass_count"].sum())
        thresholds = pd.to_numeric(group["train_threshold"], errors="coerce").dropna()
        if null_pass or nonrecovery_pass:
            status = "transfer_leakage"
        elif truth_total and truth_pass == truth_total:
            status = "transfer_full_recovery_retention"
        elif truth_pass:
            status = "transfer_partial_recovery_retention"
        elif truth_total:
            status = "transfer_no_recovery_retention"
        else:
            status = "transfer_no_truth_recovery_in_holdouts"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "split_kind": str(split_kind),
                "threshold_name": str(threshold_name),
                "metric": str(metric),
                "split_count": int(group.shape[0]),
                "defined_threshold_split_count": int(thresholds.shape[0]),
                "truth_recovery_test_total": truth_total,
                "truth_recovery_test_pass_total": truth_pass,
                "truth_recovery_retention": _safe_rate(truth_pass, truth_total),
                "null_like_test_total": null_total,
                "null_like_test_pass_total": null_pass,
                "null_like_leakage_rate": _safe_rate(null_pass, null_total),
                "nonrecovery_test_total": nonrecovery_total,
                "nonrecovery_test_pass_total": nonrecovery_pass,
                "nonrecovery_leakage_rate": _safe_rate(
                    nonrecovery_pass,
                    nonrecovery_total,
                ),
                "min_train_threshold": (
                    float(thresholds.min()) if not thresholds.empty else math.nan
                ),
                "median_train_threshold": (
                    float(thresholds.median()) if not thresholds.empty else math.nan
                ),
                "max_train_threshold": (
                    float(thresholds.max()) if not thresholds.empty else math.nan
                ),
                "transfer_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def _parse_split_columns(value: str) -> tuple[str, ...]:
    columns = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not columns:
        raise ValueError("At least one split column is required.")
    return columns


def run_overlap_residual_threshold_transfer(
    config: OverlapResidualThresholdTransferConfig,
) -> dict[str, Path]:
    """Run residual threshold transfer diagnostics and write outputs."""
    family_rows = pd.read_csv(config.residual_family_rows_path)
    transfer_rows = build_residual_threshold_transfer_rows(
        family_rows,
        split_columns=config.split_columns,
    )
    summary = summarize_residual_threshold_transfer(transfer_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    transfer_rows.to_csv(config.transfer_rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "residual_family_rows_path": str(config.residual_family_rows_path),
        "split_columns": list(config.split_columns),
        "outputs": {
            "transfer_rows": str(config.transfer_rows_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "transfer_rows": config.transfer_rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-family-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-columns", default="case_id,replicate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_residual_threshold_transfer(
        OverlapResidualThresholdTransferConfig(
            residual_family_rows_path=args.residual_family_rows_path,
            output_dir=args.output_dir,
            split_columns=_parse_split_columns(args.split_columns),
        )
    )


if __name__ == "__main__":
    main()
