"""Transfer validation for context-negative topology conditioning.

The focused topology scan found incoming/outgoing balance relations that
separate one context-negative emergent truth row from negatives. This panel
checks whether such rules transfer across held-out cases or replicates.

It is diagnostic-only. In particular, a perfect full-sample separator is not a
validated traversal law when no held-out fold contains an evaluable truth row.
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

from benchmarks.diagnostics.calibration.overlap.overlap_context_negative_topology_conditioning import (
    DEFAULT_TOPOLOGY_METRICS,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_context_negative_topology_transfer"
SCHEMA_VERSION = "overlap_context_negative_topology_transfer/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_context_negative_topology_transfer"
)

DEFAULT_TRANSFER_METRICS = (
    "balance_product",
    "outgoing_balance",
    "size_balance",
    "edge_norm_balance",
    "outgoing_edge_norm_balance",
    "outgoing_balance_edge_product",
    "outgoing_balance_edge_subspace_product",
)

TRANSFER_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "holdout_value",
    "metric",
    "train_truth_count",
    "train_negative_count",
    "train_direction",
    "train_threshold",
    "train_truth_selected_count",
    "train_truth_retention",
    "train_value_margin",
    "train_rule_status",
    "test_row_count",
    "test_truth_count",
    "test_truth_selected_count",
    "test_truth_retention",
    "test_negative_count",
    "test_negative_selected_count",
    "test_negative_selection_rate",
    "transfer_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "metric",
    "split_count",
    "train_rule_count",
    "train_truth_support_missing_count",
    "train_separator_missing_count",
    "positive_holdout_count",
    "positive_holdout_with_rule_count",
    "test_truth_total",
    "test_truth_selected_total",
    "test_truth_retention",
    "test_negative_total",
    "test_negative_selected_total",
    "test_negative_selection_rate",
    "leakage_split_count",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapContextNegativeTopologyTransferConfig:
    """Runtime contract for context-negative topology transfer diagnostics."""

    topology_rows_path: Path
    output_dir: Path
    metrics: tuple[str, ...] = DEFAULT_TRANSFER_METRICS
    split_columns: tuple[str, ...] = ("case_id", "replicate")
    min_train_truth_count: int = 1

    @property
    def transfer_rows_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_transfer_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_transfer_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one topology-transfer metric is required.")
    unknown = sorted(set(metrics) - set(DEFAULT_TOPOLOGY_METRICS))
    if unknown:
        raise ValueError(f"Unknown topology-transfer metrics: {unknown!r}")
    return metrics


def _parse_split_columns(value: str) -> tuple[str, ...]:
    columns = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not columns:
        raise ValueError("At least one split column is required.")
    return columns


def _validate_rows(
    rows: pd.DataFrame,
    *,
    metrics: Iterable[str],
    split_columns: Sequence[str],
) -> None:
    required = {"guard_truth_role", *metrics, *split_columns}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Topology rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _split_values(rows: pd.DataFrame, split_column: str) -> Iterable[Any]:
    values = rows[split_column].dropna().unique()
    return sorted(values.tolist(), key=str)


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def _learn_zero_negative_rule(
    train: pd.DataFrame,
    *,
    metric: str,
    min_train_truth_count: int = 1,
) -> dict[str, object]:
    roles = train["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    values = _numeric(train, metric)
    truth_values = values[truth].dropna().to_numpy(dtype=float)
    negative_values = values[negative].dropna().to_numpy(dtype=float)
    truth_values = truth_values[np.isfinite(truth_values)]
    negative_values = negative_values[np.isfinite(negative_values)]
    truth_count = int(truth_values.size)
    negative_count = int(negative_values.size)
    if truth_count < int(min_train_truth_count):
        return {
            "direction": "",
            "threshold": math.nan,
            "truth_count": truth_count,
            "negative_count": negative_count,
            "truth_selected_count": 0,
            "truth_retention": math.nan,
            "value_margin": math.nan,
            "status": "train_truth_support_missing",
        }
    if negative_count == 0:
        return {
            "direction": "",
            "threshold": math.nan,
            "truth_count": truth_count,
            "negative_count": negative_count,
            "truth_selected_count": 0,
            "truth_retention": math.nan,
            "value_margin": math.nan,
            "status": "train_negative_support_missing",
        }

    candidates: list[dict[str, object]] = []
    negative_max = float(np.max(negative_values))
    threshold_high = float(np.nextafter(negative_max, math.inf))
    selected_high = truth_values >= threshold_high
    candidates.append(
        {
            "direction": "greater_equal",
            "threshold": threshold_high,
            "truth_selected_count": int(selected_high.sum()),
            "truth_retention": float(selected_high.sum() / truth_count),
            "value_margin": float(np.min(truth_values) - negative_max),
        }
    )
    negative_min = float(np.min(negative_values))
    threshold_low = float(np.nextafter(negative_min, -math.inf))
    selected_low = truth_values <= threshold_low
    candidates.append(
        {
            "direction": "less_equal",
            "threshold": threshold_low,
            "truth_selected_count": int(selected_low.sum()),
            "truth_retention": float(selected_low.sum() / truth_count),
            "value_margin": float(negative_min - np.max(truth_values)),
        }
    )
    candidates.sort(
        key=lambda item: (
            float(item["truth_retention"]),
            int(item["truth_selected_count"]),
            float(item["value_margin"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    if int(best["truth_selected_count"]) == truth_count:
        status = "train_zero_negative_separator"
    elif int(best["truth_selected_count"]) > 0:
        status = "train_partial_truth_zero_negative_separator"
    else:
        status = "train_separator_missing"
    return {
        **best,
        "truth_count": truth_count,
        "negative_count": negative_count,
        "status": status,
    }


def _apply_rule(test: pd.DataFrame, *, metric: str, rule: dict[str, object]) -> dict[str, object]:
    roles = test["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    values = _numeric(test, metric)
    direction = str(rule["direction"])
    threshold = float(rule["threshold"])
    if direction == "greater_equal" and math.isfinite(threshold):
        selected = values.ge(threshold)
    elif direction == "less_equal" and math.isfinite(threshold):
        selected = values.le(threshold)
    else:
        selected = pd.Series(False, index=test.index)
    truth_count = int(truth.sum())
    negative_count = int(negative.sum())
    truth_selected = int((selected & truth).sum())
    negative_selected = int((selected & negative).sum())
    return {
        "test_row_count": int(test.shape[0]),
        "test_truth_count": truth_count,
        "test_truth_selected_count": truth_selected,
        "test_truth_retention": _safe_rate(truth_selected, truth_count),
        "test_negative_count": negative_count,
        "test_negative_selected_count": negative_selected,
        "test_negative_selection_rate": _safe_rate(negative_selected, negative_count),
    }


def _transfer_status(
    *,
    train_status: str,
    test_truth_count: int,
    test_truth_selected_count: int,
    test_negative_selected_count: int,
) -> str:
    if train_status in {
        "train_truth_support_missing",
        "train_negative_support_missing",
        "train_separator_missing",
    }:
        return f"transfer_{train_status}"
    if train_status == "train_partial_truth_zero_negative_separator":
        return "transfer_train_partial_truth_rule"
    if test_negative_selected_count:
        return "transfer_leakage"
    if test_truth_count and test_truth_selected_count == test_truth_count:
        return "transfer_full_truth_retention_zero_leakage"
    if test_truth_selected_count:
        return "transfer_partial_truth_retention_zero_leakage"
    if test_truth_count:
        return "transfer_no_truth_retention_zero_leakage"
    return "transfer_no_truth_in_holdout_zero_leakage"


def build_topology_transfer_rows(
    topology_rows: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_TRANSFER_METRICS,
    split_columns: Sequence[str] = ("case_id", "replicate"),
    min_train_truth_count: int = 1,
) -> pd.DataFrame:
    """Build leave-one-split transfer rows for topology conditioning metrics."""
    _validate_rows(topology_rows, metrics=metrics, split_columns=split_columns)
    records: list[dict[str, object]] = []
    for split_column in split_columns:
        for holdout_value in _split_values(topology_rows, split_column):
            test_mask = topology_rows[split_column].eq(holdout_value)
            train = topology_rows.loc[~test_mask].copy()
            test = topology_rows.loc[test_mask].copy()
            for metric in metrics:
                rule = _learn_zero_negative_rule(
                    train,
                    metric=metric,
                    min_train_truth_count=int(min_train_truth_count),
                )
                test_result = _apply_rule(test, metric=metric, rule=rule)
                status = _transfer_status(
                    train_status=str(rule["status"]),
                    test_truth_count=int(test_result["test_truth_count"]),
                    test_truth_selected_count=int(test_result["test_truth_selected_count"]),
                    test_negative_selected_count=int(test_result["test_negative_selected_count"]),
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "split_kind": f"leave_one_{split_column}",
                        "holdout_value": str(holdout_value),
                        "metric": metric,
                        "train_truth_count": int(rule["truth_count"]),
                        "train_negative_count": int(rule["negative_count"]),
                        "train_direction": str(rule["direction"]),
                        "train_threshold": float(rule["threshold"]),
                        "train_truth_selected_count": int(rule["truth_selected_count"]),
                        "train_truth_retention": float(rule["truth_retention"]),
                        "train_value_margin": float(rule["value_margin"]),
                        "train_rule_status": str(rule["status"]),
                        **test_result,
                        "transfer_status": status,
                    }
                )
    return pd.DataFrame.from_records(records, columns=TRANSFER_COLUMNS)


def _summary_status(group: pd.DataFrame) -> str:
    with_rule = group["train_rule_status"].astype(str).eq("train_zero_negative_separator")
    positive_holdout = group["test_truth_count"].gt(0)
    positive_with_rule = positive_holdout & with_rule
    leakage = group["test_negative_selected_count"].gt(0)
    if not bool(with_rule.any()):
        return "transfer_no_train_separator"
    if not bool(positive_with_rule.any()) and bool(leakage[with_rule].any()):
        return "transfer_unvalidated_no_truth_holdout_support_with_leakage"
    if not bool(positive_with_rule.any()):
        return "transfer_unvalidated_no_truth_holdout_support"
    if bool(leakage[with_rule].any()):
        return "transfer_leakage"
    tested_truth = int(group.loc[positive_with_rule, "test_truth_count"].sum())
    selected_truth = int(group.loc[positive_with_rule, "test_truth_selected_count"].sum())
    if tested_truth and selected_truth == tested_truth:
        return "transfer_validated_zero_leakage_full_truth_retention"
    if selected_truth:
        return "transfer_validated_zero_leakage_partial_truth_retention"
    return "transfer_validated_zero_leakage_no_truth_retention"


def summarize_topology_transfer_rows(transfer_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize topology transfer rows by split kind and metric."""
    records: list[dict[str, object]] = []
    for (split_kind, metric), group in transfer_rows.groupby(
        ["split_kind", "metric"],
        sort=True,
    ):
        with_rule = group["train_rule_status"].astype(str).eq("train_zero_negative_separator")
        truth_missing = group["train_rule_status"].astype(str).eq("train_truth_support_missing")
        separator_missing = (
            group["train_rule_status"]
            .astype(str)
            .isin(
                {
                    "train_separator_missing",
                    "train_partial_truth_zero_negative_separator",
                }
            )
        )
        positive_holdout = group["test_truth_count"].gt(0)
        positive_with_rule = positive_holdout & with_rule
        test_truth_total = int(group.loc[with_rule, "test_truth_count"].sum())
        test_truth_selected = int(group.loc[with_rule, "test_truth_selected_count"].sum())
        test_negative_total = int(group.loc[with_rule, "test_negative_count"].sum())
        test_negative_selected = int(group.loc[with_rule, "test_negative_selected_count"].sum())
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "split_kind": str(split_kind),
                "metric": str(metric),
                "split_count": int(group["holdout_value"].nunique()),
                "train_rule_count": int(with_rule.sum()),
                "train_truth_support_missing_count": int(truth_missing.sum()),
                "train_separator_missing_count": int(separator_missing.sum()),
                "positive_holdout_count": int(positive_holdout.sum()),
                "positive_holdout_with_rule_count": int(positive_with_rule.sum()),
                "test_truth_total": test_truth_total,
                "test_truth_selected_total": test_truth_selected,
                "test_truth_retention": _safe_rate(
                    test_truth_selected,
                    test_truth_total,
                ),
                "test_negative_total": test_negative_total,
                "test_negative_selected_total": test_negative_selected,
                "test_negative_selection_rate": _safe_rate(
                    test_negative_selected,
                    test_negative_total,
                ),
                "leakage_split_count": int(
                    (with_rule & group["test_negative_selected_count"].gt(0)).sum()
                ),
                "diagnostic_status": _summary_status(group),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_context_negative_topology_transfer(
    config: OverlapContextNegativeTopologyTransferConfig,
) -> dict[str, Path]:
    """Run topology transfer diagnostics and write outputs."""
    topology_rows = pd.read_csv(config.topology_rows_path)
    transfer_rows = build_topology_transfer_rows(
        topology_rows,
        metrics=config.metrics,
        split_columns=config.split_columns,
        min_train_truth_count=int(config.min_train_truth_count),
    )
    summary = summarize_topology_transfer_rows(transfer_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    transfer_rows.to_csv(config.transfer_rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "topology_rows_path": str(config.topology_rows_path),
        "metrics": list(config.metrics),
        "split_columns": list(config.split_columns),
        "min_train_truth_count": int(config.min_train_truth_count),
        "outputs": {
            "transfer_rows": str(config.transfer_rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_transfer_support_audit",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "transfer_rows": config.transfer_rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=_parse_metric_list,
        default=DEFAULT_TRANSFER_METRICS,
        help="Comma-separated topology metrics to transfer-test.",
    )
    parser.add_argument(
        "--split-columns",
        type=_parse_split_columns,
        default=("case_id", "replicate"),
        help="Comma-separated columns for leave-one-split validation.",
    )
    parser.add_argument("--min-train-truth-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_context_negative_topology_transfer(
        OverlapContextNegativeTopologyTransferConfig(
            topology_rows_path=args.topology_rows_path,
            output_dir=args.output_dir,
            metrics=tuple(args.metrics),
            split_columns=tuple(args.split_columns),
            min_train_truth_count=int(args.min_train_truth_count),
        )
    )


if __name__ == "__main__":
    main()
