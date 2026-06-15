"""Edge-test conditioning scan for context-negative emergent overlap rows.

The Bayesian incidence-mode diagnostic leaves `context_negative_emergent`
fail-closed when parent context and branch incidence do not separate truth
recovery from negatives. This post-run panel asks whether child-parent edge
tests add a useful conditioning variable inside exactly that ambiguous subset.

This is diagnostic-only. A separator is reported only when an edge-evidence
metric retains all truth-recovery rows with zero negative leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.overlap_weak_zone_separability import (
    rank_auc,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_context_negative_edge_conditioning"
SCHEMA_VERSION = "overlap_context_negative_edge_conditioning/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "overlap_context_negative_edge_conditioning"
)

CONTEXT_NEGATIVE_STATUS = "context_negative_emergent_mode_ambiguous"

BASE_EDGE_METRICS = (
    "incoming_edge_neglog10_bh_p_value",
    "incoming_sibling_edge_neglog10_bh_p_value",
    "outgoing_left_edge_neglog10_bh_p_value",
    "outgoing_right_edge_neglog10_bh_p_value",
    "min_outgoing_edge_neglog10_bh_p_value",
    "max_outgoing_edge_neglog10_bh_p_value",
    "outgoing_edge_neglog10_balance",
    "incoming_edge_rejected",
    "incoming_sibling_edge_rejected",
    "outgoing_left_edge_rejected",
    "outgoing_right_edge_rejected",
    "outgoing_edges_both_rejected",
    "incoming_edges_both_rejected",
)

DERIVED_EDGE_METRICS = (
    "incoming_edge_evidence_min",
    "incoming_edge_evidence_max",
    "outgoing_edge_evidence_min",
    "outgoing_edge_evidence_max",
    "outgoing_minus_incoming_edge_evidence_gap",
    "incoming_minus_outgoing_edge_evidence_gap",
    "incoming_edge_rejection_count",
    "outgoing_edge_rejection_count",
    "edge_rejection_excess",
)

DEFAULT_EDGE_METRICS = BASE_EDGE_METRICS + DERIVED_EDGE_METRICS

MODE_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "bayesian_incidence_mode_status",
}

BRANCH_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    *BASE_EDGE_METRICS,
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "bayesian_incidence_mode_status",
    *DEFAULT_EDGE_METRICS,
)

METRIC_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "truth_count",
    "negative_count",
    "finite_truth_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "high_direction_auc",
    "low_direction_auc",
    "truth_min",
    "truth_median",
    "truth_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_threshold",
    "zero_negative_direction",
    "zero_negative_truth_count",
    "zero_negative_truth_retention",
    "zero_negative_negative_count",
    "zero_negative_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "context_negative_emergent_count",
    "truth_recovery_count",
    "negative_count",
    "metric_count",
    "separator_metric_count",
    "best_separator_metric",
    "best_separator_direction",
    "best_separator_threshold",
    "best_separator_truth_retention",
    "diagnostic_status",
)

THRESHOLD_SCAN_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "direction",
    "threshold",
    "truth_selected_count",
    "negative_selected_count",
    "truth_total",
    "negative_total",
    "truth_retention",
    "negative_selection_rate",
)


@dataclass(frozen=True)
class OverlapContextNegativeEdgeConditioningConfig:
    """Runtime contract for context-negative edge conditioning diagnostics."""

    branch_rows_path: Path
    mode_rows_path: Path
    output_dir: Path
    metrics: tuple[str, ...] = DEFAULT_EDGE_METRICS

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_edge_conditioning_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_context_negative_edge_conditioning_metric_summary.csv"
        )

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_edge_conditioning_summary.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_context_negative_edge_conditioning_threshold_scan.csv"
        )

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one edge-conditioning metric is required.")
    unknown = sorted(set(metrics) - set(DEFAULT_EDGE_METRICS))
    if unknown:
        raise ValueError(f"Unknown edge-conditioning metrics: {unknown!r}")
    return metrics


def _validate_columns(rows: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{label} rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _safe_min(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left, right], axis=1).min(axis=1)


def _safe_max(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left, right], axis=1).max(axis=1)


def add_edge_conditioning_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with derived edge-evidence conditioning metrics."""
    enriched = rows.copy()
    incoming = _numeric(enriched, "incoming_edge_neglog10_bh_p_value")
    incoming_sibling = _numeric(
        enriched,
        "incoming_sibling_edge_neglog10_bh_p_value",
    )
    outgoing_left = _numeric(enriched, "outgoing_left_edge_neglog10_bh_p_value")
    outgoing_right = _numeric(enriched, "outgoing_right_edge_neglog10_bh_p_value")
    incoming_min = _safe_min(incoming, incoming_sibling)
    incoming_max = _safe_max(incoming, incoming_sibling)
    outgoing_min = _safe_min(outgoing_left, outgoing_right)
    outgoing_max = _safe_max(outgoing_left, outgoing_right)
    enriched["incoming_edge_evidence_min"] = incoming_min
    enriched["incoming_edge_evidence_max"] = incoming_max
    enriched["outgoing_edge_evidence_min"] = outgoing_min
    enriched["outgoing_edge_evidence_max"] = outgoing_max
    enriched["outgoing_minus_incoming_edge_evidence_gap"] = (
        outgoing_min - incoming_max
    )
    enriched["incoming_minus_outgoing_edge_evidence_gap"] = (
        incoming_min - outgoing_max
    )
    incoming_rejections = (
        _bool_series(enriched["incoming_edge_rejected"]).astype(int)
        + _bool_series(enriched["incoming_sibling_edge_rejected"]).astype(int)
    )
    outgoing_rejections = (
        _bool_series(enriched["outgoing_left_edge_rejected"]).astype(int)
        + _bool_series(enriched["outgoing_right_edge_rejected"]).astype(int)
    )
    enriched["incoming_edge_rejection_count"] = incoming_rejections
    enriched["outgoing_edge_rejection_count"] = outgoing_rejections
    enriched["edge_rejection_excess"] = outgoing_rejections - incoming_rejections
    return enriched


def build_context_negative_edge_conditioning_rows(
    *,
    branch_rows: pd.DataFrame,
    mode_rows: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_EDGE_METRICS,
) -> pd.DataFrame:
    """Join ambiguous incidence-mode rows to child-parent edge evidence."""
    requested_metrics = tuple(metrics)
    _validate_columns(mode_rows, MODE_REQUIRED_COLUMNS, "Incidence-mode")
    _validate_columns(branch_rows, BRANCH_REQUIRED_COLUMNS, "Branch-incidence")
    missing_metrics = sorted(set(requested_metrics) - set(DEFAULT_EDGE_METRICS))
    if missing_metrics:
        raise ValueError(f"Unknown edge-conditioning metrics: {missing_metrics!r}")

    keys = ["case_id", "data_role", "replicate", "node_id"]
    mode_columns = [
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "guard_truth_role",
        "bayesian_incidence_mode_status",
    ]
    ambiguous = mode_rows.loc[
        mode_rows["bayesian_incidence_mode_status"].astype(str).eq(
            CONTEXT_NEGATIVE_STATUS
        ),
        mode_columns,
    ].copy()
    branch = branch_rows[keys + list(BASE_EDGE_METRICS)].copy()
    rows = ambiguous.merge(branch, on=keys, how="left", validate="one_to_one")
    rows = add_edge_conditioning_metrics(rows)
    records = []
    for _, row in rows.iterrows():
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": str(row["case_id"]),
            "data_role": str(row["data_role"]),
            "replicate": int(row["replicate"]),
            "node_id": str(row["node_id"]),
            "guard_truth_role": str(row["guard_truth_role"]),
            "bayesian_incidence_mode_status": str(
                row["bayesian_incidence_mode_status"]
            ),
        }
        for metric in DEFAULT_EDGE_METRICS:
            value = row[metric]
            if metric.endswith("_rejected") or metric.endswith("_both_rejected"):
                record[metric] = _bool_value(value)
            else:
                try:
                    record[metric] = float(value)
                except (TypeError, ValueError):
                    record[metric] = math.nan
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _zero_negative_separator(
    *,
    truth_values: Sequence[float],
    negative_values: Sequence[float],
    direction: str,
) -> tuple[float, int, float, int, str]:
    truth = _finite(truth_values)
    negative = _finite(negative_values)
    if truth.size == 0 or negative.size == 0:
        return math.nan, 0, math.nan, 0, "zero_negative_undefined"
    if direction == "greater_equal":
        threshold = float(np.nextafter(float(np.max(negative)), math.inf))
        selected_truth = truth >= threshold
        selected_negative = negative >= threshold
    elif direction == "less_equal":
        threshold = float(np.nextafter(float(np.min(negative)), -math.inf))
        selected_truth = truth <= threshold
        selected_negative = negative <= threshold
    else:
        raise ValueError(f"Unknown separator direction: {direction!r}")
    truth_count = int(selected_truth.sum())
    negative_count = int(selected_negative.sum())
    retention = float(truth_count / truth.size)
    if truth_count == truth.size and negative_count == 0:
        status = "zero_negative_separates_all_truth"
    elif truth_count > 0 and negative_count == 0:
        status = "zero_negative_partial_truth_retention"
    elif truth_count == 0 and negative_count == 0:
        status = "zero_negative_no_truth_retention"
    else:
        status = "zero_negative_leakage"
    return threshold, truth_count, retention, negative_count, status


def threshold_scan_for_metric(rows: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Scan observed thresholds for one edge-conditioning metric."""
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    values = _numeric(rows, metric)
    finite = values[np.isfinite(values)]
    thresholds = np.sort(finite.unique())
    records: list[dict[str, object]] = []
    truth_total = int(truth.sum())
    negative_total = int(negative.sum())
    for direction in ("greater_equal", "less_equal"):
        for threshold in thresholds:
            selected = values.ge(float(threshold))
            if direction == "less_equal":
                selected = values.le(float(threshold))
            truth_selected = int((selected & truth).sum())
            negative_selected = int((selected & negative).sum())
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "metric": metric,
                    "direction": direction,
                    "threshold": float(threshold),
                    "truth_selected_count": truth_selected,
                    "negative_selected_count": negative_selected,
                    "truth_total": truth_total,
                    "negative_total": negative_total,
                    "truth_retention": (
                        float(truth_selected / truth_total)
                        if truth_total
                        else math.nan
                    ),
                    "negative_selection_rate": (
                        float(negative_selected / negative_total)
                        if negative_total
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=THRESHOLD_SCAN_COLUMNS)


def summarize_metric(
    rows: pd.DataFrame,
    *,
    metric: str,
) -> dict[str, object]:
    """Summarize one edge metric inside context-negative emergent rows."""
    roles = rows["guard_truth_role"].astype(str)
    truth_mask = roles.eq("truth_recovery")
    negative_mask = ~truth_mask
    values = _numeric(rows, metric)
    truth_values = _finite(values[truth_mask].to_numpy(dtype=float))
    negative_values = _finite(values[negative_mask].to_numpy(dtype=float))
    high_auc = rank_auc(truth_values, negative_values)
    low_auc = 1.0 - high_auc if math.isfinite(high_auc) else math.nan
    if not math.isfinite(high_auc):
        best_direction = "undefined"
        best_auc = math.nan
    elif high_auc >= low_auc:
        best_direction = "greater_equal"
        best_auc = high_auc
    else:
        best_direction = "less_equal"
        best_auc = low_auc

    high_sep = _zero_negative_separator(
        truth_values=truth_values,
        negative_values=negative_values,
        direction="greater_equal",
    )
    low_sep = _zero_negative_separator(
        truth_values=truth_values,
        negative_values=negative_values,
        direction="less_equal",
    )
    candidates = [
        ("greater_equal", *high_sep),
        ("less_equal", *low_sep),
    ]
    candidates.sort(
        key=lambda item: (
            item[3] if math.isfinite(item[3]) else -1.0,
            item[2],
            -item[4],
        ),
        reverse=True,
    )
    (
        separator_direction,
        separator_threshold,
        separator_truth_count,
        separator_retention,
        separator_negative_count,
        separator_status,
    ) = candidates[0]

    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric": metric,
        "truth_count": int(truth_mask.sum()),
        "negative_count": int(negative_mask.sum()),
        "finite_truth_count": int(truth_values.size),
        "finite_negative_count": int(negative_values.size),
        "best_direction": best_direction,
        "best_auc": float(best_auc) if math.isfinite(best_auc) else math.nan,
        "high_direction_auc": (
            float(high_auc) if math.isfinite(high_auc) else math.nan
        ),
        "low_direction_auc": float(low_auc) if math.isfinite(low_auc) else math.nan,
        "truth_min": float(np.min(truth_values)) if truth_values.size else math.nan,
        "truth_median": (
            float(np.median(truth_values)) if truth_values.size else math.nan
        ),
        "truth_max": float(np.max(truth_values)) if truth_values.size else math.nan,
        "negative_min": (
            float(np.min(negative_values)) if negative_values.size else math.nan
        ),
        "negative_median": (
            float(np.median(negative_values)) if negative_values.size else math.nan
        ),
        "negative_max": (
            float(np.max(negative_values)) if negative_values.size else math.nan
        ),
        "zero_negative_threshold": separator_threshold,
        "zero_negative_direction": separator_direction,
        "zero_negative_truth_count": separator_truth_count,
        "zero_negative_truth_retention": separator_retention,
        "zero_negative_negative_count": separator_negative_count,
        "zero_negative_status": separator_status,
    }


def summarize_context_negative_edge_conditioning(
    rows: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_EDGE_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one-row overall summary and per-metric edge summaries."""
    requested_metrics = tuple(metrics)
    if rows.empty:
        empty_metrics = pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS)
        summary = pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "context_negative_emergent_count": 0,
                    "truth_recovery_count": 0,
                    "negative_count": 0,
                    "metric_count": len(requested_metrics),
                    "separator_metric_count": 0,
                    "best_separator_metric": "",
                    "best_separator_direction": "",
                    "best_separator_threshold": math.nan,
                    "best_separator_truth_retention": math.nan,
                    "diagnostic_status": "edge_conditioning_unavailable",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
        return summary, empty_metrics

    missing = sorted(set(requested_metrics) - set(rows.columns))
    if missing:
        raise ValueError(f"Conditioning rows are missing metrics: {missing!r}")

    metric_rows = pd.DataFrame.from_records(
        [summarize_metric(rows, metric=metric) for metric in requested_metrics],
        columns=METRIC_SUMMARY_COLUMNS,
    )
    full = metric_rows["zero_negative_status"].eq(
        "zero_negative_separates_all_truth"
    )
    partial = metric_rows["zero_negative_status"].eq(
        "zero_negative_partial_truth_retention"
    )
    if bool(full.any()):
        diagnostic_status = "edge_conditioning_separator_found"
        candidates = metric_rows.loc[full].sort_values(
            ["zero_negative_truth_retention", "best_auc"],
            ascending=[False, False],
        )
    elif bool(partial.any()):
        diagnostic_status = "edge_conditioning_partial_zero_negative_retention"
        candidates = metric_rows.loc[partial].sort_values(
            ["zero_negative_truth_retention", "best_auc"],
            ascending=[False, False],
        )
    else:
        diagnostic_status = "edge_conditioning_no_zero_negative_separator"
        candidates = metric_rows.sort_values("best_auc", ascending=False)

    best = candidates.iloc[0] if not candidates.empty else pd.Series(dtype=object)
    roles = rows["guard_truth_role"].astype(str)
    truth_count = int(roles.eq("truth_recovery").sum())
    summary = pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "context_negative_emergent_count": int(rows.shape[0]),
                "truth_recovery_count": truth_count,
                "negative_count": int(rows.shape[0] - truth_count),
                "metric_count": len(requested_metrics),
                "separator_metric_count": int(full.sum()),
                "best_separator_metric": str(best.get("metric", "")),
                "best_separator_direction": str(
                    best.get("zero_negative_direction", "")
                ),
                "best_separator_threshold": float(
                    best.get("zero_negative_threshold", math.nan)
                ),
                "best_separator_truth_retention": float(
                    best.get("zero_negative_truth_retention", math.nan)
                ),
                "diagnostic_status": diagnostic_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return summary, metric_rows


def run_overlap_context_negative_edge_conditioning(
    config: OverlapContextNegativeEdgeConditioningConfig,
) -> dict[str, Path]:
    """Run context-negative edge conditioning diagnostics and write outputs."""
    branch_rows = pd.read_csv(config.branch_rows_path)
    mode_rows = pd.read_csv(config.mode_rows_path)
    rows = build_context_negative_edge_conditioning_rows(
        branch_rows=branch_rows,
        mode_rows=mode_rows,
        metrics=config.metrics,
    )
    summary, metric_summary = summarize_context_negative_edge_conditioning(
        rows,
        metrics=config.metrics,
    )
    scans = [
        threshold_scan_for_metric(rows, metric=metric)
        for metric in config.metrics
        if metric in rows.columns
    ]
    threshold_scan = (
        pd.concat(scans, ignore_index=True)
        if scans
        else pd.DataFrame(columns=THRESHOLD_SCAN_COLUMNS)
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    metric_summary.to_csv(config.metric_summary_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    threshold_scan.to_csv(config.threshold_scan_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "branch_rows_path": str(config.branch_rows_path),
        "mode_rows_path": str(config.mode_rows_path),
        "context_negative_status": CONTEXT_NEGATIVE_STATUS,
        "metrics": list(config.metrics),
        "outputs": {
            "rows": str(config.rows_path),
            "metric_summary": str(config.metric_summary_path),
            "summary": str(config.summary_path),
            "threshold_scan": str(config.threshold_scan_path),
        },
        "production_status": "diagnostic_only_fail_closed_edge_conditioning_scan",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "metric_summary": config.metric_summary_path,
        "summary": config.summary_path,
        "threshold_scan": config.threshold_scan_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-rows-path", type=Path, required=True)
    parser.add_argument("--mode-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=_parse_metric_list,
        default=DEFAULT_EDGE_METRICS,
        help="Comma-separated edge-conditioning metrics to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_context_negative_edge_conditioning(
        OverlapContextNegativeEdgeConditioningConfig(
            branch_rows_path=args.branch_rows_path,
            mode_rows_path=args.mode_rows_path,
            output_dir=args.output_dir,
            metrics=tuple(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
