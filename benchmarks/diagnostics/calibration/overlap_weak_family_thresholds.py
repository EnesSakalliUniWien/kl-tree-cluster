"""Family-wise threshold diagnostics for unstable overlap traversal zones.

This post-run panel aggregates rows in `unstable_weak_homogeneity_zone` by
`case_id`, `data_role`, and `replicate`. It tests whether family-level
statistics, rather than row-wise thresholds, can separate weak truth-aligned
signal families from selected-null or truth-misaligned families.
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

from benchmarks.diagnostics.calibration.overlap_weak_zone_separability import (
    rank_auc,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_weak_family_thresholds_not_calibration"
SCHEMA_VERSION = "overlap_weak_family_thresholds/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_weak_family_thresholds"
DEFAULT_ZONE = "unstable_weak_homogeneity_zone"
DEFAULT_METRICS = (
    "family_size",
    "min_sibling_p_value",
    "neg_log10_min_sibling_p_value",
    "max_homogeneity_gain_min",
    "max_context_homogeneity_margin",
    "max_subspace_consensus_jaccard_topk",
    "min_continuous_homogeneity_threshold",
    "max_depth",
    "median_parent_size",
    "max_barycentric_balance",
)
COMPARISONS = (
    ("signal_truth_aligned_family", "null_like_family", "aligned_family_vs_null"),
    (
        "signal_truth_aligned_family",
        "signal_truth_misaligned_family",
        "aligned_family_vs_misaligned",
    ),
    (
        "signal_truth_aligned_family",
        "null_or_misaligned_family",
        "aligned_family_vs_null_or_misaligned",
    ),
)

FAMILY_COLUMNS = (
    "schema_version",
    "study_role",
    "zone",
    "case_id",
    "data_role",
    "replicate",
    "family_truth_role",
    "family_size",
    "null_like_row_count",
    "truth_aligned_row_count",
    "truth_misaligned_row_count",
    "min_sibling_p_value",
    "neg_log10_min_sibling_p_value",
    "max_homogeneity_gain_min",
    "max_context_homogeneity_margin",
    "max_subspace_consensus_jaccard_topk",
    "min_continuous_homogeneity_threshold",
    "max_depth",
    "median_parent_size",
    "max_barycentric_balance",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "zone",
    "metric",
    "comparison",
    "positive_role",
    "negative_role",
    "positive_family_count",
    "negative_family_count",
    "best_direction",
    "best_auc",
    "high_direction_auc",
    "low_direction_auc",
    "positive_min",
    "positive_median",
    "positive_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_threshold",
    "zero_negative_direction",
    "zero_negative_positive_count",
    "zero_negative_positive_retention",
    "zero_negative_status",
)

THRESHOLD_SCAN_COLUMNS = (
    "schema_version",
    "study_role",
    "zone",
    "metric",
    "comparison",
    "direction",
    "threshold",
    "positive_selected_count",
    "negative_selected_count",
    "positive_total",
    "negative_total",
    "positive_retention",
    "negative_selection_rate",
)


@dataclass(frozen=True)
class OverlapWeakFamilyThresholdConfig:
    """Runtime contract for family-wise weak-zone diagnostics."""

    rows_path: Path
    output_dir: Path
    zone: str = DEFAULT_ZONE
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def family_rows_path(self) -> Path:
        return self.output_dir / "overlap_weak_family_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "overlap_weak_family_metric_separability.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return self.output_dir / "overlap_weak_family_threshold_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one family metric is required.")
    return metrics


def _required_columns() -> set[str]:
    return {
        "structural_decision_zone",
        "case_id",
        "data_role",
        "replicate",
        "structural_truth_role",
        "sibling_p_value",
        "homogeneity_gain_min",
        "subspace_consensus_jaccard_topk",
        "continuous_homogeneity_threshold",
        "depth",
        "n_parent",
        "barycentric_balance",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Decision-zone rows are missing columns: {missing!r}")


def _classify_family_role(group: pd.DataFrame) -> str:
    roles = set(group["structural_truth_role"].astype(str))
    data_roles = set(group["data_role"].astype(str))
    if data_roles <= {"null", "selected_null"} or roles <= {"null_like"}:
        return "null_like_family"
    has_aligned = "signal_truth_aligned" in roles
    has_misaligned = "signal_truth_misaligned" in roles
    if has_aligned:
        return "signal_truth_aligned_family"
    if has_misaligned:
        return "signal_truth_misaligned_family"
    return "signal_unlabeled_family"


def build_weak_family_rows(
    rows: pd.DataFrame,
    *,
    zone: str = DEFAULT_ZONE,
) -> pd.DataFrame:
    """Aggregate unstable weak-zone rows to selected-family rows."""
    _validate_rows(rows)
    working = rows[rows["structural_decision_zone"].astype(str).eq(str(zone))].copy()
    if working.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    working["context_homogeneity_margin"] = (
        pd.to_numeric(working["homogeneity_gain_min"], errors="coerce")
        - pd.to_numeric(
            working["continuous_homogeneity_threshold"],
            errors="coerce",
        )
    )
    records: list[dict[str, object]] = []
    group_cols = ["case_id", "data_role", "replicate"]
    for (case_id, data_role, replicate), group in working.groupby(group_cols, sort=True):
        p_values = pd.to_numeric(group["sibling_p_value"], errors="coerce")
        min_p = float(p_values.min())
        min_p_clipped = max(min_p, float(np.nextafter(0.0, 1.0)))
        roles = group["structural_truth_role"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "zone": str(zone),
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "family_truth_role": _classify_family_role(group),
                "family_size": int(group.shape[0]),
                "null_like_row_count": int(roles.eq("null_like").sum()),
                "truth_aligned_row_count": int(
                    roles.eq("signal_truth_aligned").sum()
                ),
                "truth_misaligned_row_count": int(
                    roles.eq("signal_truth_misaligned").sum()
                ),
                "min_sibling_p_value": min_p,
                "neg_log10_min_sibling_p_value": float(-math.log10(min_p_clipped)),
                "max_homogeneity_gain_min": float(
                    pd.to_numeric(group["homogeneity_gain_min"], errors="coerce").max()
                ),
                "max_context_homogeneity_margin": float(
                    pd.to_numeric(
                        group["context_homogeneity_margin"],
                        errors="coerce",
                    ).max()
                ),
                "max_subspace_consensus_jaccard_topk": float(
                    pd.to_numeric(
                        group["subspace_consensus_jaccard_topk"],
                        errors="coerce",
                    ).max()
                ),
                "min_continuous_homogeneity_threshold": float(
                    pd.to_numeric(
                        group["continuous_homogeneity_threshold"],
                        errors="coerce",
                    ).min()
                ),
                "max_depth": float(pd.to_numeric(group["depth"], errors="coerce").max()),
                "median_parent_size": float(
                    pd.to_numeric(group["n_parent"], errors="coerce").median()
                ),
                "max_barycentric_balance": float(
                    pd.to_numeric(
                        group["barycentric_balance"],
                        errors="coerce",
                    ).max()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_COLUMNS)


def _comparison_masks(
    rows: pd.DataFrame,
    *,
    positive_role: str,
    negative_role: str,
) -> tuple[pd.Series, pd.Series]:
    roles = rows["family_truth_role"].astype(str)
    positive = roles.eq(positive_role)
    if negative_role == "null_or_misaligned_family":
        negative = roles.isin({"null_like_family", "signal_truth_misaligned_family"})
    else:
        negative = roles.eq(negative_role)
    return positive, negative


def _zero_negative_threshold(
    *,
    positive_values: np.ndarray,
    negative_values: np.ndarray,
    direction: str,
) -> tuple[float, int, float, str]:
    pos = positive_values[np.isfinite(positive_values)]
    neg = negative_values[np.isfinite(negative_values)]
    if pos.size == 0 or neg.size == 0:
        return math.nan, 0, math.nan, "zero_negative_undefined"
    if direction == "greater_equal":
        threshold = float(np.nextafter(float(np.max(neg)), math.inf))
        selected = pos >= threshold
    elif direction == "less_equal":
        threshold = float(np.nextafter(float(np.min(neg)), -math.inf))
        selected = pos <= threshold
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    count = int(selected.sum())
    retention = float(count / pos.size)
    if count == pos.size:
        status = "zero_negative_separates_all_positives"
    elif count > 0:
        status = "zero_negative_partial_positive_retention"
    else:
        status = "zero_negative_no_positive_retention"
    return threshold, count, retention, status


def summarize_family_metric(
    family_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_role: str,
    negative_role: str,
    zone: str,
) -> dict[str, object]:
    """Summarize one family metric comparison."""
    positive_mask, negative_mask = _comparison_masks(
        family_rows,
        positive_role=positive_role,
        negative_role=negative_role,
    )
    values = pd.to_numeric(family_rows[metric], errors="coerce")
    pos = values[positive_mask].to_numpy(dtype=float)
    neg = values[negative_mask].to_numpy(dtype=float)
    high_auc = rank_auc(pos, neg)
    low_auc = math.nan if math.isnan(high_auc) else float(1.0 - high_auc)
    if math.isnan(high_auc) or high_auc >= low_auc:
        best_direction = "greater_equal"
        best_auc = high_auc
    else:
        best_direction = "less_equal"
        best_auc = low_auc
    threshold, count, retention, status = _zero_negative_threshold(
        positive_values=pos,
        negative_values=neg,
        direction=best_direction,
    )
    pos_finite = pos[np.isfinite(pos)]
    neg_finite = neg[np.isfinite(neg)]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "zone": zone,
        "metric": metric,
        "comparison": comparison,
        "positive_role": positive_role,
        "negative_role": negative_role,
        "positive_family_count": int(pos_finite.size),
        "negative_family_count": int(neg_finite.size),
        "best_direction": best_direction,
        "best_auc": float(best_auc) if not math.isnan(best_auc) else math.nan,
        "high_direction_auc": float(high_auc) if not math.isnan(high_auc) else math.nan,
        "low_direction_auc": float(low_auc) if not math.isnan(low_auc) else math.nan,
        "positive_min": float(np.min(pos_finite)) if pos_finite.size else math.nan,
        "positive_median": float(np.median(pos_finite)) if pos_finite.size else math.nan,
        "positive_max": float(np.max(pos_finite)) if pos_finite.size else math.nan,
        "negative_min": float(np.min(neg_finite)) if neg_finite.size else math.nan,
        "negative_median": float(np.median(neg_finite)) if neg_finite.size else math.nan,
        "negative_max": float(np.max(neg_finite)) if neg_finite.size else math.nan,
        "zero_negative_threshold": threshold,
        "zero_negative_direction": best_direction,
        "zero_negative_positive_count": count,
        "zero_negative_positive_retention": retention,
        "zero_negative_status": status,
    }


def threshold_scan_for_family_metric(
    family_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_mask: pd.Series,
    negative_mask: pd.Series,
    zone: str,
) -> pd.DataFrame:
    """Scan observed thresholds for one family metric."""
    values = pd.to_numeric(family_rows[metric], errors="coerce")
    finite = values[np.isfinite(values)]
    thresholds = np.sort(finite.unique())
    positive_total = int(positive_mask.sum())
    negative_total = int(negative_mask.sum())
    records: list[dict[str, object]] = []
    for direction in ("greater_equal", "less_equal"):
        for threshold in thresholds:
            if direction == "greater_equal":
                selected = values.ge(float(threshold))
            else:
                selected = values.le(float(threshold))
            positive_selected = int((selected & positive_mask).sum())
            negative_selected = int((selected & negative_mask).sum())
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "zone": zone,
                    "metric": metric,
                    "comparison": comparison,
                    "direction": direction,
                    "threshold": float(threshold),
                    "positive_selected_count": positive_selected,
                    "negative_selected_count": negative_selected,
                    "positive_total": positive_total,
                    "negative_total": negative_total,
                    "positive_retention": (
                        float(positive_selected / positive_total)
                        if positive_total
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


def build_weak_family_thresholds(
    rows: pd.DataFrame,
    *,
    zone: str = DEFAULT_ZONE,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build family rows, metric summaries, and threshold scans."""
    family_rows = build_weak_family_rows(rows, zone=zone)
    missing_metrics = sorted(set(metrics) - set(family_rows.columns))
    if missing_metrics:
        raise ValueError(f"Family metrics are missing: {missing_metrics!r}")
    summaries: list[dict[str, object]] = []
    scans: list[pd.DataFrame] = []
    for metric in metrics:
        for positive_role, negative_role, comparison in COMPARISONS:
            positive_mask, negative_mask = _comparison_masks(
                family_rows,
                positive_role=positive_role,
                negative_role=negative_role,
            )
            summaries.append(
                summarize_family_metric(
                    family_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_role=positive_role,
                    negative_role=negative_role,
                    zone=zone,
                )
            )
            scans.append(
                threshold_scan_for_family_metric(
                    family_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
                    zone=zone,
                )
            )
    summary = pd.DataFrame.from_records(summaries, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["comparison", "best_auc", "zero_negative_positive_retention"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    scan = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame(
        columns=THRESHOLD_SCAN_COLUMNS
    )
    return family_rows, summary, scan


def run_overlap_weak_family_thresholds(
    config: OverlapWeakFamilyThresholdConfig,
) -> dict[str, Path]:
    """Run family-wise weak-zone diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    family_rows, summary, scan = build_weak_family_thresholds(
        rows,
        zone=str(config.zone),
        metrics=config.metrics,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    family_rows.to_csv(config.family_rows_path, index=False)
    summary.to_csv(config.metric_summary_path, index=False)
    scan.to_csv(config.threshold_scan_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "zone": str(config.zone),
        "metrics": list(config.metrics),
        "outputs": {
            "family_rows": str(config.family_rows_path),
            "metric_summary": str(config.metric_summary_path),
            "threshold_scan": str(config.threshold_scan_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "family_rows": config.family_rows_path,
        "metric_summary": config.metric_summary_path,
        "threshold_scan": config.threshold_scan_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated family metrics to scan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_weak_family_thresholds(
        OverlapWeakFamilyThresholdConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            zone=str(args.zone),
            metrics=_parse_metric_list(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
