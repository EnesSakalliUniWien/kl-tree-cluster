"""Separability diagnostics inside unstable overlap structural zones.

This post-run panel asks whether any scalar structural metric can separate the
weak truth-aligned signal rows from selected-null or truth-misaligned rows
inside `unstable_weak_homogeneity_zone`. It is diagnostic-only and is meant to
quantify why a selected-family null law is still needed.
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

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_weak_zone_separability_not_calibration"
SCHEMA_VERSION = "overlap_weak_zone_separability/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_weak_zone_separability"
DEFAULT_ZONE = "unstable_weak_homogeneity_zone"
DEFAULT_METRICS = (
    "depth",
    "n_parent",
    "barycentric_balance",
    "neg_log10_sibling_p_value",
    "homogeneity_gain_min",
    "subspace_consensus_jaccard_topk",
    "continuous_homogeneity_threshold",
    "context_homogeneity_margin",
)
COMPARISONS = (
    ("signal_truth_aligned", "null_like", "aligned_vs_null"),
    ("signal_truth_aligned", "signal_truth_misaligned", "aligned_vs_misaligned"),
    ("signal_truth_aligned", "null_or_misaligned", "aligned_vs_null_or_misaligned"),
)

METRIC_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "zone",
    "metric",
    "comparison",
    "positive_role",
    "negative_role",
    "positive_count",
    "negative_count",
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
class OverlapWeakZoneSeparabilityConfig:
    """Runtime contract for unstable weak-zone separability diagnostics."""

    rows_path: Path
    output_dir: Path
    zone: str = DEFAULT_ZONE
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "overlap_weak_zone_metric_separability.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return self.output_dir / "overlap_weak_zone_threshold_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one weak-zone metric is required.")
    return metrics


def _required_columns(metrics: Iterable[str]) -> set[str]:
    base = {
        "structural_decision_zone",
        "structural_truth_role",
        "sibling_p_value",
        "homogeneity_gain_min",
        "continuous_homogeneity_threshold",
    }
    return base | {m for m in metrics if m not in _derived_metric_names()}


def _derived_metric_names() -> set[str]:
    return {"neg_log10_sibling_p_value", "context_homogeneity_margin"}


def _validate_rows(rows: pd.DataFrame, metrics: Iterable[str]) -> None:
    missing = sorted(_required_columns(metrics) - set(rows.columns))
    if missing:
        raise ValueError(f"Decision-zone rows are missing columns: {missing!r}")


def add_derived_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with derived scalar separability metrics."""
    enriched = rows.copy()
    p_values = pd.to_numeric(enriched["sibling_p_value"], errors="coerce")
    clipped = p_values.clip(lower=np.nextafter(0.0, 1.0), upper=1.0)
    enriched["neg_log10_sibling_p_value"] = -np.log10(clipped)
    enriched["context_homogeneity_margin"] = pd.to_numeric(
        enriched["homogeneity_gain_min"], errors="coerce"
    ) - pd.to_numeric(enriched["continuous_homogeneity_threshold"], errors="coerce")
    return enriched


def _comparison_masks(
    rows: pd.DataFrame,
    *,
    positive_role: str,
    negative_role: str,
) -> tuple[pd.Series, pd.Series]:
    roles = rows["structural_truth_role"].astype(str)
    positive = roles.eq(positive_role)
    if negative_role == "null_or_misaligned":
        negative = roles.isin({"null_like", "signal_truth_misaligned"})
    else:
        negative = roles.eq(negative_role)
    return positive, negative


def rank_auc(positive_values: Sequence[float], negative_values: Sequence[float]) -> float:
    """Return pairwise AUC for high values indicating positives."""
    pos = np.asarray(positive_values, dtype=float)
    neg = np.asarray(negative_values, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


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
        raise ValueError(f"Unknown threshold direction: {direction!r}")
    count = int(selected.sum())
    retention = float(count / pos.size)
    if count == pos.size:
        status = "zero_negative_separates_all_positives"
    elif count > 0:
        status = "zero_negative_partial_positive_retention"
    else:
        status = "zero_negative_no_positive_retention"
    return threshold, count, retention, status


def threshold_scan_for_metric(
    rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_mask: pd.Series,
    negative_mask: pd.Series,
    zone: str,
) -> pd.DataFrame:
    """Scan observed thresholds for one metric and comparison."""
    values = pd.to_numeric(rows[metric], errors="coerce")
    finite = values[np.isfinite(values)]
    thresholds = np.sort(finite.unique())
    records: list[dict[str, object]] = []
    pos_total = int(positive_mask.sum())
    neg_total = int(negative_mask.sum())
    for direction in ("greater_equal", "less_equal"):
        for threshold in thresholds:
            if direction == "greater_equal":
                selected = values.ge(float(threshold))
            else:
                selected = values.le(float(threshold))
            pos_selected = int((selected & positive_mask).sum())
            neg_selected = int((selected & negative_mask).sum())
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "zone": zone,
                    "metric": metric,
                    "comparison": comparison,
                    "direction": direction,
                    "threshold": float(threshold),
                    "positive_selected_count": pos_selected,
                    "negative_selected_count": neg_selected,
                    "positive_total": pos_total,
                    "negative_total": neg_total,
                    "positive_retention": (
                        float(pos_selected / pos_total) if pos_total else math.nan
                    ),
                    "negative_selection_rate": (
                        float(neg_selected / neg_total) if neg_total else math.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=THRESHOLD_SCAN_COLUMNS)


def summarize_metric_separability(
    rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_role: str,
    negative_role: str,
    zone: str,
) -> dict[str, object]:
    """Summarize separability for one metric and comparison."""
    positive_mask, negative_mask = _comparison_masks(
        rows,
        positive_role=positive_role,
        negative_role=negative_role,
    )
    values = pd.to_numeric(rows[metric], errors="coerce")
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
        "positive_count": int(pos_finite.size),
        "negative_count": int(neg_finite.size),
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


def build_weak_zone_separability(
    rows: pd.DataFrame,
    *,
    zone: str = DEFAULT_ZONE,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metric summaries and threshold scans for one unstable zone."""
    _validate_rows(rows, metrics)
    enriched = add_derived_metrics(rows)
    zone_rows = enriched[enriched["structural_decision_zone"].astype(str).eq(zone)]
    summaries: list[dict[str, object]] = []
    scans: list[pd.DataFrame] = []
    for metric in metrics:
        if metric not in zone_rows.columns:
            raise ValueError(f"Metric {metric!r} is not available in decision rows.")
        for positive_role, negative_role, comparison in COMPARISONS:
            positive_mask, negative_mask = _comparison_masks(
                zone_rows,
                positive_role=positive_role,
                negative_role=negative_role,
            )
            summaries.append(
                summarize_metric_separability(
                    zone_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_role=positive_role,
                    negative_role=negative_role,
                    zone=zone,
                )
            )
            scans.append(
                threshold_scan_for_metric(
                    zone_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
                    zone=zone,
                )
            )
    summary = pd.DataFrame.from_records(summaries, columns=METRIC_SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["comparison", "best_auc", "zero_negative_positive_retention"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    scan = (
        pd.concat(scans, ignore_index=True)
        if scans
        else pd.DataFrame(columns=THRESHOLD_SCAN_COLUMNS)
    )
    return summary, scan


def run_overlap_weak_zone_separability(
    config: OverlapWeakZoneSeparabilityConfig,
) -> dict[str, Path]:
    """Run weak-zone separability diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    summary, scan = build_weak_zone_separability(
        rows,
        zone=str(config.zone),
        metrics=config.metrics,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
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
            "metric_summary": str(config.metric_summary_path),
            "threshold_scan": str(config.threshold_scan_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
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
        help="Comma-separated scalar metrics to scan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_weak_zone_separability(
        OverlapWeakZoneSeparabilityConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            zone=str(args.zone),
            metrics=_parse_metric_list(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
