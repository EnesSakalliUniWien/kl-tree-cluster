"""Non-oracle proxy diagnostics for weak overlap structural recovery.

This panel joins weak overlap structural rows to oracle truth-geometry labels
only for evaluation. The metrics themselves are non-oracle structural proxies:
child homogeneity symmetry, child-size balance, edge-norm balance, and related
quantities. It asks whether a future production rule could approximate
balanced structural recovery without using truth labels.
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

STUDY_ROLE = "diagnostic_overlap_recovery_proxy_separability_not_calibration"
SCHEMA_VERSION = "overlap_recovery_proxy_separability/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_recovery_proxy_separability"

RECOVERY_MODES = {"balanced_truth_recovery", "partial_truth_recovery"}
FRAGMENT_MODES = {"one_sided_pure_fragment", "one_sided_mixed_remainder"}
NONRECOVERY_MODES = {
    "one_sided_pure_fragment",
    "one_sided_mixed_remainder",
    "balanced_but_wrong_granularity",
    "diffuse_truth_mismatch",
}
DEFAULT_METRICS = (
    "barycentric_balance",
    "min_child_pairwise_jaccard",
    "max_child_pairwise_jaccard",
    "child_pairwise_jaccard_gap",
    "homogeneity_gain_min",
    "max_homogeneity_gain",
    "homogeneity_gain_gap",
    "edge_norm_balance",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "balanced_recovery_proxy_score",
    "fragment_risk_proxy_score",
)
COMPARISONS = (
    ("truth_recovery", "nonrecovery", "recovery_vs_nonrecovery"),
    ("truth_recovery", "fragment_like", "recovery_vs_fragment_like"),
    ("truth_recovery", "diffuse_or_wrong", "recovery_vs_diffuse_or_wrong"),
)

PROXY_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "truth_geometry_mode",
    "proxy_truth_role",
    "depth",
    "n_parent",
    "n_left",
    "n_right",
    "barycentric_balance",
    "left_pairwise_jaccard",
    "right_pairwise_jaccard",
    "min_child_pairwise_jaccard",
    "max_child_pairwise_jaccard",
    "child_pairwise_jaccard_gap",
    "homogeneity_gain_left",
    "homogeneity_gain_right",
    "homogeneity_gain_min",
    "max_homogeneity_gain",
    "homogeneity_gain_gap",
    "left_edge_norm",
    "right_edge_norm",
    "edge_norm_balance",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "balanced_recovery_proxy_score",
    "fragment_risk_proxy_score",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
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
class OverlapRecoveryProxySeparabilityConfig:
    """Runtime contract for non-oracle recovery proxy diagnostics."""

    structural_rows_path: Path
    truth_geometry_rows_path: Path
    output_dir: Path
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def proxy_rows_path(self) -> Path:
        return self.output_dir / "overlap_recovery_proxy_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "overlap_recovery_proxy_metric_separability.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return self.output_dir / "overlap_recovery_proxy_threshold_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one proxy metric is required.")
    return metrics


def _required_structural_columns() -> set[str]:
    return {
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "depth",
        "n_parent",
        "n_left",
        "n_right",
        "barycentric_balance",
        "left_pairwise_jaccard",
        "right_pairwise_jaccard",
        "homogeneity_gain_left",
        "homogeneity_gain_right",
        "homogeneity_gain_min",
        "left_edge_norm",
        "right_edge_norm",
        "subspace_consensus_jaccard_topk",
    }


def _required_truth_columns() -> set[str]:
    return {"case_id", "data_role", "replicate", "node_id", "truth_geometry_mode"}


def _validate_inputs(structural_rows: pd.DataFrame, truth_rows: pd.DataFrame) -> None:
    missing_structural = sorted(_required_structural_columns() - set(structural_rows.columns))
    if missing_structural:
        raise ValueError(f"Structural rows are missing columns: {missing_structural!r}")
    missing_truth = sorted(_required_truth_columns() - set(truth_rows.columns))
    if missing_truth:
        raise ValueError(f"Truth-geometry rows are missing columns: {missing_truth!r}")


def _safe_balance(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce").abs()
    right_num = pd.to_numeric(right, errors="coerce").abs()
    denom = pd.concat([left_num, right_num], axis=1).max(axis=1)
    numer = pd.concat([left_num, right_num], axis=1).min(axis=1)
    return numer.divide(denom).where(denom.gt(0.0), 1.0)


def _proxy_truth_role(mode: str) -> str:
    if mode in RECOVERY_MODES:
        return "truth_recovery"
    if mode in FRAGMENT_MODES:
        return "fragment_like"
    if mode in NONRECOVERY_MODES:
        return "diffuse_or_wrong"
    return "other"


def build_recovery_proxy_rows(
    structural_rows: pd.DataFrame,
    truth_geometry_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Join structural rows to oracle labels and derive non-oracle proxies."""
    _validate_inputs(structural_rows, truth_geometry_rows)
    keys = ["case_id", "data_role", "replicate", "node_id"]
    truth = truth_geometry_rows[keys + ["truth_geometry_mode"]].copy()
    joined = truth.merge(structural_rows, on=keys, how="left", validate="one_to_one")
    left_pairwise = pd.to_numeric(joined["left_pairwise_jaccard"], errors="coerce")
    right_pairwise = pd.to_numeric(joined["right_pairwise_jaccard"], errors="coerce")
    min_pairwise = pd.concat([left_pairwise, right_pairwise], axis=1).min(axis=1)
    max_pairwise = pd.concat([left_pairwise, right_pairwise], axis=1).max(axis=1)
    left_gain = pd.to_numeric(joined["homogeneity_gain_left"], errors="coerce")
    right_gain = pd.to_numeric(joined["homogeneity_gain_right"], errors="coerce")
    max_gain = pd.concat([left_gain, right_gain], axis=1).max(axis=1)
    min_gain = pd.to_numeric(joined["homogeneity_gain_min"], errors="coerce")
    edge_balance = _safe_balance(joined["left_edge_norm"], joined["right_edge_norm"])
    n_parent = pd.to_numeric(joined["n_parent"], errors="coerce")
    size_balance = (
        pd.concat(
            [
                pd.to_numeric(joined["n_left"], errors="coerce"),
                pd.to_numeric(joined["n_right"], errors="coerce"),
            ],
            axis=1,
        )
        .min(axis=1)
        .divide(n_parent)
    )
    subspace = pd.to_numeric(joined["subspace_consensus_jaccard_topk"], errors="coerce")
    balanced_score = (
        min_pairwise
        + min_gain
        + edge_balance
        + size_balance
        + subspace
        - (max_pairwise - min_pairwise).abs()
    )
    fragment_risk = (
        (max_pairwise - min_pairwise).abs()
        + (max_gain - min_gain).abs()
        + (1.0 - size_balance)
        + (1.0 - edge_balance)
    )
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": joined["case_id"].astype(str),
            "data_role": joined["data_role"].astype(str),
            "replicate": joined["replicate"].astype(int),
            "node_id": joined["node_id"].astype(str),
            "truth_geometry_mode": joined["truth_geometry_mode"].astype(str),
            "proxy_truth_role": joined["truth_geometry_mode"]
            .astype(str)
            .map(_proxy_truth_role),
            "depth": pd.to_numeric(joined["depth"], errors="coerce"),
            "n_parent": n_parent,
            "n_left": pd.to_numeric(joined["n_left"], errors="coerce"),
            "n_right": pd.to_numeric(joined["n_right"], errors="coerce"),
            "barycentric_balance": pd.to_numeric(
                joined["barycentric_balance"],
                errors="coerce",
            ),
            "left_pairwise_jaccard": left_pairwise,
            "right_pairwise_jaccard": right_pairwise,
            "min_child_pairwise_jaccard": min_pairwise,
            "max_child_pairwise_jaccard": max_pairwise,
            "child_pairwise_jaccard_gap": (max_pairwise - min_pairwise).abs(),
            "homogeneity_gain_left": left_gain,
            "homogeneity_gain_right": right_gain,
            "homogeneity_gain_min": min_gain,
            "max_homogeneity_gain": max_gain,
            "homogeneity_gain_gap": (max_gain - min_gain).abs(),
            "left_edge_norm": pd.to_numeric(joined["left_edge_norm"], errors="coerce"),
            "right_edge_norm": pd.to_numeric(
                joined["right_edge_norm"],
                errors="coerce",
            ),
            "edge_norm_balance": edge_balance,
            "subspace_consensus_jaccard_topk": subspace,
            "size_balance": size_balance,
            "balanced_recovery_proxy_score": balanced_score,
            "fragment_risk_proxy_score": fragment_risk,
        },
        columns=PROXY_ROW_COLUMNS,
    )


def _comparison_masks(
    rows: pd.DataFrame,
    *,
    positive_role: str,
    negative_role: str,
) -> tuple[pd.Series, pd.Series]:
    roles = rows["proxy_truth_role"].astype(str)
    positive = roles.eq(positive_role)
    if negative_role == "nonrecovery":
        negative = roles.isin({"fragment_like", "diffuse_or_wrong"})
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


def summarize_proxy_metric(
    proxy_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_role: str,
    negative_role: str,
) -> dict[str, object]:
    """Summarize one proxy metric against oracle recovery labels."""
    positive_mask, negative_mask = _comparison_masks(
        proxy_rows,
        positive_role=positive_role,
        negative_role=negative_role,
    )
    values = pd.to_numeric(proxy_rows[metric], errors="coerce")
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


def threshold_scan_for_proxy_metric(
    proxy_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_mask: pd.Series,
    negative_mask: pd.Series,
) -> pd.DataFrame:
    """Scan observed thresholds for one proxy metric."""
    values = pd.to_numeric(proxy_rows[metric], errors="coerce")
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


def build_recovery_proxy_separability(
    structural_rows: pd.DataFrame,
    truth_geometry_rows: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build proxy rows, metric summaries, and threshold scans."""
    proxy_rows = build_recovery_proxy_rows(structural_rows, truth_geometry_rows)
    missing = sorted(set(metrics) - set(proxy_rows.columns))
    if missing:
        raise ValueError(f"Proxy metrics are missing: {missing!r}")
    summaries: list[dict[str, object]] = []
    scans: list[pd.DataFrame] = []
    for metric in metrics:
        for positive_role, negative_role, comparison in COMPARISONS:
            positive_mask, negative_mask = _comparison_masks(
                proxy_rows,
                positive_role=positive_role,
                negative_role=negative_role,
            )
            summaries.append(
                summarize_proxy_metric(
                    proxy_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_role=positive_role,
                    negative_role=negative_role,
                )
            )
            scans.append(
                threshold_scan_for_proxy_metric(
                    proxy_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
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
    return proxy_rows, summary, scan


def run_overlap_recovery_proxy_separability(
    config: OverlapRecoveryProxySeparabilityConfig,
) -> dict[str, Path]:
    """Run non-oracle recovery proxy diagnostics and write outputs."""
    structural_rows = pd.read_csv(config.structural_rows_path)
    truth_rows = pd.read_csv(config.truth_geometry_rows_path)
    proxy_rows, summary, scan = build_recovery_proxy_separability(
        structural_rows,
        truth_rows,
        metrics=config.metrics,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    proxy_rows.to_csv(config.proxy_rows_path, index=False)
    summary.to_csv(config.metric_summary_path, index=False)
    scan.to_csv(config.threshold_scan_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "structural_rows_path": str(config.structural_rows_path),
        "truth_geometry_rows_path": str(config.truth_geometry_rows_path),
        "metrics": list(config.metrics),
        "outputs": {
            "proxy_rows": str(config.proxy_rows_path),
            "metric_summary": str(config.metric_summary_path),
            "threshold_scan": str(config.threshold_scan_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "proxy_rows": config.proxy_rows_path,
        "metric_summary": config.metric_summary_path,
        "threshold_scan": config.threshold_scan_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-rows-path", required=True, type=Path)
    parser.add_argument("--truth-geometry-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated proxy metrics to scan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_recovery_proxy_separability(
        OverlapRecoveryProxySeparabilityConfig(
            structural_rows_path=args.structural_rows_path,
            truth_geometry_rows_path=args.truth_geometry_rows_path,
            output_dir=args.output_dir,
            metrics=_parse_metric_list(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
