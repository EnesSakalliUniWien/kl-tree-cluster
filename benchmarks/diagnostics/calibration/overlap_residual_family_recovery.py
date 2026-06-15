"""Residual selected-family recovery diagnostics after fragment blocking.

This panel examines only rows left in `weak_unstable_multiscale_zone` by the
diagnostic traversal policy. It asks whether non-oracle selected-family metrics
can separate remaining truth-recovery families from selected null or other
non-recovery families after the continuous structural rule and fragment-risk
guard have already acted.
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

STUDY_ROLE = "diagnostic_overlap_residual_family_recovery_not_calibration"
SCHEMA_VERSION = "overlap_residual_family_recovery/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_residual_family_recovery"

DEFAULT_ACTION = "weak_unstable_multiscale_zone"
DEFAULT_METRICS = (
    "residual_family_size",
    "residual_min_sibling_p_value",
    "residual_neg_log10_min_sibling_p_value",
    "residual_max_homogeneity_gain_min",
    "residual_max_continuous_context_margin",
    "residual_max_subspace_consensus_jaccard_topk",
    "residual_max_depth",
    "residual_median_parent_size",
    "residual_max_barycentric_balance",
    "residual_min_fragment_risk_proxy_score",
    "residual_max_fragment_risk_proxy_score",
    "residual_max_balanced_recovery_proxy_score",
    "residual_min_size_balance",
    "residual_min_edge_norm_balance",
)

COMPARISONS = (
    ("residual_truth_recovery_family", "residual_null_like_family", "recovery_vs_null"),
    (
        "residual_truth_recovery_family",
        "residual_nonrecovery_family",
        "recovery_vs_nonrecovery",
    ),
    (
        "residual_truth_recovery_family",
        "residual_null_or_nonrecovery_family",
        "recovery_vs_null_or_nonrecovery",
    ),
)

FAMILY_COLUMNS = (
    "schema_version",
    "study_role",
    "action",
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "residual_family_size",
    "null_like_row_count",
    "truth_recovery_row_count",
    "fragment_like_row_count",
    "diffuse_or_wrong_row_count",
    "residual_min_sibling_p_value",
    "residual_neg_log10_min_sibling_p_value",
    "residual_max_homogeneity_gain_min",
    "residual_max_continuous_context_margin",
    "residual_max_subspace_consensus_jaccard_topk",
    "residual_max_depth",
    "residual_median_parent_size",
    "residual_max_barycentric_balance",
    "residual_min_fragment_risk_proxy_score",
    "residual_max_fragment_risk_proxy_score",
    "residual_max_balanced_recovery_proxy_score",
    "residual_min_size_balance",
    "residual_min_edge_norm_balance",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "action",
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
    "action",
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
class OverlapResidualFamilyRecoveryConfig:
    """Runtime contract for residual selected-family recovery diagnostics."""

    policy_rows_path: Path
    decision_zone_rows_path: Path
    fragment_guard_rows_path: Path
    output_dir: Path
    action: str = DEFAULT_ACTION
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def family_rows_path(self) -> Path:
        return self.output_dir / "overlap_residual_family_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "overlap_residual_family_metric_separability.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return self.output_dir / "overlap_residual_family_threshold_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one residual family metric is required.")
    return metrics


def _validate_inputs(
    policy_rows: pd.DataFrame,
    zone_rows: pd.DataFrame,
    guard_rows: pd.DataFrame,
) -> None:
    missing_policy = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "diagnostic_traversal_action",
            "guard_truth_role",
        }
        - set(policy_rows.columns)
    )
    if missing_policy:
        raise ValueError(f"Policy rows are missing columns: {missing_policy!r}")
    missing_zone = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "sibling_p_value",
            "homogeneity_gain_min",
            "subspace_consensus_jaccard_topk",
            "continuous_homogeneity_threshold",
            "depth",
            "n_parent",
            "barycentric_balance",
        }
        - set(zone_rows.columns)
    )
    if missing_zone:
        raise ValueError(f"Decision-zone rows are missing columns: {missing_zone!r}")
    missing_guard = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "fragment_risk_proxy_score",
            "balanced_recovery_proxy_score",
            "size_balance",
            "edge_norm_balance",
        }
        - set(guard_rows.columns)
    )
    if missing_guard:
        raise ValueError(f"Fragment-guard rows are missing columns: {missing_guard!r}")


def _classify_residual_family(group: pd.DataFrame) -> str:
    roles = set(group["guard_truth_role"].astype(str))
    if roles <= {"null_like"}:
        return "residual_null_like_family"
    if "truth_recovery" in roles:
        return "residual_truth_recovery_family"
    if roles & {"fragment_like", "diffuse_or_wrong"}:
        return "residual_nonrecovery_family"
    return "residual_unknown_family"


def build_residual_family_rows(
    policy_rows: pd.DataFrame,
    decision_zone_rows: pd.DataFrame,
    fragment_guard_rows: pd.DataFrame,
    *,
    action: str = DEFAULT_ACTION,
) -> pd.DataFrame:
    """Aggregate residual weak policy rows into selected-family rows."""
    _validate_inputs(policy_rows, decision_zone_rows, fragment_guard_rows)
    keys = ["case_id", "data_role", "replicate", "node_id"]
    residual = policy_rows[
        policy_rows["diagnostic_traversal_action"].astype(str).eq(str(action))
    ][keys + ["guard_truth_role"]].copy()
    if residual.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    zone_subset = decision_zone_rows[
        keys
        + [
            "sibling_p_value",
            "homogeneity_gain_min",
            "subspace_consensus_jaccard_topk",
            "continuous_homogeneity_threshold",
            "depth",
            "n_parent",
            "barycentric_balance",
        ]
    ].copy()
    guard_subset = fragment_guard_rows[
        keys
        + [
            "fragment_risk_proxy_score",
            "balanced_recovery_proxy_score",
            "size_balance",
            "edge_norm_balance",
        ]
    ].copy()
    joined = residual.merge(zone_subset, on=keys, how="left", validate="one_to_one")
    joined = joined.merge(guard_subset, on=keys, how="left", validate="one_to_one")
    joined["continuous_context_margin"] = (
        pd.to_numeric(joined["homogeneity_gain_min"], errors="coerce")
        - pd.to_numeric(joined["continuous_homogeneity_threshold"], errors="coerce")
    )
    records: list[dict[str, object]] = []
    group_cols = ["case_id", "data_role", "replicate"]
    for (case_id, data_role, replicate), group in joined.groupby(group_cols, sort=True):
        p_values = pd.to_numeric(group["sibling_p_value"], errors="coerce")
        min_p = float(p_values.min())
        min_p_clipped = max(min_p, float(np.nextafter(0.0, 1.0)))
        roles = group["guard_truth_role"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "action": str(action),
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "residual_family_truth_role": _classify_residual_family(group),
                "residual_family_size": int(group.shape[0]),
                "null_like_row_count": int(roles.eq("null_like").sum()),
                "truth_recovery_row_count": int(roles.eq("truth_recovery").sum()),
                "fragment_like_row_count": int(roles.eq("fragment_like").sum()),
                "diffuse_or_wrong_row_count": int(roles.eq("diffuse_or_wrong").sum()),
                "residual_min_sibling_p_value": min_p,
                "residual_neg_log10_min_sibling_p_value": float(
                    -math.log10(min_p_clipped)
                ),
                "residual_max_homogeneity_gain_min": _max(
                    group["homogeneity_gain_min"]
                ),
                "residual_max_continuous_context_margin": _max(
                    group["continuous_context_margin"]
                ),
                "residual_max_subspace_consensus_jaccard_topk": _max(
                    group["subspace_consensus_jaccard_topk"]
                ),
                "residual_max_depth": _max(group["depth"]),
                "residual_median_parent_size": _median(group["n_parent"]),
                "residual_max_barycentric_balance": _max(group["barycentric_balance"]),
                "residual_min_fragment_risk_proxy_score": _min(
                    group["fragment_risk_proxy_score"]
                ),
                "residual_max_fragment_risk_proxy_score": _max(
                    group["fragment_risk_proxy_score"]
                ),
                "residual_max_balanced_recovery_proxy_score": _max(
                    group["balanced_recovery_proxy_score"]
                ),
                "residual_min_size_balance": _min(group["size_balance"]),
                "residual_min_edge_norm_balance": _min(group["edge_norm_balance"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_COLUMNS)


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def _min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else math.nan


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else math.nan


def _comparison_masks(
    family_rows: pd.DataFrame,
    *,
    positive_role: str,
    negative_role: str,
) -> tuple[pd.Series, pd.Series]:
    roles = family_rows["residual_family_truth_role"].astype(str)
    positive = roles.eq(positive_role)
    if negative_role == "residual_null_or_nonrecovery_family":
        negative = roles.isin(
            {"residual_null_like_family", "residual_nonrecovery_family"}
        )
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
        raise ValueError(f"Unknown direction: {direction!r}.")
    count = int(selected.sum())
    retention = float(count / pos.size)
    if count == pos.size:
        status = "zero_negative_separates_all_positives"
    elif count > 0:
        status = "zero_negative_partial_positive_retention"
    else:
        status = "zero_negative_no_positive_retention"
    return threshold, count, retention, status


def summarize_residual_family_metric(
    family_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_role: str,
    negative_role: str,
    action: str,
) -> dict[str, object]:
    """Summarize one residual selected-family metric comparison."""
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
        "action": action,
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


def threshold_scan_for_residual_metric(
    family_rows: pd.DataFrame,
    *,
    metric: str,
    comparison: str,
    positive_mask: pd.Series,
    negative_mask: pd.Series,
    action: str,
) -> pd.DataFrame:
    """Scan observed thresholds for one residual family metric."""
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
                    "action": action,
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


def build_residual_family_recovery(
    family_rows: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    action: str = DEFAULT_ACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build residual family metric summaries and threshold scans."""
    missing_metrics = sorted(set(metrics) - set(family_rows.columns))
    if missing_metrics:
        raise ValueError(f"Residual family metrics are missing: {missing_metrics!r}")
    summaries: list[dict[str, object]] = []
    scans: list[pd.DataFrame] = []
    for positive_role, negative_role, comparison in COMPARISONS:
        positive_mask, negative_mask = _comparison_masks(
            family_rows,
            positive_role=positive_role,
            negative_role=negative_role,
        )
        for metric in metrics:
            summaries.append(
                summarize_residual_family_metric(
                    family_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_role=positive_role,
                    negative_role=negative_role,
                    action=action,
                )
            )
            scans.append(
                threshold_scan_for_residual_metric(
                    family_rows,
                    metric=metric,
                    comparison=comparison,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
                    action=action,
                )
            )
    summary = pd.DataFrame.from_records(summaries, columns=SUMMARY_COLUMNS)
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


def run_overlap_residual_family_recovery(
    config: OverlapResidualFamilyRecoveryConfig,
) -> dict[str, Path]:
    """Run residual selected-family recovery diagnostics and write outputs."""
    policy_rows = pd.read_csv(config.policy_rows_path)
    zone_rows = pd.read_csv(config.decision_zone_rows_path)
    guard_rows = pd.read_csv(config.fragment_guard_rows_path)
    family_rows = build_residual_family_rows(
        policy_rows,
        zone_rows,
        guard_rows,
        action=str(config.action),
    )
    summary, scan = build_residual_family_recovery(
        family_rows,
        metrics=config.metrics,
        action=str(config.action),
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
        "policy_rows_path": str(config.policy_rows_path),
        "decision_zone_rows_path": str(config.decision_zone_rows_path),
        "fragment_guard_rows_path": str(config.fragment_guard_rows_path),
        "action": str(config.action),
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
    parser.add_argument("--policy-rows-path", required=True, type=Path)
    parser.add_argument("--decision-zone-rows-path", required=True, type=Path)
    parser.add_argument("--fragment-guard-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default=DEFAULT_ACTION)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated residual family metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_residual_family_recovery(
        OverlapResidualFamilyRecoveryConfig(
            policy_rows_path=args.policy_rows_path,
            decision_zone_rows_path=args.decision_zone_rows_path,
            fragment_guard_rows_path=args.fragment_guard_rows_path,
            output_dir=args.output_dir,
            action=str(args.action),
            metrics=_parse_metric_list(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
