"""Diagnostic fragment-risk guard scan for weak overlap traversal zones.

This panel applies non-oracle proxy metrics to all weak-zone rows, including
selected-null rows. Oracle truth-geometry labels are used only to evaluate what
candidate guard thresholds would block or retain.
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

from benchmarks.diagnostics.calibration.overlap_recovery_proxy_separability import (
    FRAGMENT_MODES,
    RECOVERY_MODES,
    _safe_balance,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_fragment_risk_guard_not_calibration"
SCHEMA_VERSION = "overlap_fragment_risk_guard/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_fragment_risk_guard"
DEFAULT_ZONE = "unstable_weak_homogeneity_zone"
DEFAULT_GUARD_SPECS = (
    ("fragment_risk_proxy_score", "greater_equal"),
    ("child_pairwise_jaccard_gap", "greater_equal"),
    ("homogeneity_gain_gap", "greater_equal"),
    ("barycentric_balance", "less_equal"),
    ("edge_norm_balance", "less_equal"),
    ("size_balance", "less_equal"),
    ("balanced_recovery_proxy_score", "less_equal"),
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "truth_geometry_mode",
    "structural_truth_role",
    "barycentric_balance",
    "min_child_pairwise_jaccard",
    "child_pairwise_jaccard_gap",
    "homogeneity_gain_min",
    "homogeneity_gain_gap",
    "edge_norm_balance",
    "size_balance",
    "subspace_consensus_jaccard_topk",
    "balanced_recovery_proxy_score",
    "fragment_risk_proxy_score",
)

SCAN_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "block_direction",
    "threshold",
    "total_rows",
    "blocked_rows",
    "blocked_rate",
    "null_like_total",
    "null_like_blocked",
    "null_like_blocked_rate",
    "truth_recovery_total",
    "truth_recovery_blocked",
    "truth_recovery_retained",
    "truth_recovery_retention",
    "fragment_like_total",
    "fragment_like_blocked",
    "fragment_like_blocked_rate",
    "diffuse_or_wrong_total",
    "diffuse_or_wrong_blocked",
    "diffuse_or_wrong_blocked_rate",
    "guard_score",
    "guard_status",
)


@dataclass(frozen=True)
class OverlapFragmentRiskGuardConfig:
    """Runtime contract for fragment-risk guard scans."""

    structural_rows_path: Path
    decision_zone_rows_path: Path
    truth_geometry_rows_path: Path
    output_dir: Path
    zone: str = DEFAULT_ZONE
    guard_specs: tuple[tuple[str, str], ...] = DEFAULT_GUARD_SPECS

    @property
    def guard_rows_path(self) -> Path:
        return self.output_dir / "overlap_fragment_risk_guard_rows.csv"

    @property
    def guard_scan_path(self) -> Path:
        return self.output_dir / "overlap_fragment_risk_guard_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_guard_specs(value: str) -> tuple[tuple[str, str], ...]:
    specs: list[tuple[str, str]] = []
    for token in str(value).split(","):
        if not token.strip():
            continue
        if ":" not in token:
            raise ValueError("Guard specs must use metric:direction syntax.")
        metric, direction = token.split(":", 1)
        direction = direction.strip()
        if direction not in {"greater_equal", "less_equal"}:
            raise ValueError(f"Unknown guard direction: {direction!r}.")
        specs.append((metric.strip(), direction))
    if not specs:
        raise ValueError("At least one guard spec is required.")
    return tuple(specs)


def _required_structural_columns() -> set[str]:
    return {
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "left_pairwise_jaccard",
        "right_pairwise_jaccard",
        "homogeneity_gain_left",
        "homogeneity_gain_right",
        "homogeneity_gain_min",
        "left_edge_norm",
        "right_edge_norm",
        "n_left",
        "n_right",
        "n_parent",
        "barycentric_balance",
        "subspace_consensus_jaccard_topk",
    }


def _validate_inputs(
    structural_rows: pd.DataFrame,
    zone_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
) -> None:
    missing_structural = sorted(_required_structural_columns() - set(structural_rows.columns))
    if missing_structural:
        raise ValueError(f"Structural rows are missing columns: {missing_structural!r}")
    missing_zone = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "structural_decision_zone",
            "structural_truth_role",
        }
        - set(zone_rows.columns)
    )
    if missing_zone:
        raise ValueError(f"Decision-zone rows are missing columns: {missing_zone!r}")
    missing_truth = sorted(
        {"case_id", "data_role", "replicate", "node_id", "truth_geometry_mode"}
        - set(truth_rows.columns)
    )
    if missing_truth:
        raise ValueError(f"Truth-geometry rows are missing columns: {missing_truth!r}")


def _guard_truth_role(structural_truth_role: str, truth_geometry_mode: object) -> str:
    if str(structural_truth_role) == "null_like":
        return "null_like"
    mode = "" if pd.isna(truth_geometry_mode) else str(truth_geometry_mode)
    if mode in RECOVERY_MODES:
        return "truth_recovery"
    if mode in FRAGMENT_MODES:
        return "fragment_like"
    if mode:
        return "diffuse_or_wrong"
    return "unknown_signal"


def build_fragment_guard_rows(
    structural_rows: pd.DataFrame,
    decision_zone_rows: pd.DataFrame,
    truth_geometry_rows: pd.DataFrame,
    *,
    zone: str = DEFAULT_ZONE,
) -> pd.DataFrame:
    """Build weak-zone rows with non-oracle proxy metrics and eval labels."""
    _validate_inputs(structural_rows, decision_zone_rows, truth_geometry_rows)
    keys = ["case_id", "data_role", "replicate", "node_id"]
    weak = decision_zone_rows[
        decision_zone_rows["structural_decision_zone"].astype(str).eq(str(zone))
    ][keys + ["structural_truth_role"]].copy()
    joined = weak.merge(structural_rows, on=keys, how="left", validate="one_to_one")
    truth = truth_geometry_rows[keys + ["truth_geometry_mode"]].copy()
    joined = joined.merge(truth, on=keys, how="left", validate="one_to_one")
    left_pairwise = pd.to_numeric(joined["left_pairwise_jaccard"], errors="coerce")
    right_pairwise = pd.to_numeric(joined["right_pairwise_jaccard"], errors="coerce")
    min_pairwise = pd.concat([left_pairwise, right_pairwise], axis=1).min(axis=1)
    max_pairwise = pd.concat([left_pairwise, right_pairwise], axis=1).max(axis=1)
    left_gain = pd.to_numeric(joined["homogeneity_gain_left"], errors="coerce")
    right_gain = pd.to_numeric(joined["homogeneity_gain_right"], errors="coerce")
    min_gain = pd.to_numeric(joined["homogeneity_gain_min"], errors="coerce")
    max_gain = pd.concat([left_gain, right_gain], axis=1).max(axis=1)
    edge_balance = _safe_balance(joined["left_edge_norm"], joined["right_edge_norm"])
    n_parent = pd.to_numeric(joined["n_parent"], errors="coerce")
    n_left = pd.to_numeric(joined["n_left"], errors="coerce")
    n_right = pd.to_numeric(joined["n_right"], errors="coerce")
    size_balance = pd.concat([n_left, n_right], axis=1).min(axis=1).divide(n_parent)
    subspace = pd.to_numeric(joined["subspace_consensus_jaccard_topk"], errors="coerce")
    pairwise_gap = (max_pairwise - min_pairwise).abs()
    gain_gap = (max_gain - min_gain).abs()
    balanced_score = (
        min_pairwise
        + min_gain
        + edge_balance
        + size_balance
        + subspace
        - pairwise_gap
    )
    fragment_risk = pairwise_gap + gain_gap + (1.0 - size_balance) + (1.0 - edge_balance)
    roles = [
        _guard_truth_role(structural_role, truth_mode)
        for structural_role, truth_mode in zip(
            joined["structural_truth_role"],
            joined["truth_geometry_mode"],
            strict=True,
        )
    ]
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": joined["case_id"].astype(str),
            "data_role": joined["data_role"].astype(str),
            "replicate": joined["replicate"].astype(int),
            "node_id": joined["node_id"].astype(str),
            "guard_truth_role": roles,
            "truth_geometry_mode": joined["truth_geometry_mode"].astype("object"),
            "structural_truth_role": joined["structural_truth_role"].astype(str),
            "barycentric_balance": pd.to_numeric(
                joined["barycentric_balance"],
                errors="coerce",
            ),
            "min_child_pairwise_jaccard": min_pairwise,
            "child_pairwise_jaccard_gap": pairwise_gap,
            "homogeneity_gain_min": min_gain,
            "homogeneity_gain_gap": gain_gap,
            "edge_norm_balance": edge_balance,
            "size_balance": size_balance,
            "subspace_consensus_jaccard_topk": subspace,
            "balanced_recovery_proxy_score": balanced_score,
            "fragment_risk_proxy_score": fragment_risk,
        },
        columns=ROW_COLUMNS,
    )


def scan_fragment_guard_thresholds(
    guard_rows: pd.DataFrame,
    *,
    guard_specs: Sequence[tuple[str, str]] = DEFAULT_GUARD_SPECS,
) -> pd.DataFrame:
    """Scan candidate guard thresholds over observed weak-zone proxy values."""
    records: list[dict[str, object]] = []
    roles = guard_rows["guard_truth_role"].astype(str)
    total_rows = int(guard_rows.shape[0])
    totals = {
        role: int(roles.eq(role).sum())
        for role in ("null_like", "truth_recovery", "fragment_like", "diffuse_or_wrong")
    }
    for metric, direction in guard_specs:
        if metric not in guard_rows.columns:
            raise ValueError(f"Guard metric {metric!r} is not available.")
        values = pd.to_numeric(guard_rows[metric], errors="coerce")
        thresholds = np.sort(values[np.isfinite(values)].unique())
        for threshold in thresholds:
            if direction == "greater_equal":
                blocked = values.ge(float(threshold))
            elif direction == "less_equal":
                blocked = values.le(float(threshold))
            else:
                raise ValueError(f"Unknown guard direction: {direction!r}.")
            counts = {
                role: int((blocked & roles.eq(role)).sum())
                for role in totals
            }
            recovery_retained = totals["truth_recovery"] - counts["truth_recovery"]
            recovery_retention = (
                recovery_retained / totals["truth_recovery"]
                if totals["truth_recovery"]
                else math.nan
            )
            fragment_block_rate = (
                counts["fragment_like"] / totals["fragment_like"]
                if totals["fragment_like"]
                else math.nan
            )
            null_block_rate = (
                counts["null_like"] / totals["null_like"]
                if totals["null_like"]
                else math.nan
            )
            diffuse_block_rate = (
                counts["diffuse_or_wrong"] / totals["diffuse_or_wrong"]
                if totals["diffuse_or_wrong"]
                else math.nan
            )
            score = (
                (0.0 if math.isnan(fragment_block_rate) else fragment_block_rate)
                + 0.5 * (0.0 if math.isnan(null_block_rate) else null_block_rate)
                + 0.25 * (0.0 if math.isnan(diffuse_block_rate) else diffuse_block_rate)
                - 1.5 * (1.0 - recovery_retention)
            )
            if recovery_retention >= 0.8 and fragment_block_rate >= 0.5:
                status = "fragment_guard_candidate"
            elif recovery_retention < 0.8:
                status = "fragment_guard_recovery_loss"
            elif fragment_block_rate < 0.5:
                status = "fragment_guard_low_fragment_block"
            else:
                status = "fragment_guard_diagnostic_only"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "metric": metric,
                    "block_direction": direction,
                    "threshold": float(threshold),
                    "total_rows": total_rows,
                    "blocked_rows": int(blocked.sum()),
                    "blocked_rate": float(blocked.mean()) if total_rows else math.nan,
                    "null_like_total": totals["null_like"],
                    "null_like_blocked": counts["null_like"],
                    "null_like_blocked_rate": null_block_rate,
                    "truth_recovery_total": totals["truth_recovery"],
                    "truth_recovery_blocked": counts["truth_recovery"],
                    "truth_recovery_retained": recovery_retained,
                    "truth_recovery_retention": recovery_retention,
                    "fragment_like_total": totals["fragment_like"],
                    "fragment_like_blocked": counts["fragment_like"],
                    "fragment_like_blocked_rate": fragment_block_rate,
                    "diffuse_or_wrong_total": totals["diffuse_or_wrong"],
                    "diffuse_or_wrong_blocked": counts["diffuse_or_wrong"],
                    "diffuse_or_wrong_blocked_rate": diffuse_block_rate,
                    "guard_score": float(score),
                    "guard_status": status,
                }
            )
    scan = pd.DataFrame.from_records(records, columns=SCAN_COLUMNS)
    return scan.sort_values(
        ["guard_status", "guard_score", "truth_recovery_retention"],
        ascending=[True, False, False],
        ignore_index=True,
    )


def run_overlap_fragment_risk_guard(
    config: OverlapFragmentRiskGuardConfig,
) -> dict[str, Path]:
    """Run fragment-risk guard diagnostics and write outputs."""
    structural_rows = pd.read_csv(config.structural_rows_path)
    zone_rows = pd.read_csv(config.decision_zone_rows_path)
    truth_rows = pd.read_csv(config.truth_geometry_rows_path)
    guard_rows = build_fragment_guard_rows(
        structural_rows,
        zone_rows,
        truth_rows,
        zone=str(config.zone),
    )
    scan = scan_fragment_guard_thresholds(guard_rows, guard_specs=config.guard_specs)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    guard_rows.to_csv(config.guard_rows_path, index=False)
    scan.to_csv(config.guard_scan_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "structural_rows_path": str(config.structural_rows_path),
        "decision_zone_rows_path": str(config.decision_zone_rows_path),
        "truth_geometry_rows_path": str(config.truth_geometry_rows_path),
        "zone": str(config.zone),
        "guard_specs": [f"{metric}:{direction}" for metric, direction in config.guard_specs],
        "outputs": {
            "guard_rows": str(config.guard_rows_path),
            "guard_scan": str(config.guard_scan_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "guard_rows": config.guard_rows_path,
        "guard_scan": config.guard_scan_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-rows-path", required=True, type=Path)
    parser.add_argument("--decision-zone-rows-path", required=True, type=Path)
    parser.add_argument("--truth-geometry-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument(
        "--guard-specs",
        default=",".join(f"{metric}:{direction}" for metric, direction in DEFAULT_GUARD_SPECS),
        help="Comma-separated metric:direction guard specs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_fragment_risk_guard(
        OverlapFragmentRiskGuardConfig(
            structural_rows_path=args.structural_rows_path,
            decision_zone_rows_path=args.decision_zone_rows_path,
            truth_geometry_rows_path=args.truth_geometry_rows_path,
            output_dir=args.output_dir,
            zone=str(args.zone),
            guard_specs=_parse_guard_specs(args.guard_specs),
        )
    )


if __name__ == "__main__":
    main()
