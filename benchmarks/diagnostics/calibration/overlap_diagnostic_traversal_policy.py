"""Compose overlap structural zones and fragment-risk guards.

This diagnostic is not a production calibration rule. It records the traversal
action implied by the current evidence stack:

1. stable same-subspace structural accepts may be exposed as stable regions;
2. weak-zone rows above a predeclared fragment-risk threshold are guard-blocked;
3. remaining weak-zone rows stay multi-scale/unstable pending a selected-family
   recovery law.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_traversal_policy_not_calibration"
SCHEMA_VERSION = "overlap_diagnostic_traversal_policy/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_diagnostic_traversal_policy"

DEFAULT_WEAK_ZONE = "unstable_weak_homogeneity_zone"
DEFAULT_FRAGMENT_GUARD_METRIC = "fragment_risk_proxy_score"
DEFAULT_FRAGMENT_GUARD_DIRECTION = "greater_equal"
DEFAULT_FRAGMENT_GUARD_THRESHOLD = 1.252728536810977

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "decision_class",
    "structural_decision_zone",
    "structural_truth_role",
    "guard_truth_role",
    "truth_geometry_mode",
    "fragment_guard_metric",
    "fragment_guard_direction",
    "fragment_guard_threshold",
    "fragment_guard_value",
    "fragment_guard_blocked",
    "continuous_context_min_margin",
    "diagnostic_traversal_action",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "diagnostic_traversal_action",
    "row_count",
    "accepted_split_count",
    "null_like_count",
    "truth_recovery_count",
    "fragment_like_count",
    "diffuse_or_wrong_count",
    "signal_truth_aligned_count",
    "signal_truth_misaligned_count",
    "median_fragment_guard_value",
    "median_continuous_context_min_margin",
)


@dataclass(frozen=True)
class OverlapDiagnosticTraversalPolicyConfig:
    """Runtime contract for diagnostic traversal policy composition."""

    decision_zone_rows_path: Path
    fragment_guard_rows_path: Path
    output_dir: Path
    weak_zone: str = DEFAULT_WEAK_ZONE
    fragment_guard_metric: str = DEFAULT_FRAGMENT_GUARD_METRIC
    fragment_guard_direction: str = DEFAULT_FRAGMENT_GUARD_DIRECTION
    fragment_guard_threshold: float = DEFAULT_FRAGMENT_GUARD_THRESHOLD

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_diagnostic_traversal_policy_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_diagnostic_traversal_policy_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_inputs(
    zone_rows: pd.DataFrame,
    guard_rows: pd.DataFrame,
    *,
    fragment_guard_metric: str,
) -> None:
    missing_zone = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "decision_class",
            "structural_decision_zone",
            "structural_truth_role",
            "continuous_context_min_margin",
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
            "guard_truth_role",
            "truth_geometry_mode",
            fragment_guard_metric,
        }
        - set(guard_rows.columns)
    )
    if missing_guard:
        raise ValueError(f"Fragment-guard rows are missing columns: {missing_guard!r}")


def _blocked_by_guard(
    values: pd.Series,
    *,
    direction: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if direction == "greater_equal":
        return numeric.ge(float(threshold))
    if direction == "less_equal":
        return numeric.le(float(threshold))
    raise ValueError(f"Unknown fragment guard direction: {direction!r}.")


def assign_diagnostic_traversal_policy(
    zone_rows: pd.DataFrame,
    guard_rows: pd.DataFrame,
    *,
    weak_zone: str = DEFAULT_WEAK_ZONE,
    fragment_guard_metric: str = DEFAULT_FRAGMENT_GUARD_METRIC,
    fragment_guard_direction: str = DEFAULT_FRAGMENT_GUARD_DIRECTION,
    fragment_guard_threshold: float = DEFAULT_FRAGMENT_GUARD_THRESHOLD,
) -> pd.DataFrame:
    """Assign diagnostic traversal actions from structural zones and guards."""
    _validate_inputs(
        zone_rows,
        guard_rows,
        fragment_guard_metric=str(fragment_guard_metric),
    )
    keys = ["case_id", "data_role", "replicate", "node_id"]
    guard_subset = guard_rows[
        keys + ["guard_truth_role", "truth_geometry_mode", str(fragment_guard_metric)]
    ].copy()
    guard_subset = guard_subset.rename(
        columns={str(fragment_guard_metric): "fragment_guard_value"}
    )
    joined = zone_rows.merge(guard_subset, on=keys, how="left", validate="one_to_one")
    zone = joined["structural_decision_zone"].astype(str)
    guard_blocked = _blocked_by_guard(
        joined["fragment_guard_value"],
        direction=str(fragment_guard_direction),
        threshold=float(fragment_guard_threshold),
    ).fillna(False)
    actions = zone.copy()
    weak = zone.eq(str(weak_zone))
    actions.loc[zone.eq("stable_structural_accept")] = "stable_region_accept"
    actions.loc[weak & guard_blocked] = "weak_fragment_guard_blocked"
    actions.loc[weak & ~guard_blocked] = "weak_unstable_multiscale_zone"
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": joined["case_id"].astype(str),
            "data_role": joined["data_role"].astype(str),
            "replicate": joined["replicate"].astype(int),
            "node_id": joined["node_id"].astype(str),
            "decision_class": joined["decision_class"].astype(str),
            "structural_decision_zone": zone,
            "structural_truth_role": joined["structural_truth_role"].astype(str),
            "guard_truth_role": joined["guard_truth_role"].fillna("not_guarded"),
            "truth_geometry_mode": joined["truth_geometry_mode"],
            "fragment_guard_metric": str(fragment_guard_metric),
            "fragment_guard_direction": str(fragment_guard_direction),
            "fragment_guard_threshold": float(fragment_guard_threshold),
            "fragment_guard_value": pd.to_numeric(
                joined["fragment_guard_value"],
                errors="coerce",
            ),
            "fragment_guard_blocked": guard_blocked.astype(bool),
            "continuous_context_min_margin": pd.to_numeric(
                joined["continuous_context_min_margin"],
                errors="coerce",
            ),
            "diagnostic_traversal_action": actions,
        },
        columns=ROW_COLUMNS,
    )


def summarize_diagnostic_traversal_policy(policy_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostic traversal actions."""
    records: list[dict[str, object]] = []
    for action, group in policy_rows.groupby("diagnostic_traversal_action", sort=True):
        structural_role = group["structural_truth_role"].astype(str)
        guard_role = group["guard_truth_role"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "diagnostic_traversal_action": str(action),
                "row_count": int(group.shape[0]),
                "accepted_split_count": int(
                    group["decision_class"].astype(str).eq("accepted_internal_split").sum()
                ),
                "null_like_count": int(
                    structural_role.eq("null_like").sum()
                    + guard_role.eq("null_like").sum()
                    - (
                        structural_role.eq("null_like") & guard_role.eq("null_like")
                    ).sum()
                ),
                "truth_recovery_count": int(guard_role.eq("truth_recovery").sum()),
                "fragment_like_count": int(guard_role.eq("fragment_like").sum()),
                "diffuse_or_wrong_count": int(guard_role.eq("diffuse_or_wrong").sum()),
                "signal_truth_aligned_count": int(
                    structural_role.eq("signal_truth_aligned").sum()
                ),
                "signal_truth_misaligned_count": int(
                    structural_role.eq("signal_truth_misaligned").sum()
                ),
                "median_fragment_guard_value": _finite_median(
                    group["fragment_guard_value"]
                ),
                "median_continuous_context_min_margin": _finite_median(
                    group["continuous_context_min_margin"]
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.median())


def run_overlap_diagnostic_traversal_policy(
    config: OverlapDiagnosticTraversalPolicyConfig,
) -> dict[str, Path]:
    """Run diagnostic traversal policy composition and write outputs."""
    zone_rows = pd.read_csv(config.decision_zone_rows_path)
    guard_rows = pd.read_csv(config.fragment_guard_rows_path)
    policy_rows = assign_diagnostic_traversal_policy(
        zone_rows,
        guard_rows,
        weak_zone=str(config.weak_zone),
        fragment_guard_metric=str(config.fragment_guard_metric),
        fragment_guard_direction=str(config.fragment_guard_direction),
        fragment_guard_threshold=float(config.fragment_guard_threshold),
    )
    summary = summarize_diagnostic_traversal_policy(policy_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy_rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "decision_zone_rows_path": str(config.decision_zone_rows_path),
        "fragment_guard_rows_path": str(config.fragment_guard_rows_path),
        "weak_zone": str(config.weak_zone),
        "fragment_guard_metric": str(config.fragment_guard_metric),
        "fragment_guard_direction": str(config.fragment_guard_direction),
        "fragment_guard_threshold": float(config.fragment_guard_threshold),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-zone-rows-path", required=True, type=Path)
    parser.add_argument("--fragment-guard-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weak-zone", default=DEFAULT_WEAK_ZONE)
    parser.add_argument("--fragment-guard-metric", default=DEFAULT_FRAGMENT_GUARD_METRIC)
    parser.add_argument(
        "--fragment-guard-direction",
        default=DEFAULT_FRAGMENT_GUARD_DIRECTION,
    )
    parser.add_argument(
        "--fragment-guard-threshold",
        default=DEFAULT_FRAGMENT_GUARD_THRESHOLD,
        type=float,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_diagnostic_traversal_policy(
        OverlapDiagnosticTraversalPolicyConfig(
            decision_zone_rows_path=args.decision_zone_rows_path,
            fragment_guard_rows_path=args.fragment_guard_rows_path,
            output_dir=args.output_dir,
            weak_zone=str(args.weak_zone),
            fragment_guard_metric=str(args.fragment_guard_metric),
            fragment_guard_direction=str(args.fragment_guard_direction),
            fragment_guard_threshold=float(args.fragment_guard_threshold),
        )
    )


if __name__ == "__main__":
    main()
