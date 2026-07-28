"""Threshold sensitivity for row-level overlap internal-node likelihood probes.

This diagnostic checks whether the row-level overlap likelihood behavior is
stable over a small predeclared threshold grid. It is diagnostic-only and does
not promote a production threshold.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_internal_node_likelihood_sensitivity"
SCHEMA_VERSION = "overlap_internal_node_likelihood_sensitivity/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_internal_node_likelihood_sensitivity"
)

DEFAULT_CONTEXT_MARGIN_FLOORS = (-0.002, 0.0, 0.002)
DEFAULT_SUBSPACE_FLOORS = (0.10, 0.15, 0.20)
DEFAULT_SIZE_BALANCE_FLOORS = (0.30, 0.33, 1.0 / 3.0, 0.35)
DEFAULT_EDGE_NORM_BALANCE_FLOORS = (0.45, 0.49, 0.50)
DEFAULT_FRAGMENT_RISK_CEILINGS = (1.10, 1.25, 1.40)
DEFAULT_P_BF_FLOOR = 3.0

REQUIRED_COLUMNS = {
    "guard_truth_role",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
}

GRID_COLUMNS = (
    "schema_version",
    "study_role",
    "rule_id",
    "p_extreme_log_bayes_factor_floor",
    "context_margin_floor",
    "soft_subspace_floor",
    "size_balance_floor",
    "edge_norm_balance_floor",
    "fragment_risk_ceiling",
    "truth_recovery_total",
    "truth_recovery_candidate_count",
    "truth_recovery_retention",
    "null_like_total",
    "null_like_candidate_count",
    "diffuse_or_wrong_total",
    "diffuse_or_wrong_candidate_count",
    "fragment_like_total",
    "fragment_like_candidate_count",
    "negative_candidate_count",
    "candidate_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "grid_count",
    "zero_negative_grid_count",
    "zero_negative_recovery_retaining_grid_count",
    "max_zero_negative_truth_recovery_retention",
    "median_zero_negative_truth_recovery_retention",
    "default_rule_status",
    "diagnostic_status",
)


@dataclass(frozen=True)
class InternalNodeLikelihoodThresholds:
    """One row-level threshold configuration."""

    context_margin_floor: float
    soft_subspace_floor: float
    size_balance_floor: float
    edge_norm_balance_floor: float
    fragment_risk_ceiling: float
    p_extreme_log_bayes_factor_floor: float = DEFAULT_P_BF_FLOOR

    @property
    def rule_id(self) -> str:
        return (
            f"ctx={self.context_margin_floor:g}"
            f"|sub={self.soft_subspace_floor:g}"
            f"|size={self.size_balance_floor:g}"
            f"|edge={self.edge_norm_balance_floor:g}"
            f"|frag={self.fragment_risk_ceiling:g}"
            f"|bf={self.p_extreme_log_bayes_factor_floor:g}"
        )


@dataclass(frozen=True)
class OverlapInternalNodeLikelihoodSensitivityConfig:
    """Runtime contract for row-level likelihood threshold sensitivity."""

    likelihood_rows_path: Path
    output_dir: Path
    context_margin_floors: tuple[float, ...] = DEFAULT_CONTEXT_MARGIN_FLOORS
    subspace_floors: tuple[float, ...] = DEFAULT_SUBSPACE_FLOORS
    size_balance_floors: tuple[float, ...] = DEFAULT_SIZE_BALANCE_FLOORS
    edge_norm_balance_floors: tuple[float, ...] = DEFAULT_EDGE_NORM_BALANCE_FLOORS
    fragment_risk_ceilings: tuple[float, ...] = DEFAULT_FRAGMENT_RISK_CEILINGS
    p_extreme_log_bayes_factor_floor: float = DEFAULT_P_BF_FLOOR
    recovery_retention_floor: float = 0.75

    @property
    def grid_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_sensitivity.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_sensitivity_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Threshold grid must contain at least one value.")
    return values


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"Likelihood rows are missing columns: {missing!r}")


def threshold_grid(
    *,
    context_margin_floors: Sequence[float] = DEFAULT_CONTEXT_MARGIN_FLOORS,
    subspace_floors: Sequence[float] = DEFAULT_SUBSPACE_FLOORS,
    size_balance_floors: Sequence[float] = DEFAULT_SIZE_BALANCE_FLOORS,
    edge_norm_balance_floors: Sequence[float] = DEFAULT_EDGE_NORM_BALANCE_FLOORS,
    fragment_risk_ceilings: Sequence[float] = DEFAULT_FRAGMENT_RISK_CEILINGS,
    p_extreme_log_bayes_factor_floor: float = DEFAULT_P_BF_FLOOR,
) -> list[InternalNodeLikelihoodThresholds]:
    """Build the predeclared threshold sensitivity grid."""
    return [
        InternalNodeLikelihoodThresholds(
            context_margin_floor=float(context),
            soft_subspace_floor=float(subspace),
            size_balance_floor=float(size_balance),
            edge_norm_balance_floor=float(edge_balance),
            fragment_risk_ceiling=float(fragment_ceiling),
            p_extreme_log_bayes_factor_floor=float(p_extreme_log_bayes_factor_floor),
        )
        for context in context_margin_floors
        for subspace in subspace_floors
        for size_balance in size_balance_floors
        for edge_balance in edge_norm_balance_floors
        for fragment_ceiling in fragment_risk_ceilings
    ]


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def apply_internal_node_thresholds(
    likelihood_rows: pd.DataFrame,
    thresholds: InternalNodeLikelihoodThresholds,
    *,
    recovery_retention_floor: float = 0.75,
) -> dict[str, object]:
    """Evaluate one threshold configuration against row labels."""
    _validate_rows(likelihood_rows)
    rows = likelihood_rows
    candidate = (
        _numeric(rows, "selected_family_log_bayes_factor_lower").ge(
            float(thresholds.p_extreme_log_bayes_factor_floor)
        )
        & _numeric(rows, "continuous_context_min_margin").ge(float(thresholds.context_margin_floor))
        & _numeric(rows, "subspace_consensus_jaccard_topk").ge(
            float(thresholds.soft_subspace_floor)
        )
        & _numeric(rows, "size_balance").ge(float(thresholds.size_balance_floor))
        & _numeric(rows, "edge_norm_balance").ge(float(thresholds.edge_norm_balance_floor))
        & _numeric(rows, "fragment_risk_proxy_score").le(float(thresholds.fragment_risk_ceiling))
    )
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    null_like = roles.eq("null_like")
    diffuse_or_wrong = roles.eq("diffuse_or_wrong")
    fragment_like = roles.eq("fragment_like")
    negative = ~truth
    truth_total = int(truth.sum())
    truth_count = int((candidate & truth).sum())
    retention = float(truth_count / truth_total) if truth_total else float("nan")
    negative_count = int((candidate & negative).sum())
    if negative_count == 0 and retention >= float(recovery_retention_floor):
        status = "zero_negative_recovery_retaining"
    elif negative_count == 0:
        status = "zero_negative_low_recovery"
    else:
        status = "negative_leakage"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "rule_id": thresholds.rule_id,
        "p_extreme_log_bayes_factor_floor": float(thresholds.p_extreme_log_bayes_factor_floor),
        "context_margin_floor": float(thresholds.context_margin_floor),
        "soft_subspace_floor": float(thresholds.soft_subspace_floor),
        "size_balance_floor": float(thresholds.size_balance_floor),
        "edge_norm_balance_floor": float(thresholds.edge_norm_balance_floor),
        "fragment_risk_ceiling": float(thresholds.fragment_risk_ceiling),
        "truth_recovery_total": truth_total,
        "truth_recovery_candidate_count": truth_count,
        "truth_recovery_retention": retention,
        "null_like_total": int(null_like.sum()),
        "null_like_candidate_count": int((candidate & null_like).sum()),
        "diffuse_or_wrong_total": int(diffuse_or_wrong.sum()),
        "diffuse_or_wrong_candidate_count": int((candidate & diffuse_or_wrong).sum()),
        "fragment_like_total": int(fragment_like.sum()),
        "fragment_like_candidate_count": int((candidate & fragment_like).sum()),
        "negative_candidate_count": negative_count,
        "candidate_status": status,
    }


def build_internal_node_likelihood_sensitivity(
    likelihood_rows: pd.DataFrame,
    *,
    grid: Sequence[InternalNodeLikelihoodThresholds],
    recovery_retention_floor: float = 0.75,
) -> pd.DataFrame:
    """Evaluate a threshold grid over row-level likelihood rows."""
    _validate_rows(likelihood_rows)
    records = [
        apply_internal_node_thresholds(
            likelihood_rows,
            thresholds,
            recovery_retention_floor=float(recovery_retention_floor),
        )
        for thresholds in grid
    ]
    return pd.DataFrame.from_records(records, columns=GRID_COLUMNS)


def summarize_internal_node_likelihood_sensitivity(
    grid_rows: pd.DataFrame,
    *,
    default_rule_id: str,
) -> pd.DataFrame:
    """Summarize the threshold sensitivity grid."""
    if grid_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    zero_negative = grid_rows[grid_rows["negative_candidate_count"].eq(0)]
    zero_retain = zero_negative[
        zero_negative["candidate_status"].eq("zero_negative_recovery_retaining")
    ]
    default = grid_rows[grid_rows["rule_id"].eq(str(default_rule_id))]
    default_status = str(default["candidate_status"].iloc[0]) if not default.empty else "missing"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "grid_count": int(grid_rows.shape[0]),
        "zero_negative_grid_count": int(zero_negative.shape[0]),
        "zero_negative_recovery_retaining_grid_count": int(zero_retain.shape[0]),
        "max_zero_negative_truth_recovery_retention": (
            float(zero_negative["truth_recovery_retention"].max())
            if not zero_negative.empty
            else float("nan")
        ),
        "median_zero_negative_truth_recovery_retention": (
            float(zero_negative["truth_recovery_retention"].median())
            if not zero_negative.empty
            else float("nan")
        ),
        "default_rule_status": default_status,
        "diagnostic_status": "diagnostic_only_threshold_sensitivity_not_calibration",
    }
    return pd.DataFrame.from_records([summary], columns=SUMMARY_COLUMNS)


def run_overlap_internal_node_likelihood_sensitivity(
    config: OverlapInternalNodeLikelihoodSensitivityConfig,
) -> dict[str, Path]:
    """Run row-level threshold sensitivity and write outputs."""
    likelihood_rows = pd.read_csv(config.likelihood_rows_path)
    grid = threshold_grid(
        context_margin_floors=config.context_margin_floors,
        subspace_floors=config.subspace_floors,
        size_balance_floors=config.size_balance_floors,
        edge_norm_balance_floors=config.edge_norm_balance_floors,
        fragment_risk_ceilings=config.fragment_risk_ceilings,
        p_extreme_log_bayes_factor_floor=config.p_extreme_log_bayes_factor_floor,
    )
    grid_rows = build_internal_node_likelihood_sensitivity(
        likelihood_rows,
        grid=grid,
        recovery_retention_floor=float(config.recovery_retention_floor),
    )
    default_rule = InternalNodeLikelihoodThresholds(
        context_margin_floor=0.0,
        soft_subspace_floor=0.15,
        size_balance_floor=0.33,
        edge_norm_balance_floor=0.49,
        fragment_risk_ceiling=1.25,
        p_extreme_log_bayes_factor_floor=float(config.p_extreme_log_bayes_factor_floor),
    )
    summary = summarize_internal_node_likelihood_sensitivity(
        grid_rows,
        default_rule_id=default_rule.rule_id,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows.to_csv(config.grid_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "likelihood_rows_path": str(config.likelihood_rows_path),
        "outputs": {
            "grid": str(config.grid_path),
            "summary": str(config.summary_path),
        },
        "grid": {
            "context_margin_floors": list(config.context_margin_floors),
            "subspace_floors": list(config.subspace_floors),
            "size_balance_floors": list(config.size_balance_floors),
            "edge_norm_balance_floors": list(config.edge_norm_balance_floors),
            "fragment_risk_ceilings": list(config.fragment_risk_ceilings),
            "p_extreme_log_bayes_factor_floor": float(config.p_extreme_log_bayes_factor_floor),
        },
        "production_status": "diagnostic_only_no_threshold_promotion",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "grid": config.grid_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--context-margin-floors", default=",".join(map(str, DEFAULT_CONTEXT_MARGIN_FLOORS))
    )
    parser.add_argument("--subspace-floors", default=",".join(map(str, DEFAULT_SUBSPACE_FLOORS)))
    parser.add_argument(
        "--size-balance-floors", default=",".join(map(str, DEFAULT_SIZE_BALANCE_FLOORS))
    )
    parser.add_argument(
        "--edge-norm-balance-floors",
        default=",".join(map(str, DEFAULT_EDGE_NORM_BALANCE_FLOORS)),
    )
    parser.add_argument(
        "--fragment-risk-ceilings",
        default=",".join(map(str, DEFAULT_FRAGMENT_RISK_CEILINGS)),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_internal_node_likelihood_sensitivity(
        OverlapInternalNodeLikelihoodSensitivityConfig(
            likelihood_rows_path=args.likelihood_rows_path,
            output_dir=args.output_dir,
            context_margin_floors=_parse_float_grid(str(args.context_margin_floors)),
            subspace_floors=_parse_float_grid(str(args.subspace_floors)),
            size_balance_floors=_parse_float_grid(str(args.size_balance_floors)),
            edge_norm_balance_floors=_parse_float_grid(str(args.edge_norm_balance_floors)),
            fragment_risk_ceilings=_parse_float_grid(str(args.fragment_risk_ceilings)),
        )
    )


if __name__ == "__main__":
    main()
