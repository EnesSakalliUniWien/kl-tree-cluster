"""Transfer diagnostics for row-level overlap internal-node likelihood rules.

This diagnostic asks whether candidate threshold rules selected on training
splits remain zero-leakage on held-out cases or replicates. It is
diagnostic-only and does not promote production thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.diagnostics.calibration.overlap.overlap_internal_node_likelihood_sensitivity import (
    DEFAULT_CONTEXT_MARGIN_FLOORS,
    DEFAULT_EDGE_NORM_BALANCE_FLOORS,
    DEFAULT_FRAGMENT_RISK_CEILINGS,
    DEFAULT_P_BF_FLOOR,
    DEFAULT_SIZE_BALANCE_FLOORS,
    DEFAULT_SUBSPACE_FLOORS,
    InternalNodeLikelihoodThresholds,
    apply_internal_node_thresholds,
    threshold_grid,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_internal_node_likelihood_transfer"
SCHEMA_VERSION = "overlap_internal_node_likelihood_transfer/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_internal_node_likelihood_transfer"
)

TRANSFER_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "holdout_value",
    "rule_id",
    "train_rule_status",
    "train_truth_recovery_total",
    "train_truth_recovery_candidate_count",
    "train_truth_recovery_retention",
    "train_negative_candidate_count",
    "test_truth_recovery_total",
    "test_truth_recovery_candidate_count",
    "test_truth_recovery_retention",
    "test_null_like_total",
    "test_null_like_candidate_count",
    "test_diffuse_or_wrong_total",
    "test_diffuse_or_wrong_candidate_count",
    "test_fragment_like_total",
    "test_fragment_like_candidate_count",
    "test_negative_candidate_count",
    "transfer_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "split_kind",
    "split_count",
    "selected_rule_evaluation_count",
    "leakage_rule_count",
    "zero_leakage_rule_count",
    "truth_recovery_test_total",
    "max_test_truth_recovery_retention",
    "median_test_truth_recovery_retention",
    "transfer_status",
)


@dataclass(frozen=True)
class OverlapInternalNodeLikelihoodTransferConfig:
    """Runtime contract for row-level likelihood transfer diagnostics."""

    likelihood_rows_path: Path
    output_dir: Path
    split_columns: tuple[str, ...] = ("case_id", "replicate")
    context_margin_floors: tuple[float, ...] = DEFAULT_CONTEXT_MARGIN_FLOORS
    subspace_floors: tuple[float, ...] = DEFAULT_SUBSPACE_FLOORS
    size_balance_floors: tuple[float, ...] = DEFAULT_SIZE_BALANCE_FLOORS
    edge_norm_balance_floors: tuple[float, ...] = DEFAULT_EDGE_NORM_BALANCE_FLOORS
    fragment_risk_ceilings: tuple[float, ...] = DEFAULT_FRAGMENT_RISK_CEILINGS
    p_extreme_log_bayes_factor_floor: float = DEFAULT_P_BF_FLOOR
    recovery_retention_floor: float = 0.75

    @property
    def transfer_rows_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_transfer_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_transfer_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Threshold grid must contain at least one value.")
    return values


def _split_values(rows: pd.DataFrame, split_column: str) -> Iterable[Any]:
    values = rows[split_column].dropna().unique()
    return sorted(values.tolist(), key=str)


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _transfer_status(test_result: dict[str, object]) -> str:
    negative_count = int(test_result["negative_candidate_count"])
    truth_total = int(test_result["truth_recovery_total"])
    truth_count = int(test_result["truth_recovery_candidate_count"])
    if negative_count:
        return "transfer_leakage"
    if truth_total and truth_count == truth_total:
        return "transfer_full_recovery_retention"
    if truth_count:
        return "transfer_partial_recovery_retention"
    if truth_total:
        return "transfer_no_recovery_retention"
    return "transfer_no_truth_recovery_in_holdout"


def _selected_train_rules(
    train: pd.DataFrame,
    grid: Sequence[InternalNodeLikelihoodThresholds],
    *,
    recovery_retention_floor: float,
) -> list[tuple[InternalNodeLikelihoodThresholds, dict[str, object]]]:
    selected: list[tuple[InternalNodeLikelihoodThresholds, dict[str, object]]] = []
    for thresholds in grid:
        result = apply_internal_node_thresholds(
            train,
            thresholds,
            recovery_retention_floor=float(recovery_retention_floor),
        )
        if result["candidate_status"] == "zero_negative_recovery_retaining":
            selected.append((thresholds, result))
    return selected


def build_internal_node_likelihood_transfer_rows(
    likelihood_rows: pd.DataFrame,
    *,
    grid: Sequence[InternalNodeLikelihoodThresholds],
    split_columns: Sequence[str] = ("case_id", "replicate"),
    recovery_retention_floor: float = 0.75,
) -> pd.DataFrame:
    """Build leave-one-split transfer rows for selected threshold rules."""
    records: list[dict[str, object]] = []
    missing = sorted(set(split_columns) - set(likelihood_rows.columns))
    if missing:
        raise ValueError(f"Likelihood rows are missing split columns: {missing!r}")
    for split_column in split_columns:
        for holdout_value in _split_values(likelihood_rows, split_column):
            test_mask = likelihood_rows[split_column].eq(holdout_value)
            train = likelihood_rows.loc[~test_mask].copy()
            test = likelihood_rows.loc[test_mask].copy()
            selected = _selected_train_rules(
                train,
                grid,
                recovery_retention_floor=float(recovery_retention_floor),
            )
            if not selected:
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "split_kind": f"leave_one_{split_column}",
                        "holdout_value": str(holdout_value),
                        "rule_id": "no_train_rule",
                        "train_rule_status": "no_zero_negative_recovery_retaining_rule",
                        "train_truth_recovery_total": int(
                            train["guard_truth_role"].astype(str).eq("truth_recovery").sum()
                        ),
                        "train_truth_recovery_candidate_count": 0,
                        "train_truth_recovery_retention": float("nan"),
                        "train_negative_candidate_count": 0,
                        "test_truth_recovery_total": int(
                            test["guard_truth_role"].astype(str).eq("truth_recovery").sum()
                        ),
                        "test_truth_recovery_candidate_count": 0,
                        "test_truth_recovery_retention": float("nan"),
                        "test_null_like_total": int(
                            test["guard_truth_role"].astype(str).eq("null_like").sum()
                        ),
                        "test_null_like_candidate_count": 0,
                        "test_diffuse_or_wrong_total": int(
                            test["guard_truth_role"].astype(str).eq("diffuse_or_wrong").sum()
                        ),
                        "test_diffuse_or_wrong_candidate_count": 0,
                        "test_fragment_like_total": int(
                            test["guard_truth_role"].astype(str).eq("fragment_like").sum()
                        ),
                        "test_fragment_like_candidate_count": 0,
                        "test_negative_candidate_count": 0,
                        "transfer_status": "transfer_no_train_candidate_rule",
                    }
                )
                continue
            for thresholds, train_result in selected:
                test_result = apply_internal_node_thresholds(
                    test,
                    thresholds,
                    recovery_retention_floor=float(recovery_retention_floor),
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "split_kind": f"leave_one_{split_column}",
                        "holdout_value": str(holdout_value),
                        "rule_id": thresholds.rule_id,
                        "train_rule_status": str(train_result["candidate_status"]),
                        "train_truth_recovery_total": int(train_result["truth_recovery_total"]),
                        "train_truth_recovery_candidate_count": int(
                            train_result["truth_recovery_candidate_count"]
                        ),
                        "train_truth_recovery_retention": float(
                            train_result["truth_recovery_retention"]
                        ),
                        "train_negative_candidate_count": int(
                            train_result["negative_candidate_count"]
                        ),
                        "test_truth_recovery_total": int(test_result["truth_recovery_total"]),
                        "test_truth_recovery_candidate_count": int(
                            test_result["truth_recovery_candidate_count"]
                        ),
                        "test_truth_recovery_retention": float(
                            test_result["truth_recovery_retention"]
                        ),
                        "test_null_like_total": int(test_result["null_like_total"]),
                        "test_null_like_candidate_count": int(
                            test_result["null_like_candidate_count"]
                        ),
                        "test_diffuse_or_wrong_total": int(test_result["diffuse_or_wrong_total"]),
                        "test_diffuse_or_wrong_candidate_count": int(
                            test_result["diffuse_or_wrong_candidate_count"]
                        ),
                        "test_fragment_like_total": int(test_result["fragment_like_total"]),
                        "test_fragment_like_candidate_count": int(
                            test_result["fragment_like_candidate_count"]
                        ),
                        "test_negative_candidate_count": int(
                            test_result["negative_candidate_count"]
                        ),
                        "transfer_status": _transfer_status(test_result),
                    }
                )
    return pd.DataFrame.from_records(records, columns=TRANSFER_COLUMNS)


def summarize_internal_node_likelihood_transfer(
    transfer_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize transfer rows by split kind."""
    records: list[dict[str, object]] = []
    for split_kind, group in transfer_rows.groupby("split_kind", sort=True):
        selected = group[~group["rule_id"].astype(str).eq("no_train_rule")]
        leakage = selected["test_negative_candidate_count"].gt(0)
        truth_total = int(group["test_truth_recovery_total"].sum())
        retentions = pd.to_numeric(
            selected["test_truth_recovery_retention"],
            errors="coerce",
        ).dropna()
        if selected.empty:
            status = "transfer_no_train_candidate_rule"
        elif leakage.any():
            status = "transfer_leakage"
        elif not retentions.empty and retentions.max() >= 0.75:
            status = "transfer_zero_leakage_recovery_retaining"
        else:
            status = "transfer_zero_leakage_low_recovery"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "split_kind": str(split_kind),
                "split_count": int(group["holdout_value"].nunique()),
                "selected_rule_evaluation_count": int(selected.shape[0]),
                "leakage_rule_count": int(leakage.sum()) if not selected.empty else 0,
                "zero_leakage_rule_count": int((~leakage).sum()) if not selected.empty else 0,
                "truth_recovery_test_total": truth_total,
                "max_test_truth_recovery_retention": (
                    float(retentions.max()) if not retentions.empty else float("nan")
                ),
                "median_test_truth_recovery_retention": (
                    float(retentions.median()) if not retentions.empty else float("nan")
                ),
                "transfer_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_internal_node_likelihood_transfer(
    config: OverlapInternalNodeLikelihoodTransferConfig,
) -> dict[str, Path]:
    """Run row-level likelihood transfer and write outputs."""
    likelihood_rows = pd.read_csv(config.likelihood_rows_path)
    grid = threshold_grid(
        context_margin_floors=config.context_margin_floors,
        subspace_floors=config.subspace_floors,
        size_balance_floors=config.size_balance_floors,
        edge_norm_balance_floors=config.edge_norm_balance_floors,
        fragment_risk_ceilings=config.fragment_risk_ceilings,
        p_extreme_log_bayes_factor_floor=config.p_extreme_log_bayes_factor_floor,
    )
    transfer_rows = build_internal_node_likelihood_transfer_rows(
        likelihood_rows,
        grid=grid,
        split_columns=config.split_columns,
        recovery_retention_floor=float(config.recovery_retention_floor),
    )
    summary = summarize_internal_node_likelihood_transfer(transfer_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    transfer_rows.to_csv(config.transfer_rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "likelihood_rows_path": str(config.likelihood_rows_path),
        "split_columns": list(config.split_columns),
        "outputs": {
            "transfer_rows": str(config.transfer_rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_no_threshold_promotion",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "transfer_rows": config.transfer_rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-columns", default="case_id,replicate")
    parser.add_argument(
        "--context-margin-floors",
        default=",".join(map(str, DEFAULT_CONTEXT_MARGIN_FLOORS)),
    )
    parser.add_argument(
        "--subspace-floors",
        default=",".join(map(str, DEFAULT_SUBSPACE_FLOORS)),
    )
    parser.add_argument(
        "--size-balance-floors",
        default=",".join(map(str, DEFAULT_SIZE_BALANCE_FLOORS)),
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


def _parse_split_columns(value: str) -> tuple[str, ...]:
    columns = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not columns:
        raise ValueError("At least one split column is required.")
    return columns


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Threshold grid must contain at least one value.")
    return values


def main() -> None:
    args = _parse_args()
    run_overlap_internal_node_likelihood_transfer(
        OverlapInternalNodeLikelihoodTransferConfig(
            likelihood_rows_path=args.likelihood_rows_path,
            output_dir=args.output_dir,
            split_columns=_parse_split_columns(str(args.split_columns)),
            context_margin_floors=_parse_float_grid(str(args.context_margin_floors)),
            subspace_floors=_parse_float_grid(str(args.subspace_floors)),
            size_balance_floors=_parse_float_grid(str(args.size_balance_floors)),
            edge_norm_balance_floors=_parse_float_grid(str(args.edge_norm_balance_floors)),
            fragment_risk_ceilings=_parse_float_grid(str(args.fragment_risk_ceilings)),
        )
    )


if __name__ == "__main__":
    main()
