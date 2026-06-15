"""Continuous structural traversal rule diagnostics for overlap rows.

This diagnostic scores smooth threshold surfaces over traversal context:
depth, parent size, and barycentric balance. It is diagnostic-only and does not
install a production traversal rule.
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

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_structural_continuous_rule_not_calibration"
SCHEMA_VERSION = "overlap_structural_continuous_rule/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_structural_continuous_rule"

DEFAULT_BASE_THRESHOLDS = (0.0025, 0.005, 0.0075)
DEFAULT_SHALLOW_PENALTIES = (0.0, 0.005, 0.01, 0.015)
DEFAULT_PARENT_SIZE_PENALTIES = (0.0, 0.005, 0.01)
DEFAULT_IMBALANCE_PENALTIES = (0.0, 0.005, 0.01)
DEFAULT_SUBSPACE_THRESHOLDS = (0.15, 0.25)
DEFAULT_SIBLING_P_THRESHOLDS = (0.001, 0.01)
DEFAULT_DEPTH_SCALE = 1.5
DEFAULT_PARENT_REFERENCE_SIZE = 800.0
DEFAULT_BALANCE_REFERENCE = 0.50

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "rule_id",
    "base_threshold",
    "shallow_penalty",
    "parent_size_penalty",
    "imbalance_penalty",
    "depth_scale",
    "parent_reference_size",
    "balance_reference",
    "subspace_consensus_threshold",
    "sibling_p_threshold",
    "null_structural_accept_count",
    "null_case_replicates_with_accept",
    "signal_truth_aligned_total",
    "signal_truth_aligned_accept_count",
    "signal_truth_aligned_retention",
    "signal_truth_aligned_case_replicates_total",
    "signal_truth_aligned_case_replicates_retained",
    "signal_truth_misaligned_accept_count",
    "signal_truth_misaligned_case_replicates_accepted",
    "median_accept_threshold",
    "median_accept_depth",
    "median_accept_parent_size",
    "median_accept_balance",
    "rule_score",
    "rule_status",
)

DECISION_COLUMNS = (
    "schema_version",
    "study_role",
    "rule_id",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "decision_class",
    "depth",
    "n_parent",
    "barycentric_balance",
    "sibling_p_value",
    "homogeneity_gain_min",
    "subspace_consensus_jaccard_topk",
    "truth_split_ari",
    "continuous_homogeneity_threshold",
    "continuous_rule_accept",
)


@dataclass(frozen=True)
class ContinuousRuleParameters:
    """Parameters for a smooth structural homogeneity threshold surface."""

    base_threshold: float
    shallow_penalty: float
    parent_size_penalty: float
    imbalance_penalty: float
    subspace_consensus_threshold: float
    sibling_p_threshold: float
    depth_scale: float = DEFAULT_DEPTH_SCALE
    parent_reference_size: float = DEFAULT_PARENT_REFERENCE_SIZE
    balance_reference: float = DEFAULT_BALANCE_REFERENCE

    @property
    def rule_id(self) -> str:
        return (
            "continuous_structural"
            f"|base={self.base_threshold:g}"
            f"|shallow={self.shallow_penalty:g}"
            f"|parent={self.parent_size_penalty:g}"
            f"|imbalance={self.imbalance_penalty:g}"
            f"|subspace={self.subspace_consensus_threshold:g}"
            f"|p={self.sibling_p_threshold:g}"
        )


@dataclass(frozen=True)
class OverlapStructuralContinuousRuleConfig:
    """Runtime contract for continuous structural rule diagnostics."""

    rows_path: Path
    output_dir: Path
    base_thresholds: tuple[float, ...] = DEFAULT_BASE_THRESHOLDS
    shallow_penalties: tuple[float, ...] = DEFAULT_SHALLOW_PENALTIES
    parent_size_penalties: tuple[float, ...] = DEFAULT_PARENT_SIZE_PENALTIES
    imbalance_penalties: tuple[float, ...] = DEFAULT_IMBALANCE_PENALTIES
    subspace_thresholds: tuple[float, ...] = DEFAULT_SUBSPACE_THRESHOLDS
    sibling_p_thresholds: tuple[float, ...] = DEFAULT_SIBLING_P_THRESHOLDS
    depth_scale: float = DEFAULT_DEPTH_SCALE
    parent_reference_size: float = DEFAULT_PARENT_REFERENCE_SIZE
    balance_reference: float = DEFAULT_BALANCE_REFERENCE
    truth_alignment_threshold: float = 0.50
    write_decision_rows: bool = True

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_structural_continuous_rule_summary.csv"

    @property
    def decision_rows_path(self) -> Path:
        return self.output_dir / "overlap_structural_continuous_rule_rows.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Continuous rule grid must contain at least one value.")
    return values


def _required_columns() -> set[str]:
    return {
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "decision_class",
        "depth",
        "n_parent",
        "barycentric_balance",
        "sibling_p_value",
        "homogeneity_gain_min",
        "subspace_consensus_jaccard_topk",
        "truth_split_ari",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Overlap structural rows are missing columns: {missing!r}")


def continuous_homogeneity_threshold(
    rows: pd.DataFrame,
    parameters: ContinuousRuleParameters,
) -> pd.Series:
    """Return smooth homogeneity thresholds for traversal rows."""
    depth = pd.to_numeric(rows["depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    parent_size = (
        pd.to_numeric(rows["n_parent"], errors="coerce").fillna(0.0).clip(lower=1.0)
    )
    balance = (
        pd.to_numeric(rows["barycentric_balance"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=0.5)
    )
    depth_scale = max(float(parameters.depth_scale), 1e-9)
    parent_ref = max(float(parameters.parent_reference_size), 1.0)
    shallow_term = np.exp(-depth / depth_scale)
    parent_term = np.log1p(parent_size) / math.log1p(parent_ref)
    imbalance_term = (float(parameters.balance_reference) - balance).clip(lower=0.0)
    threshold = (
        float(parameters.base_threshold)
        + float(parameters.shallow_penalty) * shallow_term
        + float(parameters.parent_size_penalty) * parent_term
        + float(parameters.imbalance_penalty) * imbalance_term
    )
    return pd.Series(threshold, index=rows.index, dtype=float)


def parameter_grid(
    *,
    base_thresholds: Sequence[float] = DEFAULT_BASE_THRESHOLDS,
    shallow_penalties: Sequence[float] = DEFAULT_SHALLOW_PENALTIES,
    parent_size_penalties: Sequence[float] = DEFAULT_PARENT_SIZE_PENALTIES,
    imbalance_penalties: Sequence[float] = DEFAULT_IMBALANCE_PENALTIES,
    subspace_thresholds: Sequence[float] = DEFAULT_SUBSPACE_THRESHOLDS,
    sibling_p_thresholds: Sequence[float] = DEFAULT_SIBLING_P_THRESHOLDS,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    parent_reference_size: float = DEFAULT_PARENT_REFERENCE_SIZE,
    balance_reference: float = DEFAULT_BALANCE_REFERENCE,
) -> list[ContinuousRuleParameters]:
    """Build continuous-rule parameter grid."""
    grid: list[ContinuousRuleParameters] = []
    for base in base_thresholds:
        for shallow in shallow_penalties:
            for parent in parent_size_penalties:
                for imbalance in imbalance_penalties:
                    for subspace in subspace_thresholds:
                        for sibling_p in sibling_p_thresholds:
                            grid.append(
                                ContinuousRuleParameters(
                                    base_threshold=float(base),
                                    shallow_penalty=float(shallow),
                                    parent_size_penalty=float(parent),
                                    imbalance_penalty=float(imbalance),
                                    subspace_consensus_threshold=float(subspace),
                                    sibling_p_threshold=float(sibling_p),
                                    depth_scale=float(depth_scale),
                                    parent_reference_size=float(parent_reference_size),
                                    balance_reference=float(balance_reference),
                                )
                            )
    return grid


def _is_null_role(series: pd.Series) -> pd.Series:
    return series.astype(str).isin({"null", "selected_null"})


def _is_signal_role(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("signal")


def apply_continuous_rule(
    rows: pd.DataFrame,
    parameters: ContinuousRuleParameters,
    *,
    truth_alignment_threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply one continuous rule and return row decisions plus summary."""
    _validate_rows(rows)
    threshold = continuous_homogeneity_threshold(rows, parameters)
    accepted_split = rows["decision_class"].astype(str).eq("accepted_internal_split")
    accept = (
        accepted_split
        & pd.to_numeric(rows["sibling_p_value"], errors="coerce").le(
            float(parameters.sibling_p_threshold)
        )
        & pd.to_numeric(rows["homogeneity_gain_min"], errors="coerce").ge(threshold)
        & pd.to_numeric(
            rows["subspace_consensus_jaccard_topk"],
            errors="coerce",
        ).ge(float(parameters.subspace_consensus_threshold))
    )
    truth_ari = pd.to_numeric(rows["truth_split_ari"], errors="coerce")
    truth_aligned = truth_ari.ge(float(truth_alignment_threshold))
    truth_available = truth_ari.notna()
    truth_misaligned = truth_available & ~truth_aligned
    null_mask = _is_null_role(rows["data_role"])
    signal_mask = _is_signal_role(rows["data_role"])
    case_rep = rows["case_id"].astype(str) + "::" + rows["replicate"].astype(str)
    signal_truth_aligned_total_by_case_rep = (
        accepted_split & signal_mask & truth_aligned
    ).groupby(case_rep).sum()
    signal_truth_aligned_case_replicates_total = int(
        signal_truth_aligned_total_by_case_rep.gt(0).sum()
    )
    signal_truth_aligned_accept_by_case_rep = (
        accept & signal_mask & truth_aligned
    ).groupby(case_rep).sum()
    signal_truth_aligned_case_replicates_retained = int(
        (
            signal_truth_aligned_total_by_case_rep.gt(0)
            & signal_truth_aligned_accept_by_case_rep.gt(0)
        ).sum()
    )
    null_case_replicates_with_accept = int((accept & null_mask).groupby(case_rep).sum().gt(0).sum())
    signal_truth_misaligned_case_replicates_accepted = int(
        (accept & signal_mask & truth_misaligned).groupby(case_rep).sum().gt(0).sum()
    )
    signal_truth_aligned_total = int((accepted_split & signal_mask & truth_aligned).sum())
    signal_truth_aligned_accept = int((accept & signal_mask & truth_aligned).sum())
    signal_truth_misaligned_accept = int((accept & signal_mask & truth_misaligned).sum())
    null_accept = int((accept & null_mask).sum())
    retention = (
        signal_truth_aligned_accept / signal_truth_aligned_total
        if signal_truth_aligned_total > 0
        else np.nan
    )
    if null_accept > 0:
        status = "continuous_rule_null_uncontrolled"
    elif (
        signal_truth_aligned_case_replicates_total > 0
        and signal_truth_aligned_case_replicates_retained
        < signal_truth_aligned_case_replicates_total
    ):
        status = "continuous_rule_signal_loss"
    elif signal_truth_misaligned_accept > 0:
        status = "continuous_rule_truth_misaligned_kept"
    elif signal_truth_aligned_total == 0:
        status = "continuous_rule_no_truth_aligned_signal"
    else:
        status = "continuous_rule_candidate"
    score = (
        (0.0 if np.isnan(retention) else float(retention))
        - 2.0 * null_accept
        - 0.25 * signal_truth_misaligned_accept
        - 0.50
        * (
            signal_truth_aligned_case_replicates_total
            - signal_truth_aligned_case_replicates_retained
        )
    )
    accepted_thresholds = threshold[accept]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "rule_id": parameters.rule_id,
        "base_threshold": float(parameters.base_threshold),
        "shallow_penalty": float(parameters.shallow_penalty),
        "parent_size_penalty": float(parameters.parent_size_penalty),
        "imbalance_penalty": float(parameters.imbalance_penalty),
        "depth_scale": float(parameters.depth_scale),
        "parent_reference_size": float(parameters.parent_reference_size),
        "balance_reference": float(parameters.balance_reference),
        "subspace_consensus_threshold": float(parameters.subspace_consensus_threshold),
        "sibling_p_threshold": float(parameters.sibling_p_threshold),
        "null_structural_accept_count": null_accept,
        "null_case_replicates_with_accept": null_case_replicates_with_accept,
        "signal_truth_aligned_total": signal_truth_aligned_total,
        "signal_truth_aligned_accept_count": signal_truth_aligned_accept,
        "signal_truth_aligned_retention": (
            float(retention) if not np.isnan(retention) else np.nan
        ),
        "signal_truth_aligned_case_replicates_total": (
            signal_truth_aligned_case_replicates_total
        ),
        "signal_truth_aligned_case_replicates_retained": (
            signal_truth_aligned_case_replicates_retained
        ),
        "signal_truth_misaligned_accept_count": signal_truth_misaligned_accept,
        "signal_truth_misaligned_case_replicates_accepted": (
            signal_truth_misaligned_case_replicates_accepted
        ),
        "median_accept_threshold": (
            float(accepted_thresholds.median()) if not accepted_thresholds.empty else np.nan
        ),
        "median_accept_depth": (
            float(pd.to_numeric(rows.loc[accept, "depth"], errors="coerce").median())
            if bool(accept.any())
            else np.nan
        ),
        "median_accept_parent_size": (
            float(pd.to_numeric(rows.loc[accept, "n_parent"], errors="coerce").median())
            if bool(accept.any())
            else np.nan
        ),
        "median_accept_balance": (
            float(
                pd.to_numeric(
                    rows.loc[accept, "barycentric_balance"],
                    errors="coerce",
                ).median()
            )
            if bool(accept.any())
            else np.nan
        ),
        "rule_score": float(score),
        "rule_status": status,
    }
    decisions = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "rule_id": parameters.rule_id,
            "case_id": rows["case_id"].astype(str),
            "data_role": rows["data_role"].astype(str),
            "replicate": rows["replicate"].astype(int),
            "node_id": rows["node_id"].astype(str),
            "decision_class": rows["decision_class"].astype(str),
            "depth": pd.to_numeric(rows["depth"], errors="coerce"),
            "n_parent": pd.to_numeric(rows["n_parent"], errors="coerce"),
            "barycentric_balance": pd.to_numeric(
                rows["barycentric_balance"],
                errors="coerce",
            ),
            "sibling_p_value": pd.to_numeric(rows["sibling_p_value"], errors="coerce"),
            "homogeneity_gain_min": pd.to_numeric(
                rows["homogeneity_gain_min"],
                errors="coerce",
            ),
            "subspace_consensus_jaccard_topk": pd.to_numeric(
                rows["subspace_consensus_jaccard_topk"],
                errors="coerce",
            ),
            "truth_split_ari": truth_ari,
            "continuous_homogeneity_threshold": threshold,
            "continuous_rule_accept": accept.astype(bool),
        },
        columns=DECISION_COLUMNS,
    )
    return decisions, summary


def evaluate_continuous_rules(
    rows: pd.DataFrame,
    parameters: Sequence[ContinuousRuleParameters],
    *,
    truth_alignment_threshold: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate many continuous rules."""
    summaries: list[dict[str, object]] = []
    decision_frames: list[pd.DataFrame] = []
    for params in parameters:
        decisions, summary = apply_continuous_rule(
            rows,
            params,
            truth_alignment_threshold=float(truth_alignment_threshold),
        )
        summaries.append(summary)
        decision_frames.append(decisions)
    summary_frame = pd.DataFrame.from_records(summaries, columns=SUMMARY_COLUMNS)
    summary_frame = summary_frame.sort_values(
        ["rule_status", "rule_score"],
        ascending=[True, False],
        ignore_index=True,
    )
    decision_frame = pd.concat(decision_frames, ignore_index=True)
    return decision_frame, summary_frame


def run_overlap_structural_continuous_rule(
    config: OverlapStructuralContinuousRuleConfig,
) -> dict[str, Path]:
    """Run continuous-rule diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    params = parameter_grid(
        base_thresholds=config.base_thresholds,
        shallow_penalties=config.shallow_penalties,
        parent_size_penalties=config.parent_size_penalties,
        imbalance_penalties=config.imbalance_penalties,
        subspace_thresholds=config.subspace_thresholds,
        sibling_p_thresholds=config.sibling_p_thresholds,
        depth_scale=float(config.depth_scale),
        parent_reference_size=float(config.parent_reference_size),
        balance_reference=float(config.balance_reference),
    )
    decisions, summary = evaluate_continuous_rules(
        rows,
        params,
        truth_alignment_threshold=float(config.truth_alignment_threshold),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.summary_path, index=False)
    outputs = {"summary": str(config.summary_path)}
    if bool(config.write_decision_rows):
        decisions.to_csv(config.decision_rows_path, index=False)
        outputs["decision_rows"] = str(config.decision_rows_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "base_thresholds": list(config.base_thresholds),
        "shallow_penalties": list(config.shallow_penalties),
        "parent_size_penalties": list(config.parent_size_penalties),
        "imbalance_penalties": list(config.imbalance_penalties),
        "subspace_thresholds": list(config.subspace_thresholds),
        "sibling_p_thresholds": list(config.sibling_p_thresholds),
        "depth_scale": float(config.depth_scale),
        "parent_reference_size": float(config.parent_reference_size),
        "balance_reference": float(config.balance_reference),
        "truth_alignment_threshold": float(config.truth_alignment_threshold),
        "rule_count": int(summary.shape[0]),
        "outputs": outputs,
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    result = {"summary": config.summary_path, "manifest": config.manifest_path}
    if bool(config.write_decision_rows):
        result["decision_rows"] = config.decision_rows_path
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-thresholds",
        default=",".join(str(value) for value in DEFAULT_BASE_THRESHOLDS),
    )
    parser.add_argument(
        "--shallow-penalties",
        default=",".join(str(value) for value in DEFAULT_SHALLOW_PENALTIES),
    )
    parser.add_argument(
        "--parent-size-penalties",
        default=",".join(str(value) for value in DEFAULT_PARENT_SIZE_PENALTIES),
    )
    parser.add_argument(
        "--imbalance-penalties",
        default=",".join(str(value) for value in DEFAULT_IMBALANCE_PENALTIES),
    )
    parser.add_argument(
        "--subspace-thresholds",
        default=",".join(str(value) for value in DEFAULT_SUBSPACE_THRESHOLDS),
    )
    parser.add_argument(
        "--sibling-p-thresholds",
        default=",".join(str(value) for value in DEFAULT_SIBLING_P_THRESHOLDS),
    )
    parser.add_argument("--depth-scale", default=DEFAULT_DEPTH_SCALE, type=float)
    parser.add_argument(
        "--parent-reference-size",
        default=DEFAULT_PARENT_REFERENCE_SIZE,
        type=float,
    )
    parser.add_argument(
        "--balance-reference",
        default=DEFAULT_BALANCE_REFERENCE,
        type=float,
    )
    parser.add_argument("--truth-alignment-threshold", default=0.50, type=float)
    parser.add_argument("--skip-decision-rows", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_structural_continuous_rule(
        OverlapStructuralContinuousRuleConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            base_thresholds=_parse_float_grid(args.base_thresholds),
            shallow_penalties=_parse_float_grid(args.shallow_penalties),
            parent_size_penalties=_parse_float_grid(args.parent_size_penalties),
            imbalance_penalties=_parse_float_grid(args.imbalance_penalties),
            subspace_thresholds=_parse_float_grid(args.subspace_thresholds),
            sibling_p_thresholds=_parse_float_grid(args.sibling_p_thresholds),
            depth_scale=float(args.depth_scale),
            parent_reference_size=float(args.parent_reference_size),
            balance_reference=float(args.balance_reference),
            truth_alignment_threshold=float(args.truth_alignment_threshold),
            write_decision_rows=not bool(args.skip_decision_rows),
        )
    )


if __name__ == "__main__":
    main()
