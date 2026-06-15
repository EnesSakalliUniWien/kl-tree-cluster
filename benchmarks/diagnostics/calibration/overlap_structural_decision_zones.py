"""Three-way structural traversal zones for binary overlap diagnostics.

This panel turns overlap structural sibling rows into diagnostic traversal
zones. It does not promote a production rule. The intent is to separate stable
same-subspace structural accepts from weak-homogeneity rows that should remain
multi-scale/ambiguous unless a selected-family null law supports promotion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.overlap_structural_continuous_rule import (
    ContinuousRuleParameters,
    continuous_homogeneity_threshold,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_structural_decision_zones_not_calibration"
SCHEMA_VERSION = "overlap_structural_decision_zones/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_structural_decision_zones"

DEFAULT_BASE_THRESHOLD = 0.005
DEFAULT_SHALLOW_PENALTY = 0.015
DEFAULT_PARENT_SIZE_PENALTY = 0.0
DEFAULT_IMBALANCE_PENALTY = 0.005
DEFAULT_SUBSPACE_THRESHOLD = 0.15
DEFAULT_SIBLING_P_THRESHOLD = 0.001
DEFAULT_DEPTH_SCALE = 1.5
DEFAULT_PARENT_REFERENCE_SIZE = 800.0
DEFAULT_BALANCE_REFERENCE = 0.50
DEFAULT_TRUTH_ALIGNMENT_THRESHOLD = 0.50

STABLE_STATUS = "structural_same_subspace_supported"
WEAK_OR_UNSTABLE_STATUSES = {
    "weak_homogeneity_gain",
    "same_subspace_heterogeneity_increase",
    "unrelated_subspace_heterogeneity_signal",
    "barycentric_focus_mismatch",
    "structural_homogeneity_subspace_mismatch",
    "structural_subspace_mismatch",
    "degenerate_sibling_contrast",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "structural_truth_role",
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
    "structural_sibling_status",
    "structural_change_mode",
    "continuous_homogeneity_threshold",
    "continuous_homogeneity_margin",
    "continuous_subspace_margin",
    "continuous_log_p_margin",
    "continuous_context_min_margin",
    "continuous_context_pass",
    "structural_decision_zone",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "structural_decision_zone",
    "row_count",
    "accepted_split_count",
    "null_count",
    "signal_truth_aligned_count",
    "signal_truth_misaligned_count",
    "signal_truth_unlabeled_count",
    "case_replicate_count",
    "median_depth",
    "median_parent_size",
    "median_barycentric_balance",
    "median_homogeneity_gain_min",
    "median_continuous_threshold",
    "median_continuous_context_min_margin",
)


@dataclass(frozen=True)
class OverlapStructuralDecisionZoneConfig:
    """Runtime contract for overlap structural decision zones."""

    rows_path: Path
    output_dir: Path
    base_threshold: float = DEFAULT_BASE_THRESHOLD
    shallow_penalty: float = DEFAULT_SHALLOW_PENALTY
    parent_size_penalty: float = DEFAULT_PARENT_SIZE_PENALTY
    imbalance_penalty: float = DEFAULT_IMBALANCE_PENALTY
    subspace_threshold: float = DEFAULT_SUBSPACE_THRESHOLD
    sibling_p_threshold: float = DEFAULT_SIBLING_P_THRESHOLD
    depth_scale: float = DEFAULT_DEPTH_SCALE
    parent_reference_size: float = DEFAULT_PARENT_REFERENCE_SIZE
    balance_reference: float = DEFAULT_BALANCE_REFERENCE
    truth_alignment_threshold: float = DEFAULT_TRUTH_ALIGNMENT_THRESHOLD

    @property
    def rows_output_path(self) -> Path:
        return self.output_dir / "overlap_structural_decision_zone_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_structural_decision_zone_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


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
        "structural_sibling_status",
        "structural_change_mode",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Overlap structural rows are missing columns: {missing!r}")


def _truth_roles(
    rows: pd.DataFrame,
    *,
    truth_alignment_threshold: float,
) -> pd.Series:
    data_role = rows["data_role"].astype(str)
    truth_ari = pd.to_numeric(rows["truth_split_ari"], errors="coerce")
    roles = pd.Series("other", index=rows.index, dtype=object)
    roles.loc[data_role.isin({"null", "selected_null"})] = "null_like"
    signal = data_role.eq("signal")
    roles.loc[signal & truth_ari.ge(float(truth_alignment_threshold))] = (
        "signal_truth_aligned"
    )
    roles.loc[signal & truth_ari.notna() & truth_ari.lt(float(truth_alignment_threshold))] = (
        "signal_truth_misaligned"
    )
    roles.loc[signal & truth_ari.isna()] = "signal_truth_unlabeled"
    return roles


def _continuous_parameters(
    config: OverlapStructuralDecisionZoneConfig,
) -> ContinuousRuleParameters:
    return ContinuousRuleParameters(
        base_threshold=float(config.base_threshold),
        shallow_penalty=float(config.shallow_penalty),
        parent_size_penalty=float(config.parent_size_penalty),
        imbalance_penalty=float(config.imbalance_penalty),
        subspace_consensus_threshold=float(config.subspace_threshold),
        sibling_p_threshold=float(config.sibling_p_threshold),
        depth_scale=float(config.depth_scale),
        parent_reference_size=float(config.parent_reference_size),
        balance_reference=float(config.balance_reference),
    )


def continuous_context_margins(
    rows: pd.DataFrame,
    parameters: ContinuousRuleParameters,
) -> pd.DataFrame:
    """Return signed continuous margins for the structural traversal rule.

    The minimum margin is used only as a signed pass/fail diagnostic. Its
    components have different units, so downstream code should inspect the
    component margins when explaining why a row was blocked.
    """
    threshold = continuous_homogeneity_threshold(rows, parameters)
    sibling_p = pd.to_numeric(rows["sibling_p_value"], errors="coerce")
    homogeneity = pd.to_numeric(rows["homogeneity_gain_min"], errors="coerce")
    subspace = pd.to_numeric(
        rows["subspace_consensus_jaccard_topk"],
        errors="coerce",
    )
    clipped_p = sibling_p.clip(lower=1e-300)
    log_p_margin = pd.Series(
        float("nan"),
        index=rows.index,
        dtype=float,
    )
    valid_p = clipped_p.notna()
    log_p_margin.loc[valid_p] = (
        np.log(float(parameters.sibling_p_threshold))
        - np.log(clipped_p.loc[valid_p].astype(float))
    )
    margins = pd.DataFrame(
        {
            "continuous_homogeneity_margin": homogeneity - threshold,
            "continuous_subspace_margin": (
                subspace - float(parameters.subspace_consensus_threshold)
            ),
            "continuous_log_p_margin": log_p_margin,
        },
        index=rows.index,
    )
    margins["continuous_context_min_margin"] = margins.min(axis=1, skipna=False)
    return margins


def assign_structural_decision_zones(
    rows: pd.DataFrame,
    parameters: ContinuousRuleParameters,
    *,
    truth_alignment_threshold: float = DEFAULT_TRUTH_ALIGNMENT_THRESHOLD,
) -> pd.DataFrame:
    """Assign diagnostic structural traversal zones to overlap rows."""
    _validate_rows(rows)
    threshold = continuous_homogeneity_threshold(rows, parameters)
    accepted_split = rows["decision_class"].astype(str).eq("accepted_internal_split")
    sibling_p = pd.to_numeric(rows["sibling_p_value"], errors="coerce")
    homogeneity = pd.to_numeric(rows["homogeneity_gain_min"], errors="coerce")
    subspace = pd.to_numeric(
        rows["subspace_consensus_jaccard_topk"],
        errors="coerce",
    )
    margins = continuous_context_margins(rows, parameters)
    status = rows["structural_sibling_status"].astype(str)
    context_pass = (
        accepted_split
        & margins["continuous_homogeneity_margin"].ge(0.0)
        & margins["continuous_subspace_margin"].ge(0.0)
        & margins["continuous_log_p_margin"].ge(0.0)
    )
    zones = pd.Series("nonaccepted_or_leaf", index=rows.index, dtype=object)
    zones.loc[accepted_split] = "continuous_context_blocked"
    zones.loc[accepted_split & status.isin(WEAK_OR_UNSTABLE_STATUSES)] = (
        "unstable_weak_homogeneity_zone"
    )
    zones.loc[accepted_split & status.eq(STABLE_STATUS) & context_pass] = (
        "stable_structural_accept"
    )
    zones.loc[accepted_split & status.eq(STABLE_STATUS) & ~context_pass] = (
        "stable_structure_context_blocked"
    )
    truth_roles = _truth_roles(
        rows,
        truth_alignment_threshold=float(truth_alignment_threshold),
    )
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": rows["case_id"].astype(str),
            "data_role": rows["data_role"].astype(str),
            "structural_truth_role": truth_roles,
            "replicate": rows["replicate"].astype(int),
            "node_id": rows["node_id"].astype(str),
            "decision_class": rows["decision_class"].astype(str),
            "depth": pd.to_numeric(rows["depth"], errors="coerce"),
            "n_parent": pd.to_numeric(rows["n_parent"], errors="coerce"),
            "barycentric_balance": pd.to_numeric(
                rows["barycentric_balance"],
                errors="coerce",
            ),
            "sibling_p_value": sibling_p,
            "homogeneity_gain_min": homogeneity,
            "subspace_consensus_jaccard_topk": subspace,
            "truth_split_ari": pd.to_numeric(rows["truth_split_ari"], errors="coerce"),
            "structural_sibling_status": status,
            "structural_change_mode": rows["structural_change_mode"].astype(str),
            "continuous_homogeneity_threshold": threshold,
            "continuous_homogeneity_margin": margins["continuous_homogeneity_margin"],
            "continuous_subspace_margin": margins["continuous_subspace_margin"],
            "continuous_log_p_margin": margins["continuous_log_p_margin"],
            "continuous_context_min_margin": margins["continuous_context_min_margin"],
            "continuous_context_pass": context_pass.astype(bool),
            "structural_decision_zone": zones,
        },
        columns=ROW_COLUMNS,
    )


def summarize_structural_decision_zones(zones: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostic structural zones."""
    rows: list[dict[str, object]] = []
    case_rep = zones["case_id"].astype(str) + "::" + zones["replicate"].astype(str)
    for zone, group in zones.groupby("structural_decision_zone", sort=True):
        idx = group.index
        row = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "structural_decision_zone": str(zone),
            "row_count": int(group.shape[0]),
            "accepted_split_count": int(
                group["decision_class"].astype(str).eq("accepted_internal_split").sum()
            ),
            "null_count": int(group["structural_truth_role"].eq("null_like").sum()),
            "signal_truth_aligned_count": int(
                group["structural_truth_role"].eq("signal_truth_aligned").sum()
            ),
            "signal_truth_misaligned_count": int(
                group["structural_truth_role"].eq("signal_truth_misaligned").sum()
            ),
            "signal_truth_unlabeled_count": int(
                group["structural_truth_role"].eq("signal_truth_unlabeled").sum()
            ),
            "case_replicate_count": int(case_rep.loc[idx].nunique()),
            "median_depth": float(group["depth"].median()),
            "median_parent_size": float(group["n_parent"].median()),
            "median_barycentric_balance": float(group["barycentric_balance"].median()),
            "median_homogeneity_gain_min": float(group["homogeneity_gain_min"].median()),
            "median_continuous_threshold": float(
                group["continuous_homogeneity_threshold"].median()
            ),
            "median_continuous_context_min_margin": float(
                group["continuous_context_min_margin"].median()
            ),
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def run_overlap_structural_decision_zones(
    config: OverlapStructuralDecisionZoneConfig,
) -> dict[str, Path]:
    """Run structural decision-zone diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    parameters = _continuous_parameters(config)
    zones = assign_structural_decision_zones(
        rows,
        parameters,
        truth_alignment_threshold=float(config.truth_alignment_threshold),
    )
    summary = summarize_structural_decision_zones(zones)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    zones.to_csv(config.rows_output_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "parameters": {
            "base_threshold": float(config.base_threshold),
            "shallow_penalty": float(config.shallow_penalty),
            "parent_size_penalty": float(config.parent_size_penalty),
            "imbalance_penalty": float(config.imbalance_penalty),
            "subspace_threshold": float(config.subspace_threshold),
            "sibling_p_threshold": float(config.sibling_p_threshold),
            "depth_scale": float(config.depth_scale),
            "parent_reference_size": float(config.parent_reference_size),
            "balance_reference": float(config.balance_reference),
            "truth_alignment_threshold": float(config.truth_alignment_threshold),
        },
        "outputs": {
            "rows": str(config.rows_output_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_output_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-threshold", default=DEFAULT_BASE_THRESHOLD, type=float)
    parser.add_argument("--shallow-penalty", default=DEFAULT_SHALLOW_PENALTY, type=float)
    parser.add_argument(
        "--parent-size-penalty",
        default=DEFAULT_PARENT_SIZE_PENALTY,
        type=float,
    )
    parser.add_argument(
        "--imbalance-penalty",
        default=DEFAULT_IMBALANCE_PENALTY,
        type=float,
    )
    parser.add_argument(
        "--subspace-threshold",
        default=DEFAULT_SUBSPACE_THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--sibling-p-threshold",
        default=DEFAULT_SIBLING_P_THRESHOLD,
        type=float,
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
    parser.add_argument(
        "--truth-alignment-threshold",
        default=DEFAULT_TRUTH_ALIGNMENT_THRESHOLD,
        type=float,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_structural_decision_zones(
        OverlapStructuralDecisionZoneConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            base_threshold=float(args.base_threshold),
            shallow_penalty=float(args.shallow_penalty),
            parent_size_penalty=float(args.parent_size_penalty),
            imbalance_penalty=float(args.imbalance_penalty),
            subspace_threshold=float(args.subspace_threshold),
            sibling_p_threshold=float(args.sibling_p_threshold),
            depth_scale=float(args.depth_scale),
            parent_reference_size=float(args.parent_reference_size),
            balance_reference=float(args.balance_reference),
            truth_alignment_threshold=float(args.truth_alignment_threshold),
        )
    )


if __name__ == "__main__":
    main()
