"""Overlap-aware internal-node likelihood probe for residual traversal rows.

This diagnostic tests whether row-level internal-node neighborhood evidence can
recover overlap structure that family-level aggregates mark as unstable. It is
diagnostic-only: thresholds are transparent probes, not calibrated production
constants.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.overlap_conditional_bayesian_traversal_law import (
    p_value_log_bayes_factor_lower_bound,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_internal_node_bayesian_likelihood_probe"
SCHEMA_VERSION = "overlap_internal_node_bayesian_likelihood_probe/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap_internal_node_bayesian_likelihood_probe"
)

DEFAULT_ACTION = "weak_unstable_multiscale_zone"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "truth_geometry_mode",
    "residual_action",
    "sibling_p_value",
    "selected_family_log_bayes_factor_lower",
    "homogeneity_gain_min",
    "continuous_context_min_margin",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "context_margin_pass",
    "soft_subspace_pass",
    "size_balance_pass",
    "edge_norm_balance_pass",
    "fragment_risk_pass",
    "row_overlap_neighborhood_log_bayes_factor",
    "row_overlap_coherent_candidate",
    "row_overlap_likelihood_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "guard_truth_role",
    "row_count",
    "candidate_count",
    "candidate_rate",
    "median_row_overlap_neighborhood_log_bayes_factor",
    "context_blocked_count",
    "subspace_blocked_count",
    "balance_or_fragment_blocked_count",
    "diagnostic_status",
)

FAMILY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "family_row_count",
    "candidate_row_count",
    "truth_recovery_candidate_row_count",
    "nonrecovery_or_null_candidate_row_count",
    "family_overlap_likelihood_status",
)


@dataclass(frozen=True)
class OverlapInternalNodeLikelihoodParameters:
    """Transparent row-level thresholds for overlap internal-node probes."""

    p_extreme_log_bayes_factor_threshold: float = 3.0
    context_margin_floor: float = 0.0
    soft_subspace_floor: float = 0.15
    size_balance_floor: float = 0.33
    edge_norm_balance_floor: float = 0.49
    fragment_risk_ceiling: float = 1.25
    homogeneity_weight: float = 80.0
    context_margin_weight: float = 80.0
    subspace_weight: float = 1.5
    size_balance_weight: float = 2.0
    edge_norm_balance_weight: float = 1.0
    fragment_risk_weight: float = 1.5


@dataclass(frozen=True)
class OverlapInternalNodeBayesianLikelihoodProbeConfig:
    """Runtime contract for the row-level overlap likelihood probe."""

    policy_rows_path: Path
    decision_zone_rows_path: Path
    fragment_guard_rows_path: Path
    output_dir: Path
    action: str = DEFAULT_ACTION
    parameters: OverlapInternalNodeLikelihoodParameters = (
        OverlapInternalNodeLikelihoodParameters()
    )

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_summary.csv"

    @property
    def family_summary_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_likelihood_families.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_inputs(
    policy_rows: pd.DataFrame,
    decision_zone_rows: pd.DataFrame,
    fragment_guard_rows: pd.DataFrame,
) -> None:
    missing_policy = sorted(
        {
            "case_id",
            "data_role",
            "replicate",
            "node_id",
            "diagnostic_traversal_action",
            "guard_truth_role",
            "truth_geometry_mode",
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
            "continuous_context_min_margin",
            "subspace_consensus_jaccard_topk",
        }
        - set(decision_zone_rows.columns)
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
            "size_balance",
            "edge_norm_balance",
        }
        - set(fragment_guard_rows.columns)
    )
    if missing_guard:
        raise ValueError(f"Fragment-guard rows are missing columns: {missing_guard!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def build_internal_node_likelihood_rows(
    policy_rows: pd.DataFrame,
    decision_zone_rows: pd.DataFrame,
    fragment_guard_rows: pd.DataFrame,
    *,
    action: str = DEFAULT_ACTION,
    parameters: OverlapInternalNodeLikelihoodParameters = (
        OverlapInternalNodeLikelihoodParameters()
    ),
) -> pd.DataFrame:
    """Build row-level overlap-aware likelihood probe rows."""
    _validate_inputs(policy_rows, decision_zone_rows, fragment_guard_rows)
    keys = ["case_id", "data_role", "replicate", "node_id"]
    residual = policy_rows[
        policy_rows["diagnostic_traversal_action"].astype(str).eq(str(action))
    ][keys + ["guard_truth_role", "truth_geometry_mode"]].copy()
    zone = decision_zone_rows[
        keys
        + [
            "sibling_p_value",
            "homogeneity_gain_min",
            "continuous_context_min_margin",
            "subspace_consensus_jaccard_topk",
        ]
    ].copy()
    guard = fragment_guard_rows[
        keys + ["fragment_risk_proxy_score", "size_balance", "edge_norm_balance"]
    ].copy()
    rows = residual.merge(zone, on=keys, how="left", validate="one_to_one")
    rows = rows.merge(guard, on=keys, how="left", validate="one_to_one")
    sibling_p = _numeric(rows, "sibling_p_value")
    selected_bf = p_value_log_bayes_factor_lower_bound(sibling_p)
    homogeneity = _numeric(rows, "homogeneity_gain_min")
    context = _numeric(rows, "continuous_context_min_margin")
    subspace = _numeric(rows, "subspace_consensus_jaccard_topk")
    size_balance = _numeric(rows, "size_balance")
    edge_balance = _numeric(rows, "edge_norm_balance")
    fragment_risk = _numeric(rows, "fragment_risk_proxy_score")

    context_pass = context.ge(float(parameters.context_margin_floor))
    subspace_pass = subspace.ge(float(parameters.soft_subspace_floor))
    size_pass = size_balance.ge(float(parameters.size_balance_floor))
    edge_pass = edge_balance.ge(float(parameters.edge_norm_balance_floor))
    fragment_pass = fragment_risk.le(float(parameters.fragment_risk_ceiling))
    p_pass = selected_bf.ge(float(parameters.p_extreme_log_bayes_factor_threshold))
    likelihood = (
        float(parameters.homogeneity_weight) * homogeneity.fillna(0.0)
        + float(parameters.context_margin_weight) * context.fillna(0.0)
        + float(parameters.subspace_weight)
        * (subspace.fillna(0.0) - float(parameters.soft_subspace_floor))
        + float(parameters.size_balance_weight)
        * (size_balance.fillna(0.0) - float(parameters.size_balance_floor))
        + float(parameters.edge_norm_balance_weight)
        * (edge_balance.fillna(0.0) - float(parameters.edge_norm_balance_floor))
        + float(parameters.fragment_risk_weight)
        * (float(parameters.fragment_risk_ceiling) - fragment_risk.fillna(0.0))
    )
    candidate = p_pass & context_pass & subspace_pass & size_pass & edge_pass & fragment_pass
    status = pd.Series("row_overlap_candidate", index=rows.index, dtype=object)
    status.loc[~candidate] = "row_overlap_blocked"
    status.loc[p_pass & ~context_pass] = "row_overlap_context_blocked"
    status.loc[p_pass & context_pass & ~subspace_pass] = "row_overlap_subspace_blocked"
    status.loc[
        p_pass
        & context_pass
        & subspace_pass
        & (~size_pass | ~edge_pass | ~fragment_pass)
    ] = "row_overlap_balance_or_fragment_blocked"
    output = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": rows["case_id"].astype(str),
            "data_role": rows["data_role"].astype(str),
            "replicate": rows["replicate"].astype(int),
            "node_id": rows["node_id"].astype(str),
            "guard_truth_role": rows["guard_truth_role"].astype(str),
            "truth_geometry_mode": rows["truth_geometry_mode"].astype(str),
            "residual_action": str(action),
            "sibling_p_value": sibling_p,
            "selected_family_log_bayes_factor_lower": selected_bf,
            "homogeneity_gain_min": homogeneity,
            "continuous_context_min_margin": context,
            "subspace_consensus_jaccard_topk": subspace,
            "size_balance": size_balance,
            "edge_norm_balance": edge_balance,
            "fragment_risk_proxy_score": fragment_risk,
            "context_margin_pass": context_pass,
            "soft_subspace_pass": subspace_pass,
            "size_balance_pass": size_pass,
            "edge_norm_balance_pass": edge_pass,
            "fragment_risk_pass": fragment_pass,
            "row_overlap_neighborhood_log_bayes_factor": likelihood,
            "row_overlap_coherent_candidate": candidate,
            "row_overlap_likelihood_status": status,
        },
        columns=ROW_COLUMNS,
    )
    return output


def summarize_internal_node_likelihood_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-level likelihood probe results by evaluation role."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for role, group in rows.groupby("guard_truth_role", sort=True):
        status = group["row_overlap_likelihood_status"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "guard_truth_role": str(role),
                "row_count": int(group.shape[0]),
                "candidate_count": int(group["row_overlap_coherent_candidate"].sum()),
                "candidate_rate": float(group["row_overlap_coherent_candidate"].mean()),
                "median_row_overlap_neighborhood_log_bayes_factor": float(
                    group["row_overlap_neighborhood_log_bayes_factor"].median()
                ),
                "context_blocked_count": int(
                    status.eq("row_overlap_context_blocked").sum()
                ),
                "subspace_blocked_count": int(
                    status.eq("row_overlap_subspace_blocked").sum()
                ),
                "balance_or_fragment_blocked_count": int(
                    status.eq("row_overlap_balance_or_fragment_blocked").sum()
                ),
                "diagnostic_status": (
                    "diagnostic_only_internal_node_likelihood_probe_not_calibration"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def summarize_internal_node_likelihood_families(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-level candidates by selected residual family."""
    if rows.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    records: list[dict[str, object]] = []
    group_cols = ["case_id", "data_role", "replicate"]
    for (case_id, data_role, replicate), group in rows.groupby(group_cols, sort=True):
        candidate = group["row_overlap_coherent_candidate"].astype(bool)
        candidate_rows = group[candidate]
        recovery_candidates = candidate_rows["guard_truth_role"].astype(str).eq(
            "truth_recovery"
        )
        nonrecovery_candidates = candidate_rows["guard_truth_role"].astype(str).ne(
            "truth_recovery"
        )
        if candidate_rows.empty:
            status = "family_no_internal_node_candidate"
        elif recovery_candidates.any() and not nonrecovery_candidates.any():
            status = "family_truth_recovery_internal_node_candidate"
        elif recovery_candidates.any() and nonrecovery_candidates.any():
            status = "family_mixed_internal_node_candidates"
        else:
            status = "family_nonrecovery_or_null_internal_node_candidate"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "family_row_count": int(group.shape[0]),
                "candidate_row_count": int(candidate.sum()),
                "truth_recovery_candidate_row_count": int(recovery_candidates.sum()),
                "nonrecovery_or_null_candidate_row_count": int(
                    nonrecovery_candidates.sum()
                ),
                "family_overlap_likelihood_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_COLUMNS)


def run_overlap_internal_node_bayesian_likelihood_probe(
    config: OverlapInternalNodeBayesianLikelihoodProbeConfig,
) -> dict[str, Path]:
    """Run the internal-node likelihood probe and write outputs."""
    policy_rows = pd.read_csv(config.policy_rows_path)
    decision_zone_rows = pd.read_csv(config.decision_zone_rows_path)
    fragment_guard_rows = pd.read_csv(config.fragment_guard_rows_path)
    rows = build_internal_node_likelihood_rows(
        policy_rows,
        decision_zone_rows,
        fragment_guard_rows,
        action=config.action,
        parameters=config.parameters,
    )
    summary = summarize_internal_node_likelihood_rows(rows)
    family_summary = summarize_internal_node_likelihood_families(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    family_summary.to_csv(config.family_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "policy_rows_path": str(config.policy_rows_path),
        "decision_zone_rows_path": str(config.decision_zone_rows_path),
        "fragment_guard_rows_path": str(config.fragment_guard_rows_path),
        "action": str(config.action),
        "parameters": config.parameters.__dict__,
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "family_summary": str(config.family_summary_path),
        },
        "production_status": "diagnostic_only_no_production_promotion",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "family_summary": config.family_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-rows-path", required=True, type=Path)
    parser.add_argument("--decision-zone-rows-path", required=True, type=Path)
    parser.add_argument("--fragment-guard-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default=DEFAULT_ACTION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_internal_node_bayesian_likelihood_probe(
        OverlapInternalNodeBayesianLikelihoodProbeConfig(
            policy_rows_path=args.policy_rows_path,
            decision_zone_rows_path=args.decision_zone_rows_path,
            fragment_guard_rows_path=args.fragment_guard_rows_path,
            output_dir=args.output_dir,
            action=str(args.action),
        )
    )


if __name__ == "__main__":
    main()
