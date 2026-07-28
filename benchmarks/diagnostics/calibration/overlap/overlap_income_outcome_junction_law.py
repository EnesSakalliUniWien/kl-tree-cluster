"""Income/outcome-aware junction diagnostics for overlap traversal rows.

This diagnostic is the next layer after the internal-node transfer-gap audit.
It does not treat a node only by degree. It separates the role of the incoming
selected parent relation from the outgoing child-sibling decomposition relation,
then classifies whether the unresolved overlap row requires a transition law
between those two roles.

The current overlap artifacts contain parent IDs and parent-level outgoing
evidence, but not branch-specific incoming child-edge coordinates. This runner
therefore builds a first income/outcome proxy: a node's incoming evidence is
the selected context of its parent junction, and its outgoing evidence is the
node's own internal-node likelihood row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_income_outcome_junction_law"
SCHEMA_VERSION = "overlap_income_outcome_junction_law/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_income_outcome_junction_law"

GAP_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "truth_geometry_mode",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "context_margin_pass",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "conditional_bayesian_gap_status",
}

STRUCTURAL_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "parent_id",
    "depth",
    "n_parent",
    "n_left",
    "n_right",
    "barycentric_balance",
    "sibling_p_value",
    "homogeneity_gain_min",
    "subspace_consensus_jaccard_topk",
    "structural_sibling_status",
    "structural_change_mode",
}

DECISION_ZONE_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "continuous_context_min_margin",
    "structural_decision_zone",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "parent_id",
    "guard_truth_role",
    "truth_geometry_mode",
    "incidence_signature",
    "incoming_relation_available",
    "incoming_parent_depth",
    "incoming_parent_structural_decision_zone",
    "incoming_parent_context_margin",
    "incoming_parent_context_pass",
    "incoming_parent_homogeneity_gain_min",
    "incoming_parent_subspace_consensus_jaccard_topk",
    "outgoing_depth",
    "outgoing_structural_decision_zone",
    "outgoing_context_margin",
    "outgoing_context_pass",
    "outgoing_soft_structure_pass",
    "outgoing_default_candidate",
    "outgoing_homogeneity_gain_min",
    "outgoing_subspace_consensus_jaccard_topk",
    "outgoing_size_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "context_transition_delta",
    "subspace_transition_delta",
    "homogeneity_transition_delta",
    "income_outcome_transition_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "truth_recovery_total",
    "negative_total",
    "income_outcome_junction_count",
    "root_or_unobserved_income_count",
    "incoming_context_pass_count",
    "truth_outcome_supported_count",
    "truth_transition_required_count",
    "negative_default_candidate_count",
    "negative_outgoing_context_blocked_count",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapIncomeOutcomeJunctionLawConfig:
    """Runtime contract for income/outcome-aware junction diagnostics."""

    structural_rows_path: Path
    decision_zone_rows_path: Path
    transfer_gap_rows_path: Path
    output_dir: Path

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_income_outcome_junction_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_income_outcome_junction_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_columns(rows: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{label} rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _bool(rows: pd.DataFrame, column: str) -> pd.Series:
    return rows[column].astype(bool)


def _finite_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    return right_numeric - left_numeric


def _incidence_signature(parent_available: bool) -> str:
    if parent_available:
        return "income1_outcome2_internal_junction"
    return "income0_outcome2_root_or_unobserved_junction"


def _transition_status(
    *,
    truth_recovery: bool,
    incoming_available: bool,
    outgoing_context_pass: bool,
    outgoing_soft_structure_pass: bool,
    outgoing_default_candidate: bool,
    gap_status: str,
) -> str:
    if not incoming_available:
        if truth_recovery and outgoing_default_candidate:
            return "truth_recovery_root_or_unobserved_income_outcome_supported"
        return "root_or_unobserved_income_outcome_only"
    if truth_recovery:
        if outgoing_default_candidate:
            return "truth_recovery_outcome_supported_under_income_context"
        if (
            gap_status == "context_negative_but_soft_structure_supported"
            and outgoing_soft_structure_pass
            and not outgoing_context_pass
        ):
            return "truth_recovery_income_outcome_context_transition_required"
        if not outgoing_soft_structure_pass:
            return "truth_recovery_outcome_soft_structure_blocked"
        if not outgoing_context_pass:
            return "truth_recovery_outgoing_context_blocked"
        return "truth_recovery_unresolved_income_outcome"
    if outgoing_default_candidate:
        return "negative_income_outcome_leakage"
    if not outgoing_context_pass:
        return "negative_blocked_by_outgoing_context"
    if not outgoing_soft_structure_pass:
        return "negative_blocked_by_outgoing_soft_structure"
    return "negative_blocked_by_other"


def _prepare_node_structural_rows(structural_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "data_role", "replicate", "node_id"]
    return structural_rows[
        keys
        + [
            "parent_id",
            "depth",
            "n_parent",
            "n_left",
            "n_right",
            "barycentric_balance",
            "sibling_p_value",
            "homogeneity_gain_min",
            "subspace_consensus_jaccard_topk",
            "structural_sibling_status",
            "structural_change_mode",
        ]
    ].rename(
        columns={
            "depth": "outgoing_depth",
            "homogeneity_gain_min": "outgoing_homogeneity_gain_min",
            "subspace_consensus_jaccard_topk": ("outgoing_subspace_consensus_jaccard_topk"),
        }
    )


def _prepare_parent_structural_rows(structural_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "data_role", "replicate"]
    return structural_rows[
        keys
        + [
            "node_id",
            "depth",
            "homogeneity_gain_min",
            "subspace_consensus_jaccard_topk",
            "structural_sibling_status",
            "structural_change_mode",
        ]
    ].rename(
        columns={
            "node_id": "incoming_parent_id",
            "depth": "incoming_parent_depth",
            "homogeneity_gain_min": "incoming_parent_homogeneity_gain_min",
            "subspace_consensus_jaccard_topk": ("incoming_parent_subspace_consensus_jaccard_topk"),
            "structural_sibling_status": "incoming_parent_structural_sibling_status",
            "structural_change_mode": "incoming_parent_structural_change_mode",
        }
    )


def _prepare_decision_zone_rows(
    decision_zone_rows: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    keys = ["case_id", "data_role", "replicate", "node_id"]
    return decision_zone_rows[
        keys + ["continuous_context_min_margin", "structural_decision_zone"]
    ].rename(
        columns={
            "continuous_context_min_margin": f"{prefix}_context_margin",
            "structural_decision_zone": f"{prefix}_structural_decision_zone",
        }
    )


def build_income_outcome_junction_rows(
    *,
    structural_rows: pd.DataFrame,
    decision_zone_rows: pd.DataFrame,
    transfer_gap_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build income/outcome-aware rows for current overlap residual nodes."""
    _validate_columns(transfer_gap_rows, GAP_REQUIRED_COLUMNS, "Transfer-gap")
    _validate_columns(structural_rows, STRUCTURAL_REQUIRED_COLUMNS, "Structural")
    _validate_columns(decision_zone_rows, DECISION_ZONE_REQUIRED_COLUMNS, "Decision-zone")
    keys = ["case_id", "data_role", "replicate", "node_id"]
    gap = transfer_gap_rows.copy()
    node_structural = _prepare_node_structural_rows(structural_rows)
    outgoing_zone = _prepare_decision_zone_rows(
        decision_zone_rows,
        prefix="outgoing",
    )
    parent_structural = _prepare_parent_structural_rows(structural_rows)
    parent_zone = _prepare_decision_zone_rows(
        decision_zone_rows,
        prefix="incoming_parent",
    ).rename(columns={"node_id": "incoming_parent_id"})

    joined = gap.merge(node_structural, on=keys, how="left", validate="one_to_one")
    joined = joined.merge(outgoing_zone, on=keys, how="left", validate="one_to_one")
    parent_keys = ["case_id", "data_role", "replicate"]
    joined = joined.merge(
        parent_structural,
        left_on=parent_keys + ["parent_id"],
        right_on=parent_keys + ["incoming_parent_id"],
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        parent_zone,
        on=parent_keys + ["incoming_parent_id"],
        how="left",
        validate="many_to_one",
    )

    incoming_available = joined["incoming_parent_id"].notna()
    incoming_context = _numeric(joined, "incoming_parent_context_margin")
    outgoing_context = _numeric(joined, "continuous_context_min_margin")
    outgoing_context_pass = _bool(joined, "context_margin_pass")
    outgoing_soft_pass = _bool(joined, "soft_structure_pass")
    outgoing_candidate = _bool(joined, "default_internal_node_candidate")
    truth_recovery = joined["guard_truth_role"].astype(str).eq("truth_recovery")
    incoming_context_pass = incoming_available & incoming_context.ge(0.0)

    records: list[dict[str, object]] = []
    for idx, row in joined.iterrows():
        parent_available = bool(incoming_available.loc[idx])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "parent_id": (str(row["parent_id"]) if pd.notna(row["parent_id"]) else ""),
                "guard_truth_role": str(row["guard_truth_role"]),
                "truth_geometry_mode": str(row["truth_geometry_mode"]),
                "incidence_signature": _incidence_signature(parent_available),
                "incoming_relation_available": parent_available,
                "incoming_parent_depth": (
                    float(row["incoming_parent_depth"])
                    if pd.notna(row["incoming_parent_depth"])
                    else np.nan
                ),
                "incoming_parent_structural_decision_zone": (
                    str(row["incoming_parent_structural_decision_zone"])
                    if pd.notna(row["incoming_parent_structural_decision_zone"])
                    else "missing"
                ),
                "incoming_parent_context_margin": float(incoming_context.loc[idx]),
                "incoming_parent_context_pass": bool(incoming_context_pass.loc[idx]),
                "incoming_parent_homogeneity_gain_min": (
                    float(row["incoming_parent_homogeneity_gain_min"])
                    if pd.notna(row["incoming_parent_homogeneity_gain_min"])
                    else np.nan
                ),
                "incoming_parent_subspace_consensus_jaccard_topk": (
                    float(row["incoming_parent_subspace_consensus_jaccard_topk"])
                    if pd.notna(row["incoming_parent_subspace_consensus_jaccard_topk"])
                    else np.nan
                ),
                "outgoing_depth": (
                    float(row["outgoing_depth"]) if pd.notna(row["outgoing_depth"]) else np.nan
                ),
                "outgoing_structural_decision_zone": (
                    str(row["outgoing_structural_decision_zone"])
                    if pd.notna(row["outgoing_structural_decision_zone"])
                    else "missing"
                ),
                "outgoing_context_margin": float(outgoing_context.loc[idx]),
                "outgoing_context_pass": bool(outgoing_context_pass.loc[idx]),
                "outgoing_soft_structure_pass": bool(outgoing_soft_pass.loc[idx]),
                "outgoing_default_candidate": bool(outgoing_candidate.loc[idx]),
                "outgoing_homogeneity_gain_min": (
                    float(row["outgoing_homogeneity_gain_min"])
                    if pd.notna(row["outgoing_homogeneity_gain_min"])
                    else np.nan
                ),
                "outgoing_subspace_consensus_jaccard_topk": float(
                    row["subspace_consensus_jaccard_topk"]
                ),
                "outgoing_size_balance": float(row["size_balance"]),
                "outgoing_edge_norm_balance": float(row["edge_norm_balance"]),
                "outgoing_fragment_risk_proxy_score": float(row["fragment_risk_proxy_score"]),
                "context_transition_delta": float(
                    outgoing_context.loc[idx] - incoming_context.loc[idx]
                ),
                "subspace_transition_delta": float(
                    row["subspace_consensus_jaccard_topk"]
                    - row["incoming_parent_subspace_consensus_jaccard_topk"]
                )
                if pd.notna(row["incoming_parent_subspace_consensus_jaccard_topk"])
                else np.nan,
                "homogeneity_transition_delta": float(
                    row["outgoing_homogeneity_gain_min"]
                    - row["incoming_parent_homogeneity_gain_min"]
                )
                if pd.notna(row["incoming_parent_homogeneity_gain_min"])
                and pd.notna(row["outgoing_homogeneity_gain_min"])
                else np.nan,
                "income_outcome_transition_status": _transition_status(
                    truth_recovery=bool(truth_recovery.loc[idx]),
                    incoming_available=parent_available,
                    outgoing_context_pass=bool(outgoing_context_pass.loc[idx]),
                    outgoing_soft_structure_pass=bool(outgoing_soft_pass.loc[idx]),
                    outgoing_default_candidate=bool(outgoing_candidate.loc[idx]),
                    gap_status=str(row["conditional_bayesian_gap_status"]),
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_income_outcome_junction_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize income/outcome transition-law requirements."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    truth = rows["guard_truth_role"].astype(str).eq("truth_recovery")
    negative = ~truth
    status = rows["income_outcome_transition_status"].astype(str)
    incoming_available = rows["incoming_relation_available"].astype(bool)
    incoming_context_pass = rows["incoming_parent_context_pass"].astype(bool)
    outgoing_candidate = rows["outgoing_default_candidate"].astype(bool)
    outgoing_context_pass = rows["outgoing_context_pass"].astype(bool)
    transition_required = status.eq("truth_recovery_income_outcome_context_transition_required")
    negative_leakage = negative & outgoing_candidate
    if negative_leakage.any():
        diagnostic_status = "income_outcome_negative_leakage_detected"
    elif transition_required.any():
        diagnostic_status = "income_outcome_transition_law_required"
    elif (truth & outgoing_candidate).all():
        diagnostic_status = "income_outcome_local_outcome_sufficient"
    else:
        diagnostic_status = "income_outcome_unresolved"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "truth_recovery_total": int(truth.sum()),
                "negative_total": int(negative.sum()),
                "income_outcome_junction_count": int(incoming_available.sum()),
                "root_or_unobserved_income_count": int((~incoming_available).sum()),
                "incoming_context_pass_count": int(incoming_context_pass.sum()),
                "truth_outcome_supported_count": int((truth & outgoing_candidate).sum()),
                "truth_transition_required_count": int((truth & transition_required).sum()),
                "negative_default_candidate_count": int(negative_leakage.sum()),
                "negative_outgoing_context_blocked_count": int(
                    (negative & ~outgoing_context_pass).sum()
                ),
                "diagnostic_status": diagnostic_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_income_outcome_junction_law(
    config: OverlapIncomeOutcomeJunctionLawConfig,
) -> dict[str, Path]:
    """Run income/outcome-aware junction diagnostics and write outputs."""
    structural_rows = pd.read_csv(config.structural_rows_path)
    decision_zone_rows = pd.read_csv(config.decision_zone_rows_path)
    transfer_gap_rows = pd.read_csv(config.transfer_gap_rows_path)
    rows = build_income_outcome_junction_rows(
        structural_rows=structural_rows,
        decision_zone_rows=decision_zone_rows,
        transfer_gap_rows=transfer_gap_rows,
    )
    summary = summarize_income_outcome_junction_rows(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "structural_rows_path": str(config.structural_rows_path),
        "decision_zone_rows_path": str(config.decision_zone_rows_path),
        "transfer_gap_rows_path": str(config.transfer_gap_rows_path),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_income_outcome_transition_law",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-rows-path", required=True, type=Path)
    parser.add_argument("--decision-zone-rows-path", required=True, type=Path)
    parser.add_argument("--transfer-gap-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_income_outcome_junction_law(
        OverlapIncomeOutcomeJunctionLawConfig(
            structural_rows_path=args.structural_rows_path,
            decision_zone_rows_path=args.decision_zone_rows_path,
            transfer_gap_rows_path=args.transfer_gap_rows_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
