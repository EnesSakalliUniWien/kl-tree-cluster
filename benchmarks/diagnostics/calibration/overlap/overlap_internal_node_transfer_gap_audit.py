"""Audit unresolved transfer gaps in overlap internal-node likelihood rows.

This diagnostic separates threshold tuning from the remaining conditional-law
problem. In particular, it marks truth-recovery rows that pass the soft
internal-node structural evidence but fail the nonnegative local context gate.
Those rows should not be recovered by relaxing the local context threshold if
relaxed-context transfer leaks; they require a higher-order conditional or
Bayesian traversal law.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_internal_node_transfer_gap_audit"
SCHEMA_VERSION = "overlap_internal_node_transfer_gap_audit/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_internal_node_transfer_gap_audit"

DEFAULT_CONTEXT_MARGIN_FLOOR = 0.0
DEFAULT_SUBSPACE_FLOOR = 0.15
DEFAULT_SIZE_BALANCE_FLOOR = 0.33
DEFAULT_EDGE_NORM_BALANCE_FLOOR = 0.49
DEFAULT_FRAGMENT_RISK_CEILING = 1.25
DEFAULT_BAYES_FACTOR_FLOOR = 3.0

REQUIRED_LIKELIHOOD_COLUMNS = {
    "case_id",
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
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
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
    "bayes_evidence_pass",
    "context_margin_pass",
    "soft_subspace_pass",
    "size_balance_pass",
    "edge_norm_balance_pass",
    "fragment_risk_pass",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "blocking_components",
    "conditional_bayesian_gap_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "truth_recovery_total",
    "truth_recovered_by_default_count",
    "truth_context_negative_soft_supported_count",
    "truth_soft_structure_blocked_count",
    "truth_bayes_evidence_blocked_count",
    "negative_total",
    "negative_default_candidate_count",
    "relaxed_context_leakage_rule_count",
    "nonnegative_context_leakage_rule_count",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapInternalNodeTransferGapAuditConfig:
    """Runtime contract for the internal-node transfer-gap audit."""

    likelihood_rows_path: Path
    output_dir: Path
    transfer_rows_path: Path | None = None
    bayes_factor_floor: float = DEFAULT_BAYES_FACTOR_FLOOR
    context_margin_floor: float = DEFAULT_CONTEXT_MARGIN_FLOOR
    subspace_floor: float = DEFAULT_SUBSPACE_FLOOR
    size_balance_floor: float = DEFAULT_SIZE_BALANCE_FLOOR
    edge_norm_balance_floor: float = DEFAULT_EDGE_NORM_BALANCE_FLOOR
    fragment_risk_ceiling: float = DEFAULT_FRAGMENT_RISK_CEILING

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_transfer_gap_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_internal_node_transfer_gap_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _validate_likelihood_rows(rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_LIKELIHOOD_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"Likelihood rows are missing columns: {missing!r}")


def _blocking_components(
    *,
    bayes_pass: bool,
    context_pass: bool,
    subspace_pass: bool,
    size_pass: bool,
    edge_pass: bool,
    fragment_pass: bool,
) -> str:
    blocks: list[str] = []
    if not bayes_pass:
        blocks.append("selected_family_bayes_evidence")
    if not context_pass:
        blocks.append("local_context_margin")
    if not subspace_pass:
        blocks.append("subspace_consensus")
    if not size_pass:
        blocks.append("size_balance")
    if not edge_pass:
        blocks.append("edge_norm_balance")
    if not fragment_pass:
        blocks.append("fragment_risk")
    return "|".join(blocks) if blocks else "none"


def _gap_status(
    *,
    truth_recovery: bool,
    candidate: bool,
    bayes_pass: bool,
    context_pass: bool,
    soft_structure_pass: bool,
) -> str:
    if truth_recovery:
        if candidate:
            return "recovered_by_default_internal_node_likelihood"
        if not bayes_pass:
            return "selected_family_bayes_evidence_blocked"
        if not context_pass and soft_structure_pass:
            return "context_negative_but_soft_structure_supported"
        if not context_pass:
            return "context_negative_and_soft_structure_blocked"
        if not soft_structure_pass:
            return "soft_structure_blocked"
        return "transfer_unrecovered_by_nonnegative_context_rules"
    if candidate:
        return "negative_default_candidate_leakage"
    if not bayes_pass:
        return "negative_blocked_by_bayes_evidence"
    if not context_pass:
        return "negative_blocked_by_context"
    if not soft_structure_pass:
        return "negative_blocked_by_soft_structure"
    return "negative_blocked_by_other"


def build_internal_node_transfer_gap_rows(
    likelihood_rows: pd.DataFrame,
    *,
    bayes_factor_floor: float = DEFAULT_BAYES_FACTOR_FLOOR,
    context_margin_floor: float = DEFAULT_CONTEXT_MARGIN_FLOOR,
    subspace_floor: float = DEFAULT_SUBSPACE_FLOOR,
    size_balance_floor: float = DEFAULT_SIZE_BALANCE_FLOOR,
    edge_norm_balance_floor: float = DEFAULT_EDGE_NORM_BALANCE_FLOOR,
    fragment_risk_ceiling: float = DEFAULT_FRAGMENT_RISK_CEILING,
) -> pd.DataFrame:
    """Classify row-level likelihood misses by conditional/Bayesian component."""
    _validate_likelihood_rows(likelihood_rows)
    rows = likelihood_rows.copy()
    if "data_role" not in rows.columns:
        rows["data_role"] = "unknown"
    bayes = _numeric(rows, "selected_family_log_bayes_factor_lower")
    context = _numeric(rows, "continuous_context_min_margin")
    subspace = _numeric(rows, "subspace_consensus_jaccard_topk")
    size = _numeric(rows, "size_balance")
    edge = _numeric(rows, "edge_norm_balance")
    fragment = _numeric(rows, "fragment_risk_proxy_score")

    bayes_pass = bayes.ge(float(bayes_factor_floor))
    context_pass = context.ge(float(context_margin_floor))
    subspace_pass = subspace.ge(float(subspace_floor))
    size_pass = size.ge(float(size_balance_floor))
    edge_pass = edge.ge(float(edge_norm_balance_floor))
    fragment_pass = fragment.le(float(fragment_risk_ceiling))
    soft_structure_pass = subspace_pass & size_pass & edge_pass & fragment_pass
    candidate = bayes_pass & context_pass & soft_structure_pass
    truth_recovery = rows["guard_truth_role"].astype(str).eq("truth_recovery")

    records: list[dict[str, object]] = []
    for idx, row in rows.iterrows():
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": str(row["case_id"]),
            "data_role": str(row["data_role"]),
            "replicate": int(row["replicate"]),
            "node_id": str(row["node_id"]),
            "guard_truth_role": str(row["guard_truth_role"]),
            "truth_geometry_mode": str(row["truth_geometry_mode"]),
            "selected_family_log_bayes_factor_lower": float(bayes.loc[idx]),
            "continuous_context_min_margin": float(context.loc[idx]),
            "subspace_consensus_jaccard_topk": float(subspace.loc[idx]),
            "size_balance": float(size.loc[idx]),
            "edge_norm_balance": float(edge.loc[idx]),
            "fragment_risk_proxy_score": float(fragment.loc[idx]),
            "bayes_evidence_pass": bool(bayes_pass.loc[idx]),
            "context_margin_pass": bool(context_pass.loc[idx]),
            "soft_subspace_pass": bool(subspace_pass.loc[idx]),
            "size_balance_pass": bool(size_pass.loc[idx]),
            "edge_norm_balance_pass": bool(edge_pass.loc[idx]),
            "fragment_risk_pass": bool(fragment_pass.loc[idx]),
            "soft_structure_pass": bool(soft_structure_pass.loc[idx]),
            "default_internal_node_candidate": bool(candidate.loc[idx]),
            "blocking_components": _blocking_components(
                bayes_pass=bool(bayes_pass.loc[idx]),
                context_pass=bool(context_pass.loc[idx]),
                subspace_pass=bool(subspace_pass.loc[idx]),
                size_pass=bool(size_pass.loc[idx]),
                edge_pass=bool(edge_pass.loc[idx]),
                fragment_pass=bool(fragment_pass.loc[idx]),
            ),
            "conditional_bayesian_gap_status": _gap_status(
                truth_recovery=bool(truth_recovery.loc[idx]),
                candidate=bool(candidate.loc[idx]),
                bayes_pass=bool(bayes_pass.loc[idx]),
                context_pass=bool(context_pass.loc[idx]),
                soft_structure_pass=bool(soft_structure_pass.loc[idx]),
            ),
        }
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _context_floor_from_rule_id(rule_id: object) -> float | None:
    match = re.search(r"(?:^|\|)ctx=([-+0-9.eE]+)", str(rule_id))
    if match is None:
        return None
    return float(match.group(1))


def summarize_transfer_gap_rows(
    gap_rows: pd.DataFrame,
    transfer_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize conditional/Bayesian gaps and transfer leakage evidence."""
    if gap_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    roles = gap_rows["guard_truth_role"].astype(str)
    status = gap_rows["conditional_bayesian_gap_status"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    negative_candidate = gap_rows["default_internal_node_candidate"].astype(bool)
    relaxed_leakage_count = 0
    nonnegative_leakage_count = 0
    if transfer_rows is not None and not transfer_rows.empty:
        required = {"rule_id", "test_negative_candidate_count"}
        missing = sorted(required - set(transfer_rows.columns))
        if missing:
            raise ValueError(f"Transfer rows are missing columns: {missing!r}")
        transfer = transfer_rows.copy()
        transfer["context_floor"] = transfer["rule_id"].map(_context_floor_from_rule_id)
        leakage = pd.to_numeric(
            transfer["test_negative_candidate_count"],
            errors="coerce",
        ).gt(0)
        selected_rule = ~transfer["rule_id"].astype(str).eq("no_train_rule")
        relaxed_leakage_count = int(
            (
                selected_rule
                & leakage
                & transfer["context_floor"].notna()
                & transfer["context_floor"].lt(0.0)
            ).sum()
        )
        nonnegative_leakage_count = int(
            (
                selected_rule
                & leakage
                & transfer["context_floor"].notna()
                & transfer["context_floor"].ge(0.0)
            ).sum()
        )

    context_exception_count = int(
        (truth & status.eq("context_negative_but_soft_structure_supported")).sum()
    )
    soft_blocked_count = int(
        (
            truth
            & status.isin(
                [
                    "soft_structure_blocked",
                    "context_negative_and_soft_structure_blocked",
                ]
            )
        ).sum()
    )
    bayes_blocked_count = int((truth & status.eq("selected_family_bayes_evidence_blocked")).sum())
    negative_default_count = int((negative & negative_candidate).sum())
    if negative_default_count:
        diagnostic_status = "gap_audit_detected_default_negative_leakage"
    elif context_exception_count and relaxed_leakage_count:
        diagnostic_status = "context_exception_requires_higher_order_conditional_law"
    elif context_exception_count:
        diagnostic_status = "context_exception_without_transfer_leakage_evidence"
    elif soft_blocked_count or bayes_blocked_count:
        diagnostic_status = "truth_recovery_blocked_by_noncontext_components"
    else:
        diagnostic_status = "all_truth_recovery_supported_by_local_likelihood"

    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "truth_recovery_total": int(truth.sum()),
                "truth_recovered_by_default_count": int(
                    (truth & status.eq("recovered_by_default_internal_node_likelihood")).sum()
                ),
                "truth_context_negative_soft_supported_count": (context_exception_count),
                "truth_soft_structure_blocked_count": soft_blocked_count,
                "truth_bayes_evidence_blocked_count": bayes_blocked_count,
                "negative_total": int(negative.sum()),
                "negative_default_candidate_count": negative_default_count,
                "relaxed_context_leakage_rule_count": relaxed_leakage_count,
                "nonnegative_context_leakage_rule_count": nonnegative_leakage_count,
                "diagnostic_status": diagnostic_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_internal_node_transfer_gap_audit(
    config: OverlapInternalNodeTransferGapAuditConfig,
) -> dict[str, Path]:
    """Run the transfer-gap audit and write outputs."""
    likelihood_rows = pd.read_csv(config.likelihood_rows_path)
    transfer_rows = (
        pd.read_csv(config.transfer_rows_path) if config.transfer_rows_path is not None else None
    )
    gap_rows = build_internal_node_transfer_gap_rows(
        likelihood_rows,
        bayes_factor_floor=float(config.bayes_factor_floor),
        context_margin_floor=float(config.context_margin_floor),
        subspace_floor=float(config.subspace_floor),
        size_balance_floor=float(config.size_balance_floor),
        edge_norm_balance_floor=float(config.edge_norm_balance_floor),
        fragment_risk_ceiling=float(config.fragment_risk_ceiling),
    )
    summary = summarize_transfer_gap_rows(gap_rows, transfer_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    gap_rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "likelihood_rows_path": str(config.likelihood_rows_path),
        "transfer_rows_path": (
            str(config.transfer_rows_path) if config.transfer_rows_path is not None else None
        ),
        "parameters": {
            "bayes_factor_floor": float(config.bayes_factor_floor),
            "context_margin_floor": float(config.context_margin_floor),
            "subspace_floor": float(config.subspace_floor),
            "size_balance_floor": float(config.size_balance_floor),
            "edge_norm_balance_floor": float(config.edge_norm_balance_floor),
            "fragment_risk_ceiling": float(config.fragment_risk_ceiling),
        },
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_conditional_bayesian_gap_audit",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--transfer-rows-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_internal_node_transfer_gap_audit(
        OverlapInternalNodeTransferGapAuditConfig(
            likelihood_rows_path=args.likelihood_rows_path,
            transfer_rows_path=args.transfer_rows_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
