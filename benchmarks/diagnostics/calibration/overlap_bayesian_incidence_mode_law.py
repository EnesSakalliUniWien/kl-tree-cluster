"""Bayesian incidence-mode law diagnostic for overlap traversal rows.

This diagnostic consumes the true branch-incidence panel and transfer-gap rows
to express the next traversal law as two latent modes:

* continuation mode: outgoing structure is compatible with the incoming branch;
* emergent local-outcome mode: outgoing structure is locally coherent in a
  different subspace.

The law remains fail-closed for context-negative emergent rows when the current
conditioning variables do not separate truth recovery from negatives.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_bayesian_incidence_mode_law"
SCHEMA_VERSION = "overlap_bayesian_incidence_mode_law/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap_bayesian_incidence_mode_law"
)

BRANCH_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "branch_incidence_geometry_status",
    "incoming_family_outgoing_jaccard_topk",
    "incoming_family_outgoing_abs_cosine",
    "incoming_edge_outgoing_jaccard_topk",
    "incoming_edge_outgoing_abs_cosine",
}

GAP_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "conditional_bayesian_gap_status",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "context_margin_pass",
    "soft_structure_pass",
    "default_internal_node_candidate",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "conditional_bayesian_gap_status",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "context_margin_pass",
    "soft_structure_pass",
    "branch_incidence_geometry_status",
    "branch_alignment_score",
    "continuation_mode_log_score",
    "local_outcome_mode_log_score",
    "context_negative_emergent_log_score",
    "bayesian_incidence_mode_status",
    "production_admissibility_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "truth_recovery_total",
    "negative_total",
    "continuation_candidate_count",
    "local_outcome_candidate_count",
    "local_outcome_truth_count",
    "local_outcome_negative_count",
    "context_negative_emergent_ambiguous_count",
    "context_negative_emergent_truth_count",
    "context_negative_emergent_negative_count",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapBayesianIncidenceModeLawConfig:
    """Runtime contract for Bayesian incidence-mode diagnostics."""

    branch_rows_path: Path
    transfer_gap_rows_path: Path
    output_dir: Path
    selected_bayes_factor_floor: float = 3.0
    branch_alignment_floor: float = 0.25

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_bayesian_incidence_mode_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_bayesian_incidence_mode_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate(rows: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{label} rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _alignment_score(rows: pd.DataFrame) -> pd.Series:
    if "metric_family_alignment_score" in rows.columns:
        return _numeric(rows, "metric_family_alignment_score").fillna(0.0)
    parts = [
        _numeric(rows, "incoming_family_outgoing_jaccard_topk"),
        _numeric(rows, "incoming_family_outgoing_abs_cosine"),
        _numeric(rows, "incoming_edge_outgoing_jaccard_topk"),
        _numeric(rows, "incoming_edge_outgoing_abs_cosine"),
    ]
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def _mode_status(
    *,
    selected_bayes_pass: bool,
    continuation_candidate: bool,
    local_outcome_candidate: bool,
    context_negative_emergent: bool,
    soft_structure_pass: bool,
    context_margin_pass: bool,
) -> str:
    if not selected_bayes_pass:
        return "selected_bayes_evidence_blocked"
    if continuation_candidate:
        return "continuation_mode_candidate"
    if local_outcome_candidate:
        return "local_outcome_mode_candidate"
    if context_negative_emergent:
        return "context_negative_emergent_mode_ambiguous"
    if not soft_structure_pass:
        return "soft_structure_blocked"
    if not context_margin_pass:
        return "context_negative_blocked"
    return "incidence_mode_unresolved"


def build_bayesian_incidence_mode_rows(
    *,
    branch_rows: pd.DataFrame,
    transfer_gap_rows: pd.DataFrame,
    selected_bayes_factor_floor: float = 3.0,
    branch_alignment_floor: float = 0.25,
) -> pd.DataFrame:
    """Build incidence-mode rows from true branch geometry and gap evidence."""
    _validate(branch_rows, BRANCH_REQUIRED_COLUMNS, "Branch-incidence")
    _validate(transfer_gap_rows, GAP_REQUIRED_COLUMNS, "Transfer-gap")
    keys = ["case_id", "data_role", "replicate", "node_id"]
    optional_branch_columns = [
        "metric_family_alignment_score",
        "metric_family_compatibility_status",
    ]
    branch_columns = list(BRANCH_REQUIRED_COLUMNS) + [
        column for column in optional_branch_columns if column in branch_rows.columns
    ]
    branch = branch_rows[branch_columns].copy()
    gap = transfer_gap_rows[list(GAP_REQUIRED_COLUMNS)].copy()
    rows = gap.merge(branch, on=keys, how="left", validate="one_to_one")
    rows["branch_incidence_geometry_status"] = rows[
        "branch_incidence_geometry_status"
    ].fillna("root_or_unobserved_branch_incidence")
    alignment = _alignment_score(rows)
    selected_bf = _numeric(rows, "selected_family_log_bayes_factor_lower")
    context = _numeric(rows, "continuous_context_min_margin")
    context_pass = rows["context_margin_pass"].astype(bool)
    soft_pass = rows["soft_structure_pass"].astype(bool)
    selected_pass = selected_bf.ge(float(selected_bayes_factor_floor))
    branch_aligned = alignment.ge(float(branch_alignment_floor))
    continuation_candidate = selected_pass & soft_pass & branch_aligned
    local_outcome_candidate = selected_pass & soft_pass & context_pass
    context_negative_emergent = (
        selected_pass & soft_pass & ~context_pass & ~branch_aligned
    )

    continuation_score = selected_bf + 4.0 * alignment
    local_outcome_score = selected_bf + 80.0 * context
    emergent_score = selected_bf + 4.0 * alignment + 80.0 * context

    records: list[dict[str, object]] = []
    for idx, row in rows.iterrows():
        mode_status = _mode_status(
            selected_bayes_pass=bool(selected_pass.loc[idx]),
            continuation_candidate=bool(continuation_candidate.loc[idx]),
            local_outcome_candidate=bool(local_outcome_candidate.loc[idx]),
            context_negative_emergent=bool(context_negative_emergent.loc[idx]),
            soft_structure_pass=bool(soft_pass.loc[idx]),
            context_margin_pass=bool(context_pass.loc[idx]),
        )
        production_status = (
            "diagnostic_only_candidate"
            if mode_status in {
                "continuation_mode_candidate",
                "local_outcome_mode_candidate",
            }
            else "fail_closed_requires_additional_conditioning"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "guard_truth_role": str(row["guard_truth_role"]),
                "conditional_bayesian_gap_status": str(
                    row["conditional_bayesian_gap_status"]
                ),
                "selected_family_log_bayes_factor_lower": float(selected_bf.loc[idx]),
                "continuous_context_min_margin": float(context.loc[idx]),
                "context_margin_pass": bool(context_pass.loc[idx]),
                "soft_structure_pass": bool(soft_pass.loc[idx]),
                "branch_incidence_geometry_status": str(
                    row["branch_incidence_geometry_status"]
                ),
                "branch_alignment_score": float(alignment.loc[idx]),
                "continuation_mode_log_score": float(continuation_score.loc[idx]),
                "local_outcome_mode_log_score": float(local_outcome_score.loc[idx]),
                "context_negative_emergent_log_score": float(emergent_score.loc[idx]),
                "bayesian_incidence_mode_status": mode_status,
                "production_admissibility_status": production_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_bayesian_incidence_mode_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize mode separation and fail-closed status."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    status = rows["bayesian_incidence_mode_status"].astype(str)
    continuation = status.eq("continuation_mode_candidate")
    local = status.eq("local_outcome_mode_candidate")
    emergent = status.eq("context_negative_emergent_mode_ambiguous")
    local_negative = int((local & negative).sum())
    emergent_truth = int((emergent & truth).sum())
    emergent_negative = int((emergent & negative).sum())
    if local_negative:
        diagnostic_status = "local_outcome_mode_leakage"
    elif emergent_truth and emergent_negative:
        diagnostic_status = "context_negative_emergent_mode_not_identified"
    elif emergent_truth:
        diagnostic_status = "context_negative_emergent_mode_candidate"
    elif int((local & truth).sum()):
        diagnostic_status = "local_outcome_mode_zero_negative_candidate"
    else:
        diagnostic_status = "incidence_mode_unresolved"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "truth_recovery_total": int(truth.sum()),
                "negative_total": int(negative.sum()),
                "continuation_candidate_count": int(continuation.sum()),
                "local_outcome_candidate_count": int(local.sum()),
                "local_outcome_truth_count": int((local & truth).sum()),
                "local_outcome_negative_count": local_negative,
                "context_negative_emergent_ambiguous_count": int(emergent.sum()),
                "context_negative_emergent_truth_count": emergent_truth,
                "context_negative_emergent_negative_count": emergent_negative,
                "diagnostic_status": diagnostic_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_bayesian_incidence_mode_law(
    config: OverlapBayesianIncidenceModeLawConfig,
) -> dict[str, Path]:
    """Run Bayesian incidence-mode diagnostics and write outputs."""
    branch_rows = pd.read_csv(config.branch_rows_path)
    gap_rows = pd.read_csv(config.transfer_gap_rows_path)
    rows = build_bayesian_incidence_mode_rows(
        branch_rows=branch_rows,
        transfer_gap_rows=gap_rows,
        selected_bayes_factor_floor=float(config.selected_bayes_factor_floor),
        branch_alignment_floor=float(config.branch_alignment_floor),
    )
    summary = summarize_bayesian_incidence_mode_rows(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "branch_rows_path": str(config.branch_rows_path),
        "transfer_gap_rows_path": str(config.transfer_gap_rows_path),
        "selected_bayes_factor_floor": float(config.selected_bayes_factor_floor),
        "branch_alignment_floor": float(config.branch_alignment_floor),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_fail_closed_mode_law",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-rows-path", required=True, type=Path)
    parser.add_argument("--transfer-gap-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selected-bayes-factor-floor", default=3.0, type=float)
    parser.add_argument("--branch-alignment-floor", default=0.25, type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_bayesian_incidence_mode_law(
        OverlapBayesianIncidenceModeLawConfig(
            branch_rows_path=args.branch_rows_path,
            transfer_gap_rows_path=args.transfer_gap_rows_path,
            output_dir=args.output_dir,
            selected_bayes_factor_floor=float(args.selected_bayes_factor_floor),
            branch_alignment_floor=float(args.branch_alignment_floor),
        )
    )


if __name__ == "__main__":
    main()
