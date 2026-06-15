"""Audit limiting terms in the overlap Bayesian neighborhood likelihood.

This diagnostic reads posterior-style component rows and reports which
neighborhood likelihood terms are preventing coherent-split promotion. It is a
model-specification audit only; it does not fit weights or promote thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_bayesian_neighborhood_component_audit"
SCHEMA_VERSION = "overlap_bayesian_neighborhood_component_audit/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap_bayesian_neighborhood_component_audit"
)

NEIGHBORHOOD_COMPONENTS = (
    "homogeneity_log_bayes_factor",
    "context_margin_log_bayes_factor",
    "subspace_log_bayes_factor",
    "balance_log_bayes_factor",
    "balanced_recovery_log_bayes_factor",
    "fragment_risk_log_contribution",
)

REQUIRED_BAYESIAN_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "residual_family_size",
    "selected_family_log_bayes_factor_lower",
    "selection_context_log_penalty",
    "context_prior_log_odds",
    "homogeneity_log_bayes_factor",
    "context_margin_log_bayes_factor",
    "subspace_log_bayes_factor",
    "balance_log_bayes_factor",
    "balanced_recovery_log_bayes_factor",
    "fragment_risk_log_penalty",
    "neighborhood_log_bayes_factor",
    "conditional_coherent_posterior",
    "conditional_bayesian_status",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "conditional_bayesian_status",
    "selected_family_log_bayes_factor_lower",
    "selection_context_log_penalty",
    "neighborhood_log_bayes_factor",
    "strong_neighborhood_threshold",
    "strong_neighborhood_gap",
    "negative_neighborhood_component_count",
    "limiting_component",
    "limiting_component_value",
    "homogeneity_log_bayes_factor",
    "context_margin_log_bayes_factor",
    "subspace_log_bayes_factor",
    "balance_log_bayes_factor",
    "balanced_recovery_log_bayes_factor",
    "fragment_risk_log_contribution",
    "component_audit_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "residual_family_truth_role",
    "family_count",
    "median_selected_family_log_bayes_factor_lower",
    "median_neighborhood_log_bayes_factor",
    "median_strong_neighborhood_gap",
    "coherent_candidate_count",
    "p_value_evidence_without_strong_neighborhood_count",
    "structurally_incoherent_count",
    "most_common_limiting_component",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapBayesianNeighborhoodComponentAuditConfig:
    """Runtime contract for Bayesian neighborhood component audits."""

    bayesian_rows_path: Path
    output_dir: Path
    strong_neighborhood_threshold: float = 2.0
    p_extreme_log_bayes_factor_threshold: float = 3.0

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_bayesian_neighborhood_component_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_bayesian_neighborhood_component_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_bayesian_rows(rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_BAYESIAN_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"Bayesian rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _limiting_component(component_rows: pd.DataFrame) -> pd.DataFrame:
    numeric = component_rows.apply(pd.to_numeric, errors="coerce")
    limiting_names: list[str] = []
    limiting_values: list[float] = []
    negative_counts: list[int] = []
    for _, row in numeric.iterrows():
        finite = row.dropna()
        negative_counts.append(int(finite.lt(0.0).sum()))
        if finite.empty:
            limiting_names.append("none")
            limiting_values.append(math.nan)
            continue
        limiting_names.append(str(finite.idxmin()))
        limiting_values.append(float(finite.min()))
    return pd.DataFrame(
        {
            "negative_neighborhood_component_count": negative_counts,
            "limiting_component": limiting_names,
            "limiting_component_value": limiting_values,
        },
        index=component_rows.index,
    )


def build_bayesian_neighborhood_component_rows(
    bayesian_rows: pd.DataFrame,
    *,
    strong_neighborhood_threshold: float = 2.0,
    p_extreme_log_bayes_factor_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return row-level audit of the Bayesian neighborhood components."""
    _validate_bayesian_rows(bayesian_rows)
    rows = bayesian_rows.copy()
    rows["fragment_risk_log_contribution"] = -_numeric(
        rows,
        "fragment_risk_log_penalty",
    )
    component_rows = rows[list(NEIGHBORHOOD_COMPONENTS)]
    limiting = _limiting_component(component_rows)
    neighborhood = _numeric(rows, "neighborhood_log_bayes_factor")
    selected_bf = _numeric(rows, "selected_family_log_bayes_factor_lower")
    gap = float(strong_neighborhood_threshold) - neighborhood
    status = pd.Series("component_audit_coherent_candidate", index=rows.index, dtype=object)
    status.loc[neighborhood.lt(0.0)] = "component_audit_structurally_incoherent"
    status.loc[
        neighborhood.ge(0.0)
        & neighborhood.lt(float(strong_neighborhood_threshold))
        & selected_bf.ge(float(p_extreme_log_bayes_factor_threshold))
    ] = "component_audit_p_value_evidence_without_strong_neighborhood"
    status.loc[
        neighborhood.ge(0.0)
        & neighborhood.lt(float(strong_neighborhood_threshold))
        & selected_bf.lt(float(p_extreme_log_bayes_factor_threshold))
    ] = "component_audit_weak_evidence_and_weak_neighborhood"
    output = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": rows["case_id"].astype(str),
            "data_role": rows["data_role"].astype(str),
            "replicate": rows["replicate"].astype(int),
            "residual_family_truth_role": rows["residual_family_truth_role"].astype(str),
            "conditional_bayesian_status": rows[
                "conditional_bayesian_status"
            ].astype(str),
            "selected_family_log_bayes_factor_lower": selected_bf,
            "selection_context_log_penalty": _numeric(
                rows,
                "selection_context_log_penalty",
            ),
            "neighborhood_log_bayes_factor": neighborhood,
            "strong_neighborhood_threshold": float(strong_neighborhood_threshold),
            "strong_neighborhood_gap": gap.clip(lower=0.0),
            "negative_neighborhood_component_count": limiting[
                "negative_neighborhood_component_count"
            ],
            "limiting_component": limiting["limiting_component"],
            "limiting_component_value": limiting["limiting_component_value"],
            "homogeneity_log_bayes_factor": _numeric(
                rows,
                "homogeneity_log_bayes_factor",
            ),
            "context_margin_log_bayes_factor": _numeric(
                rows,
                "context_margin_log_bayes_factor",
            ),
            "subspace_log_bayes_factor": _numeric(rows, "subspace_log_bayes_factor"),
            "balance_log_bayes_factor": _numeric(rows, "balance_log_bayes_factor"),
            "balanced_recovery_log_bayes_factor": _numeric(
                rows,
                "balanced_recovery_log_bayes_factor",
            ),
            "fragment_risk_log_contribution": _numeric(
                rows,
                "fragment_risk_log_contribution",
            ),
            "component_audit_status": status,
        },
        columns=ROW_COLUMNS,
    )
    return output


def summarize_bayesian_neighborhood_components(
    component_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Bayesian neighborhood component blockers by family role."""
    if component_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for role, group in component_rows.groupby("residual_family_truth_role", sort=True):
        status = group["component_audit_status"].astype(str)
        limiting = group["limiting_component"].astype(str)
        common_limiting = (
            str(limiting.value_counts().idxmax()) if not limiting.empty else "none"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "residual_family_truth_role": str(role),
                "family_count": int(group.shape[0]),
                "median_selected_family_log_bayes_factor_lower": float(
                    group["selected_family_log_bayes_factor_lower"].median()
                ),
                "median_neighborhood_log_bayes_factor": float(
                    group["neighborhood_log_bayes_factor"].median()
                ),
                "median_strong_neighborhood_gap": float(
                    group["strong_neighborhood_gap"].median()
                ),
                "coherent_candidate_count": int(
                    status.eq("component_audit_coherent_candidate").sum()
                ),
                "p_value_evidence_without_strong_neighborhood_count": int(
                    status.eq(
                        "component_audit_p_value_evidence_without_strong_neighborhood"
                    ).sum()
                ),
                "structurally_incoherent_count": int(
                    status.eq("component_audit_structurally_incoherent").sum()
                ),
                "most_common_limiting_component": common_limiting,
                "diagnostic_status": (
                    "diagnostic_only_component_audit_not_calibration"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_bayesian_neighborhood_component_audit(
    config: OverlapBayesianNeighborhoodComponentAuditConfig,
) -> dict[str, Path]:
    """Run the component audit and write outputs."""
    bayesian_rows = pd.read_csv(config.bayesian_rows_path)
    component_rows = build_bayesian_neighborhood_component_rows(
        bayesian_rows,
        strong_neighborhood_threshold=config.strong_neighborhood_threshold,
        p_extreme_log_bayes_factor_threshold=(
            config.p_extreme_log_bayes_factor_threshold
        ),
    )
    summary = summarize_bayesian_neighborhood_components(component_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    component_rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "bayesian_rows_path": str(config.bayesian_rows_path),
        "strong_neighborhood_threshold": float(config.strong_neighborhood_threshold),
        "p_extreme_log_bayes_factor_threshold": float(
            config.p_extreme_log_bayes_factor_threshold
        ),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_no_threshold_promotion",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesian-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strong-neighborhood-threshold", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_bayesian_neighborhood_component_audit(
        OverlapBayesianNeighborhoodComponentAuditConfig(
            bayesian_rows_path=args.bayesian_rows_path,
            output_dir=args.output_dir,
            strong_neighborhood_threshold=float(args.strong_neighborhood_threshold),
        )
    )


if __name__ == "__main__":
    main()
