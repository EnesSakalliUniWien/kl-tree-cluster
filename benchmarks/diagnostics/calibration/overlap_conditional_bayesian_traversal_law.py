"""Conditional Bayesian traversal-law diagnostic for overlap residual families.

This diagnostic turns the residual selected-family evidence into transparent
posterior-style log odds for a coherent structural split. It is deliberately
not a production calibration rule: no permutations are run, no labels are used
to fit weights, and every component is written out for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_conditional_bayesian_traversal_law_not_calibration"
SCHEMA_VERSION = "overlap_conditional_bayesian_traversal_law/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap_conditional_bayesian_traversal_law"
)

REQUIRED_FAMILY_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "residual_family_size",
    "residual_min_sibling_p_value",
    "residual_neg_log10_min_sibling_p_value",
    "residual_max_homogeneity_gain_min",
    "residual_max_continuous_context_margin",
    "residual_max_subspace_consensus_jaccard_topk",
    "residual_max_depth",
    "residual_median_parent_size",
    "residual_max_barycentric_balance",
    "residual_min_fragment_risk_proxy_score",
    "residual_max_balanced_recovery_proxy_score",
    "residual_min_size_balance",
    "residual_min_edge_norm_balance",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "residual_family_truth_role",
    "residual_family_size",
    "residual_min_sibling_p_value",
    "residual_neg_log10_min_sibling_p_value",
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
    "conditional_coherent_log_odds",
    "conditional_coherent_posterior",
    "conditional_bayesian_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "residual_family_truth_role",
    "family_count",
    "median_selected_family_log_bayes_factor_lower",
    "median_neighborhood_log_bayes_factor",
    "median_conditional_coherent_posterior",
    "max_conditional_coherent_posterior",
    "coherent_candidate_count",
    "p_value_extreme_structurally_incoherent_count",
    "p_value_extreme_neighborhood_insufficient_count",
    "unstable_multiscale_count",
    "diagnostic_status",
)


@dataclass(frozen=True)
class ConditionalBayesianTraversalParameters:
    """Transparent weights for a diagnostic posterior-style score."""

    base_log_prior_odds: float = -2.0
    depth_credit_weight: float = 0.45
    depth_scale: float = 2.0
    parent_size_penalty_weight: float = 0.55
    parent_reference_size: float = 800.0
    imbalance_penalty_weight: float = 2.0
    family_size_log_penalty_weight: float = 1.0
    homogeneity_gain_weight: float = 80.0
    context_margin_weight: float = 80.0
    subspace_weight: float = 1.5
    subspace_reference: float = 0.50
    size_balance_weight: float = 2.0
    size_balance_reference: float = 1.0 / 3.0
    edge_norm_balance_weight: float = 1.0
    edge_norm_balance_reference: float = 0.50
    balanced_recovery_weight: float = 0.75
    balanced_recovery_reference: float = 2.0
    fragment_risk_weight: float = 1.5
    fragment_risk_reference: float = 1.0
    coherent_posterior_threshold: float = 0.90
    strong_neighborhood_log_bayes_factor_threshold: float = 2.0
    ambiguous_posterior_threshold: float = 0.50
    p_extreme_log_bayes_factor_threshold: float = 3.0
    incoherent_neighborhood_threshold: float = 0.0


@dataclass(frozen=True)
class OverlapConditionalBayesianTraversalLawConfig:
    """Runtime contract for the conditional Bayesian traversal diagnostic."""

    residual_family_rows_path: Path
    output_dir: Path
    parameters: ConditionalBayesianTraversalParameters = (
        ConditionalBayesianTraversalParameters()
    )

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_conditional_bayesian_traversal_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_conditional_bayesian_traversal_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_family_rows(family_rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_FAMILY_COLUMNS - set(family_rows.columns))
    if missing:
        raise ValueError(f"Residual family rows are missing columns: {missing!r}")


def p_value_log_bayes_factor_lower_bound(p_values: pd.Series) -> pd.Series:
    """Return a lower-bound log Bayes factor against a point null from p-values.

    For p <= 1/e, this uses the common minimum Bayes-factor bound
    BF_01 >= -e p log(p), then returns -log(BF_01). For larger p-values the
    diagnostic reports zero positive evidence.
    """
    p = pd.to_numeric(p_values, errors="coerce").clip(
        lower=float(np.nextafter(0.0, 1.0)),
        upper=1.0,
    )
    bf01 = pd.Series(1.0, index=p.index, dtype=float)
    small = p.le(1.0 / math.e)
    bf01.loc[small] = -math.e * p.loc[small] * np.log(p.loc[small])
    bounded = bf01.clip(lower=float(np.nextafter(0.0, 1.0)), upper=1.0)
    return -np.log(bounded)


def _as_numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _safe_log1p_positive(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0.0))


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").clip(lower=-700.0, upper=700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_conditional_bayesian_traversal_rows(
    family_rows: pd.DataFrame,
    *,
    parameters: ConditionalBayesianTraversalParameters = (
        ConditionalBayesianTraversalParameters()
    ),
) -> pd.DataFrame:
    """Score residual selected families with a transparent conditional law."""
    _validate_family_rows(family_rows)
    rows = family_rows.copy()
    selected_bf = p_value_log_bayes_factor_lower_bound(
        rows["residual_min_sibling_p_value"]
    )
    family_size = _as_numeric(rows, "residual_family_size").fillna(1.0).clip(lower=1.0)
    depth = _as_numeric(rows, "residual_max_depth").fillna(0.0).clip(lower=0.0)
    parent_size = (
        _as_numeric(rows, "residual_median_parent_size").fillna(1.0).clip(lower=1.0)
    )
    balance = (
        _as_numeric(rows, "residual_max_barycentric_balance")
        .fillna(0.0)
        .clip(lower=0.0, upper=0.5)
    )
    parent_ref = max(float(parameters.parent_reference_size), 1.0)
    depth_scale = max(float(parameters.depth_scale), 1e-9)
    depth_credit = float(parameters.depth_credit_weight) * (1.0 - np.exp(-depth / depth_scale))
    parent_penalty = float(parameters.parent_size_penalty_weight) * (
        np.log1p(parent_size) / math.log1p(parent_ref)
    )
    imbalance_penalty = float(parameters.imbalance_penalty_weight) * (0.5 - balance)
    family_penalty = float(parameters.family_size_log_penalty_weight) * np.log(
        family_size
    )
    selection_context_penalty = parent_penalty + imbalance_penalty + family_penalty
    context_prior = (
        float(parameters.base_log_prior_odds)
        + depth_credit
        - selection_context_penalty
    )

    homogeneity_bf = (
        float(parameters.homogeneity_gain_weight)
        * _as_numeric(rows, "residual_max_homogeneity_gain_min").fillna(0.0)
    )
    context_margin_bf = (
        float(parameters.context_margin_weight)
        * _as_numeric(rows, "residual_max_continuous_context_margin").fillna(0.0)
    )
    subspace_bf = float(parameters.subspace_weight) * (
        _as_numeric(rows, "residual_max_subspace_consensus_jaccard_topk").fillna(0.0)
        - float(parameters.subspace_reference)
    )
    size_balance_bf = float(parameters.size_balance_weight) * (
        _as_numeric(rows, "residual_min_size_balance").fillna(0.0)
        - float(parameters.size_balance_reference)
    )
    edge_balance_bf = float(parameters.edge_norm_balance_weight) * (
        _as_numeric(rows, "residual_min_edge_norm_balance").fillna(0.0)
        - float(parameters.edge_norm_balance_reference)
    )
    balance_bf = size_balance_bf + edge_balance_bf
    balanced_recovery_bf = float(parameters.balanced_recovery_weight) * (
        _as_numeric(rows, "residual_max_balanced_recovery_proxy_score").fillna(0.0)
        - float(parameters.balanced_recovery_reference)
    )
    fragment_penalty = float(parameters.fragment_risk_weight) * (
        _as_numeric(rows, "residual_min_fragment_risk_proxy_score").fillna(0.0)
        - float(parameters.fragment_risk_reference)
    ).clip(lower=0.0)
    neighborhood_bf = (
        homogeneity_bf
        + context_margin_bf
        + subspace_bf
        + balance_bf
        + balanced_recovery_bf
        - fragment_penalty
    )
    log_odds = context_prior + selected_bf + neighborhood_bf
    posterior = _sigmoid(log_odds)
    status = pd.Series("conditional_unstable_multiscale", index=rows.index, dtype=object)
    status.loc[
        posterior.ge(float(parameters.coherent_posterior_threshold))
        & neighborhood_bf.ge(
            float(parameters.strong_neighborhood_log_bayes_factor_threshold)
        )
    ] = "conditional_coherent_candidate"
    status.loc[
        posterior.ge(float(parameters.ambiguous_posterior_threshold))
        & status.eq("conditional_unstable_multiscale")
    ] = "conditional_ambiguous_candidate"
    status.loc[
        selected_bf.ge(float(parameters.p_extreme_log_bayes_factor_threshold))
        & neighborhood_bf.lt(float(parameters.incoherent_neighborhood_threshold))
    ] = "p_value_extreme_structurally_incoherent"
    status.loc[
        selected_bf.ge(float(parameters.p_extreme_log_bayes_factor_threshold))
        & neighborhood_bf.ge(float(parameters.incoherent_neighborhood_threshold))
        & neighborhood_bf.lt(
            float(parameters.strong_neighborhood_log_bayes_factor_threshold)
        )
    ] = "p_value_extreme_neighborhood_insufficient"

    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": rows["case_id"].astype(str),
            "data_role": rows["data_role"].astype(str),
            "replicate": rows["replicate"].astype(int),
            "residual_family_truth_role": rows["residual_family_truth_role"].astype(str),
            "residual_family_size": family_size.astype(int),
            "residual_min_sibling_p_value": _as_numeric(
                rows,
                "residual_min_sibling_p_value",
            ),
            "residual_neg_log10_min_sibling_p_value": _as_numeric(
                rows,
                "residual_neg_log10_min_sibling_p_value",
            ),
            "selected_family_log_bayes_factor_lower": selected_bf,
            "selection_context_log_penalty": selection_context_penalty,
            "context_prior_log_odds": context_prior,
            "homogeneity_log_bayes_factor": homogeneity_bf,
            "context_margin_log_bayes_factor": context_margin_bf,
            "subspace_log_bayes_factor": subspace_bf,
            "balance_log_bayes_factor": balance_bf,
            "balanced_recovery_log_bayes_factor": balanced_recovery_bf,
            "fragment_risk_log_penalty": fragment_penalty,
            "neighborhood_log_bayes_factor": neighborhood_bf,
            "conditional_coherent_log_odds": log_odds,
            "conditional_coherent_posterior": posterior,
            "conditional_bayesian_status": status,
        },
        columns=ROW_COLUMNS,
    )


def summarize_conditional_bayesian_rows(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize posterior-style diagnostic rows by residual-family role."""
    if scored_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for role, group in scored_rows.groupby("residual_family_truth_role", sort=True):
        status = group["conditional_bayesian_status"].astype(str)
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
                "median_conditional_coherent_posterior": float(
                    group["conditional_coherent_posterior"].median()
                ),
                "max_conditional_coherent_posterior": float(
                    group["conditional_coherent_posterior"].max()
                ),
                "coherent_candidate_count": int(
                    status.eq("conditional_coherent_candidate").sum()
                ),
                "p_value_extreme_structurally_incoherent_count": int(
                    status.eq("p_value_extreme_structurally_incoherent").sum()
                ),
                "p_value_extreme_neighborhood_insufficient_count": int(
                    status.eq("p_value_extreme_neighborhood_insufficient").sum()
                ),
                "unstable_multiscale_count": int(
                    status.isin(
                        {
                            "conditional_unstable_multiscale",
                            "conditional_ambiguous_candidate",
                            "p_value_extreme_structurally_incoherent",
                            "p_value_extreme_neighborhood_insufficient",
                        }
                    ).sum()
                ),
                "diagnostic_status": (
                    "diagnostic_only_conditional_model_specification_not_calibration"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_conditional_bayesian_traversal_law(
    config: OverlapConditionalBayesianTraversalLawConfig,
) -> dict[str, Path]:
    """Run the conditional Bayesian traversal-law diagnostic and write outputs."""
    family_rows = pd.read_csv(config.residual_family_rows_path)
    scored_rows = build_conditional_bayesian_traversal_rows(
        family_rows,
        parameters=config.parameters,
    )
    summary = summarize_conditional_bayesian_rows(scored_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scored_rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "residual_family_rows_path": str(config.residual_family_rows_path),
        "parameters": config.parameters.__dict__,
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": (
            "diagnostic_only_no_permutation_no_production_promotion"
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-family-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_conditional_bayesian_traversal_law(
        OverlapConditionalBayesianTraversalLawConfig(
            residual_family_rows_path=args.residual_family_rows_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
