"""Bayesian topology law diagnostic for context-negative overlap rows.

The edge-conditioning scan shows that raw edge-test significance does not
identify the remaining context-negative emergent truth row. The topology scan
then finds a single-positive incoming/outgoing balance separator, but transfer
validation rejects that as a threshold rule.

This panel keeps the useful direction without promoting a threshold: it scores
context-negative rows with fixed, interpretable component likelihood ratios for
selected-neighborhood topology. The components encode income/outcome structure
around the selected family: incoming branch balance, outgoing sibling balance,
outgoing edge-norm balance, anti-fragment evidence, selected-family evidence,
and a soft context penalty.

The output is diagnostic-only. It can identify whether a smooth conditional
topology law is plausible, but it cannot promote production calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_context_negative_bayesian_topology_law"
SCHEMA_VERSION = "overlap_context_negative_bayesian_topology_law/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_context_negative_bayesian_topology_law"
)

DEFAULT_PRIOR_TRUTH_PROBABILITY = 0.05
DEFAULT_CONTEXT_PENALTY_WEIGHT = 50.0

REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "incoming_branch_balance",
    "outgoing_balance",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "incoming_balance_log_lr",
    "outgoing_balance_log_lr",
    "outgoing_edge_norm_log_lr",
    "anti_fragment_log_lr",
    "selected_family_evidence_log",
    "context_negative_soft_penalty",
    "posterior_log_odds",
    "posterior_probability",
    "posterior_rank",
    "bayesian_topology_status",
)

COMPONENT_COLUMNS = (
    "schema_version",
    "study_role",
    "component",
    "truth_count",
    "negative_count",
    "truth_min",
    "truth_median",
    "truth_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "truth_rank_best",
    "truth_rank_worst",
    "negative_above_truth_min_count",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "truth_recovery_count",
    "negative_count",
    "prior_truth_probability",
    "context_penalty_weight",
    "truth_posterior_log_odds_min",
    "truth_posterior_log_odds_median",
    "truth_posterior_log_odds_max",
    "negative_posterior_log_odds_max",
    "truth_rank_best",
    "truth_rank_worst",
    "truth_above_all_negatives_count",
    "negative_above_truth_min_count",
    "posterior_log_odds_margin",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapContextNegativeBayesianTopologyLawConfig:
    """Runtime contract for context-negative Bayesian topology diagnostics."""

    topology_rows_path: Path
    output_dir: Path
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_bayesian_topology_rows.csv"

    @property
    def component_summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_bayesian_topology_component_summary.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_bayesian_topology_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate(rows: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = sorted(set(required) - set(rows.columns))
    if missing:
        raise ValueError(f"Topology rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _clip_unit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _beta_log_pdf(values: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    log_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (alpha - 1.0) * np.log(values) + (beta - 1.0) * np.log1p(-values) - log_norm


def beta_log_likelihood_ratio(
    values: pd.Series | np.ndarray | float,
    *,
    signal_alpha: float = 6.0,
    signal_beta: float = 2.0,
    background_alpha: float = 2.0,
    background_beta: float = 5.0,
) -> np.ndarray:
    """Return a beta-density log likelihood ratio on the unit interval."""
    unit = _clip_unit(values)
    signal = _beta_log_pdf(unit, signal_alpha, signal_beta)
    background = _beta_log_pdf(unit, background_alpha, background_beta)
    return signal - background


def _balance_unit(balance: pd.Series) -> np.ndarray:
    return _clip_unit(2.0 * _numeric(pd.DataFrame({"balance": balance}), "balance"))


def _anti_fragment_unit(fragment_risk: pd.Series) -> np.ndarray:
    risk = np.maximum(np.asarray(fragment_risk, dtype=float), 0.0)
    return _clip_unit(1.0 / (1.0 + risk))


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=-700.0, upper=700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _neutralize_missing(component: np.ndarray) -> np.ndarray:
    return np.nan_to_num(component, nan=0.0, posinf=0.0, neginf=0.0)


def _status_for_row(role: str, rank: int) -> str:
    if role == "truth_recovery" and rank == 1:
        return "truth_top_rank_bayesian_topology_candidate"
    if role == "truth_recovery":
        return "truth_ranked_below_negative_bayesian_topology"
    return "negative_or_nonrecovery_bayesian_topology_row"


def build_context_negative_bayesian_topology_rows(
    topology_rows: pd.DataFrame,
    *,
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY,
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT,
) -> pd.DataFrame:
    """Build row-level Bayesian topology component scores."""
    _validate(topology_rows)
    rows = topology_rows.copy()
    incoming_balance = _numeric(rows, "incoming_branch_balance")
    outgoing_balance = _numeric(rows, "outgoing_balance")
    outgoing_edge_norm = _numeric(rows, "outgoing_edge_norm_balance").clip(0.0, 1.0)
    fragment_risk = _numeric(rows, "outgoing_fragment_risk_proxy_score")
    selected_family = _numeric(rows, "selected_family_log_bayes_factor_lower")
    context = _numeric(rows, "continuous_context_min_margin")

    incoming_balance_lr = _neutralize_missing(
        beta_log_likelihood_ratio(_balance_unit(incoming_balance))
    )
    outgoing_balance_lr = _neutralize_missing(
        beta_log_likelihood_ratio(_balance_unit(outgoing_balance))
    )
    edge_norm_lr = _neutralize_missing(beta_log_likelihood_ratio(outgoing_edge_norm))
    anti_fragment_lr = _neutralize_missing(
        beta_log_likelihood_ratio(_anti_fragment_unit(fragment_risk))
    )
    selected_evidence = _neutralize_missing(
        np.log1p(np.maximum(selected_family.to_numpy(dtype=float), 0.0))
    )
    context_values = context.to_numpy(dtype=float)
    context_penalty = _neutralize_missing(
        np.minimum(context_values, 0.0) * float(context_penalty_weight)
    )
    posterior_log_odds = (
        _logit(float(prior_truth_probability))
        + incoming_balance_lr
        + outgoing_balance_lr
        + edge_norm_lr
        + anti_fragment_lr
        + selected_evidence
        + context_penalty
    )
    posterior = _sigmoid(pd.Series(posterior_log_odds, index=rows.index))
    rank = (
        pd.Series(posterior_log_odds, index=rows.index)
        .rank(method="first", ascending=False)
        .astype(int)
    )

    records: list[dict[str, object]] = []
    for position, (idx, row) in enumerate(rows.iterrows()):
        row_rank = int(rank.loc[idx])
        role = str(row["guard_truth_role"])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "guard_truth_role": role,
                "incoming_branch_balance": float(incoming_balance.loc[idx]),
                "outgoing_balance": float(outgoing_balance.loc[idx]),
                "outgoing_edge_norm_balance": float(outgoing_edge_norm.loc[idx]),
                "outgoing_fragment_risk_proxy_score": float(fragment_risk.loc[idx]),
                "selected_family_log_bayes_factor_lower": float(selected_family.loc[idx]),
                "continuous_context_min_margin": float(context.loc[idx]),
                "incoming_balance_log_lr": float(incoming_balance_lr[position]),
                "outgoing_balance_log_lr": float(outgoing_balance_lr[position]),
                "outgoing_edge_norm_log_lr": float(edge_norm_lr[position]),
                "anti_fragment_log_lr": float(anti_fragment_lr[position]),
                "selected_family_evidence_log": float(selected_evidence[position]),
                "context_negative_soft_penalty": float(context_penalty[position]),
                "posterior_log_odds": float(posterior_log_odds[position]),
                "posterior_probability": float(posterior.loc[idx]),
                "posterior_rank": row_rank,
                "bayesian_topology_status": _status_for_row(role, row_rank),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _rank_bounds(values: pd.Series, roles: pd.Series) -> tuple[int, int, int]:
    truth = roles.eq("truth_recovery")
    if not bool(truth.any()):
        return 0, 0, 0
    ranks = values.rank(method="first", ascending=False).astype(int)
    truth_ranks = ranks[truth]
    truth_min = float(values[truth].min())
    negative_above = int((values[~truth] > truth_min).sum())
    return int(truth_ranks.min()), int(truth_ranks.max()), negative_above


def summarize_bayesian_topology_components(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize each component's truth/negative ranking behavior."""
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    component_columns = (
        "incoming_balance_log_lr",
        "outgoing_balance_log_lr",
        "outgoing_edge_norm_log_lr",
        "anti_fragment_log_lr",
        "selected_family_evidence_log",
        "context_negative_soft_penalty",
        "posterior_log_odds",
    )
    records: list[dict[str, object]] = []
    for component in component_columns:
        values = _numeric(rows, component)
        truth_values = values[truth]
        negative_values = values[~truth]
        rank_best, rank_worst, negative_above = _rank_bounds(values, roles)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "component": component,
                "truth_count": int(truth.sum()),
                "negative_count": int((~truth).sum()),
                "truth_min": float(truth_values.min()) if not truth_values.empty else math.nan,
                "truth_median": float(truth_values.median())
                if not truth_values.empty
                else math.nan,
                "truth_max": float(truth_values.max()) if not truth_values.empty else math.nan,
                "negative_min": float(negative_values.min())
                if not negative_values.empty
                else math.nan,
                "negative_median": float(negative_values.median())
                if not negative_values.empty
                else math.nan,
                "negative_max": float(negative_values.max())
                if not negative_values.empty
                else math.nan,
                "truth_rank_best": rank_best,
                "truth_rank_worst": rank_worst,
                "negative_above_truth_min_count": negative_above,
            }
        )
    return pd.DataFrame.from_records(records, columns=COMPONENT_COLUMNS)


def summarize_bayesian_topology_rows(
    rows: pd.DataFrame,
    *,
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY,
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT,
) -> pd.DataFrame:
    """Summarize the posterior-style topology score."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "truth_recovery_count": 0,
                    "negative_count": 0,
                    "prior_truth_probability": float(prior_truth_probability),
                    "context_penalty_weight": float(context_penalty_weight),
                    "truth_posterior_log_odds_min": math.nan,
                    "truth_posterior_log_odds_median": math.nan,
                    "truth_posterior_log_odds_max": math.nan,
                    "negative_posterior_log_odds_max": math.nan,
                    "truth_rank_best": 0,
                    "truth_rank_worst": 0,
                    "truth_above_all_negatives_count": 0,
                    "negative_above_truth_min_count": 0,
                    "posterior_log_odds_margin": math.nan,
                    "diagnostic_status": "bayesian_topology_unavailable",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    scores = _numeric(rows, "posterior_log_odds")
    truth_scores = scores[truth]
    negative_scores = scores[~truth]
    rank_best, rank_worst, negative_above = _rank_bounds(scores, roles)
    negative_max = float(negative_scores.max()) if not negative_scores.empty else math.nan
    truth_min = float(truth_scores.min()) if not truth_scores.empty else math.nan
    truth_above_all = int((truth_scores > negative_max).sum()) if math.isfinite(negative_max) else 0
    margin = (
        truth_min - negative_max
        if math.isfinite(truth_min) and math.isfinite(negative_max)
        else math.nan
    )
    if not bool(truth.any()):
        status = "bayesian_topology_no_truth_support"
    elif truth_above_all == int(truth.sum()):
        status = "bayesian_topology_score_separates_focused_slice"
    elif truth_above_all:
        status = "bayesian_topology_score_partial_truth_separation"
    else:
        status = "bayesian_topology_score_not_separating"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "truth_recovery_count": int(truth.sum()),
                "negative_count": int((~truth).sum()),
                "prior_truth_probability": float(prior_truth_probability),
                "context_penalty_weight": float(context_penalty_weight),
                "truth_posterior_log_odds_min": truth_min,
                "truth_posterior_log_odds_median": float(truth_scores.median())
                if not truth_scores.empty
                else math.nan,
                "truth_posterior_log_odds_max": float(truth_scores.max())
                if not truth_scores.empty
                else math.nan,
                "negative_posterior_log_odds_max": negative_max,
                "truth_rank_best": rank_best,
                "truth_rank_worst": rank_worst,
                "truth_above_all_negatives_count": truth_above_all,
                "negative_above_truth_min_count": negative_above,
                "posterior_log_odds_margin": margin,
                "diagnostic_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_context_negative_bayesian_topology_law(
    config: OverlapContextNegativeBayesianTopologyLawConfig,
) -> dict[str, Path]:
    """Run the Bayesian topology diagnostic and write outputs."""
    topology_rows = pd.read_csv(config.topology_rows_path)
    rows = build_context_negative_bayesian_topology_rows(
        topology_rows,
        prior_truth_probability=float(config.prior_truth_probability),
        context_penalty_weight=float(config.context_penalty_weight),
    )
    component_summary = summarize_bayesian_topology_components(rows)
    summary = summarize_bayesian_topology_rows(
        rows,
        prior_truth_probability=float(config.prior_truth_probability),
        context_penalty_weight=float(config.context_penalty_weight),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    component_summary.to_csv(config.component_summary_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "topology_rows_path": str(config.topology_rows_path),
        "prior_truth_probability": float(config.prior_truth_probability),
        "context_penalty_weight": float(config.context_penalty_weight),
        "component_priors": {
            "balanced_topology_signal_beta": [6.0, 2.0],
            "background_topology_beta": [2.0, 5.0],
            "anti_fragment_unit": "1 / (1 + outgoing_fragment_risk_proxy_score)",
            "selected_family_evidence": "log1p(selected_family_log_bayes_factor_lower)",
            "context_penalty": "min(continuous_context_min_margin, 0) * weight",
        },
        "outputs": {
            "rows": str(config.rows_path),
            "component_summary": str(config.component_summary_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_bayesian_topology_law",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "component_summary": config.component_summary_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prior-truth-probability",
        type=float,
        default=DEFAULT_PRIOR_TRUTH_PROBABILITY,
    )
    parser.add_argument(
        "--context-penalty-weight",
        type=float,
        default=DEFAULT_CONTEXT_PENALTY_WEIGHT,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_context_negative_bayesian_topology_law(
        OverlapContextNegativeBayesianTopologyLawConfig(
            topology_rows_path=args.topology_rows_path,
            output_dir=args.output_dir,
            prior_truth_probability=float(args.prior_truth_probability),
            context_penalty_weight=float(args.context_penalty_weight),
        )
    )


if __name__ == "__main__":
    main()
