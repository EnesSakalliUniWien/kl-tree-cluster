"""Sensitivity audit for the context-negative Bayesian topology law.

The Bayesian topology law identifies a vector-valued selected-neighborhood
conditioning variable. This panel checks whether that finding is robust to
component ablations and context-penalty settings, or whether it depends on one
hand-tuned component.

The audit is diagnostic-only. It is meant to distinguish structural topology
evidence from selected-family/context evidence; it does not estimate production
calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.overlap_context_negative_bayesian_topology_law import (
    build_context_negative_bayesian_topology_rows,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_context_negative_bayesian_topology_sensitivity"
SCHEMA_VERSION = "overlap_context_negative_bayesian_topology_sensitivity/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "overlap_context_negative_bayesian_topology_sensitivity"
)

COMPONENT_COLUMNS = {
    "incoming": "incoming_balance_log_lr",
    "outgoing": "outgoing_balance_log_lr",
    "edge_norm": "outgoing_edge_norm_log_lr",
    "anti_fragment": "anti_fragment_log_lr",
    "selected": "selected_family_evidence_log",
    "context": "context_negative_soft_penalty",
}

DEFAULT_CONTEXT_PENALTY_WEIGHTS = (0.0, 25.0, 50.0, 75.0, 100.0)

DEFAULT_COMPONENT_PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full_v1", ("incoming", "outgoing", "edge_norm", "anti_fragment", "selected", "context")),
    ("no_incoming_balance", ("outgoing", "edge_norm", "anti_fragment", "selected", "context")),
    ("no_outgoing_balance", ("incoming", "edge_norm", "anti_fragment", "selected", "context")),
    ("no_outgoing_edge_norm", ("incoming", "outgoing", "anti_fragment", "selected", "context")),
    ("no_anti_fragment", ("incoming", "outgoing", "edge_norm", "selected", "context")),
    ("no_selected_family", ("incoming", "outgoing", "edge_norm", "anti_fragment", "context")),
    ("no_context_penalty", ("incoming", "outgoing", "edge_norm", "anti_fragment", "selected")),
    ("topology_only", ("incoming", "outgoing", "edge_norm", "anti_fragment")),
    ("outgoing_topology_only", ("outgoing", "edge_norm", "anti_fragment")),
    ("balance_only", ("incoming", "outgoing")),
    ("outgoing_balance_only", ("outgoing",)),
    ("outgoing_edge_norm_only", ("edge_norm",)),
    ("selected_context_only", ("selected", "context")),
)

SENSITIVITY_COLUMNS = (
    "schema_version",
    "study_role",
    "context_penalty_weight",
    "profile",
    "components",
    "row_count",
    "truth_recovery_count",
    "negative_count",
    "truth_score_min",
    "truth_score_median",
    "truth_score_max",
    "negative_score_max",
    "truth_rank_best",
    "truth_rank_worst",
    "truth_above_all_negatives_count",
    "negative_above_truth_min_count",
    "score_margin",
    "profile_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "context_penalty_weight_count",
    "profile_count",
    "separating_profile_count",
    "top_rank_profile_count",
    "single_component_separating_count",
    "topology_only_separates_count",
    "outgoing_topology_only_separates_count",
    "selected_context_only_separates_count",
    "minimum_full_profile_margin",
    "minimum_topology_only_margin",
    "minimum_outgoing_topology_only_margin",
    "diagnostic_status",
)


@dataclass(frozen=True)
class OverlapContextNegativeBayesianTopologySensitivityConfig:
    """Runtime contract for Bayesian topology sensitivity diagnostics."""

    topology_rows_path: Path
    output_dir: Path
    context_penalty_weights: tuple[float, ...] = DEFAULT_CONTEXT_PENALTY_WEIGHTS

    @property
    def sensitivity_rows_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_context_negative_bayesian_topology_sensitivity_rows.csv"
        )

    @property
    def summary_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_context_negative_bayesian_topology_sensitivity_summary.csv"
        )

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("At least one context penalty weight is required.")
    return values


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan


def _profile_status(
    *,
    truth_count: int,
    truth_above_all_negatives_count: int,
    truth_rank_best: int,
) -> str:
    if truth_count == 0:
        return "sensitivity_no_truth_support"
    if truth_above_all_negatives_count == truth_count and truth_rank_best == 1:
        return "sensitivity_truth_top_rank_separates"
    if truth_above_all_negatives_count:
        return "sensitivity_truth_partial_separation"
    return "sensitivity_not_separating"


def _score_profile(rows: pd.DataFrame, components: Sequence[str]) -> pd.Series:
    columns = [COMPONENT_COLUMNS[component] for component in components]
    return rows[columns].sum(axis=1)


def _build_profile_record(
    *,
    rows: pd.DataFrame,
    score: pd.Series,
    context_penalty_weight: float,
    profile: str,
    components: Sequence[str],
) -> dict[str, object]:
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    truth_scores = score[truth]
    negative_scores = score[negative]
    ranks = score.rank(method="first", ascending=False).astype(int)
    truth_rank_best = int(ranks[truth].min()) if bool(truth.any()) else 0
    truth_rank_worst = int(ranks[truth].max()) if bool(truth.any()) else 0
    truth_min = float(truth_scores.min()) if not truth_scores.empty else math.nan
    negative_max = float(negative_scores.max()) if not negative_scores.empty else math.nan
    truth_above_all = (
        int((truth_scores > negative_max).sum()) if math.isfinite(negative_max) else 0
    )
    negative_above = int((negative_scores > truth_min).sum()) if math.isfinite(truth_min) else 0
    margin = truth_min - negative_max if math.isfinite(truth_min) and math.isfinite(negative_max) else math.nan
    truth_count = int(truth.sum())
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "context_penalty_weight": float(context_penalty_weight),
        "profile": str(profile),
        "components": ",".join(components),
        "row_count": int(rows.shape[0]),
        "truth_recovery_count": truth_count,
        "negative_count": int(negative.sum()),
        "truth_score_min": _safe_float(truth_min),
        "truth_score_median": float(truth_scores.median()) if not truth_scores.empty else math.nan,
        "truth_score_max": float(truth_scores.max()) if not truth_scores.empty else math.nan,
        "negative_score_max": _safe_float(negative_max),
        "truth_rank_best": truth_rank_best,
        "truth_rank_worst": truth_rank_worst,
        "truth_above_all_negatives_count": truth_above_all,
        "negative_above_truth_min_count": negative_above,
        "score_margin": _safe_float(margin),
        "profile_status": _profile_status(
            truth_count=truth_count,
            truth_above_all_negatives_count=truth_above_all,
            truth_rank_best=truth_rank_best,
        ),
    }


def build_bayesian_topology_sensitivity_rows(
    topology_rows: pd.DataFrame,
    *,
    context_penalty_weights: Sequence[float] = DEFAULT_CONTEXT_PENALTY_WEIGHTS,
) -> pd.DataFrame:
    """Build ablation/profile sensitivity rows for topology-law components."""
    records: list[dict[str, object]] = []
    for weight in context_penalty_weights:
        component_rows = build_context_negative_bayesian_topology_rows(
            topology_rows,
            context_penalty_weight=float(weight),
        )
        for profile, components in DEFAULT_COMPONENT_PROFILES:
            score = _score_profile(component_rows, components)
            records.append(
                _build_profile_record(
                    rows=component_rows,
                    score=score,
                    context_penalty_weight=float(weight),
                    profile=profile,
                    components=components,
                )
            )
    return pd.DataFrame.from_records(records, columns=SENSITIVITY_COLUMNS)


def _separates(rows: pd.DataFrame, profile: str) -> pd.Series:
    return rows["profile"].eq(profile) & rows["profile_status"].eq(
        "sensitivity_truth_top_rank_separates"
    )


def summarize_bayesian_topology_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize component-profile robustness."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "context_penalty_weight_count": 0,
                    "profile_count": 0,
                    "separating_profile_count": 0,
                    "top_rank_profile_count": 0,
                    "single_component_separating_count": 0,
                    "topology_only_separates_count": 0,
                    "outgoing_topology_only_separates_count": 0,
                    "selected_context_only_separates_count": 0,
                    "minimum_full_profile_margin": math.nan,
                    "minimum_topology_only_margin": math.nan,
                    "minimum_outgoing_topology_only_margin": math.nan,
                    "diagnostic_status": "bayesian_topology_sensitivity_unavailable",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
    separates = rows["profile_status"].eq("sensitivity_truth_top_rank_separates")
    top_rank = rows["truth_rank_best"].eq(1)
    single_component = rows["components"].str.contains(",", regex=False).eq(False)
    topology_only = _separates(rows, "topology_only")
    outgoing_topology_only = _separates(rows, "outgoing_topology_only")
    selected_context_only = _separates(rows, "selected_context_only")
    full_margins = rows.loc[rows["profile"].eq("full_v1"), "score_margin"]
    topology_margins = rows.loc[rows["profile"].eq("topology_only"), "score_margin"]
    outgoing_margins = rows.loc[
        rows["profile"].eq("outgoing_topology_only"),
        "score_margin",
    ]
    weight_count = int(rows["context_penalty_weight"].nunique())
    topology_count = int(topology_only.sum())
    outgoing_topology_count = int(outgoing_topology_only.sum())
    selected_context_count = int(selected_context_only.sum())
    if (
        topology_count == weight_count
        and outgoing_topology_count == weight_count
        and selected_context_count == 0
    ):
        status = "bayesian_topology_structural_signal_robust_not_selected_context"
    elif topology_count > 0 and selected_context_count == 0:
        status = "bayesian_topology_structural_signal_partial"
    elif selected_context_count > 0:
        status = "bayesian_topology_selected_context_confounded"
    else:
        status = "bayesian_topology_sensitivity_not_separating"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "context_penalty_weight_count": int(
                    rows["context_penalty_weight"].nunique()
                ),
                "profile_count": int(rows["profile"].nunique()),
                "separating_profile_count": int(separates.sum()),
                "top_rank_profile_count": int(top_rank.sum()),
                "single_component_separating_count": int(
                    (single_component & separates).sum()
                ),
                "topology_only_separates_count": int(topology_only.sum()),
                "outgoing_topology_only_separates_count": int(
                    outgoing_topology_only.sum()
                ),
                "selected_context_only_separates_count": int(
                    selected_context_only.sum()
                ),
                "minimum_full_profile_margin": float(full_margins.min())
                if not full_margins.empty
                else math.nan,
                "minimum_topology_only_margin": float(topology_margins.min())
                if not topology_margins.empty
                else math.nan,
                "minimum_outgoing_topology_only_margin": float(
                    outgoing_margins.min()
                )
                if not outgoing_margins.empty
                else math.nan,
                "diagnostic_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_context_negative_bayesian_topology_sensitivity(
    config: OverlapContextNegativeBayesianTopologySensitivityConfig,
) -> dict[str, Path]:
    """Run topology-law sensitivity diagnostics and write outputs."""
    topology_rows = pd.read_csv(config.topology_rows_path)
    rows = build_bayesian_topology_sensitivity_rows(
        topology_rows,
        context_penalty_weights=config.context_penalty_weights,
    )
    summary = summarize_bayesian_topology_sensitivity(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.sensitivity_rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "topology_rows_path": str(config.topology_rows_path),
        "context_penalty_weights": list(config.context_penalty_weights),
        "component_profiles": {
            profile: list(components)
            for profile, components in DEFAULT_COMPONENT_PROFILES
        },
        "outputs": {
            "sensitivity_rows": str(config.sensitivity_rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_bayesian_topology_sensitivity",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "sensitivity_rows": config.sensitivity_rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-penalty-weights",
        type=_parse_float_list,
        default=DEFAULT_CONTEXT_PENALTY_WEIGHTS,
        help="Comma-separated context penalty weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_context_negative_bayesian_topology_sensitivity(
        OverlapContextNegativeBayesianTopologySensitivityConfig(
            topology_rows_path=args.topology_rows_path,
            output_dir=args.output_dir,
            context_penalty_weights=tuple(args.context_penalty_weights),
        )
    )


if __name__ == "__main__":
    main()
