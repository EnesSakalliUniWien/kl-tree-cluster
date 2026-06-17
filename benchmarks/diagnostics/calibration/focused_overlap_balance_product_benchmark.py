"""Focused overlap-positive benchmark for the balance-product topology law.

This diagnostic creates a small row-level fixture with multiple
context-negative truth-recovery rows. It tests whether the candidate
income/outcome topology term

    incoming_branch_balance * outgoing_balance

generalizes beyond the original single-positive overlap row. It is not a
production calibration rule and it does not replace full data-generation
benchmarks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.overlap_conditional_topology_law_panel import (
    build_conditional_topology_law_rows,
    summarize_conditional_topology_components,
    summarize_conditional_topology_law_rows,
)
from benchmarks.diagnostics.calibration.overlap_context_negative_topology_conditioning import (
    DEFAULT_TOPOLOGY_METRICS,
    summarize_metric,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_focused_overlap_balance_product_not_calibration"
SCHEMA_VERSION = "focused_overlap_balance_product_benchmark/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "focused_overlap_balance_product_benchmark"
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "parent_id",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "bayesian_incidence_mode_status",
    "decision_class",
    "traversal_decision",
    "sibling_open",
    "selected_family_guard_blocked",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "blocking_components",
    "depth",
    "incoming_parent_depth",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "incoming_branch_balance",
    "outgoing_balance",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "branch_alignment_score",
    "metric_family_alignment_score",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "incoming_parent_context_margin",
    "incoming_parent_homogeneity_gain_min",
    "incoming_parent_subspace_consensus_jaccard_topk",
    "outgoing_homogeneity_gain_min",
    "outgoing_subspace_consensus_jaccard_topk",
    "outgoing_size_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "context_transition_delta",
    "subspace_transition_delta",
    "homogeneity_transition_delta",
    "neighborhood_scale",
    "neighborhood_scale_source",
    "distance_to_stopping_edge",
    "balance_product",
    "outgoing_balance_edge_product",
)

METRIC_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "truth_count",
    "negative_count",
    "finite_truth_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "high_direction_auc",
    "low_direction_auc",
    "truth_min",
    "truth_median",
    "truth_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_threshold",
    "zero_negative_direction",
    "zero_negative_truth_count",
    "zero_negative_truth_retention",
    "zero_negative_negative_count",
    "zero_negative_value_margin",
    "zero_negative_status",
)

DEFAULT_METRICS = (
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "balance_product",
    "outgoing_balance_edge_product",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
)


@dataclass(frozen=True)
class FocusedOverlapBalanceProductBenchmarkConfig:
    """Runtime contract for the focused balance-product benchmark."""

    output_dir: Path
    metrics: tuple[str, ...] = DEFAULT_METRICS
    min_truth_support_per_stratum: int = 2

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "focused_overlap_positive_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "focused_overlap_balance_metric_summary.csv"

    @property
    def conditional_law_rows_path(self) -> Path:
        return self.output_dir / "focused_overlap_conditional_law_rows.csv"

    @property
    def conditional_law_component_summary_path(self) -> Path:
        return self.output_dir / "focused_overlap_conditional_law_component_summary.csv"

    @property
    def conditional_law_summary_path(self) -> Path:
        return self.output_dir / "focused_overlap_conditional_law_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _row(
    *,
    case_id: str,
    data_role: str,
    replicate: int,
    node_id: str,
    parent_id: str,
    guard_truth_role: str,
    incoming: float,
    outgoing: float,
    edge_norm: float,
    fragment: float,
    selected: float,
    context: float,
    subspace: float,
) -> dict[str, object]:
    topology_signal_role = "signal" if guard_truth_role == "truth_recovery" else ""
    if guard_truth_role == "truth_recovery":
        topology_support_role = ""
    elif data_role == "selected_null" or guard_truth_role == "null_like":
        topology_support_role = "strict_null"
    else:
        topology_support_role = "selected_nonnull"
    n_node = 200
    n_left = int(round(float(outgoing) * n_node))
    n_right = n_node - n_left
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": data_role,
        "replicate": int(replicate),
        "node_id": node_id,
        "parent_id": parent_id,
        "guard_truth_role": guard_truth_role,
        "topology_support_role": topology_support_role,
        "topology_signal_role": topology_signal_role,
        "bayesian_incidence_mode_status": "context_negative_emergent_mode_ambiguous",
        "decision_class": "accepted_internal_split",
        "traversal_decision": "split",
        "sibling_open": True,
        "selected_family_guard_blocked": False,
        "soft_structure_pass": True,
        "default_internal_node_candidate": False,
        "blocking_components": "local_context_margin",
        "depth": 1,
        "incoming_parent_depth": 0,
        "n_parent_context": 400,
        "n_node": n_node,
        "n_incoming_sibling": 200,
        "n_left": n_left,
        "n_right": n_right,
        "incoming_branch_balance": float(incoming),
        "outgoing_balance": float(outgoing),
        "selected_family_log_bayes_factor_lower": float(selected),
        "continuous_context_min_margin": float(context),
        "branch_alignment_score": 0.10,
        "metric_family_alignment_score": 0.10,
        "subspace_consensus_jaccard_topk": float(subspace),
        "size_balance": float(outgoing),
        "edge_norm_balance": float(edge_norm),
        "fragment_risk_proxy_score": float(fragment),
        "incoming_parent_context_margin": -0.015,
        "incoming_parent_homogeneity_gain_min": 0.002,
        "incoming_parent_subspace_consensus_jaccard_topk": 0.30,
        "outgoing_homogeneity_gain_min": 0.005,
        "outgoing_subspace_consensus_jaccard_topk": float(subspace),
        "outgoing_size_balance": float(outgoing),
        "outgoing_edge_norm_balance": float(edge_norm),
        "outgoing_fragment_risk_proxy_score": float(fragment),
        "context_transition_delta": 0.009,
        "subspace_transition_delta": 0.08 if guard_truth_role == "truth_recovery" else -0.08,
        "homogeneity_transition_delta": 0.002,
        "neighborhood_scale": 8.0 + float(replicate),
        "neighborhood_scale_source": "focused_fixture_projection_dimension",
        "distance_to_stopping_edge": 1.0,
        "balance_product": float(incoming) * float(outgoing),
        "outgoing_balance_edge_product": float(outgoing) * float(edge_norm),
    }


def build_focused_overlap_positive_rows() -> pd.DataFrame:
    """Return multiple focused positives plus hard overlap negatives."""
    rows = [
        _row(
            case_id="focused_positive_balanced",
            data_role="signal",
            replicate=0,
            node_id="T0",
            parent_id="P0",
            guard_truth_role="truth_recovery",
            incoming=0.4725,
            outgoing=0.492891,
            edge_norm=0.971963,
            fragment=0.571760,
            selected=18.6,
            context=-0.0057,
            subspace=0.41,
        ),
        _row(
            case_id="focused_positive_income_dominant",
            data_role="signal",
            replicate=1,
            node_id="T1",
            parent_id="P1",
            guard_truth_role="truth_recovery",
            incoming=0.4920,
            outgoing=0.4600,
            edge_norm=0.9600,
            fragment=0.5900,
            selected=14.0,
            context=-0.0060,
            subspace=0.36,
        ),
        _row(
            case_id="focused_positive_balanced_low_outgoing",
            data_role="signal",
            replicate=2,
            node_id="T2",
            parent_id="P2",
            guard_truth_role="truth_recovery",
            incoming=0.4850,
            outgoing=0.4680,
            edge_norm=0.9550,
            fragment=0.6000,
            selected=12.5,
            context=-0.0062,
            subspace=0.34,
        ),
        _row(
            case_id="focused_negative_high_outgoing_weak_income",
            data_role="signal",
            replicate=3,
            node_id="N0",
            parent_id="P0",
            guard_truth_role="diffuse_or_wrong",
            incoming=0.4250,
            outgoing=0.491304,
            edge_norm=0.965812,
            fragment=0.565164,
            selected=12.8,
            context=-0.0052,
            subspace=0.50,
        ),
        _row(
            case_id="focused_negative_high_income_weak_outgoing",
            data_role="signal",
            replicate=4,
            node_id="N1",
            parent_id="P1",
            guard_truth_role="diffuse_or_wrong",
            incoming=0.4900,
            outgoing=0.4300,
            edge_norm=0.9300,
            fragment=0.6200,
            selected=9.0,
            context=-0.0048,
            subspace=0.30,
        ),
        _row(
            case_id="focused_null_structural_block",
            data_role="selected_null",
            replicate=5,
            node_id="N2",
            parent_id="P2",
            guard_truth_role="null_like",
            incoming=0.4550,
            outgoing=0.4500,
            edge_norm=0.9000,
            fragment=0.7200,
            selected=7.0,
            context=-0.0040,
            subspace=0.25,
        ),
        _row(
            case_id="focused_fragment_high_edge_low_product",
            data_role="signal",
            replicate=6,
            node_id="N3",
            parent_id="P3",
            guard_truth_role="fragment_like",
            incoming=0.4100,
            outgoing=0.4800,
            edge_norm=0.9720,
            fragment=1.1000,
            selected=10.0,
            context=-0.0045,
            subspace=0.45,
        ),
    ]
    return pd.DataFrame.from_records(rows, columns=ROW_COLUMNS)


def summarize_focused_overlap_metrics(
    rows: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Summarize focused topology metrics using the existing separator audit."""
    unknown = sorted(set(metrics) - (set(DEFAULT_TOPOLOGY_METRICS) | set(ROW_COLUMNS)))
    if unknown:
        raise ValueError(f"Unknown focused overlap metrics: {unknown!r}")
    records = []
    for metric in metrics:
        record = summarize_metric(rows, metric=metric)
        record["schema_version"] = SCHEMA_VERSION
        record["study_role"] = STUDY_ROLE
        records.append(record)
    return pd.DataFrame.from_records(records, columns=METRIC_SUMMARY_COLUMNS)


def run_focused_overlap_balance_product_benchmark(
    config: FocusedOverlapBalanceProductBenchmarkConfig,
) -> dict[str, Path]:
    """Run the focused balance-product benchmark and write outputs."""
    rows = build_focused_overlap_positive_rows()
    metric_summary = summarize_focused_overlap_metrics(rows, metrics=config.metrics)
    conditional_rows = build_conditional_topology_law_rows(
        rows,
        min_truth_support_per_stratum=int(config.min_truth_support_per_stratum),
    )
    component_summary = summarize_conditional_topology_components(conditional_rows)
    law_summary = summarize_conditional_topology_law_rows(
        conditional_rows,
        min_truth_support_per_stratum=int(config.min_truth_support_per_stratum),
    )
    balance = metric_summary.loc[metric_summary["metric"].eq("balance_product")]
    balance_status = (
        str(balance["zero_negative_status"].iloc[0])
        if not balance.empty
        else "balance_product_missing"
    )
    diagnostic_status = (
        "balance_product_generalizes_on_focused_positives"
        if balance_status == "zero_negative_separates_all_truth"
        else "balance_product_not_validated_on_focused_positives"
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    metric_summary.to_csv(config.metric_summary_path, index=False)
    conditional_rows.to_csv(config.conditional_law_rows_path, index=False)
    component_summary.to_csv(
        config.conditional_law_component_summary_path,
        index=False,
    )
    law_summary.to_csv(config.conditional_law_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "metrics": list(config.metrics),
        "min_truth_support_per_stratum": int(config.min_truth_support_per_stratum),
        "diagnostic_status": diagnostic_status,
        "production_status": "diagnostic_only_focused_row_level_benchmark",
        "outputs": {
            "rows": str(config.rows_path),
            "metric_summary": str(config.metric_summary_path),
            "conditional_law_rows": str(config.conditional_law_rows_path),
            "conditional_law_component_summary": str(
                config.conditional_law_component_summary_path
            ),
            "conditional_law_summary": str(config.conditional_law_summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "metric_summary": config.metric_summary_path,
        "conditional_law_rows": config.conditional_law_rows_path,
        "conditional_law_component_summary": config.conditional_law_component_summary_path,
        "conditional_law_summary": config.conditional_law_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated focused metrics.",
    )
    parser.add_argument("--min-truth-support-per-stratum", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = tuple(token.strip() for token in str(args.metrics).split(",") if token.strip())
    run_focused_overlap_balance_product_benchmark(
        FocusedOverlapBalanceProductBenchmarkConfig(
            output_dir=args.output_dir,
            metrics=metrics,
            min_truth_support_per_stratum=int(args.min_truth_support_per_stratum),
        )
    )


if __name__ == "__main__":
    main()
