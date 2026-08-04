"""Coupling-equation panel for root tie-rank proposal frontiers.

Continuous dominance checks show that proposal roots can separately create
large selected action or spectral excess without reproducing observed roots.
This panel makes the missing equation explicit by comparing observed targets
against generated rows in coupled spectral-action coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.reporting import (
    print_diagnostic_output_paths,
    write_diagnostic_bundle,
)
from benchmarks.diagnostics.calibration.root.tie_rank.cli import parse_proposal_panel_args
from benchmarks.diagnostics.calibration.root.tie_rank.comparison import (
    partition_target_and_generated_rows,
    row_bandwidth_gap_status,
)

SCHEMA_VERSION = "root_tie_rank_coupling_equation_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_coupling_equation_panel_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_coupling_equation_panel"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_null_proposal_frontier_two_case_smoke"
    / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_coupling_equation_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_coupling_equation_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "proposal_family",
    "best_generated_case_id",
    "target_tie_fraction",
    "generated_tie_fraction",
    "target_action_log",
    "generated_action_log",
    "target_edge_log",
    "generated_edge_log",
    "target_spectral_excess_log",
    "generated_spectral_excess_log",
    "target_action_edge_bottleneck",
    "generated_action_edge_bottleneck",
    "action_edge_bottleneck_deficit",
    "target_action_spectral_coupling",
    "generated_action_spectral_coupling",
    "action_spectral_coupling_deficit",
    "target_edge_spectral_coupling",
    "generated_edge_spectral_coupling",
    "edge_spectral_coupling_deficit",
    "target_bottleneck_coupling",
    "generated_bottleneck_coupling",
    "bottleneck_coupling_deficit",
    "bottleneck_coupling_ratio",
    "target_neighborhood_weight",
    "generated_neighborhood_weight",
    "target_measured_neighborhood_coupling",
    "generated_measured_neighborhood_coupling",
    "measured_neighborhood_coupling_deficit",
    "measured_neighborhood_coupling_ratio",
    "action_edge_bottleneck_dominates",
    "spectral_excess_dominates",
    "action_spectral_coupling_dominates",
    "edge_spectral_coupling_dominates",
    "bottleneck_coupling_dominates",
    "measured_neighborhood_coupling_dominates",
    "generated_neighborhood_measured",
    "bandwidth_measured_match",
    "bandwidth_gap_status",
    "coupling_deficit_score",
    "coupling_pattern",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "target_count",
    "bottleneck_coupling_dominance_count",
    "measured_neighborhood_coupling_dominance_count",
    "action_spectral_coupling_dominance_count",
    "edge_spectral_coupling_dominance_count",
    "action_edge_bottleneck_dominance_count",
    "spectral_excess_dominance_count",
    "generated_neighborhood_measured_count",
    "bandwidth_measured_match_count",
    "median_bottleneck_coupling_ratio",
    "max_bottleneck_coupling_ratio",
    "median_measured_neighborhood_coupling_ratio",
    "median_bottleneck_coupling_deficit",
    "median_measured_neighborhood_coupling_deficit",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankCouplingEquationConfig:
    """Input/output paths for the root tie-rank coupling panel."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _safe_log1p(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log1p(max(numeric, 0.0)))


def _spectral_excess_log(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(max(math.log(max(numeric, 1e-12)), 0.0))


def _deficit(*, target: float, generated: float, missing_value: float = 10.0) -> float:
    if not (math.isfinite(target) and math.isfinite(generated)):
        return float(missing_value)
    return float(max(target - generated, 0.0))


def _ratio(*, target: float, generated: float) -> float:
    if not (math.isfinite(target) and math.isfinite(generated)):
        return math.nan
    if target <= 0.0:
        return 1.0 if generated <= 0.0 else math.inf
    return float(generated / target)


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _neighborhood_weight(row: pd.Series) -> float:
    bandwidth_band = _string_value(row, "root_bandwidth_reopen_band")
    if bandwidth_band == "bandwidth_reopen_missing" or not bandwidth_band:
        return 0.0
    return 1.0


def _coupling_values(row: pd.Series) -> dict[str, float]:
    tie = _finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = _safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = _safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    spectral = _spectral_excess_log(
        row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    neighborhood = _neighborhood_weight(row)
    if not math.isfinite(tie):
        tie = math.nan
    action_edge_bottleneck = (
        float(tie * min(action, edge))
        if math.isfinite(tie) and math.isfinite(action) and math.isfinite(edge)
        else math.nan
    )
    action_spectral = (
        float(tie * action * spectral)
        if math.isfinite(tie) and math.isfinite(action) and math.isfinite(spectral)
        else math.nan
    )
    edge_spectral = (
        float(tie * edge * spectral)
        if math.isfinite(tie) and math.isfinite(edge) and math.isfinite(spectral)
        else math.nan
    )
    bottleneck = (
        float(action_edge_bottleneck * spectral)
        if math.isfinite(action_edge_bottleneck) and math.isfinite(spectral)
        else math.nan
    )
    measured = (
        float(neighborhood * bottleneck)
        if math.isfinite(neighborhood) and math.isfinite(bottleneck)
        else math.nan
    )
    return {
        "tie": tie,
        "action": action,
        "edge": edge,
        "spectral": spectral,
        "action_edge_bottleneck": action_edge_bottleneck,
        "action_spectral": action_spectral,
        "edge_spectral": edge_spectral,
        "bottleneck": bottleneck,
        "neighborhood": neighborhood,
        "measured": measured,
    }


def _coupling_pattern(
    *,
    measured_dominates: bool,
    bottleneck_dominates: bool,
    action_edge_dominates: bool,
    spectral_dominates: bool,
    action_spectral_dominates: bool,
    edge_spectral_dominates: bool,
    generated_neighborhood_measured: bool,
    bottleneck_ratio: float,
) -> str:
    if measured_dominates:
        return "measured_neighborhood_coupling_dominates"
    if bottleneck_dominates and not generated_neighborhood_measured:
        return "coupling_dominates_bandwidth_unmeasured"
    if bottleneck_dominates:
        return "bottleneck_coupling_dominates"
    if action_edge_dominates and not spectral_dominates:
        return "action_edge_high_spectral_low"
    if spectral_dominates and not action_edge_dominates:
        return "spectral_high_action_edge_low"
    if action_spectral_dominates and not edge_spectral_dominates:
        return "action_spectral_partial_edge_low"
    if edge_spectral_dominates and not action_spectral_dominates:
        return "edge_spectral_partial_action_low"
    if math.isfinite(bottleneck_ratio) and bottleneck_ratio >= 0.5:
        return "partial_coupling_match"
    return "low_coupling_all_axes"


def _coupling_record(target: pd.Series, generated: pd.Series) -> dict[str, object]:
    target_values = _coupling_values(target)
    generated_values = _coupling_values(generated)

    action_edge_deficit = _deficit(
        target=target_values["action_edge_bottleneck"],
        generated=generated_values["action_edge_bottleneck"],
    )
    action_spectral_deficit = _deficit(
        target=target_values["action_spectral"],
        generated=generated_values["action_spectral"],
    )
    edge_spectral_deficit = _deficit(
        target=target_values["edge_spectral"],
        generated=generated_values["edge_spectral"],
    )
    bottleneck_deficit = _deficit(
        target=target_values["bottleneck"],
        generated=generated_values["bottleneck"],
    )
    measured_deficit = _deficit(
        target=target_values["measured"],
        generated=generated_values["measured"],
    )
    spectral_deficit = _deficit(
        target=target_values["spectral"],
        generated=generated_values["spectral"],
    )
    bottleneck_ratio = _ratio(
        target=target_values["bottleneck"],
        generated=generated_values["bottleneck"],
    )
    measured_ratio = _ratio(
        target=target_values["measured"],
        generated=generated_values["measured"],
    )

    action_edge_dominates = action_edge_deficit <= 0.0
    spectral_dominates = spectral_deficit <= 0.0
    action_spectral_dominates = action_spectral_deficit <= 0.0
    edge_spectral_dominates = edge_spectral_deficit <= 0.0
    bottleneck_dominates = bottleneck_deficit <= 0.0
    measured_dominates = measured_deficit <= 0.0
    generated_neighborhood_measured = generated_values["neighborhood"] > 0.0
    bandwidth_status = row_bandwidth_gap_status(target, generated)
    bandwidth_match = bandwidth_status == "bandwidth_band_match"
    score = (
        min(bottleneck_deficit, 10.0)
        + min(measured_deficit, 10.0)
        + min(action_edge_deficit, 10.0)
        + min(spectral_deficit, 10.0)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": _string_value(target, "case_id"),
        "proposal_family": _string_value(generated, "proposal_family"),
        "best_generated_case_id": _string_value(generated, "case_id"),
        "target_tie_fraction": target_values["tie"],
        "generated_tie_fraction": generated_values["tie"],
        "target_action_log": target_values["action"],
        "generated_action_log": generated_values["action"],
        "target_edge_log": target_values["edge"],
        "generated_edge_log": generated_values["edge"],
        "target_spectral_excess_log": target_values["spectral"],
        "generated_spectral_excess_log": generated_values["spectral"],
        "target_action_edge_bottleneck": target_values["action_edge_bottleneck"],
        "generated_action_edge_bottleneck": generated_values["action_edge_bottleneck"],
        "action_edge_bottleneck_deficit": action_edge_deficit,
        "target_action_spectral_coupling": target_values["action_spectral"],
        "generated_action_spectral_coupling": generated_values["action_spectral"],
        "action_spectral_coupling_deficit": action_spectral_deficit,
        "target_edge_spectral_coupling": target_values["edge_spectral"],
        "generated_edge_spectral_coupling": generated_values["edge_spectral"],
        "edge_spectral_coupling_deficit": edge_spectral_deficit,
        "target_bottleneck_coupling": target_values["bottleneck"],
        "generated_bottleneck_coupling": generated_values["bottleneck"],
        "bottleneck_coupling_deficit": bottleneck_deficit,
        "bottleneck_coupling_ratio": bottleneck_ratio,
        "target_neighborhood_weight": target_values["neighborhood"],
        "generated_neighborhood_weight": generated_values["neighborhood"],
        "target_measured_neighborhood_coupling": target_values["measured"],
        "generated_measured_neighborhood_coupling": generated_values["measured"],
        "measured_neighborhood_coupling_deficit": measured_deficit,
        "measured_neighborhood_coupling_ratio": measured_ratio,
        "action_edge_bottleneck_dominates": action_edge_dominates,
        "spectral_excess_dominates": spectral_dominates,
        "action_spectral_coupling_dominates": action_spectral_dominates,
        "edge_spectral_coupling_dominates": edge_spectral_dominates,
        "bottleneck_coupling_dominates": bottleneck_dominates,
        "measured_neighborhood_coupling_dominates": measured_dominates,
        "generated_neighborhood_measured": generated_neighborhood_measured,
        "bandwidth_measured_match": bandwidth_match,
        "bandwidth_gap_status": bandwidth_status,
        "coupling_deficit_score": float(score),
        "coupling_pattern": _coupling_pattern(
            measured_dominates=measured_dominates,
            bottleneck_dominates=bottleneck_dominates,
            action_edge_dominates=action_edge_dominates,
            spectral_dominates=spectral_dominates,
            action_spectral_dominates=action_spectral_dominates,
            edge_spectral_dominates=edge_spectral_dominates,
            generated_neighborhood_measured=generated_neighborhood_measured,
            bottleneck_ratio=bottleneck_ratio,
        ),
    }


def build_root_tie_rank_coupling_equation_rows(
    combined_feasibility_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return best coupling-equation row for each target and proposal family."""
    targets, generated = partition_target_and_generated_rows(
        combined_feasibility_rows,
        required_columns={
            "case_id",
            "calibration_role",
            "proposal_family",
            "root_bandwidth_reopen_band",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        label="combined feasibility rows",
    )
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        for family, family_rows in generated.groupby("proposal_family", sort=True):
            candidate_records = [
                _coupling_record(target, generated_row)
                for _, generated_row in family_rows.iterrows()
            ]
            if not candidate_records:
                continue
            best = min(
                candidate_records,
                key=lambda record: (
                    float(record["bottleneck_coupling_deficit"]),
                    float(record["measured_neighborhood_coupling_deficit"]),
                    float(record["coupling_deficit_score"]),
                    str(record["best_generated_case_id"]),
                ),
            )
            best["proposal_family"] = str(family)
            records.append(best)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_tie_rank_coupling_equation_rows(
    coupling_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return family summary over coupling-equation rows."""
    if coupling_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for family, group in coupling_rows.groupby("proposal_family", sort=True):
        bottleneck_count = int(group["bottleneck_coupling_dominates"].sum())
        measured_count = int(group["measured_neighborhood_coupling_dominates"].sum())
        action_spectral_count = int(group["action_spectral_coupling_dominates"].sum())
        edge_spectral_count = int(group["edge_spectral_coupling_dominates"].sum())
        action_edge_count = int(group["action_edge_bottleneck_dominates"].sum())
        spectral_count = int(group["spectral_excess_dominates"].sum())
        generated_neighborhood_count = int(group["generated_neighborhood_measured"].sum())
        bandwidth_match_count = int(group["bandwidth_measured_match"].sum())
        if measured_count > 0:
            status = "measured_neighborhood_coupling_reached"
        elif bottleneck_count > 0 and generated_neighborhood_count == 0:
            status = "coupling_reached_but_bandwidth_unmeasured"
        elif bottleneck_count > 0:
            status = "bottleneck_coupling_reached"
        elif action_edge_count > 0 and spectral_count > 0:
            status = "separable_action_edge_and_spectral_no_coupling"
        elif action_edge_count > 0:
            status = "action_edge_bottleneck_spectral_deficit"
        elif spectral_count > 0:
            status = "spectral_excess_action_edge_deficit"
        else:
            status = "no_coupling_support"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": str(family),
                "target_count": int(group.shape[0]),
                "bottleneck_coupling_dominance_count": bottleneck_count,
                "measured_neighborhood_coupling_dominance_count": measured_count,
                "action_spectral_coupling_dominance_count": action_spectral_count,
                "edge_spectral_coupling_dominance_count": edge_spectral_count,
                "action_edge_bottleneck_dominance_count": action_edge_count,
                "spectral_excess_dominance_count": spectral_count,
                "generated_neighborhood_measured_count": generated_neighborhood_count,
                "bandwidth_measured_match_count": bandwidth_match_count,
                "median_bottleneck_coupling_ratio": float(
                    group["bottleneck_coupling_ratio"].median()
                ),
                "max_bottleneck_coupling_ratio": float(group["bottleneck_coupling_ratio"].max()),
                "median_measured_neighborhood_coupling_ratio": float(
                    group["measured_neighborhood_coupling_ratio"].median()
                ),
                "median_bottleneck_coupling_deficit": float(
                    group["bottleneck_coupling_deficit"].median()
                ),
                "median_measured_neighborhood_coupling_deficit": float(
                    group["measured_neighborhood_coupling_deficit"].median()
                ),
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_root_tie_rank_coupling_equation_panel(
    config: RootTieRankCouplingEquationConfig,
) -> dict[str, pd.DataFrame]:
    """Read proposal frontier rows and return coupling-equation tables."""
    combined = pd.read_csv(config.proposal_feasibility_rows_path)
    rows = build_root_tie_rank_coupling_equation_rows(combined)
    summary = summarize_root_tie_rank_coupling_equation_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_tie_rank_coupling_equation_panel(
    config: RootTieRankCouplingEquationConfig,
) -> dict[str, Path]:
    """Run the coupling-equation panel and write outputs."""
    tables = evaluate_root_tie_rank_coupling_equation_panel(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
        tables=tables,
        filenames={"rows": ROWS_OUTPUT, "summary": SUMMARY_OUTPUT},
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_proposal_panel_args(
        description=__doc__,
        default_rows_path=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    outputs = run_root_tie_rank_coupling_equation_panel(
        RootTieRankCouplingEquationConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
