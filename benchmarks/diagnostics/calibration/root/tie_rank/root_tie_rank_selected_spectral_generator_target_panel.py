"""Generator targets for the selected root spectral-excess law.

The selected spectral-excess panel localized the hard overlap roots to a
specific missing object: selected spectral excess under measured high
action-edge and high tie-rank conditioning. This panel turns that diagnosis
into generator targets. For each observed root and proposal family, it asks:

* does the family produce roots in the measured high action-edge/tie stratum?
* does any such root reach the observed selected spectral excess?
* if not, how much spectral lift is still required before calibration can even
  be attempted?

The output remains diagnostic. A spectral-lift target is not a p-value and is
not a traversal rescue rule.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import (
    print_diagnostic_output_paths,
    write_diagnostic_bundle,
)
from benchmarks.diagnostics.calibration.root.root_tail_values import (
    finite_float,
    finite_int,
    is_calibration_support,
    is_observed_target,
    require_columns,
    safe_log1p,
    spectral_excess_log,
    string_value,
)

SCHEMA_VERSION = "root_tie_rank_selected_spectral_generator_target_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_selected_spectral_generator_targets_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_selected_spectral_generator_target_panel"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_null_proposal_frontier_with_generated_replay"
    / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_selected_spectral_generator_target_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_selected_spectral_generator_target_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "proposal_family",
    "proposal_role",
    "target_spectral_excess_log",
    "target_action_edge_bottleneck",
    "target_tie_fraction",
    "target_bandwidth_band",
    "eligible_generated_count",
    "eligible_calibration_support_count",
    "best_generated_case_id",
    "best_support_role",
    "best_generated_spectral_excess_log",
    "best_generated_action_edge_bottleneck",
    "best_generated_tie_fraction",
    "best_generated_bandwidth_band",
    "spectral_lift_log_required",
    "spectral_lift_multiplier_required",
    "spectral_reach_after_current_generator",
    "conditioning_stratum_status",
    "generator_target_status",
    "minimum_null_roots_for_alpha_resolution",
    "minimum_null_roots_for_tail_precision",
    "additional_null_roots_for_alpha_resolution",
    "additional_null_roots_for_tail_precision",
    "next_generator_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "proposal_role",
    "target_count",
    "conditioning_stratum_target_count",
    "calibration_supported_target_count",
    "spectral_reach_target_count",
    "diagnostic_only_target_count",
    "median_spectral_lift_log_required",
    "max_spectral_lift_log_required",
    "median_spectral_lift_multiplier_required",
    "max_spectral_lift_multiplier_required",
    "minimum_null_roots_for_alpha_resolution_total",
    "minimum_null_roots_for_tail_precision_total",
    "additional_null_roots_for_alpha_resolution_total",
    "additional_null_roots_for_tail_precision_total",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankSelectedSpectralGeneratorTargetConfig:
    """Input/output contract for selected spectral generator targets."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS
    min_action_edge_fraction: float = 0.95
    min_tie_fraction_floor: float = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-feasibility-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    parser.add_argument("--min-action-edge-fraction", type=float, default=0.95)
    parser.add_argument("--min-tie-fraction-floor", type=float, default=0.70)
    return parser.parse_args()


def _action_edge_bottleneck(row: pd.Series) -> float:
    tie = finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    if not (math.isfinite(tie) and math.isfinite(action) and math.isfinite(edge)):
        return math.nan
    return float(tie * min(action, edge))


def _is_observed_target(row: pd.Series) -> bool:
    return is_observed_target(row)


def _support_role(row: pd.Series) -> str:
    if is_calibration_support(row):
        return "selected_null_candidate_support"
    return "diagnostic_proposal_not_calibration"


def _proposal_role(family_rows: pd.DataFrame) -> str:
    if family_rows.empty:
        return "no_generated_rows"
    support = family_rows.apply(is_calibration_support, axis=1)
    if bool(support.all()):
        return "selected_null_candidate_support"
    if bool(support.any()):
        return "mixed_support_diagnostic_and_calibration"
    return "diagnostic_proposal_not_calibration"


def _is_neighborhood_measured(row: pd.Series) -> bool:
    band = string_value(row, "root_bandwidth_reopen_band")
    return bool(band) and band != "bandwidth_reopen_missing"


def _best_spectral_row(eligible: pd.DataFrame) -> pd.Series | None:
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        [
            "_spectral_excess_log",
            "_action_edge_bottleneck",
            "_tie_fraction",
            "case_id",
        ],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def _eligible_family_rows(
    *,
    target: pd.Series,
    family_rows: pd.DataFrame,
    min_action_edge_fraction: float,
    min_tie_fraction_floor: float,
) -> pd.DataFrame:
    target_bottleneck = _action_edge_bottleneck(target)
    if not math.isfinite(target_bottleneck):
        return family_rows.iloc[0:0].copy()
    rows = family_rows.copy()
    if "conditioning_target_case_id" in rows.columns:
        target_id = string_value(target, "case_id")
        conditioning_target = rows["conditioning_target_case_id"].fillna("")
        rows = rows.loc[
            conditioning_target.astype(str).eq("") | conditioning_target.astype(str).eq(target_id)
        ].copy()
    rows["_spectral_excess_log"] = rows["root_selected_eigenvalue_over_mp_upper_bound"].map(
        spectral_excess_log
    )
    rows["_action_edge_bottleneck"] = rows.apply(_action_edge_bottleneck, axis=1)
    rows["_tie_fraction"] = pd.to_numeric(
        rows["root_tie_rank_median_fraction"],
        errors="coerce",
    )
    rows["_neighborhood_measured"] = rows.apply(_is_neighborhood_measured, axis=1)
    threshold = float(min_action_edge_fraction) * target_bottleneck
    return rows.loc[
        rows["_neighborhood_measured"].astype(bool)
        & rows["_action_edge_bottleneck"].ge(threshold)
        & rows["_tie_fraction"].ge(float(min_tie_fraction_floor))
    ].copy()


def _conditioning_stratum_status(
    *,
    eligible_count: int,
    calibration_support_count: int,
) -> str:
    if int(eligible_count) <= 0:
        return "family_high_action_edge_tie_measured_stratum_missing"
    if int(calibration_support_count) <= 0:
        return "family_stratum_diagnostic_only_external_null_missing"
    return "family_stratum_has_selected_null_support"


def _generator_target_status(
    *,
    eligible_count: int,
    calibration_support_count: int,
    best_support_role: str,
    reached: bool,
) -> str:
    if int(eligible_count) <= 0:
        return "conditioning_stratum_missing_for_family"
    if reached and best_support_role == "selected_null_candidate_support":
        return "calibration_generator_reaches_spectral_target"
    if reached:
        return "diagnostic_generator_reaches_spectral_target_not_calibration"
    if int(calibration_support_count) > 0:
        return "calibration_generator_needs_spectral_lift"
    return "diagnostic_generator_needs_spectral_lift_and_external_null_support"


def _next_generator_step(status: str) -> str:
    if status == "conditioning_stratum_missing_for_family":
        return "generate_high_action_edge_tie_measured_roots_for_family"
    if status == "calibration_generator_reaches_spectral_target":
        return "estimate_selected_spectral_tail_in_calibration_stratum"
    if status == "diagnostic_generator_reaches_spectral_target_not_calibration":
        return "convert_reaching_family_to_external_selected_null_or_reject"
    if status == "calibration_generator_needs_spectral_lift":
        return "increase_spectral_excess_inside_selected_null_generator"
    return "derive_spectral_lift_generator_with_external_null_semantics"


def _generator_target_record(
    *,
    target: pd.Series,
    family: str,
    family_rows: pd.DataFrame,
    min_action_edge_fraction: float,
    min_tie_fraction_floor: float,
) -> dict[str, object]:
    target_spectral = spectral_excess_log(
        target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    target_bottleneck = _action_edge_bottleneck(target)
    target_tie = finite_float(target.get("root_tie_rank_median_fraction", math.nan))
    target_band = string_value(target, "root_bandwidth_reopen_band")
    eligible = _eligible_family_rows(
        target=target,
        family_rows=family_rows,
        min_action_edge_fraction=float(min_action_edge_fraction),
        min_tie_fraction_floor=float(min_tie_fraction_floor),
    )
    eligible_count = int(eligible.shape[0])
    calibration_mask = (
        eligible.apply(is_calibration_support, axis=1) if eligible_count else pd.Series(dtype=bool)
    )
    calibration_count = int(calibration_mask.sum()) if eligible_count else 0
    best = _best_spectral_row(eligible)
    if best is None:
        best_case_id = ""
        best_support_role = ""
        best_spectral = math.nan
        best_bottleneck = math.nan
        best_tie = math.nan
        best_band = ""
    else:
        best_case_id = string_value(best, "case_id")
        best_support_role = _support_role(best)
        best_spectral = finite_float(best.get("_spectral_excess_log", math.nan))
        best_bottleneck = finite_float(best.get("_action_edge_bottleneck", math.nan))
        best_tie = finite_float(best.get("_tie_fraction", math.nan))
        best_band = string_value(best, "root_bandwidth_reopen_band")
    if math.isfinite(target_spectral) and math.isfinite(best_spectral):
        lift_log = float(max(target_spectral - best_spectral, 0.0))
    else:
        lift_log = math.nan
    reached = math.isfinite(lift_log) and lift_log <= 0.0
    status = _generator_target_status(
        eligible_count=eligible_count,
        calibration_support_count=calibration_count,
        best_support_role=best_support_role,
        reached=reached,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": string_value(target, "case_id"),
        "proposal_family": family,
        "proposal_role": _proposal_role(family_rows),
        "target_spectral_excess_log": target_spectral,
        "target_action_edge_bottleneck": target_bottleneck,
        "target_tie_fraction": target_tie,
        "target_bandwidth_band": target_band,
        "eligible_generated_count": eligible_count,
        "eligible_calibration_support_count": calibration_count,
        "best_generated_case_id": best_case_id,
        "best_support_role": best_support_role,
        "best_generated_spectral_excess_log": best_spectral,
        "best_generated_action_edge_bottleneck": best_bottleneck,
        "best_generated_tie_fraction": best_tie,
        "best_generated_bandwidth_band": best_band,
        "spectral_lift_log_required": lift_log,
        "spectral_lift_multiplier_required": float(math.exp(lift_log))
        if math.isfinite(lift_log)
        else math.nan,
        "spectral_reach_after_current_generator": bool(reached),
        "conditioning_stratum_status": _conditioning_stratum_status(
            eligible_count=eligible_count,
            calibration_support_count=calibration_count,
        ),
        "generator_target_status": status,
        "minimum_null_roots_for_alpha_resolution": finite_int(
            target.get("alpha_resolution_required_null_count", 0)
        ),
        "minimum_null_roots_for_tail_precision": finite_int(
            target.get("tail_precision_required_null_count", 0)
        ),
        "additional_null_roots_for_alpha_resolution": finite_int(
            target.get("additional_null_count_for_alpha_resolution", 0)
        ),
        "additional_null_roots_for_tail_precision": finite_int(
            target.get("additional_null_count_for_tail_precision", 0)
        ),
        "next_generator_step": _next_generator_step(status),
    }


def build_selected_spectral_generator_target_rows(
    combined_feasibility_rows: pd.DataFrame,
    *,
    min_action_edge_fraction: float = 0.95,
    min_tie_fraction_floor: float = 0.70,
) -> pd.DataFrame:
    """Return target-by-family spectral generator requirements."""
    require_columns(
        combined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_bandwidth_reopen_band",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "combined feasibility rows",
    )
    rows = combined_feasibility_rows.copy()
    target_mask = rows.apply(_is_observed_target, axis=1)
    targets = rows[target_mask].copy()
    generated = rows[~target_mask].copy()
    families = sorted(generated["proposal_family"].dropna().astype(str).unique())
    records = []
    for _, target in targets.sort_values("case_id").iterrows():
        for family in families:
            family_rows = generated[generated["proposal_family"].astype(str).eq(family)].copy()
            records.append(
                _generator_target_record(
                    target=target,
                    family=family,
                    family_rows=family_rows,
                    min_action_edge_fraction=float(min_action_edge_fraction),
                    min_tie_fraction_floor=float(min_tie_fraction_floor),
                )
            )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.median()) if not finite.empty else math.nan


def _safe_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.max()) if not finite.empty else math.nan


def _family_summary_status(group: pd.DataFrame) -> str:
    covered = int(group["eligible_generated_count"].gt(0).sum())
    calibration = int(group["eligible_calibration_support_count"].gt(0).sum())
    reached = int(group["spectral_reach_after_current_generator"].sum())
    if covered <= 0:
        return "no_high_action_edge_tie_measured_generator_targets"
    if calibration > 0 and reached > 0:
        return "calibration_family_has_reaching_spectral_support"
    if reached > 0:
        return "diagnostic_family_reaches_some_targets_external_null_missing"
    if calibration > 0:
        return "calibration_family_requires_spectral_lift"
    if covered == int(group.shape[0]):
        return "diagnostic_family_requires_spectral_lift_all_targets"
    return "diagnostic_family_requires_spectral_lift_partial_targets"


def summarize_selected_spectral_generator_target_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize spectral generator requirements by proposal family."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records = []
    for family, group in rows.groupby("proposal_family", sort=True):
        covered = int(group["eligible_generated_count"].gt(0).sum())
        calibration_supported = int(group["eligible_calibration_support_count"].gt(0).sum())
        reached = int(group["spectral_reach_after_current_generator"].sum())
        diagnostic_only = int(
            group["conditioning_stratum_status"]
            .astype(str)
            .eq("family_stratum_diagnostic_only_external_null_missing")
            .sum()
        )
        uncovered = group[group["eligible_generated_count"].gt(0)]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": family,
                "proposal_role": str(group["proposal_role"].iloc[0]),
                "target_count": int(group.shape[0]),
                "conditioning_stratum_target_count": covered,
                "calibration_supported_target_count": calibration_supported,
                "spectral_reach_target_count": reached,
                "diagnostic_only_target_count": diagnostic_only,
                "median_spectral_lift_log_required": _safe_median(
                    uncovered["spectral_lift_log_required"]
                ),
                "max_spectral_lift_log_required": _safe_max(
                    uncovered["spectral_lift_log_required"]
                ),
                "median_spectral_lift_multiplier_required": _safe_median(
                    uncovered["spectral_lift_multiplier_required"]
                ),
                "max_spectral_lift_multiplier_required": _safe_max(
                    uncovered["spectral_lift_multiplier_required"]
                ),
                "minimum_null_roots_for_alpha_resolution_total": int(
                    group["minimum_null_roots_for_alpha_resolution"].sum()
                ),
                "minimum_null_roots_for_tail_precision_total": int(
                    group["minimum_null_roots_for_tail_precision"].sum()
                ),
                "additional_null_roots_for_alpha_resolution_total": int(
                    group["additional_null_roots_for_alpha_resolution"].sum()
                ),
                "additional_null_roots_for_tail_precision_total": int(
                    group["additional_null_roots_for_tail_precision"].sum()
                ),
                "summary_status": _family_summary_status(group),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_selected_spectral_generator_target_panel(
    config: RootTieRankSelectedSpectralGeneratorTargetConfig,
) -> dict[str, pd.DataFrame]:
    """Read feasibility rows and return generator target tables."""
    combined = pd.read_csv(config.proposal_feasibility_rows_path)
    rows = build_selected_spectral_generator_target_rows(
        combined,
        min_action_edge_fraction=float(config.min_action_edge_fraction),
        min_tie_fraction_floor=float(config.min_tie_fraction_floor),
    )
    summary = summarize_selected_spectral_generator_target_rows(rows)
    return {"rows": rows, "summary": summary}


def run_selected_spectral_generator_target_panel(
    config: RootTieRankSelectedSpectralGeneratorTargetConfig,
) -> dict[str, Path]:
    """Run the selected spectral generator target panel and write outputs."""
    tables = evaluate_selected_spectral_generator_target_panel(config)
    return write_diagnostic_bundle(
        output_dir=Path(config.output_dir),
        tables=tables,
        filenames={
            "rows": ROWS_OUTPUT,
            "summary": SUMMARY_OUTPUT,
        },
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_args()
    outputs = run_selected_spectral_generator_target_panel(
        RootTieRankSelectedSpectralGeneratorTargetConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
            min_action_edge_fraction=float(args.min_action_edge_fraction),
            min_tie_fraction_floor=float(args.min_tie_fraction_floor),
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
