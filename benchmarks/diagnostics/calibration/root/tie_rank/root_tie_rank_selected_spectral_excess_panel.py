"""Selected spectral-excess diagnostic for root tie-rank coupling.

The measured coupling residual panel shows that hard overlap roots are not
blocked by action-edge mass once generated neighborhoods are measured. They are
blocked by selected spectral excess. This panel conditions generated roots on
measured bandwidth and high tie-weighted action-edge bottleneck, then asks
whether the remaining spectral excess has calibration support or only
diagnostic proposal support.
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

SCHEMA_VERSION = "root_tie_rank_selected_spectral_excess_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_selected_spectral_excess_panel_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_selected_spectral_excess_panel"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_null_proposal_frontier_with_generated_replay"
    / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_selected_spectral_excess_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_selected_spectral_excess_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

CALIBRATION_SUPPORT_ROLES = {
    "selected_null",
    "selected_null_candidate_support",
    "external_selected_null",
    "external_null_support",
    "calibration_null",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_spectral_excess_log",
    "target_action_edge_bottleneck",
    "eligible_generated_count",
    "eligible_calibration_support_count",
    "eligible_diagnostic_count",
    "spectral_exceedance_count",
    "spectral_calibration_exceedance_count",
    "empirical_conservative_spectral_tail_p_value",
    "best_generated_case_id",
    "best_proposal_family",
    "best_proposal_support_role",
    "best_generated_spectral_excess_log",
    "best_generated_action_edge_bottleneck",
    "best_generated_tie_fraction",
    "best_generated_bandwidth_band",
    "spectral_excess_log_deficit",
    "spectral_excess_log_ratio",
    "required_spectral_excess_multiplier",
    "conditioning_support_status",
    "spectral_excess_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "spectral_excess_status",
    "conditioning_support_status",
    "row_count",
    "target_reached_count",
    "calibration_supported_count",
    "diagnostic_best_count",
    "median_spectral_excess_log_ratio",
    "median_required_spectral_excess_multiplier",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankSelectedSpectralExcessConfig:
    """Input/output contract for selected spectral-excess diagnostics."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS
    min_action_edge_fraction: float = 0.95
    min_tie_fraction_floor: float = 0.70
    partial_spectral_ratio_floor: float = 0.50


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
    parser.add_argument("--partial-spectral-ratio-floor", type=float, default=0.50)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _string_value(
    row: pd.Series | dict[str, object],
    column: str,
    default: str = "",
) -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


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


def _action_edge_bottleneck(row: pd.Series) -> float:
    tie = _finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = _safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = _safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    if not (math.isfinite(tie) and math.isfinite(action) and math.isfinite(edge)):
        return math.nan
    return float(tie * min(action, edge))


def _is_observed_target(row: pd.Series) -> bool:
    family = _string_value(row, "proposal_family")
    role = _string_value(row, "calibration_role")
    data_role = _string_value(row, "data_role")
    return (
        family == "observed_target"
        or role == "observed_target_not_null_support"
        or data_role == "observed_target"
    )


def _is_calibration_support(row: pd.Series) -> bool:
    return (
        _string_value(row, "data_role") in CALIBRATION_SUPPORT_ROLES
        or _string_value(row, "calibration_role") in CALIBRATION_SUPPORT_ROLES
    )


def _is_neighborhood_measured(row: pd.Series) -> bool:
    band = _string_value(row, "root_bandwidth_reopen_band")
    return bool(band) and band != "bandwidth_reopen_missing"


def _support_role(row: pd.Series) -> str:
    if _is_calibration_support(row):
        return "selected_null_candidate_support"
    return "diagnostic_proposal_not_calibration"


def _spectral_ratio(generated_spectral: float, target_spectral: float) -> float:
    if not (math.isfinite(generated_spectral) and math.isfinite(target_spectral)):
        return math.nan
    if target_spectral <= 1e-12:
        return 1.0 if generated_spectral <= 1e-12 else math.inf
    return float(generated_spectral / target_spectral)


def _conservative_tail_p_value(exceedances: int, support: int) -> float:
    if int(support) <= 0:
        return math.nan
    return float((int(exceedances) + 1) / (int(support) + 1))


def _eligible_rows(
    *,
    target: pd.Series,
    generated: pd.DataFrame,
    min_action_edge_fraction: float,
    min_tie_fraction_floor: float,
) -> pd.DataFrame:
    target_bottleneck = _action_edge_bottleneck(target)
    if not math.isfinite(target_bottleneck):
        return generated.iloc[0:0].copy()
    rows = generated.copy()
    rows["_spectral_excess_log"] = rows["root_selected_eigenvalue_over_mp_upper_bound"].map(
        _spectral_excess_log
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


def _conditioning_support_status(
    *,
    eligible_count: int,
    calibration_support_count: int,
) -> str:
    if int(eligible_count) <= 0:
        return "high_action_edge_tie_measured_support_missing"
    if int(calibration_support_count) <= 0:
        return "diagnostic_only_support_external_null_missing"
    return "selected_null_support_observed_diagnostic_only"


def _spectral_status(
    *,
    best_reaches: bool,
    best_support_role: str,
    support_status: str,
    spectral_ratio: float,
    partial_floor: float,
) -> str:
    if support_status == "high_action_edge_tie_measured_support_missing":
        return "spectral_excess_conditioning_support_missing"
    if best_reaches:
        if best_support_role == "selected_null_candidate_support":
            return "spectral_excess_reached_by_calibration_candidate"
        return "spectral_excess_reached_by_diagnostic_proposal_not_calibration"
    if math.isfinite(spectral_ratio) and spectral_ratio >= float(partial_floor):
        return "spectral_excess_partial_residual"
    return "spectral_excess_hard_residual"


def _next_step(*, spectral_status: str, support_status: str) -> str:
    if spectral_status == "spectral_excess_reached_by_calibration_candidate":
        return "estimate_selected_spectral_excess_tail_in_null_stratum"
    if spectral_status == "spectral_excess_reached_by_diagnostic_proposal_not_calibration":
        return "convert_diagnostic_spectral_family_to_external_null_support_or_reject"
    if support_status == "high_action_edge_tie_measured_support_missing":
        return "generate_high_action_edge_tie_measured_selected_null_roots"
    if support_status == "diagnostic_only_support_external_null_missing":
        return "derive_external_selected_spectral_excess_law_for_diagnostic_stratum"
    return "increase_selected_spectral_excess_under_conditioning_stratum"


def _best_spectral_row(eligible: pd.DataFrame) -> pd.Series | None:
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        [
            "_spectral_excess_log",
            "_action_edge_bottleneck",
            "_tie_fraction",
            "proposal_family",
            "case_id",
        ],
        ascending=[False, False, False, True, True],
    )
    return ranked.iloc[0]


def _spectral_law_record(
    *,
    target: pd.Series,
    generated: pd.DataFrame,
    min_action_edge_fraction: float,
    min_tie_fraction_floor: float,
    partial_spectral_ratio_floor: float,
) -> dict[str, object]:
    target_id = _string_value(target, "case_id")
    target_spectral = _spectral_excess_log(
        target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    target_bottleneck = _action_edge_bottleneck(target)
    eligible = _eligible_rows(
        target=target,
        generated=generated,
        min_action_edge_fraction=float(min_action_edge_fraction),
        min_tie_fraction_floor=float(min_tie_fraction_floor),
    )
    eligible_count = int(eligible.shape[0])
    calibration_mask = (
        eligible.apply(_is_calibration_support, axis=1) if eligible_count else pd.Series(dtype=bool)
    )
    calibration_count = int(calibration_mask.sum()) if eligible_count else 0
    diagnostic_count = int(eligible_count - calibration_count)
    if eligible_count:
        exceedance_mask = eligible["_spectral_excess_log"].ge(target_spectral)
        exceedance_count = int(exceedance_mask.sum())
        calibration_exceedance_count = int(
            (exceedance_mask & calibration_mask.reindex(eligible.index, fill_value=False)).sum()
        )
    else:
        exceedance_count = 0
        calibration_exceedance_count = 0

    support_status = _conditioning_support_status(
        eligible_count=eligible_count,
        calibration_support_count=calibration_count,
    )
    best = _best_spectral_row(eligible)
    if best is None:
        best_spectral = math.nan
        best_case_id = ""
        best_family = ""
        best_support_role = ""
        best_bottleneck = math.nan
        best_tie = math.nan
        best_band = ""
    else:
        best_spectral = _finite_float(best.get("_spectral_excess_log", math.nan))
        best_case_id = _string_value(best, "case_id")
        best_family = _string_value(best, "proposal_family")
        best_support_role = _support_role(best)
        best_bottleneck = _finite_float(best.get("_action_edge_bottleneck", math.nan))
        best_tie = _finite_float(best.get("_tie_fraction", math.nan))
        best_band = _string_value(best, "root_bandwidth_reopen_band")
    spectral_deficit = (
        float(max(target_spectral - best_spectral, 0.0))
        if math.isfinite(target_spectral) and math.isfinite(best_spectral)
        else math.nan
    )
    spectral_ratio = _spectral_ratio(best_spectral, target_spectral)
    best_reaches = math.isfinite(spectral_deficit) and spectral_deficit <= 0.0
    spectral_status = _spectral_status(
        best_reaches=best_reaches,
        best_support_role=best_support_role,
        support_status=support_status,
        spectral_ratio=spectral_ratio,
        partial_floor=float(partial_spectral_ratio_floor),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": target_id,
        "target_spectral_excess_log": target_spectral,
        "target_action_edge_bottleneck": target_bottleneck,
        "eligible_generated_count": eligible_count,
        "eligible_calibration_support_count": calibration_count,
        "eligible_diagnostic_count": diagnostic_count,
        "spectral_exceedance_count": exceedance_count,
        "spectral_calibration_exceedance_count": calibration_exceedance_count,
        "empirical_conservative_spectral_tail_p_value": (
            _conservative_tail_p_value(calibration_exceedance_count, calibration_count)
        ),
        "best_generated_case_id": best_case_id,
        "best_proposal_family": best_family,
        "best_proposal_support_role": best_support_role,
        "best_generated_spectral_excess_log": best_spectral,
        "best_generated_action_edge_bottleneck": best_bottleneck,
        "best_generated_tie_fraction": best_tie,
        "best_generated_bandwidth_band": best_band,
        "spectral_excess_log_deficit": spectral_deficit,
        "spectral_excess_log_ratio": spectral_ratio,
        "required_spectral_excess_multiplier": (
            float(math.exp(spectral_deficit)) if math.isfinite(spectral_deficit) else math.nan
        ),
        "conditioning_support_status": support_status,
        "spectral_excess_status": spectral_status,
        "next_mathematical_step": _next_step(
            spectral_status=spectral_status,
            support_status=support_status,
        ),
    }


def build_selected_spectral_excess_rows(
    combined_feasibility_rows: pd.DataFrame,
    *,
    min_action_edge_fraction: float = 0.95,
    min_tie_fraction_floor: float = 0.70,
    partial_spectral_ratio_floor: float = 0.50,
) -> pd.DataFrame:
    """Return target-level selected spectral-excess support diagnostics."""
    _require_columns(
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
    records = [
        _spectral_law_record(
            target=target,
            generated=generated,
            min_action_edge_fraction=float(min_action_edge_fraction),
            min_tie_fraction_floor=float(min_tie_fraction_floor),
            partial_spectral_ratio_floor=float(partial_spectral_ratio_floor),
        )
        for _, target in targets.sort_values("case_id").iterrows()
    ]
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.median()) if not finite.empty else math.nan


def summarize_selected_spectral_excess_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected spectral-excess diagnostics."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(
        ["spectral_excess_status", "conditioning_support_status"],
        dropna=False,
        sort=True,
    ):
        target_reached = int(group["spectral_exceedance_count"].gt(0).sum())
        calibration_supported = int(group["eligible_calibration_support_count"].gt(0).sum())
        diagnostic_best = int(
            group["best_proposal_support_role"]
            .astype(str)
            .eq("diagnostic_proposal_not_calibration")
            .sum()
        )
        if str(keys[0]).endswith("not_calibration"):
            status = "diagnostic_spectral_reach_requires_external_null_support"
        elif str(keys[0]) == "spectral_excess_partial_residual":
            status = "partial_spectral_residual_requires_stronger_spectral_generator"
        elif str(keys[0]) == "spectral_excess_hard_residual":
            status = "hard_spectral_residual_requires_selected_spectral_law"
        elif str(keys[1]) == "high_action_edge_tie_measured_support_missing":
            status = "conditioning_support_missing"
        else:
            status = "selected_spectral_excess_diagnostic_only"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "spectral_excess_status": keys[0],
                "conditioning_support_status": keys[1],
                "row_count": int(group.shape[0]),
                "target_reached_count": target_reached,
                "calibration_supported_count": calibration_supported,
                "diagnostic_best_count": diagnostic_best,
                "median_spectral_excess_log_ratio": _safe_median(
                    group["spectral_excess_log_ratio"]
                ),
                "median_required_spectral_excess_multiplier": _safe_median(
                    group["required_spectral_excess_multiplier"]
                ),
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_selected_spectral_excess_panel(
    config: RootTieRankSelectedSpectralExcessConfig,
) -> dict[str, pd.DataFrame]:
    """Read feasibility rows and return selected spectral-excess tables."""
    combined = pd.read_csv(config.proposal_feasibility_rows_path)
    rows = build_selected_spectral_excess_rows(
        combined,
        min_action_edge_fraction=float(config.min_action_edge_fraction),
        min_tie_fraction_floor=float(config.min_tie_fraction_floor),
        partial_spectral_ratio_floor=float(config.partial_spectral_ratio_floor),
    )
    summary = summarize_selected_spectral_excess_rows(rows)
    return {"rows": rows, "summary": summary}


def run_selected_spectral_excess_panel(
    config: RootTieRankSelectedSpectralExcessConfig,
) -> dict[str, Path]:
    """Run the selected spectral-excess panel and write outputs."""
    tables = evaluate_selected_spectral_excess_panel(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
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
    outputs = run_selected_spectral_excess_panel(
        RootTieRankSelectedSpectralExcessConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
            min_action_edge_fraction=float(args.min_action_edge_fraction),
            min_tie_fraction_floor=float(args.min_tie_fraction_floor),
            partial_spectral_ratio_floor=float(args.partial_spectral_ratio_floor),
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
