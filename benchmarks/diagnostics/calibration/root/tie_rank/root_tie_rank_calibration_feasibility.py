"""Calibration feasibility for the root discrete tie-rank law.

The mixed root selected-region diagnostic identifies the event

    E_root = E_margin intersect E_tie_cell intersect E_tie_rank.

This panel turns that event into explicit calibration strata and reports
whether the available rows contain enough selected-null support to estimate a
conditional tail law. It does not produce production p-values; it names the
missing simulation support.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import write_diagnostic_bundle
from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float, require_columns

SCHEMA_VERSION = "root_tie_rank_calibration_feasibility/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_calibration_feasibility_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_calibration_feasibility"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_MIXED_REGION_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_mixed_region_law_overlap_case_family"
    / "root_selected_mixed_region_law_rows.csv"
)
DEFAULT_TARGET_ALPHA = 0.01
DEFAULT_RELATIVE_SE_TARGET = 0.25

ROWS_OUTPUT = "root_tie_rank_calibration_feasibility_rows.csv"
STRATA_OUTPUT = "root_tie_rank_calibration_feasibility_strata.csv"
SUMMARY_OUTPUT = "root_tie_rank_calibration_feasibility_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "calibration_role",
    "root_mixed_region_component",
    "root_tie_rank_band",
    "root_edge_margin_band",
    "root_spectral_ratio_band",
    "root_bandwidth_reopen_band",
    "root_conditioning_stratum_key",
    "root_sibling_selected_ratio",
    "root_tie_rank_median_fraction",
    "root_edge_path_statistic_margin",
    "root_selected_eigenvalue_over_mp_upper_bound",
    "root_bandwidth_reopen_count",
    "stratum_observed_count",
    "stratum_calibration_null_support_count",
    "stratum_calibration_null_exceedance_count",
    "empirical_conservative_tail_p_value",
    "target_alpha",
    "alpha_resolution_required_null_count",
    "tail_precision_required_null_count",
    "additional_null_count_for_alpha_resolution",
    "additional_null_count_for_tail_precision",
    "calibration_feasibility_status",
)

STRATA_COLUMNS = (
    "schema_version",
    "study_role",
    "root_conditioning_stratum_key",
    "root_mixed_region_component",
    "root_tie_rank_band",
    "root_edge_margin_band",
    "root_spectral_ratio_band",
    "root_bandwidth_reopen_band",
    "stratum_observed_count",
    "stratum_calibration_null_support_count",
    "target_alpha",
    "alpha_resolution_required_null_count",
    "tail_precision_required_null_count",
    "additional_null_count_for_alpha_resolution",
    "additional_null_count_for_tail_precision",
    "median_root_sibling_selected_ratio",
    "max_root_sibling_selected_ratio",
    "stratum_feasibility_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "stratum_count",
    "calibration_null_support_count",
    "alpha_resolution_required_null_count",
    "tail_precision_required_null_count",
    "alpha_resolution_ready_stratum_count",
    "tail_precision_ready_stratum_count",
    "missing_alpha_resolution_null_count_total",
    "missing_tail_precision_null_count_total",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankCalibrationFeasibilityConfig:
    """Input path and calibration support requirements."""

    output_dir: Path
    mixed_region_rows_path: Path = DEFAULT_MIXED_REGION_ROWS
    target_alpha: float = DEFAULT_TARGET_ALPHA
    relative_se_target: float = DEFAULT_RELATIVE_SE_TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mixed-region-rows-path",
        type=Path,
        default=DEFAULT_MIXED_REGION_ROWS,
    )
    parser.add_argument("--target-alpha", type=float, default=DEFAULT_TARGET_ALPHA)
    parser.add_argument(
        "--relative-se-target",
        type=float,
        default=DEFAULT_RELATIVE_SE_TARGET,
    )
    return parser.parse_args()


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if not numeric.empty else math.nan


def alpha_resolution_required_null_count(target_alpha: float) -> int:
    """Null count needed so min plus-one empirical p-value is <= alpha."""
    if not (0.0 < float(target_alpha) < 1.0):
        raise ValueError("target_alpha must be in (0, 1).")
    return max(int(math.ceil(1.0 / float(target_alpha))) - 1, 1)


def tail_precision_required_null_count(
    *,
    target_alpha: float,
    relative_se_target: float,
) -> int:
    """Binomial tail count for relative SE near the target alpha."""
    if not (0.0 < float(target_alpha) < 1.0):
        raise ValueError("target_alpha must be in (0, 1).")
    if float(relative_se_target) <= 0.0:
        raise ValueError("relative_se_target must be positive.")
    return int(
        math.ceil(
            (1.0 - float(target_alpha)) / (float(target_alpha) * float(relative_se_target) ** 2)
        )
    )


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _data_role(row: pd.Series) -> str:
    return _string_value(row, "data_role", "unlabeled")


def _calibration_role(row: pd.Series) -> str:
    if "calibration_role" in row and not pd.isna(row["calibration_role"]):
        return str(row["calibration_role"])
    data_role = _data_role(row)
    if data_role == "selected_null":
        return "selected_null_candidate_support"
    if data_role == "signal":
        return "signal_diagnostic_not_null_support"
    return "unlabeled_diagnostic_not_null_support"


def _is_calibration_null_support(row: pd.Series) -> bool:
    role = _calibration_role(row)
    return role in {
        "selected_null",
        "selected_null_candidate_support",
        "external_selected_null",
        "external_null_support",
        "calibration_null",
    }


def _tie_rank_band(value: float) -> str:
    if not math.isfinite(value):
        return "tie_rank_missing"
    if value < 0.50:
        return "tie_rank_low_lt_0_50"
    if value < 0.75:
        return "tie_rank_mid_0_50_0_75"
    if value < 0.90:
        return "tie_rank_high_0_75_0_90"
    return "tie_rank_extreme_ge_0_90"


def _edge_margin_band(value: float) -> str:
    if not math.isfinite(value):
        return "edge_margin_missing"
    log_margin = math.log1p(max(value, 0.0))
    if log_margin < 5.0:
        return "edge_log_margin_low_lt_5"
    if log_margin < 7.0:
        return "edge_log_margin_mid_5_7"
    return "edge_log_margin_high_ge_7"


def _spectral_ratio_band(value: float) -> str:
    if not math.isfinite(value):
        return "spectral_ratio_missing"
    if value <= 1.0:
        return "spectral_ratio_le_1"
    if value <= 2.0:
        return "spectral_ratio_1_2"
    if value <= 4.0:
        return "spectral_ratio_2_4"
    return "spectral_ratio_gt_4"


def _bandwidth_reopen_band(value: float) -> str:
    if not math.isfinite(value):
        return "bandwidth_reopen_missing"
    if value <= 0.0:
        return "bandwidth_no_root_reopen"
    return "bandwidth_root_reopen_observed"


def _bandwidth_count_for_stratum(row: pd.Series) -> float:
    locality_status = _string_value(row, "root_bandwidth_locality_status", "")
    if locality_status == "topology_frontier_not_joined":
        return math.nan
    return finite_float(row.get("root_bandwidth_reopen_count", 0))


def _stratum_key(
    *,
    component: str,
    tie_band: str,
    edge_band: str,
    spectral_band: str,
    bandwidth_band: str,
) -> str:
    return "|".join(
        (
            f"component={component}",
            f"tie={tie_band}",
            f"edge={edge_band}",
            f"spectral={spectral_band}",
            f"bandwidth={bandwidth_band}",
        )
    )


def _feasibility_status(
    *,
    null_count: int,
    resolution_required: int,
    precision_required: int,
) -> str:
    if null_count >= precision_required:
        return "tail_precision_ready_diagnostic_only"
    if null_count >= resolution_required:
        return "alpha_resolution_ready_precision_missing"
    if null_count > 0:
        return "insufficient_null_support_for_alpha_resolution"
    return "external_null_support_missing"


def _annotate_conditioning_strata(mixed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in mixed_rows.iterrows():
        component = _string_value(
            row,
            "root_mixed_region_component",
            "component_missing",
        )
        tie_fraction = finite_float(row.get("root_tie_rank_median_fraction", math.nan))
        edge_margin = finite_float(row.get("root_edge_path_statistic_margin", math.nan))
        spectral_ratio = finite_float(
            row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        )
        bandwidth_count = _bandwidth_count_for_stratum(row)
        tie_band = _tie_rank_band(tie_fraction)
        edge_band = _edge_margin_band(edge_margin)
        spectral_band = _spectral_ratio_band(spectral_ratio)
        bandwidth_band = _bandwidth_reopen_band(bandwidth_count)
        records.append(
            {
                "case_id": _string_value(row, "case_id", "case_missing"),
                "data_role": _data_role(row),
                "calibration_role": _calibration_role(row),
                "root_mixed_region_component": component,
                "root_tie_rank_band": tie_band,
                "root_edge_margin_band": edge_band,
                "root_spectral_ratio_band": spectral_band,
                "root_bandwidth_reopen_band": bandwidth_band,
                "root_conditioning_stratum_key": _stratum_key(
                    component=component,
                    tie_band=tie_band,
                    edge_band=edge_band,
                    spectral_band=spectral_band,
                    bandwidth_band=bandwidth_band,
                ),
                "root_sibling_selected_ratio": finite_float(
                    row.get("root_sibling_selected_ratio", math.nan)
                ),
                "root_tie_rank_median_fraction": tie_fraction,
                "root_edge_path_statistic_margin": edge_margin,
                "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
                "root_bandwidth_reopen_count": bandwidth_count,
                "_is_calibration_null_support": _is_calibration_null_support(row),
            }
        )
    return pd.DataFrame.from_records(records)


def build_root_tie_rank_calibration_feasibility_rows(
    *,
    mixed_region_rows: pd.DataFrame,
    target_alpha: float = DEFAULT_TARGET_ALPHA,
    relative_se_target: float = DEFAULT_RELATIVE_SE_TARGET,
) -> pd.DataFrame:
    """Return row-level feasibility annotations for mixed root-law rows."""
    require_columns(
        mixed_region_rows,
        {
            "case_id",
            "root_mixed_region_component",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
            "root_bandwidth_reopen_count",
        },
        "mixed root-law rows",
    )
    resolution_required = alpha_resolution_required_null_count(target_alpha)
    precision_required = tail_precision_required_null_count(
        target_alpha=target_alpha,
        relative_se_target=relative_se_target,
    )
    annotated = _annotate_conditioning_strata(mixed_region_rows)
    records: list[dict[str, object]] = []
    for _, row in annotated.iterrows():
        key = str(row["root_conditioning_stratum_key"])
        stratum = annotated[annotated["root_conditioning_stratum_key"].eq(key)]
        null_support = stratum[stratum["_is_calibration_null_support"]]
        observed_ratio = finite_float(row["root_sibling_selected_ratio"])
        null_ratios = pd.to_numeric(
            null_support["root_sibling_selected_ratio"],
            errors="coerce",
        )
        null_ratios = null_ratios[np.isfinite(null_ratios)]
        if math.isfinite(observed_ratio) and not null_ratios.empty:
            exceedance_count = int((null_ratios >= observed_ratio).sum())
            empirical_p = (float(exceedance_count) + 1.0) / (float(null_ratios.shape[0]) + 1.0)
        else:
            exceedance_count = 0
            empirical_p = math.nan
        null_count = int(null_ratios.shape[0])
        status = _feasibility_status(
            null_count=null_count,
            resolution_required=resolution_required,
            precision_required=precision_required,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "calibration_role": str(row["calibration_role"]),
                "root_mixed_region_component": str(row["root_mixed_region_component"]),
                "root_tie_rank_band": str(row["root_tie_rank_band"]),
                "root_edge_margin_band": str(row["root_edge_margin_band"]),
                "root_spectral_ratio_band": str(row["root_spectral_ratio_band"]),
                "root_bandwidth_reopen_band": str(row["root_bandwidth_reopen_band"]),
                "root_conditioning_stratum_key": key,
                "root_sibling_selected_ratio": observed_ratio,
                "root_tie_rank_median_fraction": finite_float(row["root_tie_rank_median_fraction"]),
                "root_edge_path_statistic_margin": finite_float(
                    row["root_edge_path_statistic_margin"]
                ),
                "root_selected_eigenvalue_over_mp_upper_bound": finite_float(
                    row["root_selected_eigenvalue_over_mp_upper_bound"]
                ),
                "root_bandwidth_reopen_count": finite_float(row["root_bandwidth_reopen_count"]),
                "stratum_observed_count": int(stratum.shape[0]),
                "stratum_calibration_null_support_count": null_count,
                "stratum_calibration_null_exceedance_count": exceedance_count,
                "empirical_conservative_tail_p_value": empirical_p,
                "target_alpha": float(target_alpha),
                "alpha_resolution_required_null_count": resolution_required,
                "tail_precision_required_null_count": precision_required,
                "additional_null_count_for_alpha_resolution": max(
                    resolution_required - null_count,
                    0,
                ),
                "additional_null_count_for_tail_precision": max(
                    precision_required - null_count,
                    0,
                ),
                "calibration_feasibility_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_tie_rank_calibration_strata(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one feasibility row per conditioning stratum."""
    if rows.empty:
        return pd.DataFrame(columns=STRATA_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = [
        "root_conditioning_stratum_key",
        "root_mixed_region_component",
        "root_tie_rank_band",
        "root_edge_margin_band",
        "root_spectral_ratio_band",
        "root_bandwidth_reopen_band",
    ]
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        (
            stratum_key,
            component,
            tie_band,
            edge_band,
            spectral_band,
            bandwidth_band,
        ) = keys
        null_count = int(group["stratum_calibration_null_support_count"].max())
        resolution_required = int(group["alpha_resolution_required_null_count"].max())
        precision_required = int(group["tail_precision_required_null_count"].max())
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "root_conditioning_stratum_key": str(stratum_key),
                "root_mixed_region_component": str(component),
                "root_tie_rank_band": str(tie_band),
                "root_edge_margin_band": str(edge_band),
                "root_spectral_ratio_band": str(spectral_band),
                "root_bandwidth_reopen_band": str(bandwidth_band),
                "stratum_observed_count": int(group.shape[0]),
                "stratum_calibration_null_support_count": null_count,
                "target_alpha": float(group["target_alpha"].iloc[0]),
                "alpha_resolution_required_null_count": resolution_required,
                "tail_precision_required_null_count": precision_required,
                "additional_null_count_for_alpha_resolution": max(
                    resolution_required - null_count,
                    0,
                ),
                "additional_null_count_for_tail_precision": max(
                    precision_required - null_count,
                    0,
                ),
                "median_root_sibling_selected_ratio": _finite_median(
                    group["root_sibling_selected_ratio"]
                ),
                "max_root_sibling_selected_ratio": float(
                    pd.to_numeric(
                        group["root_sibling_selected_ratio"],
                        errors="coerce",
                    ).max()
                ),
                "stratum_feasibility_status": str(group["calibration_feasibility_status"].iloc[0]),
            }
        )
    return pd.DataFrame.from_records(records, columns=STRATA_COLUMNS)


def summarize_root_tie_rank_calibration_feasibility(
    rows: pd.DataFrame,
    strata: pd.DataFrame,
) -> pd.DataFrame:
    """Return global feasibility summary."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    resolution_required = int(rows["alpha_resolution_required_null_count"].max())
    precision_required = int(rows["tail_precision_required_null_count"].max())
    null_support_count = int(strata["stratum_calibration_null_support_count"].sum())
    alpha_ready = int(
        strata["stratum_feasibility_status"]
        .astype(str)
        .isin(
            {
                "alpha_resolution_ready_precision_missing",
                "tail_precision_ready_diagnostic_only",
            }
        )
        .sum()
    )
    precision_ready = int(
        strata["stratum_feasibility_status"]
        .astype(str)
        .eq("tail_precision_ready_diagnostic_only")
        .sum()
    )
    missing_resolution = int(strata["additional_null_count_for_alpha_resolution"].sum())
    missing_precision = int(strata["additional_null_count_for_tail_precision"].sum())
    if precision_ready == int(strata.shape[0]):
        status = "all_strata_tail_precision_ready_diagnostic_only"
    elif alpha_ready > 0:
        status = "partial_alpha_resolution_support"
    elif null_support_count > 0:
        status = "calibration_null_support_observed_below_alpha_resolution"
    else:
        status = "external_null_support_missing_for_all_strata"
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "row_count": int(rows.shape[0]),
        "stratum_count": int(strata.shape[0]),
        "calibration_null_support_count": null_support_count,
        "alpha_resolution_required_null_count": resolution_required,
        "tail_precision_required_null_count": precision_required,
        "alpha_resolution_ready_stratum_count": alpha_ready,
        "tail_precision_ready_stratum_count": precision_ready,
        "missing_alpha_resolution_null_count_total": missing_resolution,
        "missing_tail_precision_null_count_total": missing_precision,
        "summary_status": status,
    }
    return pd.DataFrame.from_records([record], columns=SUMMARY_COLUMNS)


def evaluate_root_tie_rank_calibration_feasibility(
    config: RootTieRankCalibrationFeasibilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read inputs and return row, stratum, and global feasibility tables."""
    mixed_rows = pd.read_csv(config.mixed_region_rows_path)
    rows = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=mixed_rows,
        target_alpha=float(config.target_alpha),
        relative_se_target=float(config.relative_se_target),
    )
    strata = summarize_root_tie_rank_calibration_strata(rows)
    summary = summarize_root_tie_rank_calibration_feasibility(rows, strata)
    return rows, strata, summary


def run_root_tie_rank_calibration_feasibility(
    config: RootTieRankCalibrationFeasibilityConfig,
) -> dict[str, Path]:
    """Write root tie-rank calibration feasibility outputs."""
    rows, strata, summary = evaluate_root_tie_rank_calibration_feasibility(config)
    return write_diagnostic_bundle(
        output_dir=Path(config.output_dir),
        tables={"rows": rows, "strata": strata, "summary": summary},
        filenames={
            "rows": ROWS_OUTPUT,
            "strata": STRATA_OUTPUT,
            "summary": SUMMARY_OUTPUT,
        },
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
            "row_count": int(rows.shape[0]),
            "stratum_count": int(strata.shape[0]),
            "summary_row_count": int(summary.shape[0]),
        },
        manifest_filename=MANIFEST_OUTPUT,
        include_row_counts=False,
    )


def main() -> None:
    args = parse_args()
    run_root_tie_rank_calibration_feasibility(
        RootTieRankCalibrationFeasibilityConfig(
            output_dir=args.output_dir,
            mixed_region_rows_path=args.mixed_region_rows_path,
            target_alpha=float(args.target_alpha),
            relative_se_target=float(args.relative_se_target),
        )
    )


if __name__ == "__main__":
    main()
