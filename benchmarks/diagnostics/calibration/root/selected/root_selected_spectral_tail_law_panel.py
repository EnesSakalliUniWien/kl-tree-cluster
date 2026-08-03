"""Support-aware selected-root spectral tail diagnostic.

This panel expresses the current root inference target explicitly:

* S_root is log(lambda / lambda_MP), the selected root spectral excess.
* T is selected tie-rank fraction.
* A is log selected-ratio action.
* E is log edge-margin action.
* B is measured root bandwidth/topology status.
* H_u is the local null-whitened spectral law status.

Rows are diagnostic only. They estimate a conservative empirical tail p-value
only when selected-null/external-null support exists in the same coarse root
stratum. Otherwise they fail closed.
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
    is_calibration_support,
    is_observed_target,
    lookup_numeric_by_key,
    require_columns,
    root_tail_stratum_key,
    safe_log1p,
    string_value,
    tail_excess_for_case,
)

SCHEMA_VERSION = "root_selected_spectral_tail_law_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_spectral_tail_law_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_conditioned_coherent_topology_join_after_replay"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_selected_spectral_tail_law_rows.csv"
SUMMARY_OUTPUT = "root_selected_spectral_tail_law_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "root_event_condition",
    "s_root_spectral_excess_log",
    "spectral_tail_variable",
    "s_root_identity_excess_log",
    "s_root_deformed_excess_log",
    "t_selected_tie_rank_fraction",
    "a_selected_ratio_action_log1p",
    "e_edge_margin_action_log1p",
    "b_bandwidth_topology_status",
    "h_u_population_law_status",
    "root_tail_stratum_key",
    "selected_null_support_count",
    "selected_null_exceedance_count",
    "selected_null_importance_effective_sample_size",
    "selected_null_importance_weighted_exceedance_fraction",
    "conservative_spectral_tail_p_value",
    "spectral_tail_p_value_status",
    "root_tail_inference_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "calibrated_tail_count",
    "fail_closed_missing_support_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedSpectralTailLawConfig:
    """Input/output contract for the selected-root spectral tail diagnostic."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    deformed_mp_edge_rows_path: Path | None = None
    deformed_mp_edge_support_rows_path: Path | None = None
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument("--deformed-mp-edge-rows-path", type=Path, default=None)
    parser.add_argument(
        "--deformed-mp-edge-support-rows-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--h-u-population-law-status",
        default="identity_mp_assumed_deformed_mp_unestimated",
    )
    return parser.parse_args()


def _support_in_target_stratum(
    *,
    target: pd.Series,
    generated: pd.DataFrame,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    if generated.empty:
        return generated.copy()
    target_key = root_tail_stratum_key(
        target=target,
        h_u_population_law_status=h_u_population_law_status,
    )
    rows = generated.copy()
    rows["_root_tail_stratum_key"] = rows.apply(
        lambda row: root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
        axis=1,
    )
    rows = rows.loc[rows["_root_tail_stratum_key"].eq(target_key)].copy()
    support_mask = rows.apply(is_calibration_support, axis=1)
    return rows.loc[support_mask].copy()


def _conservative_tail_p_value(exceedance_count: int, support_count: int) -> float:
    if int(support_count) <= 0:
        return math.nan
    return float((int(exceedance_count) + 1) / (int(support_count) + 1))


def _importance_log_weights(support: pd.DataFrame) -> np.ndarray:
    if support.empty:
        return np.asarray([], dtype=float)
    if "importance_log_weight" not in support.columns:
        return np.zeros(int(support.shape[0]), dtype=float)
    raw = pd.to_numeric(support["importance_log_weight"], errors="coerce")
    weights = raw.to_numpy(dtype=float)
    return np.where(np.isfinite(weights), weights, 0.0)


def _weighted_tail_summary(
    *,
    support: pd.DataFrame,
    exceedance_mask: np.ndarray,
) -> tuple[float, float, float, str]:
    """Return conservative weighted p, ESS, weighted exceedance, and status."""
    if support.empty:
        return math.nan, 0.0, math.nan, "selected_root_spectral_tail_support_missing"
    log_weights = _importance_log_weights(support)
    finite = np.isfinite(log_weights)
    if not bool(np.all(finite)):
        return (
            math.nan,
            0.0,
            math.nan,
            "invalid_importance_weight_support",
        )
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    weight_sum = float(np.sum(weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return math.nan, 0.0, math.nan, "invalid_importance_weight_support"
    weighted_exceedance = float(
        np.sum(weights * np.asarray(exceedance_mask, dtype=float)) / weight_sum
    )
    squared_sum = float(np.sum(weights * weights))
    effective_n = (
        float(weight_sum * weight_sum / squared_sum)
        if math.isfinite(squared_sum) and squared_sum > 0.0
        else 0.0
    )
    if effective_n <= 0.0:
        return math.nan, 0.0, math.nan, "invalid_importance_weight_support"
    conservative_p = float((effective_n * weighted_exceedance + 1.0) / (effective_n + 1.0))
    has_importance = (
        "importance_log_weight" in support.columns
        and pd.to_numeric(support["importance_log_weight"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .any()
    )
    status = (
        "importance_weighted_external_selected_null_tail"
        if has_importance
        else "direct_selected_null_empirical_tail"
    )
    return conservative_p, effective_n, weighted_exceedance, status


def build_root_selected_spectral_tail_law_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    deformed_mp_edge_rows: pd.DataFrame | None = None,
    deformed_mp_edge_support_rows: pd.DataFrame | None = None,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Return support-aware root spectral tail rows."""
    require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "joined feasibility rows",
    )
    rows = joined_feasibility_rows.copy()
    if "root_bandwidth_reopen_band" not in rows.columns:
        rows["root_bandwidth_reopen_band"] = ""
    if "root_mixed_region_component" not in rows.columns:
        rows["root_mixed_region_component"] = "root_component_missing"
    target_mask = rows.apply(is_observed_target, axis=1)
    targets = rows[target_mask].copy()
    generated = rows[~target_mask].copy()
    target_deformed = deformed_mp_edge_rows if deformed_mp_edge_rows is not None else pd.DataFrame()
    support_deformed = (
        deformed_mp_edge_support_rows
        if deformed_mp_edge_support_rows is not None
        else pd.DataFrame()
    )
    deformed_excess_by_case = {
        **lookup_numeric_by_key(
            target_deformed,
            key_column="target_case_id",
            value_column="s_root_deformed_excess_log",
        ),
        **lookup_numeric_by_key(
            support_deformed,
            key_column="case_id",
            value_column="s_root_deformed_excess_log",
        ),
    }
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = string_value(target, "case_id")
        (
            s_root,
            identity_excess,
            deformed_excess,
            spectral_tail_variable,
        ) = tail_excess_for_case(
            target,
            deformed_excess_by_case=deformed_excess_by_case,
        )
        t_rank = finite_float(target.get("root_tie_rank_median_fraction", math.nan))
        action = safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
        edge = safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
        support = _support_in_target_stratum(
            target=target,
            generated=generated,
            h_u_population_law_status=str(h_u_population_law_status),
        )
        support_count = int(support.shape[0])
        if support_count:
            support_s = support.apply(
                lambda row: tail_excess_for_case(
                    row,
                    deformed_excess_by_case=deformed_excess_by_case,
                )[0],
                axis=1,
            )
            exceedance_mask = support_s.ge(s_root).to_numpy(dtype=bool)
            exceedances = int(np.sum(exceedance_mask))
        else:
            exceedance_mask = np.asarray([], dtype=bool)
            exceedances = 0
        if support_count:
            p_value, effective_n, weighted_exceedance, p_value_status = _weighted_tail_summary(
                support=support,
                exceedance_mask=exceedance_mask,
            )
        else:
            p_value = _conservative_tail_p_value(exceedances, support_count)
            effective_n = 0.0
            weighted_exceedance = math.nan
            p_value_status = "selected_root_spectral_tail_support_missing"
        inference_status = (
            "calibrated_selected_root_spectral_tail_available"
            if support_count > 0
            else "fail_closed_selected_root_spectral_tail_support_missing"
        )
        next_step = (
            "use_conservative_empirical_tail_p_value"
            if support_count > 0
            else "generate_selected_null_roots_in_same_root_tail_stratum"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "root_event_condition": "E_root = E_margin intersect E_tie intersect E_rank",
                "s_root_spectral_excess_log": s_root,
                "spectral_tail_variable": spectral_tail_variable,
                "s_root_identity_excess_log": identity_excess,
                "s_root_deformed_excess_log": deformed_excess,
                "t_selected_tie_rank_fraction": t_rank,
                "a_selected_ratio_action_log1p": action,
                "e_edge_margin_action_log1p": edge,
                "b_bandwidth_topology_status": string_value(
                    target,
                    "root_bandwidth_reopen_band",
                ),
                "h_u_population_law_status": str(h_u_population_law_status),
                "root_tail_stratum_key": root_tail_stratum_key(
                    target=target,
                    h_u_population_law_status=str(h_u_population_law_status),
                ),
                "selected_null_support_count": support_count,
                "selected_null_exceedance_count": exceedances,
                "selected_null_importance_effective_sample_size": effective_n,
                "selected_null_importance_weighted_exceedance_fraction": (weighted_exceedance),
                "conservative_spectral_tail_p_value": p_value,
                "spectral_tail_p_value_status": p_value_status,
                "root_tail_inference_status": inference_status,
                "next_mathematical_step": next_step,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_spectral_tail_law_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize root spectral-tail support."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    calibrated = int(
        rows["root_tail_inference_status"]
        .astype(str)
        .eq("calibrated_selected_root_spectral_tail_available")
        .sum()
    )
    fail_closed = int(
        rows["root_tail_inference_status"]
        .astype(str)
        .eq("fail_closed_selected_root_spectral_tail_support_missing")
        .sum()
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "calibrated_tail_count": calibrated,
                "fail_closed_missing_support_count": fail_closed,
                "summary_status": (
                    "selected_root_spectral_tail_support_missing"
                    if fail_closed
                    else "selected_root_spectral_tail_support_available"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def evaluate_root_selected_spectral_tail_law_panel(
    config: RootSelectedSpectralTailLawConfig,
) -> dict[str, pd.DataFrame]:
    """Read inputs and return root spectral-tail law tables."""
    joined = pd.read_csv(config.joined_feasibility_rows_path, low_memory=False)
    deformed_rows = _read_optional_csv(config.deformed_mp_edge_rows_path)
    deformed_support_rows = _read_optional_csv(config.deformed_mp_edge_support_rows_path)
    rows = build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=joined,
        deformed_mp_edge_rows=deformed_rows,
        deformed_mp_edge_support_rows=deformed_support_rows,
        h_u_population_law_status=str(config.h_u_population_law_status),
    )
    summary = summarize_root_selected_spectral_tail_law_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_spectral_tail_law_panel(
    config: RootSelectedSpectralTailLawConfig,
) -> dict[str, Path]:
    """Run the panel and write outputs."""
    tables = evaluate_root_selected_spectral_tail_law_panel(config)
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
    outputs = run_root_selected_spectral_tail_law_panel(
        RootSelectedSpectralTailLawConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            deformed_mp_edge_rows_path=args.deformed_mp_edge_rows_path,
            deformed_mp_edge_support_rows_path=args.deformed_mp_edge_support_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
