"""Kernel-weighted selected-root spectral tail diagnostic.

This is a candidate bridge between neighborhood smoothing and the current
fail-closed root-tail law. It does not alter clustering. It uses local
smoothing only as admissible support weights for the selected root spectral
tail:

    P(S_Hu >= s | R_root, T, A, E, B, H_u, N_tau).

Rows remain diagnostic-only. A weighted p-value is reported only when the
kernel support is admissible and not concentrated on a single row.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    _finite_float,
    _finite_int,
    _is_calibration_support,
    _is_observed_target,
    _lookup_numeric_by_key,
    _safe_log1p,
    _string_value,
    _tail_excess_for_case,
)

SCHEMA_VERSION = "root_selected_kernel_spectral_tail_law_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_kernel_spectral_tail_law_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_kernel_spectral_tail_law_panel"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_INPUT_ROOT = (
    DEFAULT_RESULT_ROOT
    / "root_selected_same_geometry_external_support_attempt_five_target_tiny_smoke"
)
DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_INPUT_ROOT / "same_geometry_external_joined_feasibility_rows.csv"
)
DEFAULT_STRICT_TAIL_ROWS = DEFAULT_INPUT_ROOT / "root_selected_spectral_tail_law_rows.csv"
DEFAULT_DEFORMED_MP_EDGE_ROWS = DEFAULT_INPUT_ROOT / "root_selected_deformed_mp_edge_rows.csv"
DEFAULT_DEFORMED_MP_EDGE_SUPPORT_ROWS = (
    DEFAULT_INPUT_ROOT / "root_selected_deformed_mp_edge_support_rows.csv"
)
DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_region_margins_overlap_case_family"
    / "root_selected_region_summary.csv"
)

ROWS_OUTPUT = "root_selected_kernel_spectral_tail_law_rows.csv"
SUMMARY_OUTPUT = "root_selected_kernel_spectral_tail_law_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_s_h_u_excess_log",
    "target_spectral_tail_variable",
    "target_t_selected_tie_rank_fraction",
    "target_a_selected_ratio_action_log1p",
    "target_e_edge_margin_action_log1p",
    "target_b_bandwidth_topology_status",
    "target_log_active_feature_count",
    "target_log_effective_rows",
    "target_log_h_u_edge",
    "target_root_topology_signature",
    "target_root_topology_coarse_signature",
    "strict_selected_null_support_count",
    "strict_conservative_spectral_tail_p_value",
    "strict_root_tail_inference_status",
    "admissible_kernel_support_count",
    "non_support_neighbor_excluded_count",
    "kernel_positive_s_h_u_support_count",
    "kernel_exceedance_count",
    "kernel_weight_sum",
    "kernel_effective_sample_size",
    "kernel_max_weight_share",
    "kernel_weighted_exceedance_fraction",
    "kernel_conservative_tail_p_value",
    "kernel_tail_status",
    "kernel_candidate_decision",
    "kernel_bandwidths_json",
    "nearest_support_case_id",
    "nearest_support_weight_share",
    "nearest_support_s_h_u_excess_log",
    "topology_exact_support_count",
    "topology_coarsened_support_count",
    "topology_admissible_support_count",
    "topology_positive_s_h_u_support_count",
    "topology_exceedance_count",
    "topology_weight_sum",
    "topology_effective_sample_size",
    "topology_max_weight_share",
    "topology_weighted_exceedance_fraction",
    "topology_conservative_tail_p_value",
    "topology_kernel_status",
    "topology_kernel_candidate_decision",
    "topology_nearest_support_case_id",
    "topology_nearest_support_weight_share",
    "topology_nearest_support_s_h_u_excess_log",
    "topology_nearest_support_signature",
    "comparison_to_current",
    "next_tracking_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "strict_calibrated_count",
    "strict_fail_closed_count",
    "kernel_available_count",
    "kernel_fail_closed_count",
    "strict_fail_closed_kernel_available_count",
    "kernel_nonzero_support_target_count",
    "topology_kernel_available_count",
    "topology_kernel_fail_closed_count",
    "topology_strict_fail_closed_kernel_available_count",
    "topology_target_signature_missing_count",
    "topology_support_missing_count",
    "topology_degenerate_support_count",
    "topology_nonzero_support_target_count",
    "kernel_selected_null_leakage_flag_count",
    "summary_status",
)

OBSERVED_ROOT_TOPOLOGY_COLUMNS = (
    "root_selected_region_law_status",
    "root_calibration_status",
    "root_child_balance",
    "root_child_construction_merge_count",
    "root_child_near_active_merge_count",
    "root_child_tied_minimum_merge_count",
    "root_child_discrete_tie_cell_count",
    "root_child_smooth_constraint_count",
    "root_child_min_merge_margin",
    "left_root_child_construction_merge_count",
    "right_root_child_construction_merge_count",
)


@dataclass(frozen=True)
class RootSelectedKernelSpectralTailLawConfig:
    """Input/output contract for the kernel-spectral root-tail diagnostic."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    strict_tail_rows_path: Path = DEFAULT_STRICT_TAIL_ROWS
    deformed_mp_edge_rows_path: Path = DEFAULT_DEFORMED_MP_EDGE_ROWS
    deformed_mp_edge_support_rows_path: Path = DEFAULT_DEFORMED_MP_EDGE_SUPPORT_ROWS
    observed_root_summary_rows_path: Path | None = DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS
    min_kernel_effective_sample_size: float = 1.0
    max_kernel_weight_share: float = 1.0
    min_kernel_weight: float = 1e-12
    bandwidth_mismatch_weight: float = 0.35
    topology_coarsening_weight: float = 0.25
    topology_balance_bin_width: float = 0.10
    topology_tie_density_bin_width: float = 0.05
    min_topology_effective_sample_size: float = 2.0
    max_topology_weight_share: float = 0.75
    require_deformed_support: bool = True
    h_u_population_law_status: str = "deformed_mp_edge_measured_support_side"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument("--strict-tail-rows-path", type=Path, default=DEFAULT_STRICT_TAIL_ROWS)
    parser.add_argument(
        "--deformed-mp-edge-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_MP_EDGE_ROWS,
    )
    parser.add_argument(
        "--deformed-mp-edge-support-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_MP_EDGE_SUPPORT_ROWS,
    )
    parser.add_argument(
        "--observed-root-summary-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS,
    )
    parser.add_argument("--no-observed-root-summary", action="store_true")
    parser.add_argument("--min-kernel-effective-sample-size", type=float, default=1.0)
    parser.add_argument("--max-kernel-weight-share", type=float, default=1.0)
    parser.add_argument("--min-kernel-weight", type=float, default=1e-12)
    parser.add_argument("--bandwidth-mismatch-weight", type=float, default=0.35)
    parser.add_argument("--topology-coarsening-weight", type=float, default=0.25)
    parser.add_argument("--topology-balance-bin-width", type=float, default=0.10)
    parser.add_argument("--topology-tie-density-bin-width", type=float, default=0.05)
    parser.add_argument("--min-topology-effective-sample-size", type=float, default=2.0)
    parser.add_argument("--max-topology-weight-share", type=float, default=0.75)
    parser.add_argument("--allow-identity-support", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RootSelectedKernelSpectralTailLawConfig:
    return RootSelectedKernelSpectralTailLawConfig(
        output_dir=args.output_dir,
        joined_feasibility_rows_path=args.joined_feasibility_rows_path,
        strict_tail_rows_path=args.strict_tail_rows_path,
        deformed_mp_edge_rows_path=args.deformed_mp_edge_rows_path,
        deformed_mp_edge_support_rows_path=args.deformed_mp_edge_support_rows_path,
        observed_root_summary_rows_path=(
            None if bool(args.no_observed_root_summary) else args.observed_root_summary_rows_path
        ),
        min_kernel_effective_sample_size=float(args.min_kernel_effective_sample_size),
        max_kernel_weight_share=float(args.max_kernel_weight_share),
        min_kernel_weight=float(args.min_kernel_weight),
        bandwidth_mismatch_weight=float(args.bandwidth_mismatch_weight),
        topology_coarsening_weight=float(args.topology_coarsening_weight),
        topology_balance_bin_width=float(args.topology_balance_bin_width),
        topology_tie_density_bin_width=float(args.topology_tie_density_bin_width),
        min_topology_effective_sample_size=float(args.min_topology_effective_sample_size),
        max_topology_weight_share=float(args.max_topology_weight_share),
        require_deformed_support=not bool(args.allow_identity_support),
    )


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedKernelSpectralTailLawConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fill_observed_root_topology(
    *,
    rows: pd.DataFrame,
    observed_root_summary_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Fill observed target topology fields from root selected-region replay."""
    if observed_root_summary_rows is None or observed_root_summary_rows.empty:
        return rows
    if "case_id" not in observed_root_summary_rows.columns or "case_id" not in rows.columns:
        return rows
    summary_by_case = {str(row["case_id"]): row for _, row in observed_root_summary_rows.iterrows()}
    enriched = rows.copy()
    observed_mask = enriched.apply(_is_observed_target, axis=1)
    for column in OBSERVED_ROOT_TOPOLOGY_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = math.nan
    for index, row in enriched.loc[observed_mask].iterrows():
        summary = summary_by_case.get(str(row["case_id"]))
        if summary is None:
            continue
        for column in OBSERVED_ROOT_TOPOLOGY_COLUMNS:
            if column not in summary:
                continue
            current = enriched.at[index, column]
            if _is_missing_value(current) or str(current).strip() == "":
                enriched.at[index, column] = summary[column]
    return enriched


def _positive_bandwidth(values: pd.Series, default: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite.loc[finite.gt(0.0)]
    if finite.empty:
        return float(default)
    return max(float(finite.median()), 1e-12)


def _coordinate_bandwidths(targets: pd.DataFrame, support: pd.DataFrame) -> dict[str, float]:
    combined = pd.concat([targets, support], ignore_index=True, sort=False)
    action = combined["root_sibling_selected_ratio"].map(_safe_log1p)
    edge = combined["root_edge_path_statistic_margin"].map(_safe_log1p)
    tie = pd.to_numeric(combined["root_tie_rank_median_fraction"], errors="coerce")
    log_k = combined["root_active_feature_count"].map(
        lambda value: math.log(max(_finite_float(value), 1.0))
    )
    log_rows = combined["root_effective_independent_rows"].map(
        lambda value: math.log(max(_finite_float(value), 1.0))
    )
    log_h = combined["root_mp_upper_bound"].map(
        lambda value: math.log(max(_finite_float(value), 1e-12))
    )
    return {
        "tau_tie": _positive_bandwidth(tie.diff().abs(), 0.12),
        "tau_action": _positive_bandwidth(action.diff().abs(), 1.0),
        "tau_edge": _positive_bandwidth(edge.diff().abs(), 1.0),
        "h_log_k": _positive_bandwidth(log_k.diff().abs(), 0.75),
        "h_log_rows": _positive_bandwidth(log_rows.diff().abs(), 0.75),
        "h_log_h_u": _positive_bandwidth(log_h.diff().abs(), 0.35),
    }


def _safe_log_feature(row: pd.Series, column: str) -> float:
    value = _finite_float(row.get(column, math.nan))
    if not math.isfinite(value):
        return math.nan
    return float(math.log(max(value, 1.0)))


def _safe_log_positive(row: pd.Series, column: str) -> float:
    value = _finite_float(row.get(column, math.nan))
    if not math.isfinite(value):
        return math.nan
    return float(math.log(max(value, 1e-12)))


def _floor_bin(value: float, *, width: float, upper: float | None = None) -> str:
    if not math.isfinite(value):
        return "missing"
    width = max(float(width), 1e-12)
    lower = math.floor(max(value, 0.0) / width) * width
    if upper is not None:
        lower = min(lower, float(upper) - width)
    high = lower + width
    return f"{lower:.2f}_{high:.2f}"


def _count_bin(value: float) -> str:
    if not math.isfinite(value):
        return "missing"
    integer = int(round(max(value, 0.0)))
    if integer == 0:
        return "0"
    if integer == 1:
        return "1"
    if integer <= 5:
        return "2_5"
    if integer <= 20:
        return "6_20"
    if integer <= 100:
        return "21_100"
    if integer <= 300:
        return "101_300"
    if integer <= 600:
        return "301_600"
    return "gt_600"


def _merge_count_signature(row: pd.Series, *, coarse: bool) -> str:
    value = _finite_float(row.get("root_child_construction_merge_count", math.nan))
    if not math.isfinite(value):
        return "missing"
    integer = int(round(max(value, 0.0)))
    if not coarse:
        return str(integer)
    if integer <= 450:
        return "small_le_450"
    if integer <= 550:
        return "medium_451_550"
    return "large_gt_550"


def _balance_signature(
    row: pd.Series,
    *,
    coarse: bool,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> str:
    value = _finite_float(row.get("root_child_balance", math.nan))
    if not math.isfinite(value):
        return "missing"
    bounded = min(max(value, 0.0), 0.5)
    if coarse:
        if bounded < 0.20:
            return "unbalanced_lt_0_20"
        if bounded < 0.40:
            return "mid_0_20_0_40"
        return "balanced_ge_0_40"
    return _floor_bin(
        bounded,
        width=config.topology_balance_bin_width,
        upper=0.50 + config.topology_balance_bin_width,
    )


def _tie_density(row: pd.Series) -> float:
    fraction = _finite_float(row.get("root_tie_step_fraction", math.nan))
    if math.isfinite(fraction):
        return min(max(fraction, 0.0), 1.0)
    tied = _finite_float(row.get("root_child_tied_minimum_merge_count", math.nan))
    merges = _finite_float(row.get("root_child_construction_merge_count", math.nan))
    if math.isfinite(tied) and math.isfinite(merges) and merges > 0.0:
        return min(max(tied / merges, 0.0), 1.0)
    return math.nan


def _tie_density_signature(
    row: pd.Series,
    *,
    coarse: bool,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> str:
    value = _tie_density(row)
    if not math.isfinite(value):
        return "missing"
    if coarse:
        if value < 0.45:
            return "tie_density_low_lt_0_45"
        if value < 0.65:
            return "tie_density_mid_0_45_0_65"
        return "tie_density_high_ge_0_65"
    return _floor_bin(
        value,
        width=config.topology_tie_density_bin_width,
        upper=1.0 + config.topology_tie_density_bin_width,
    )


def _root_topology_signature(
    row: pd.Series,
    *,
    coarse: bool,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> str:
    parts = {
        "component": _string_value(row, "root_mixed_region_component"),
        "balance": _balance_signature(row, coarse=coarse, config=config),
        "merges": _merge_count_signature(row, coarse=coarse),
        "tie_density": _tie_density_signature(row, coarse=coarse, config=config),
        "tie_cell": _count_bin(
            _finite_float(row.get("root_child_discrete_tie_cell_count", math.nan))
        ),
        "smooth": _count_bin(
            _finite_float(row.get("root_child_smooth_constraint_count", math.nan))
        ),
    }
    if coarse:
        parts.pop("tie_cell")
        parts["tie_cell_presence"] = (
            "tie_cell_present"
            if _finite_float(row.get("root_child_discrete_tie_cell_count", math.nan)) > 0.0
            else "tie_cell_absent"
        )
        parts["smooth_presence"] = (
            "smooth_present"
            if _finite_float(row.get("root_child_smooth_constraint_count", math.nan)) > 0.0
            else "smooth_absent"
        )
        parts.pop("smooth")
    return "|".join(f"{key}={value}" for key, value in parts.items())


def _topology_signature_available(signature: str) -> bool:
    return bool(signature) and "missing" not in signature and "component=" in signature


def _topology_match_kind(
    *,
    target: pd.Series,
    support: pd.Series,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> str:
    target_exact = _root_topology_signature(target, coarse=False, config=config)
    support_exact = _root_topology_signature(support, coarse=False, config=config)
    if not _topology_signature_available(target_exact):
        return "target_topology_missing"
    if not _topology_signature_available(support_exact):
        return "support_topology_missing"
    if target_exact == support_exact:
        return "exact"
    target_coarse = _root_topology_signature(target, coarse=True, config=config)
    support_coarse = _root_topology_signature(support, coarse=True, config=config)
    if target_coarse == support_coarse:
        return "coarsened"
    return "mismatch"


def _squared_scaled_delta(
    left: float,
    right: float,
    scale: float,
    *,
    missing_penalty: float = 1.0,
) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return float(missing_penalty)
    return float(((left - right) / max(scale, 1e-12)) ** 2)


def _importance_log_weight(row: pd.Series) -> float:
    value = _finite_float(row.get("importance_log_weight", 0.0))
    return value if math.isfinite(value) else 0.0


def _tail_lookup(
    target_deformed: pd.DataFrame,
    support_deformed: pd.DataFrame,
) -> dict[str, float]:
    return {
        **_lookup_numeric_by_key(
            target_deformed,
            key_column="target_case_id",
            value_column="s_root_deformed_excess_log",
        ),
        **_lookup_numeric_by_key(
            support_deformed,
            key_column="case_id",
            value_column="s_root_deformed_excess_log",
        ),
    }


def _prepare_joined_rows(
    joined: pd.DataFrame,
    deformed_lookup: dict[str, float],
) -> pd.DataFrame:
    rows = joined.copy()
    for column in (
        "root_bandwidth_reopen_band",
        "root_mixed_region_component",
        "root_active_feature_count",
        "root_effective_independent_rows",
        "root_mp_upper_bound",
    ):
        if column not in rows.columns:
            rows[column] = math.nan if column.startswith("root_") else ""
    rows["s_h_u_excess_log"] = rows.apply(
        lambda row: _tail_excess_for_case(
            row,
            deformed_excess_by_case=deformed_lookup,
        )[0],
        axis=1,
    )
    rows["s_h_u_deformed_available"] = (
        rows["case_id"]
        .astype(str)
        .map(lambda case_id: math.isfinite(deformed_lookup.get(case_id, math.nan)))
    )
    return rows


def _strict_tail_lookup(strict_tail_rows: pd.DataFrame) -> dict[str, pd.Series]:
    if strict_tail_rows.empty or "target_case_id" not in strict_tail_rows.columns:
        return {}
    return {str(row["target_case_id"]): row for _, row in strict_tail_rows.iterrows()}


def _kernel_log_weight(
    *,
    target: pd.Series,
    support: pd.Series,
    bandwidths: dict[str, float],
    config: RootSelectedKernelSpectralTailLawConfig,
) -> float:
    if _string_value(target, "root_mixed_region_component") != _string_value(
        support,
        "root_mixed_region_component",
    ):
        return -math.inf

    target_bandwidth = _string_value(target, "root_bandwidth_reopen_band")
    support_bandwidth = _string_value(support, "root_bandwidth_reopen_band")
    bandwidth_weight = (
        1.0
        if target_bandwidth == support_bandwidth
        else max(float(config.bandwidth_mismatch_weight), 1e-12)
    )

    target_action = _safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
    support_action = _safe_log1p(support.get("root_sibling_selected_ratio", math.nan))
    target_edge = _safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
    support_edge = _safe_log1p(support.get("root_edge_path_statistic_margin", math.nan))
    target_tie = _finite_float(target.get("root_tie_rank_median_fraction", math.nan))
    support_tie = _finite_float(support.get("root_tie_rank_median_fraction", math.nan))
    target_log_k = _safe_log_feature(target, "root_active_feature_count")
    support_log_k = _safe_log_feature(support, "root_active_feature_count")
    target_log_rows = _safe_log_feature(target, "root_effective_independent_rows")
    support_log_rows = _safe_log_feature(support, "root_effective_independent_rows")
    target_log_h = _safe_log_positive(target, "root_mp_upper_bound")
    support_log_h = _safe_log_positive(support, "root_mp_upper_bound")

    distance = (
        _squared_scaled_delta(target_tie, support_tie, bandwidths["tau_tie"])
        + _squared_scaled_delta(target_action, support_action, bandwidths["tau_action"])
        + _squared_scaled_delta(target_edge, support_edge, bandwidths["tau_edge"])
        + _squared_scaled_delta(target_log_k, support_log_k, bandwidths["h_log_k"])
        + _squared_scaled_delta(
            target_log_rows,
            support_log_rows,
            bandwidths["h_log_rows"],
        )
        + _squared_scaled_delta(target_log_h, support_log_h, bandwidths["h_log_h_u"])
    )
    return float(math.log(bandwidth_weight) - 0.5 * distance + _importance_log_weight(support))


def _weighted_support_summary(
    *,
    target: pd.Series,
    support: pd.DataFrame,
    bandwidths: dict[str, float],
    config: RootSelectedKernelSpectralTailLawConfig,
) -> dict[str, object]:
    target_s = _finite_float(target.get("s_h_u_excess_log", math.nan))
    if support.empty:
        return {
            "support_count": 0,
            "positive_count": 0,
            "exceedance_count": 0,
            "weight_sum": 0.0,
            "effective_n": 0.0,
            "max_share": math.nan,
            "weighted_exceedance": math.nan,
            "p_value": math.nan,
            "status": "kernel_support_missing",
            "decision": "fail_closed_kernel_support_missing",
            "nearest_case_id": "",
            "nearest_weight_share": math.nan,
            "nearest_s": math.nan,
        }

    log_weights = np.asarray(
        [
            _kernel_log_weight(
                target=target,
                support=row,
                bandwidths=bandwidths,
                config=config,
            )
            for _, row in support.iterrows()
        ],
        dtype=float,
    )
    finite = np.isfinite(log_weights)
    if not bool(np.any(finite)):
        return {
            "support_count": int(support.shape[0]),
            "positive_count": int(
                pd.to_numeric(support["s_h_u_excess_log"], errors="coerce").gt(0.0).sum()
            ),
            "exceedance_count": 0,
            "weight_sum": 0.0,
            "effective_n": 0.0,
            "max_share": math.nan,
            "weighted_exceedance": math.nan,
            "p_value": math.nan,
            "status": "kernel_weight_support_missing",
            "decision": "fail_closed_kernel_weight_support_missing",
            "nearest_case_id": "",
            "nearest_weight_share": math.nan,
            "nearest_s": math.nan,
        }

    valid_support = support.loc[finite].copy()
    valid_log_weights = log_weights[finite]
    shifted = valid_log_weights - float(np.max(valid_log_weights))
    weights = np.exp(shifted)
    keep = weights >= float(config.min_kernel_weight)
    if not bool(np.any(keep)):
        return {
            "support_count": int(valid_support.shape[0]),
            "positive_count": int(
                pd.to_numeric(valid_support["s_h_u_excess_log"], errors="coerce").gt(0.0).sum()
            ),
            "exceedance_count": 0,
            "weight_sum": 0.0,
            "effective_n": 0.0,
            "max_share": math.nan,
            "weighted_exceedance": math.nan,
            "p_value": math.nan,
            "status": "kernel_weight_below_threshold",
            "decision": "fail_closed_kernel_weight_below_threshold",
            "nearest_case_id": "",
            "nearest_weight_share": math.nan,
            "nearest_s": math.nan,
        }

    valid_support = valid_support.loc[keep].copy()
    weights = weights[keep]
    support_s = pd.to_numeric(valid_support["s_h_u_excess_log"], errors="coerce").to_numpy(
        dtype=float,
    )
    finite_s = np.isfinite(support_s)
    weights = weights[finite_s]
    valid_support = valid_support.loc[finite_s].copy()
    support_s = support_s[finite_s]
    if len(support_s) == 0:
        return {
            "support_count": 0,
            "positive_count": 0,
            "exceedance_count": 0,
            "weight_sum": 0.0,
            "effective_n": 0.0,
            "max_share": math.nan,
            "weighted_exceedance": math.nan,
            "p_value": math.nan,
            "status": "kernel_tail_values_missing",
            "decision": "fail_closed_kernel_tail_values_missing",
            "nearest_case_id": "",
            "nearest_weight_share": math.nan,
            "nearest_s": math.nan,
        }

    weight_sum = float(np.sum(weights))
    shares = weights / weight_sum
    squared_sum = float(np.sum(weights * weights))
    effective_n = float(weight_sum * weight_sum / squared_sum) if squared_sum > 0 else 0.0
    exceedance = support_s >= target_s
    weighted_exceedance = float(np.sum(shares * exceedance.astype(float)))
    p_value = float((effective_n * weighted_exceedance + 1.0) / (effective_n + 1.0))
    positive_count = int(np.sum(support_s > 0.0))
    exceedance_count = int(np.sum(exceedance))
    max_index = int(np.argmax(shares))

    if effective_n < float(config.min_kernel_effective_sample_size):
        status = "kernel_effective_support_insufficient"
        decision = "fail_closed_kernel_effective_support_insufficient"
    elif float(np.max(shares)) > float(config.max_kernel_weight_share):
        status = "kernel_weight_too_concentrated"
        decision = "fail_closed_kernel_weight_too_concentrated"
    elif target_s > 0.0 and positive_count == 0:
        status = "kernel_nonzero_s_h_u_support_missing"
        decision = "fail_closed_kernel_nonzero_s_h_u_support_missing"
        p_value = math.nan
    else:
        status = "kernel_weighted_tail_available_diagnostic_only"
        decision = "candidate_tail_available_diagnostic_only"

    return {
        "support_count": int(valid_support.shape[0]),
        "positive_count": positive_count,
        "exceedance_count": exceedance_count,
        "weight_sum": weight_sum,
        "effective_n": effective_n,
        "max_share": float(np.max(shares)),
        "weighted_exceedance": weighted_exceedance,
        "p_value": p_value,
        "status": status,
        "decision": decision,
        "nearest_case_id": _string_value(valid_support.iloc[max_index], "case_id"),
        "nearest_weight_share": float(shares[max_index]),
        "nearest_s": float(support_s[max_index]),
    }


def _empty_topology_summary(
    *,
    exact_count: int,
    coarsened_count: int,
    status: str,
    decision: str,
    target: pd.Series,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> dict[str, object]:
    return {
        "target_signature": _root_topology_signature(
            target,
            coarse=False,
            config=config,
        ),
        "target_coarse_signature": _root_topology_signature(
            target,
            coarse=True,
            config=config,
        ),
        "exact_count": exact_count,
        "coarsened_count": coarsened_count,
        "support_count": exact_count + coarsened_count,
        "positive_count": 0,
        "exceedance_count": 0,
        "weight_sum": 0.0,
        "effective_n": 0.0,
        "max_share": math.nan,
        "weighted_exceedance": math.nan,
        "p_value": math.nan,
        "status": status,
        "decision": decision,
        "nearest_case_id": "",
        "nearest_weight_share": math.nan,
        "nearest_s": math.nan,
        "nearest_signature": "",
    }


def _topology_status_from_kernel(status: object) -> str:
    text = str(status)
    if text.startswith("kernel_"):
        return f"topology_{text}"
    return text


def _topology_decision_from_kernel(decision: object) -> str:
    text = str(decision)
    if text == "candidate_tail_available_diagnostic_only":
        return "topology_candidate_tail_available_diagnostic_only"
    if text.startswith("fail_closed_kernel_"):
        return "fail_closed_topology_" + text.removeprefix("fail_closed_kernel_")
    return text


def _topology_weighted_support_summary(
    *,
    target: pd.Series,
    support: pd.DataFrame,
    bandwidths: dict[str, float],
    config: RootSelectedKernelSpectralTailLawConfig,
) -> dict[str, object]:
    target_signature = _root_topology_signature(target, coarse=False, config=config)
    if not _topology_signature_available(target_signature):
        return _empty_topology_summary(
            exact_count=0,
            coarsened_count=0,
            status="topology_target_signature_missing",
            decision="fail_closed_topology_target_signature_missing",
            target=target,
            config=config,
        )

    if support.empty:
        return _empty_topology_summary(
            exact_count=0,
            coarsened_count=0,
            status="topology_support_missing",
            decision="fail_closed_topology_support_missing",
            target=target,
            config=config,
        )

    support_with_match = support.copy()
    support_with_match["_topology_match_kind"] = [
        _topology_match_kind(target=target, support=row, config=config)
        for _, row in support.iterrows()
    ]
    exact_mask = support_with_match["_topology_match_kind"].eq("exact")
    coarsened_mask = support_with_match["_topology_match_kind"].eq("coarsened")
    exact_count = int(exact_mask.sum())
    coarsened_count = int(coarsened_mask.sum())
    matched = support_with_match.loc[exact_mask | coarsened_mask].copy()
    if matched.empty:
        return _empty_topology_summary(
            exact_count=exact_count,
            coarsened_count=coarsened_count,
            status="topology_support_missing",
            decision="fail_closed_topology_support_missing",
            target=target,
            config=config,
        )

    coarsening_weight = max(float(config.topology_coarsening_weight), 1e-12)
    adjusted_log_weights: list[float] = []
    for _, row in matched.iterrows():
        topology_log_weight = 0.0
        if str(row["_topology_match_kind"]) == "coarsened":
            topology_log_weight = math.log(coarsening_weight)
        adjusted_log_weights.append(_importance_log_weight(row) + topology_log_weight)
    matched["importance_log_weight"] = adjusted_log_weights

    topology_config = replace(
        config,
        min_kernel_effective_sample_size=config.min_topology_effective_sample_size,
        max_kernel_weight_share=config.max_topology_weight_share,
    )
    summary = _weighted_support_summary(
        target=target,
        support=matched,
        bandwidths=bandwidths,
        config=topology_config,
    )
    nearest_case_id = str(summary["nearest_case_id"])
    nearest_signature = ""
    if nearest_case_id:
        nearest_rows = matched.loc[matched["case_id"].astype(str).eq(nearest_case_id)]
        if not nearest_rows.empty:
            nearest_signature = _root_topology_signature(
                nearest_rows.iloc[0],
                coarse=False,
                config=config,
            )
    return {
        "target_signature": target_signature,
        "target_coarse_signature": _root_topology_signature(
            target,
            coarse=True,
            config=config,
        ),
        "exact_count": exact_count,
        "coarsened_count": coarsened_count,
        "support_count": summary["support_count"],
        "positive_count": summary["positive_count"],
        "exceedance_count": summary["exceedance_count"],
        "weight_sum": summary["weight_sum"],
        "effective_n": summary["effective_n"],
        "max_share": summary["max_share"],
        "weighted_exceedance": summary["weighted_exceedance"],
        "p_value": summary["p_value"],
        "status": _topology_status_from_kernel(summary["status"]),
        "decision": _topology_decision_from_kernel(summary["decision"]),
        "nearest_case_id": summary["nearest_case_id"],
        "nearest_weight_share": summary["nearest_weight_share"],
        "nearest_s": summary["nearest_s"],
        "nearest_signature": nearest_signature,
    }


def _comparison_text(
    *,
    strict_status: str,
    kernel_decision: str,
) -> str:
    if strict_status.startswith("fail_closed") and kernel_decision.startswith("candidate"):
        return "kernel_candidate_adds_support_against_fail_closed_current"
    return "no_kernel_candidate_change_against_current"


def build_kernel_spectral_tail_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    strict_tail_rows: pd.DataFrame,
    deformed_mp_edge_rows: pd.DataFrame,
    deformed_mp_edge_support_rows: pd.DataFrame,
    observed_root_summary_rows: pd.DataFrame | None = None,
    config: RootSelectedKernelSpectralTailLawConfig,
) -> pd.DataFrame:
    """Return candidate kernel-spectral tail rows."""
    deformed_lookup = _tail_lookup(
        target_deformed=deformed_mp_edge_rows,
        support_deformed=deformed_mp_edge_support_rows,
    )
    rows = _prepare_joined_rows(joined_feasibility_rows, deformed_lookup)
    rows = _fill_observed_root_topology(
        rows=rows,
        observed_root_summary_rows=observed_root_summary_rows,
    )
    target_mask = rows.apply(_is_observed_target, axis=1)
    targets = rows.loc[target_mask].copy()
    support = rows.loc[~target_mask].copy()
    support = support.loc[support.apply(_is_calibration_support, axis=1)].copy()
    if config.require_deformed_support:
        support = support.loc[support["s_h_u_deformed_available"].astype(bool)].copy()
    non_support = rows.loc[~target_mask].copy()
    non_support = non_support.loc[~non_support.apply(_is_calibration_support, axis=1)].copy()
    bandwidths = _coordinate_bandwidths(targets, support)
    strict_lookup = _strict_tail_lookup(strict_tail_rows)

    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = _string_value(target, "case_id")
        strict = strict_lookup.get(case_id, pd.Series(dtype=object))
        support_summary = _weighted_support_summary(
            target=target,
            support=support,
            bandwidths=bandwidths,
            config=config,
        )
        topology_summary = _topology_weighted_support_summary(
            target=target,
            support=support,
            bandwidths=bandwidths,
            config=config,
        )
        excluded_count = int(non_support.shape[0])
        strict_status = _string_value(strict, "root_tail_inference_status")
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "target_s_h_u_excess_log": _finite_float(target.get("s_h_u_excess_log", math.nan)),
                "target_spectral_tail_variable": (
                    "deformed_mp_s_h_u"
                    if bool(target.get("s_h_u_deformed_available", False))
                    else "identity_mp_s_root"
                ),
                "target_t_selected_tie_rank_fraction": _finite_float(
                    target.get("root_tie_rank_median_fraction", math.nan)
                ),
                "target_a_selected_ratio_action_log1p": _safe_log1p(
                    target.get("root_sibling_selected_ratio", math.nan)
                ),
                "target_e_edge_margin_action_log1p": _safe_log1p(
                    target.get("root_edge_path_statistic_margin", math.nan)
                ),
                "target_b_bandwidth_topology_status": _string_value(
                    target,
                    "root_bandwidth_reopen_band",
                ),
                "target_log_active_feature_count": _safe_log_feature(
                    target,
                    "root_active_feature_count",
                ),
                "target_log_effective_rows": _safe_log_feature(
                    target,
                    "root_effective_independent_rows",
                ),
                "target_log_h_u_edge": _safe_log_positive(target, "root_mp_upper_bound"),
                "target_root_topology_signature": topology_summary["target_signature"],
                "target_root_topology_coarse_signature": topology_summary[
                    "target_coarse_signature"
                ],
                "strict_selected_null_support_count": _finite_int(
                    strict.get("selected_null_support_count", math.nan)
                ),
                "strict_conservative_spectral_tail_p_value": _finite_float(
                    strict.get("conservative_spectral_tail_p_value", math.nan)
                ),
                "strict_root_tail_inference_status": strict_status,
                "admissible_kernel_support_count": support_summary["support_count"],
                "non_support_neighbor_excluded_count": excluded_count,
                "kernel_positive_s_h_u_support_count": support_summary["positive_count"],
                "kernel_exceedance_count": support_summary["exceedance_count"],
                "kernel_weight_sum": support_summary["weight_sum"],
                "kernel_effective_sample_size": support_summary["effective_n"],
                "kernel_max_weight_share": support_summary["max_share"],
                "kernel_weighted_exceedance_fraction": support_summary["weighted_exceedance"],
                "kernel_conservative_tail_p_value": support_summary["p_value"],
                "kernel_tail_status": support_summary["status"],
                "kernel_candidate_decision": support_summary["decision"],
                "kernel_bandwidths_json": json.dumps(
                    bandwidths,
                    sort_keys=True,
                ),
                "nearest_support_case_id": support_summary["nearest_case_id"],
                "nearest_support_weight_share": support_summary["nearest_weight_share"],
                "nearest_support_s_h_u_excess_log": support_summary["nearest_s"],
                "topology_exact_support_count": topology_summary["exact_count"],
                "topology_coarsened_support_count": topology_summary["coarsened_count"],
                "topology_admissible_support_count": topology_summary["support_count"],
                "topology_positive_s_h_u_support_count": topology_summary["positive_count"],
                "topology_exceedance_count": topology_summary["exceedance_count"],
                "topology_weight_sum": topology_summary["weight_sum"],
                "topology_effective_sample_size": topology_summary["effective_n"],
                "topology_max_weight_share": topology_summary["max_share"],
                "topology_weighted_exceedance_fraction": topology_summary["weighted_exceedance"],
                "topology_conservative_tail_p_value": topology_summary["p_value"],
                "topology_kernel_status": topology_summary["status"],
                "topology_kernel_candidate_decision": topology_summary["decision"],
                "topology_nearest_support_case_id": topology_summary["nearest_case_id"],
                "topology_nearest_support_weight_share": topology_summary["nearest_weight_share"],
                "topology_nearest_support_s_h_u_excess_log": topology_summary["nearest_s"],
                "topology_nearest_support_signature": topology_summary["nearest_signature"],
                "comparison_to_current": _comparison_text(
                    strict_status=strict_status,
                    kernel_decision=str(support_summary["decision"]),
                ),
                "next_tracking_step": (
                    "promote_only_after_selected_null_false_split_check"
                    if str(support_summary["decision"]).startswith("candidate")
                    else "generate_or_reweight_admissible_nonzero_s_h_u_support"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_kernel_spectral_tail_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate kernel-spectral tail rows."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    strict_status = rows["strict_root_tail_inference_status"].astype(str)
    kernel_decision = rows["kernel_candidate_decision"].astype(str)
    topology_decision = rows["topology_kernel_candidate_decision"].astype(str)
    topology_status = rows["topology_kernel_status"].astype(str)
    kernel_available = kernel_decision.eq("candidate_tail_available_diagnostic_only")
    topology_available = topology_decision.eq("topology_candidate_tail_available_diagnostic_only")
    strict_fail_closed = strict_status.str.startswith("fail_closed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "row_count": int(rows.shape[0]),
        "strict_calibrated_count": int(
            strict_status.eq("calibrated_selected_root_spectral_tail_available").sum()
        ),
        "strict_fail_closed_count": int(strict_fail_closed.sum()),
        "kernel_available_count": int(kernel_available.sum()),
        "kernel_fail_closed_count": int((~kernel_available).sum()),
        "strict_fail_closed_kernel_available_count": int(
            (strict_fail_closed & kernel_available).sum()
        ),
        "kernel_nonzero_support_target_count": int(
            pd.to_numeric(
                rows["kernel_positive_s_h_u_support_count"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "topology_kernel_available_count": int(topology_available.sum()),
        "topology_kernel_fail_closed_count": int((~topology_available).sum()),
        "topology_strict_fail_closed_kernel_available_count": int(
            (strict_fail_closed & topology_available).sum()
        ),
        "topology_target_signature_missing_count": int(
            topology_status.eq("topology_target_signature_missing").sum()
        ),
        "topology_support_missing_count": int(topology_status.eq("topology_support_missing").sum()),
        "topology_degenerate_support_count": int(
            (
                topology_status.eq("topology_kernel_weight_too_concentrated")
                | topology_status.eq("topology_kernel_effective_support_insufficient")
            ).sum()
        ),
        "topology_nonzero_support_target_count": int(
            pd.to_numeric(
                rows["topology_positive_s_h_u_support_count"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "kernel_selected_null_leakage_flag_count": int(
            rows["non_support_neighbor_excluded_count"].fillna(0).astype(float).lt(0).sum()
        ),
        "summary_status": (
            "topology_kernel_adds_candidate_support_diagnostic_only"
            if int(topology_available.sum()) > 0
            else (
                "scalar_kernel_support_but_topology_fail_closed"
                if int(kernel_available.sum()) > 0
                else "kernel_support_still_fail_closed"
            )
        ),
    }
    return pd.DataFrame.from_records([summary], columns=SUMMARY_COLUMNS)


def evaluate_kernel_spectral_tail_law_panel(
    config: RootSelectedKernelSpectralTailLawConfig,
) -> dict[str, pd.DataFrame]:
    """Read inputs and return candidate kernel-spectral tail tables."""
    joined = pd.read_csv(config.joined_feasibility_rows_path, low_memory=False)
    strict = _read_optional_csv(config.strict_tail_rows_path)
    deformed_rows = _read_optional_csv(config.deformed_mp_edge_rows_path)
    deformed_support = _read_optional_csv(config.deformed_mp_edge_support_rows_path)
    observed_root_summary = _read_optional_csv(config.observed_root_summary_rows_path)
    rows = build_kernel_spectral_tail_rows(
        joined_feasibility_rows=joined,
        strict_tail_rows=strict,
        deformed_mp_edge_rows=deformed_rows,
        deformed_mp_edge_support_rows=deformed_support,
        observed_root_summary_rows=observed_root_summary,
        config=config,
    )
    summary = summarize_kernel_spectral_tail_rows(rows)
    return {"rows": rows, "summary": summary}


def run_kernel_spectral_tail_law_panel(
    config: RootSelectedKernelSpectralTailLawConfig,
) -> dict[str, Path]:
    """Run the diagnostic and write outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_kernel_spectral_tail_law_panel(config)
    paths = {
        "rows": config.output_dir / ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    outputs = run_kernel_spectral_tail_law_panel(config_from_args(parse_args()))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
