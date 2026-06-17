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
stratum. Otherwise they fail closed. Optional legacy pairwise outputs are
joined to show how the old commit behaved on the same cases.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "root_selected_spectral_tail_law_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_spectral_tail_law_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root_selected_spectral_tail_law_panel"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_conditioned_coherent_topology_join_after_replay"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_LEGACY_FULL_PAIRWISE_ROWS = (
    DEFAULT_RESULT_ROOT
    / "legacy_c2ef9a69_method_comparison_panel"
    / "legacy_c2ef9a69_method_comparison_pairwise.csv"
)
DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS = (
    DEFAULT_RESULT_ROOT
    / "legacy_internal_spectral_comparison_panel"
    / "legacy_internal_spectral_comparison_pairwise.csv"
)

ROWS_OUTPUT = "root_selected_spectral_tail_law_rows.csv"
SUMMARY_OUTPUT = "root_selected_spectral_tail_law_summary.csv"
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
    "legacy_full_selected_null_current_clusters",
    "legacy_full_selected_null_legacy_clusters",
    "legacy_full_selected_null_legacy_false_split",
    "legacy_full_signal_current_clusters",
    "legacy_full_signal_legacy_clusters",
    "legacy_full_signal_delta_ari",
    "legacy_internal_selected_null_delta_raw_mp_signal_count_sum",
    "legacy_internal_signal_delta_raw_mp_signal_count_sum",
    "legacy_comparison_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "calibrated_tail_count",
    "fail_closed_missing_support_count",
    "legacy_full_selected_null_false_split_count",
    "legacy_full_signal_improvement_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedSpectralTailLawConfig:
    """Input/output contract for the selected-root spectral tail diagnostic."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    legacy_full_pairwise_rows_path: Path | None = DEFAULT_LEGACY_FULL_PAIRWISE_ROWS
    legacy_internal_pairwise_rows_path: Path | None = DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS
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
    parser.add_argument(
        "--legacy-full-pairwise-rows-path",
        type=Path,
        default=DEFAULT_LEGACY_FULL_PAIRWISE_ROWS,
    )
    parser.add_argument(
        "--legacy-internal-pairwise-rows-path",
        type=Path,
        default=DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS,
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


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedSpectralTailLawConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def _finite_int(value: object) -> int:
    numeric = _finite_float(value)
    return int(numeric) if math.isfinite(numeric) else 0


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


def _lookup_numeric_by_key(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
) -> dict[str, float]:
    if frame.empty or key_column not in frame.columns or value_column not in frame.columns:
        return {}
    lookup: dict[str, float] = {}
    for _, row in frame.iterrows():
        key = _string_value(row, key_column)
        if not key:
            continue
        value = _finite_float(row.get(value_column, math.nan))
        if math.isfinite(value):
            lookup[key] = value
    return lookup


def _tail_excess_for_case(
    row: pd.Series,
    *,
    deformed_excess_by_case: dict[str, float],
) -> tuple[float, float, float, str]:
    case_id = _string_value(row, "case_id")
    identity_excess = _spectral_excess_log(
        row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    deformed_excess = deformed_excess_by_case.get(case_id, math.nan)
    if math.isfinite(deformed_excess):
        return (
            float(deformed_excess),
            identity_excess,
            float(deformed_excess),
            "deformed_mp_s_h_u",
        )
    return identity_excess, identity_excess, math.nan, "identity_mp_s_root"


def _is_observed_target(row: pd.Series) -> bool:
    return (
        _string_value(row, "proposal_family") == "observed_target"
        or _string_value(row, "calibration_role")
        == "observed_target_not_null_support"
        or _string_value(row, "data_role") == "observed_target"
    )


def _is_calibration_support(row: pd.Series) -> bool:
    return (
        _string_value(row, "data_role") in CALIBRATION_SUPPORT_ROLES
        or _string_value(row, "calibration_role") in CALIBRATION_SUPPORT_ROLES
    )


def _tie_band(value: float) -> str:
    if not math.isfinite(value):
        return "tie_missing"
    if value < 0.70:
        return "tie_low_lt_0_70"
    if value < 0.85:
        return "tie_mid_0_70_0_85"
    return "tie_high_ge_0_85"


def _action_band(value: float) -> str:
    if not math.isfinite(value):
        return "action_missing"
    if value < 5.0:
        return "action_log_low_lt_5"
    if value < 7.0:
        return "action_log_mid_5_7"
    return "action_log_high_ge_7"


def _root_tail_stratum_key(
    *,
    target: pd.Series,
    h_u_population_law_status: str,
) -> str:
    tie = _finite_float(target.get("root_tie_rank_median_fraction", math.nan))
    action = _safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
    edge = _safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
    bandwidth = _string_value(target, "root_bandwidth_reopen_band", "")
    return "|".join(
        [
            _string_value(target, "root_mixed_region_component", "root_component_missing"),
            _tie_band(tie),
            _action_band(action),
            _action_band(edge),
            bandwidth or "bandwidth_missing",
            str(h_u_population_law_status),
        ]
    )


def _support_in_target_stratum(
    *,
    target: pd.Series,
    generated: pd.DataFrame,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    if generated.empty:
        return generated.copy()
    target_key = _root_tail_stratum_key(
        target=target,
        h_u_population_law_status=h_u_population_law_status,
    )
    rows = generated.copy()
    rows["_root_tail_stratum_key"] = rows.apply(
        lambda row: _root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
        axis=1,
    )
    rows = rows.loc[rows["_root_tail_stratum_key"].eq(target_key)].copy()
    support_mask = rows.apply(_is_calibration_support, axis=1)
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
    conservative_p = float(
        (effective_n * weighted_exceedance + 1.0) / (effective_n + 1.0)
    )
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


def _legacy_row(
    legacy_pairwise: pd.DataFrame,
    *,
    case_id: str,
    data_role: str,
) -> pd.Series | None:
    if legacy_pairwise.empty:
        return None
    rows = legacy_pairwise.loc[
        legacy_pairwise["case_id"].astype(str).eq(str(case_id))
        & legacy_pairwise["data_role"].astype(str).eq(str(data_role))
    ]
    return None if rows.empty else rows.iloc[0]


def _legacy_float(row: pd.Series | None, column: str) -> float:
    if row is None:
        return math.nan
    return _finite_float(row.get(column, math.nan))


def _legacy_bool(row: pd.Series | None, column: str) -> bool:
    if row is None:
        return False
    return bool(row.get(column, False))


def _legacy_interpretation(
    *,
    selected_null_row: pd.Series | None,
    signal_row: pd.Series | None,
    internal_selected_null_row: pd.Series | None,
    internal_signal_row: pd.Series | None,
) -> str:
    legacy_false_split = _legacy_bool(selected_null_row, "legacy_false_split")
    signal_delta = _legacy_float(signal_row, "delta_ari_legacy_minus_current")
    internal_delta = max(
        _legacy_float(internal_selected_null_row, "delta_raw_mp_signal_count_sum"),
        _legacy_float(internal_signal_row, "delta_raw_mp_signal_count_sum"),
    )
    if legacy_false_split:
        return "legacy_full_method_leaks_selected_null_root_risk"
    if math.isfinite(signal_delta) and signal_delta > 1e-12:
        return "legacy_full_method_signal_improves_but_not_calibration"
    if math.isfinite(internal_delta) and internal_delta > 0:
        return "legacy_internal_spectral_changes_mp_counts_only"
    return "legacy_comparison_neutral_or_missing"


def build_root_selected_spectral_tail_law_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    legacy_full_pairwise_rows: pd.DataFrame | None = None,
    legacy_internal_pairwise_rows: pd.DataFrame | None = None,
    deformed_mp_edge_rows: pd.DataFrame | None = None,
    deformed_mp_edge_support_rows: pd.DataFrame | None = None,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Return support-aware root spectral tail rows with legacy comparison."""
    _require_columns(
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
    target_mask = rows.apply(_is_observed_target, axis=1)
    targets = rows[target_mask].copy()
    generated = rows[~target_mask].copy()
    legacy_full = legacy_full_pairwise_rows if legacy_full_pairwise_rows is not None else pd.DataFrame()
    legacy_internal = (
        legacy_internal_pairwise_rows
        if legacy_internal_pairwise_rows is not None
        else pd.DataFrame()
    )
    target_deformed = (
        deformed_mp_edge_rows if deformed_mp_edge_rows is not None else pd.DataFrame()
    )
    support_deformed = (
        deformed_mp_edge_support_rows
        if deformed_mp_edge_support_rows is not None
        else pd.DataFrame()
    )
    deformed_excess_by_case = {
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
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = _string_value(target, "case_id")
        (
            s_root,
            identity_excess,
            deformed_excess,
            spectral_tail_variable,
        ) = _tail_excess_for_case(
            target,
            deformed_excess_by_case=deformed_excess_by_case,
        )
        t_rank = _finite_float(target.get("root_tie_rank_median_fraction", math.nan))
        action = _safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
        edge = _safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
        support = _support_in_target_stratum(
            target=target,
            generated=generated,
            h_u_population_law_status=str(h_u_population_law_status),
        )
        support_count = int(support.shape[0])
        if support_count:
            support_s = support.apply(
                lambda row: _tail_excess_for_case(
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
            p_value, effective_n, weighted_exceedance, p_value_status = (
                _weighted_tail_summary(
                    support=support,
                    exceedance_mask=exceedance_mask,
                )
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
        full_null = _legacy_row(legacy_full, case_id=case_id, data_role="selected_null")
        full_signal = _legacy_row(legacy_full, case_id=case_id, data_role="signal")
        internal_null = _legacy_row(
            legacy_internal,
            case_id=case_id,
            data_role="selected_null",
        )
        internal_signal = _legacy_row(legacy_internal, case_id=case_id, data_role="signal")
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
                "b_bandwidth_topology_status": _string_value(
                    target,
                    "root_bandwidth_reopen_band",
                ),
                "h_u_population_law_status": str(h_u_population_law_status),
                "root_tail_stratum_key": _root_tail_stratum_key(
                    target=target,
                    h_u_population_law_status=str(h_u_population_law_status),
                ),
                "selected_null_support_count": support_count,
                "selected_null_exceedance_count": exceedances,
                "selected_null_importance_effective_sample_size": effective_n,
                "selected_null_importance_weighted_exceedance_fraction": (
                    weighted_exceedance
                ),
                "conservative_spectral_tail_p_value": p_value,
                "spectral_tail_p_value_status": p_value_status,
                "root_tail_inference_status": inference_status,
                "next_mathematical_step": next_step,
                "legacy_full_selected_null_current_clusters": _finite_int(
                    full_null.get("current_found_clusters", math.nan)
                    if full_null is not None
                    else math.nan
                ),
                "legacy_full_selected_null_legacy_clusters": _finite_int(
                    full_null.get("legacy_found_clusters", math.nan)
                    if full_null is not None
                    else math.nan
                ),
                "legacy_full_selected_null_legacy_false_split": _legacy_bool(
                    full_null,
                    "legacy_false_split",
                ),
                "legacy_full_signal_current_clusters": _finite_int(
                    full_signal.get("current_found_clusters", math.nan)
                    if full_signal is not None
                    else math.nan
                ),
                "legacy_full_signal_legacy_clusters": _finite_int(
                    full_signal.get("legacy_found_clusters", math.nan)
                    if full_signal is not None
                    else math.nan
                ),
                "legacy_full_signal_delta_ari": _legacy_float(
                    full_signal,
                    "delta_ari_legacy_minus_current",
                ),
                "legacy_internal_selected_null_delta_raw_mp_signal_count_sum": _legacy_float(
                    internal_null,
                    "delta_raw_mp_signal_count_sum",
                ),
                "legacy_internal_signal_delta_raw_mp_signal_count_sum": _legacy_float(
                    internal_signal,
                    "delta_raw_mp_signal_count_sum",
                ),
                "legacy_comparison_interpretation": _legacy_interpretation(
                    selected_null_row=full_null,
                    signal_row=full_signal,
                    internal_selected_null_row=internal_null,
                    internal_signal_row=internal_signal,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_spectral_tail_law_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize root spectral-tail support and legacy overlay."""
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
    legacy_false = int(rows["legacy_full_selected_null_legacy_false_split"].sum())
    legacy_signal_improve = int(
        pd.to_numeric(rows["legacy_full_signal_delta_ari"], errors="coerce")
        .fillna(0.0)
        .gt(1e-12)
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
                "legacy_full_selected_null_false_split_count": legacy_false,
                "legacy_full_signal_improvement_count": legacy_signal_improve,
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
    legacy_full = _read_optional_csv(config.legacy_full_pairwise_rows_path)
    legacy_internal = _read_optional_csv(config.legacy_internal_pairwise_rows_path)
    deformed_rows = _read_optional_csv(config.deformed_mp_edge_rows_path)
    deformed_support_rows = _read_optional_csv(
        config.deformed_mp_edge_support_rows_path
    )
    rows = build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=joined,
        legacy_full_pairwise_rows=legacy_full,
        legacy_internal_pairwise_rows=legacy_internal,
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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_spectral_tail_law_panel(config)
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
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    outputs = run_root_selected_spectral_tail_law_panel(
        RootSelectedSpectralTailLawConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            legacy_full_pairwise_rows_path=args.legacy_full_pairwise_rows_path,
            legacy_internal_pairwise_rows_path=args.legacy_internal_pairwise_rows_path,
            deformed_mp_edge_rows_path=args.deformed_mp_edge_rows_path,
            deformed_mp_edge_support_rows_path=args.deformed_mp_edge_support_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
