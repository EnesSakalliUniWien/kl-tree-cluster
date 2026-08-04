"""Shared selected-root action-support geometry."""

from __future__ import annotations

import math

import pandas as pd

from benchmarks.diagnostics.calibration.root.root_tail_values import (
    action_band,
    finite_float,
    is_calibration_support,
    is_observed_target,
    require_columns,
    root_tail_stratum_key,
    safe_log1p,
    spectral_excess_log,
    string_value,
    tie_band,
)

ROOT_TAIL_ACTION_SUPPORT_COLUMNS = {
    "case_id",
    "data_role",
    "calibration_role",
    "proposal_family",
    "root_sibling_selected_ratio",
    "root_tie_rank_median_fraction",
    "root_edge_path_statistic_margin",
    "root_selected_eigenvalue_over_mp_upper_bound",
}


def calibration_support_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return calibration support while excluding observed targets."""
    support_mask = rows.apply(is_calibration_support, axis=1)
    target_mask = rows.apply(is_observed_target, axis=1)
    return rows.loc[support_mask & ~target_mask].copy()


def root_tail_coordinates(
    row: pd.Series,
    *,
    h_u_population_law_status: str,
) -> dict[str, object]:
    """Return the shared T, A, E, B, H_u root-tail coordinates."""
    tie = finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    return {
        "tie": tie,
        "action": action,
        "edge": edge,
        "tie_band": tie_band(tie),
        "action_band": action_band(action),
        "edge_band": action_band(edge),
        "bandwidth": string_value(row, "root_bandwidth_reopen_band", ""),
        "h_u": str(h_u_population_law_status),
        "s_root": spectral_excess_log(
            row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        ),
        "stratum": root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
    }


def annotate_root_tail_support(
    support: pd.DataFrame,
    *,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    """Attach shared root-tail coordinates to candidate support rows."""
    rows = support.copy()
    if rows.empty:
        return rows
    rows["_tie_band"] = rows["root_tie_rank_median_fraction"].map(
        lambda value: tie_band(finite_float(value))
    )
    rows["_action_log1p"] = rows["root_sibling_selected_ratio"].map(safe_log1p)
    rows["_action_band"] = rows["_action_log1p"].map(action_band)
    rows["_edge_log1p"] = rows["root_edge_path_statistic_margin"].map(safe_log1p)
    rows["_edge_band"] = rows["_edge_log1p"].map(action_band)
    rows["_bandwidth"] = rows.get("root_bandwidth_reopen_band", "").astype(str)
    rows["_h_u"] = str(h_u_population_law_status)
    rows["_s_root"] = rows["root_selected_eigenvalue_over_mp_upper_bound"].map(
        spectral_excess_log
    )
    rows["_root_tail_stratum_key"] = rows.apply(
        lambda row: root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
        axis=1,
    )
    return rows


def prepare_root_tail_action_support(
    joined_feasibility_rows: pd.DataFrame,
    *,
    h_u_population_law_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed targets and annotated admissible support rows.

    This is the shared preparation seam for selected-root action diagnostics;
    individual panels retain their own conditioning and inference laws.
    """
    require_columns(
        joined_feasibility_rows,
        ROOT_TAIL_ACTION_SUPPORT_COLUMNS,
        "joined feasibility rows",
    )
    rows = joined_feasibility_rows.copy()
    if "root_bandwidth_reopen_band" not in rows.columns:
        rows["root_bandwidth_reopen_band"] = ""
    if "root_mixed_region_component" not in rows.columns:
        rows["root_mixed_region_component"] = "root_component_missing"
    targets = rows.loc[rows.apply(is_observed_target, axis=1)].copy()
    support = annotate_root_tail_support(
        calibration_support_rows(rows),
        h_u_population_law_status=h_u_population_law_status,
    )
    return targets, support


__all__ = [
    "annotate_root_tail_support",
    "calibration_support_rows",
    "prepare_root_tail_action_support",
    "root_tail_coordinates",
    "string_value",
]
