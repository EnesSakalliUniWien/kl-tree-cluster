"""Shared target-versus-generated mechanics for root tie-rank diagnostics."""

from __future__ import annotations

import pandas as pd

from benchmarks.diagnostics.calibration.root.root_tail_values import (
    bandwidth_gap_status,
    is_observed_target,
    require_columns,
    string_value,
)

ROOT_TIE_RANK_COMPARISON_COLUMNS = {
    "case_id",
    "data_role",
    "calibration_role",
    "proposal_family",
    "root_bandwidth_reopen_band",
    "root_sibling_selected_ratio",
    "root_tie_rank_median_fraction",
    "root_edge_path_statistic_margin",
    "root_selected_eigenvalue_over_mp_upper_bound",
}


def partition_target_and_generated_rows(
    rows: pd.DataFrame,
    *,
    required_columns: set[str] | None = None,
    label: str = "proposal feasibility rows",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and partition observed target rows from generated proposals."""
    require_columns(rows, required_columns or ROOT_TIE_RANK_COMPARISON_COLUMNS, label)
    target_mask = rows.apply(is_observed_target, axis=1)
    return rows.loc[target_mask].copy(), rows.loc[~target_mask].copy()


def row_bandwidth_gap_status(target: pd.Series, generated: pd.Series) -> str:
    """Return the measured-bandwidth relation for a target/generated pair."""
    return bandwidth_gap_status(
        target_band=string_value(target, "root_bandwidth_reopen_band"),
        generated_band=string_value(generated, "root_bandwidth_reopen_band"),
    )


__all__ = [
    "ROOT_TIE_RANK_COMPARISON_COLUMNS",
    "partition_target_and_generated_rows",
    "row_bandwidth_gap_status",
]
