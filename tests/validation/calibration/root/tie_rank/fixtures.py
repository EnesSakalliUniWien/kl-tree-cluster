"""Shared input fixtures for root tie-rank panel tests."""

from __future__ import annotations

import pandas as pd


def feasibility_rows(
    *,
    full: tuple[float, float, float, float],
    dense: tuple[float, float, float, float],
    sparse: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Build a target plus three deliberately contrasting proposal rows."""
    records = [
        _row("target", "observed_target", "observed_target_not_null_support", "bandwidth_root_reopen_observed", (100.0, 0.80, 1000.0, 5.0)),
        _row("full", "full_family", "diagnostic_proposal_not_null_support", "bandwidth_root_reopen_observed", full),
        _row("dense", "dense_family", "diagnostic_proposal_not_null_support", "bandwidth_reopen_missing", dense),
        _row("sparse", "sparse_family", "diagnostic_proposal_not_null_support", "bandwidth_reopen_missing", sparse),
    ]
    return pd.DataFrame.from_records(records)


def _row(
    case_id: str,
    proposal_family: str,
    calibration_role: str,
    bandwidth_band: str,
    values: tuple[float, float, float, float],
) -> dict[str, object]:
    selected_ratio, tie_fraction, edge_margin, spectral_ratio = values
    return {
        "case_id": case_id,
        "data_role": "observed_target" if proposal_family == "observed_target" else "diagnostic_proposal",
        "calibration_role": calibration_role,
        "proposal_family": proposal_family,
        "root_bandwidth_reopen_band": bandwidth_band,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
    }
