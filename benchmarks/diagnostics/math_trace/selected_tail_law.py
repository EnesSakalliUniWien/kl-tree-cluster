"""Selected-tail ratio summaries for benchmark trace rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_selected_tail_variables(table: pd.DataFrame) -> pd.DataFrame:
    """Return selected-tail variables used by traceable calibration diagnostics."""
    result = table.copy()
    if "selected_ratio" not in result.columns:
        stat = pd.to_numeric(result.get("sibling_raw_stat"), errors="coerce")
        scale = pd.to_numeric(result.get("reference_scale"), errors="coerce")
        df = pd.to_numeric(result.get("degrees_of_freedom"), errors="coerce")
        result["selected_ratio"] = stat / (scale * df)
    min_edge_p = pd.concat(
        [
            pd.to_numeric(result.get("edge_left_raw_p"), errors="coerce"),
            pd.to_numeric(result.get("edge_right_raw_p"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    result["edge_action_from_p"] = -np.log(np.clip(min_edge_p, 1e-300, 1.0))
    cos2 = pd.to_numeric(result.get("selected_subspace_cos2"), errors="coerce")
    result["selected_subspace_tan2_from_cos2"] = (1.0 - cos2) / cos2.clip(lower=1e-12)
    return result


def summarize_tail_law(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Summarize selected-ratio tail exceedance by case and projection dimension."""
    variables = compute_selected_tail_variables(table)
    group_columns = [
        column
        for column in ("case_id", "sibling_projection_dimension", "internal_support_status")
        if column in variables.columns
    ]
    if not group_columns:
        group_columns = ["case_id"] if "case_id" in variables.columns else []
    rows: list[dict[str, object]] = []
    grouped = variables.groupby(group_columns, dropna=False) if group_columns else [((), variables)]
    for keys, group in grouped:
        ratio = pd.to_numeric(group["selected_ratio"], errors="coerce").dropna()
        exceedance = ratio > ratio.quantile(1.0 - alpha) if not ratio.empty else pd.Series(dtype=bool)
        row: dict[str, object] = {
            "n_records": int(len(group)),
            "n_ratio_records": int(len(ratio)),
            "heldout_exceedance_rate_at_alpha": float(exceedance.mean()) if len(exceedance) else np.nan,
            "heldout_exceedance_se": (
                float(np.sqrt(alpha * (1.0 - alpha) / len(exceedance)))
                if len(exceedance)
                else np.nan
            ),
        }
        if group_columns:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row.update(dict(zip(group_columns, keys, strict=False)))
        rows.append(row)
    return pd.DataFrame.from_records(rows)
