"""Projection-law audit helpers for selected MP and sibling dimensions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_projection_law(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize projection covariates by case."""
    rows: list[dict[str, object]] = []
    group_columns = ["case_id"] if "case_id" in table.columns else []
    grouped = table.groupby(group_columns, dropna=False) if group_columns else [((), table)]
    for keys, group in grouped:
        k = pd.to_numeric(group.get("sibling_projection_dimension"), errors="coerce")
        lambda_ratio = pd.to_numeric(group.get("lambda_k_over_mp"), errors="coerce")
        mass = pd.to_numeric(group.get("selected_eigenvalue_mass"), errors="coerce")
        cos2 = pd.to_numeric(group.get("selected_subspace_cos2"), errors="coerce")
        row: dict[str, object] = {
            "n_records": int(len(group)),
            "median_projection_dimension": float(k.median()) if k.notna().any() else np.nan,
            "share_lambda_over_mp": float((lambda_ratio > 1.0).mean())
            if lambda_ratio.notna().any()
            else np.nan,
            "median_selected_eigenvalue_mass": float(mass.median())
            if mass.notna().any()
            else np.nan,
            "median_selected_subspace_cos2": float(cos2.median()) if cos2.notna().any() else np.nan,
        }
        if group_columns:
            row["case_id"] = keys if not isinstance(keys, tuple) else keys[0]
        rows.append(row)
    return pd.DataFrame.from_records(rows)
