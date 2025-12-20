"""Utility functions for KL divergence-based tests."""

from __future__ import annotations

import pandas as pd


def get_local_kl_series(df: pd.DataFrame | pd.Series | None) -> pd.Series:
    """Extract the local KL divergence column as a float Series.

    Parameters
    ----------
    df : pd.DataFrame, pd.Series, or None
        DataFrame containing 'kl_divergence_local' column, or a Series, or None.

    Returns
    -------
    pd.Series
        Float Series with local KL divergence values.
    """
    if df is None:
        raise ValueError("Expected a DataFrame/Series with 'kl_divergence_local'.")

    if isinstance(df, pd.Series):
        return df.astype(float, copy=False)

    series = df.get("kl_divergence_local")
    if isinstance(series, pd.Series):
        return series.astype(float, copy=False)

    raise KeyError("Missing required column 'kl_divergence_local'.")


__all__ = ["get_local_kl_series"]
