"""Helpers for accessing and computing child-vs-parent KL divergences.

These utilities hide the bookkeeping around cached values stored on the tree
and provide a robust Series accessor used by multiple statistical routines.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import networkx as nx

from .decomposition_utils import binary_kl


def get_local_kl_value(
    tree: nx.DiGraph,
    child_id: str,
    parent_id: str | None = None,
    *,
    child_dist: Iterable[float] | None = None,
    parent_dist: Iterable[float] | None = None,
    default: float | None = np.nan,
) -> float | None:
    """Return local KL(child‖parent) using stored value or on-the-fly recompute."""

    node_data = tree.nodes[child_id]
    stored = node_data.get("kl_divergence_local")
    if stored is not None and np.isfinite(stored):
        return float(stored)

    if parent_id is None:
        return default

    if child_dist is None:
        child_dist = node_data.get("distribution")
    if parent_dist is None:
        parent_dist = tree.nodes[parent_id].get("distribution")
    if child_dist is None or parent_dist is None:
        return default

    child_arr = np.asarray(child_dist, dtype=float)
    parent_arr = np.asarray(parent_dist, dtype=float)
    return float(binary_kl(child_arr, parent_arr))


def get_local_kl_series(df: pd.DataFrame | pd.Series | None) -> pd.Series:
    """Extract the local KL divergence column as a float Series with safe defaults."""

    if df is None:
        return pd.Series(dtype=float)

    if isinstance(df, pd.Series):
        # Caller already supplied the Series; coerce to float dtype.
        return df.astype(float, copy=False)

    series = df.get("kl_divergence_local")
    if isinstance(series, pd.Series):
        return series.astype(float, copy=False)

    return pd.Series(index=getattr(df, "index", []), dtype=float)
