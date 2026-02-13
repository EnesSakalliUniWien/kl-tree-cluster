"""Metric helpers for benchmarking pipeline.

Contains helpers for computing clustering metrics (ARI, NMI, Purity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score


def _calculate_ari_nmi_purity_metrics(
    num_clusters: int,
    report_df: pd.DataFrame | None,
    sample_names: pd.Index,
    true_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate clustering metrics (ARI, NMI, Purity).

    This mirrors the implementation that was previously in
    `benchmarking.pipeline` and is exported with the same name for
    backward compatibility.
    """
    _ = num_clusters  # Kept for backward-compatible signature.

    if report_df is None or report_df.empty:
        return np.nan, np.nan, np.nan
    if "cluster_id" not in report_df.columns:
        return np.nan, np.nan, np.nan

    sample_index = pd.Index(sample_names)
    true_arr = np.asarray(true_labels)

    # Metrics are only valid when predicted labels and truth align by sample.
    if len(sample_index) != len(true_arr):
        return np.nan, np.nan, np.nan
    if len(report_df.index) != len(sample_index):
        return np.nan, np.nan, np.nan
    if sample_index.has_duplicates or report_df.index.has_duplicates:
        return np.nan, np.nan, np.nan

    true_lookup = pd.Series(true_arr, index=sample_index)
    true_cluster = report_df.index.to_series().map(true_lookup)
    pred_cluster = report_df["cluster_id"]

    if true_cluster.isna().any() or pred_cluster.isna().any():
        return np.nan, np.nan, np.nan

    y_true = true_cluster.to_numpy()
    y_pred = pred_cluster.to_numpy()
    try:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        purity = homogeneity_score(y_true, y_pred)
    except ValueError:
        return np.nan, np.nan, np.nan

    return float(ari), float(nmi), float(purity)
