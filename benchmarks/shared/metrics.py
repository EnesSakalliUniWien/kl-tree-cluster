"""Metric helpers for benchmarking pipeline.

Contains helpers for computing clustering metrics (ARI, NMI, Purity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class ClusteringMetrics:
    """Clustering metrics computed for a single method run."""

    ari: float
    nmi: float
    purity: float
    macro_recall: float
    macro_f1: float
    worst_cluster_recall: float


def _nan_metrics() -> ClusteringMetrics:
    return ClusteringMetrics(
        ari=np.nan,
        nmi=np.nan,
        purity=np.nan,
        macro_recall=np.nan,
        macro_f1=np.nan,
        worst_cluster_recall=np.nan,
    )


def _remap_predicted_labels_to_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map predicted cluster ids to true-label ids for per-class metrics."""
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_true_enc = le_true.fit_transform(y_true)
    y_pred_enc = le_pred.fit_transform(y_pred)

    cm = contingency_matrix(y_true_enc, y_pred_enc)
    if cm.size == 0:
        return y_true_enc, y_pred_enc

    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize agreement
    mapping_enc = {pred: true for true, pred in zip(row_ind, col_ind)}

    # Map any extra predicted clusters to their majority true class.
    for pred in range(cm.shape[1]):
        if pred not in mapping_enc:
            mapping_enc[pred] = int(np.argmax(cm[:, pred]))

    y_pred_remapped_enc = np.asarray([mapping_enc[p] for p in y_pred_enc], dtype=int)
    return y_true_enc, y_pred_remapped_enc


def _calculate_ari_nmi_purity_metrics(
    report_df: pd.DataFrame | None,
    sample_names: pd.Index,
    true_labels: np.ndarray,
) -> ClusteringMetrics:
    """Calculate clustering metrics (ARI, NMI, Purity + label-matched class metrics).
    """

    if report_df is None or report_df.empty:
        return _nan_metrics()
    if "cluster_id" not in report_df.columns:
        return _nan_metrics()

    sample_index = pd.Index(sample_names)
    true_arr = np.asarray(true_labels)

    # Metrics are only valid when predicted labels and truth align by sample.
    if len(sample_index) != len(true_arr):
        return _nan_metrics()
    if len(report_df.index) != len(sample_index):
        return _nan_metrics()
    if sample_index.has_duplicates or report_df.index.has_duplicates:
        return _nan_metrics()

    true_lookup = pd.Series(true_arr, index=sample_index)
    true_cluster = report_df.index.to_series().map(true_lookup)
    pred_cluster = report_df["cluster_id"]

    if true_cluster.isna().any() or pred_cluster.isna().any():
        return _nan_metrics()

    y_true = true_cluster.to_numpy()
    y_pred = pred_cluster.to_numpy()
    try:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        purity = homogeneity_score(y_true, y_pred)
        y_true_enc, y_pred_remapped_enc = _remap_predicted_labels_to_truth(y_true, y_pred)
        labels = np.arange(len(np.unique(y_true_enc)), dtype=int)
        if labels.size == 0:
            macro_recall = np.nan
            macro_f1 = np.nan
            worst_cluster_recall = np.nan
        else:
            per_class_recall = recall_score(
                y_true_enc,
                y_pred_remapped_enc,
                labels=labels,
                average=None,
                zero_division=0,
            )
            macro_recall = float(np.mean(per_class_recall))
            macro_f1 = float(
                f1_score(
                    y_true_enc,
                    y_pred_remapped_enc,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            worst_cluster_recall = float(np.min(per_class_recall))
    except ValueError:
        return _nan_metrics()

    return ClusteringMetrics(
        ari=float(ari),
        nmi=float(nmi),
        purity=float(purity),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        worst_cluster_recall=float(worst_cluster_recall),
    )
