"""Metric helpers for benchmarking pipeline.

Contains helpers for computing clustering metrics and outlier-isolation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class ClusteringMetrics:
    """Clustering metrics computed for a single method run."""

    ari: float
    nmi: float
    ami: float
    purity: float
    homogeneity: float
    completeness: float
    v_measure: float
    fowlkes_mallows: float
    macro_recall: float
    macro_f1: float
    worst_cluster_recall: float
    n_singleton_clusters: float
    singleton_fraction: float
    median_cluster_size: float
    largest_cluster_fraction: float
    effective_cluster_count: float
    cluster_size_entropy: float
    cluster_size_gini: float
    noise_label_fraction: float
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    outlier_precision: float
    outlier_recall: float
    outlier_f1: float
    singleton_outlier_isolated: float
    grouped_outlier_cluster_recovered: float


def _nan_metrics() -> ClusteringMetrics:
    return ClusteringMetrics(
        ari=np.nan,
        nmi=np.nan,
        ami=np.nan,
        purity=np.nan,
        homogeneity=np.nan,
        completeness=np.nan,
        v_measure=np.nan,
        fowlkes_mallows=np.nan,
        macro_recall=np.nan,
        macro_f1=np.nan,
        worst_cluster_recall=np.nan,
        n_singleton_clusters=np.nan,
        singleton_fraction=np.nan,
        median_cluster_size=np.nan,
        largest_cluster_fraction=np.nan,
        effective_cluster_count=np.nan,
        cluster_size_entropy=np.nan,
        cluster_size_gini=np.nan,
        noise_label_fraction=np.nan,
        silhouette_score=np.nan,
        davies_bouldin_index=np.nan,
        calinski_harabasz_index=np.nan,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
    )


def _unit_interval_score(value: float) -> float:
    """Normalize bounded scores against tiny floating-point overshoots."""
    score = float(value)
    if not np.isfinite(score):
        return score
    return float(np.clip(score, 0.0, 1.0))


def _is_noise_label(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        return bool(float(value) == -1.0)
    except (TypeError, ValueError):
        return str(value).strip() == "-1"


def _effective_cluster_count_from_sizes(sizes: pd.Series) -> tuple[float, float]:
    if sizes.empty or float(sizes.sum()) <= 0.0:
        return np.nan, np.nan
    probabilities = sizes.to_numpy(dtype=float) / float(sizes.sum())
    probabilities = probabilities[probabilities > 0.0]
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return float(np.exp(entropy)), entropy


def _cluster_size_gini(sizes: pd.Series) -> float:
    values = np.sort(sizes.to_numpy(dtype=float))
    if values.size == 0 or np.isclose(values.sum(), 0.0):
        return np.nan
    index = np.arange(1, values.size + 1)
    numerator = ((2.0 * index - values.size - 1.0) * values).sum()
    denominator = values.size * values.sum()
    return float(numerator / denominator)


def _cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return majority-label purity across predicted clusters."""
    matrix = contingency_matrix(y_true, y_pred)
    if matrix.size == 0 or matrix.sum() <= 0:
        return np.nan
    return float(np.max(matrix, axis=0).sum() / matrix.sum())


def _calculate_fragmentation_metrics(
    pred_cluster: pd.Series,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return cluster-size diagnostics excluding DBSCAN-style noise label -1."""
    if pred_cluster.empty or pred_cluster.isna().any():
        return (np.nan,) * 8

    noise_mask = pred_cluster.map(_is_noise_label).to_numpy(dtype=bool)
    noise_label_fraction = float(noise_mask.mean()) if noise_mask.size else np.nan
    non_noise = pred_cluster.loc[~noise_mask]
    if non_noise.empty:
        return (
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            noise_label_fraction,
        )

    sizes = non_noise.value_counts()
    n_samples = float(len(pred_cluster))
    effective_count, entropy = _effective_cluster_count_from_sizes(sizes)
    return (
        float((sizes == 1).sum()),
        float(sizes[sizes == 1].sum() / n_samples),
        float(sizes.median()),
        float(sizes.max() / n_samples),
        effective_count,
        entropy,
        _cluster_size_gini(sizes),
        noise_label_fraction,
    )


def _calculate_internal_geometry_metrics(
    feature_matrix: np.ndarray | pd.DataFrame | None,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Return silhouette, Davies-Bouldin, and Calinski-Harabasz when defined."""
    if feature_matrix is None:
        return np.nan, np.nan, np.nan
    values = np.asarray(feature_matrix, dtype=float)
    labels = np.asarray(y_pred)
    if values.ndim != 2 or values.shape[0] != labels.shape[0] or values.shape[0] == 0:
        return np.nan, np.nan, np.nan
    if not np.isfinite(values).all():
        return np.nan, np.nan, np.nan

    non_noise = np.array([not _is_noise_label(label) for label in labels], dtype=bool)
    values = values[non_noise]
    labels = labels[non_noise]
    n_samples = labels.shape[0]
    n_labels = np.unique(labels).size
    if not 1 < n_labels < n_samples:
        return np.nan, np.nan, np.nan

    silhouette = np.nan
    davies_bouldin = np.nan
    calinski_harabasz = np.nan
    try:
        sample_size = min(2000, n_samples)
        silhouette = float(
            silhouette_score(
                values,
                labels,
                sample_size=sample_size if sample_size < n_samples else None,
                random_state=0,
            )
        )
    except ValueError:
        silhouette = np.nan
    try:
        davies_bouldin = float(davies_bouldin_score(values, labels))
    except ValueError:
        davies_bouldin = np.nan
    try:
        calinski_harabasz = float(calinski_harabasz_score(values, labels))
    except ValueError:
        calinski_harabasz = np.nan
    return silhouette, davies_bouldin, calinski_harabasz


def _calculate_outlier_isolation_metrics(
    report_df: pd.DataFrame,
    sample_names: pd.Index,
    outlier_indices: list[int] | tuple[int, ...] | np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute outlier-isolation metrics from predicted clusters.

    A predicted cluster counts as an outlier cluster when a strict majority of its
    members are true outliers. This makes the metric meaningful even when methods
    do not emit a dedicated noise label.
    """

    if len(outlier_indices) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sample_index = pd.Index(sample_names)
    if sample_index.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    outlier_mask_by_position = np.zeros(len(sample_index), dtype=bool)
    outlier_positions = np.asarray(outlier_indices, dtype=int)
    if np.any(outlier_positions < 0) or np.any(outlier_positions >= len(sample_index)):
        raise ValueError("outlier_indices contains positions outside sample_names")
    outlier_mask_by_position[outlier_positions] = True

    outlier_lookup = pd.Series(outlier_mask_by_position, index=sample_index)
    true_outlier = report_df.index.to_series().map(outlier_lookup)
    pred_cluster = report_df["cluster_id"]

    if true_outlier.isna().any() or pred_cluster.isna().any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    cluster_outlier_fraction = true_outlier.groupby(pred_cluster).mean()
    predicted_outlier_clusters = cluster_outlier_fraction[cluster_outlier_fraction > 0.5].index
    predicted_outlier = pred_cluster.isin(predicted_outlier_clusters).to_numpy(dtype=bool)
    true_outlier_arr = true_outlier.to_numpy(dtype=bool)

    outlier_precision = float(precision_score(true_outlier_arr, predicted_outlier, zero_division=0))
    outlier_recall = float(recall_score(true_outlier_arr, predicted_outlier, zero_division=0))
    outlier_f1 = float(f1_score(true_outlier_arr, predicted_outlier, zero_division=0))

    singleton_outlier_isolated = np.nan
    if len(outlier_positions) == 1:
        outlier_sample_name = sample_index[int(outlier_positions[0])]
        if outlier_sample_name in pred_cluster.index:
            outlier_cluster = pred_cluster.loc[outlier_sample_name]
            singleton_outlier_isolated = float((pred_cluster == outlier_cluster).sum() == 1)

    grouped_outlier_cluster_recovered = np.nan
    if len(outlier_positions) > 1:
        outlier_sample_names = sample_index[outlier_positions]
        outlier_cluster_ids = pred_cluster.loc[outlier_sample_names]
        if outlier_cluster_ids.nunique(dropna=False) == 1:
            grouped_cluster_id = outlier_cluster_ids.iloc[0]
            grouped_cluster_mask = pred_cluster == grouped_cluster_id
            grouped_outlier_cluster_recovered = float(
                true_outlier[grouped_cluster_mask].all()
                and int(grouped_cluster_mask.sum()) == len(outlier_positions)
            )
        else:
            grouped_outlier_cluster_recovered = 0.0

    return (
        outlier_precision,
        outlier_recall,
        outlier_f1,
        singleton_outlier_isolated,
        grouped_outlier_cluster_recovered,
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
    metadata: dict[str, object] | None = None,
    feature_matrix: np.ndarray | pd.DataFrame | None = None,
) -> ClusteringMetrics:
    """Calculate external, fragmentation, internal, and outlier metrics."""

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
    (
        n_singleton_clusters,
        singleton_fraction,
        median_cluster_size,
        largest_cluster_fraction,
        effective_cluster_count,
        cluster_size_entropy,
        cluster_size_gini,
        noise_label_fraction,
    ) = _calculate_fragmentation_metrics(pred_cluster)
    (
        internal_silhouette,
        davies_bouldin,
        calinski_harabasz,
    ) = _calculate_internal_geometry_metrics(feature_matrix, y_pred)
    outlier_precision = np.nan
    outlier_recall = np.nan
    outlier_f1 = np.nan
    singleton_outlier_isolated = np.nan
    grouped_outlier_cluster_recovered = np.nan
    try:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        purity = _cluster_purity(y_true, y_pred)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            y_true,
            y_pred,
        )
        fowlkes_mallows = fowlkes_mallows_score(y_true, y_pred)
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
        outlier_indices_raw = None if metadata is None else metadata.get("outlier_indices")
        if (
            isinstance(outlier_indices_raw, (list, tuple, np.ndarray))
            and len(outlier_indices_raw) > 0
        ):
            (
                outlier_precision,
                outlier_recall,
                outlier_f1,
                singleton_outlier_isolated,
                grouped_outlier_cluster_recovered,
            ) = _calculate_outlier_isolation_metrics(
                report_df,
                sample_names,
                outlier_indices_raw,
            )
    except ValueError:
        return _nan_metrics()

    return ClusteringMetrics(
        ari=float(ari),
        nmi=_unit_interval_score(nmi),
        ami=float(ami),
        purity=_unit_interval_score(purity),
        homogeneity=_unit_interval_score(homogeneity),
        completeness=_unit_interval_score(completeness),
        v_measure=_unit_interval_score(v_measure),
        fowlkes_mallows=_unit_interval_score(fowlkes_mallows),
        macro_recall=_unit_interval_score(macro_recall),
        macro_f1=_unit_interval_score(macro_f1),
        worst_cluster_recall=_unit_interval_score(worst_cluster_recall),
        n_singleton_clusters=float(n_singleton_clusters),
        singleton_fraction=_unit_interval_score(singleton_fraction),
        median_cluster_size=float(median_cluster_size),
        largest_cluster_fraction=_unit_interval_score(largest_cluster_fraction),
        effective_cluster_count=float(effective_cluster_count),
        cluster_size_entropy=float(cluster_size_entropy),
        cluster_size_gini=_unit_interval_score(cluster_size_gini),
        noise_label_fraction=_unit_interval_score(noise_label_fraction),
        silhouette_score=float(internal_silhouette),
        davies_bouldin_index=float(davies_bouldin),
        calinski_harabasz_index=float(calinski_harabasz),
        outlier_precision=float(outlier_precision),
        outlier_recall=float(outlier_recall),
        outlier_f1=float(outlier_f1),
        singleton_outlier_isolated=float(singleton_outlier_isolated),
        grouped_outlier_cluster_recovered=float(grouped_outlier_cluster_recovered),
    )
