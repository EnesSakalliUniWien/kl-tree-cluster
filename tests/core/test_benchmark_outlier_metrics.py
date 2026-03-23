from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.shared.metrics import _calculate_ari_nmi_purity_metrics


def test_outlier_metrics_report_perfect_singleton_isolation() -> None:
    sample_names = pd.Index(["S0", "S1", "S2", "S3", "S4"])
    report_df = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 1, 2]},
        index=sample_names,
    )
    y_true = np.array([0, 0, 1, 1, 2])

    metrics = _calculate_ari_nmi_purity_metrics(
        report_df,
        sample_names,
        y_true,
        {"outlier_indices": [4]},
    )

    assert metrics.outlier_precision == 1.0
    assert metrics.outlier_recall == 1.0
    assert metrics.outlier_f1 == 1.0
    assert metrics.singleton_outlier_isolated == 1.0
    assert np.isnan(metrics.grouped_outlier_cluster_recovered)


def test_outlier_metrics_report_missed_singleton_outlier() -> None:
    sample_names = pd.Index(["S0", "S1", "S2", "S3", "S4"])
    report_df = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 1, 1]},
        index=sample_names,
    )
    y_true = np.array([0, 0, 1, 1, 2])

    metrics = _calculate_ari_nmi_purity_metrics(
        report_df,
        sample_names,
        y_true,
        {"outlier_indices": [4]},
    )

    assert metrics.outlier_precision == 0.0
    assert metrics.outlier_recall == 0.0
    assert metrics.outlier_f1 == 0.0
    assert metrics.singleton_outlier_isolated == 0.0
    assert np.isnan(metrics.grouped_outlier_cluster_recovered)


def test_grouped_outlier_metric_requires_one_pure_predicted_cluster() -> None:
    sample_names = pd.Index(["S0", "S1", "S2", "S3", "S4", "S5"])
    y_true = np.array([0, 0, 1, 1, 2, 2])

    perfect_report = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 1, 2, 2]},
        index=sample_names,
    )
    perfect_metrics = _calculate_ari_nmi_purity_metrics(
        perfect_report,
        sample_names,
        y_true,
        {"outlier_indices": [4, 5]},
    )
    assert perfect_metrics.grouped_outlier_cluster_recovered == 1.0

    split_report = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 1, 2, 3]},
        index=sample_names,
    )
    split_metrics = _calculate_ari_nmi_purity_metrics(
        split_report,
        sample_names,
        y_true,
        {"outlier_indices": [4, 5]},
    )
    assert split_metrics.grouped_outlier_cluster_recovered == 0.0

    impure_report = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 2, 2, 2]},
        index=sample_names,
    )
    impure_metrics = _calculate_ari_nmi_purity_metrics(
        impure_report,
        sample_names,
        y_true,
        {"outlier_indices": [4, 5]},
    )
    assert impure_metrics.grouped_outlier_cluster_recovered == 0.0


def test_outlier_metrics_are_nan_for_non_outlier_cases() -> None:
    sample_names = pd.Index(["S0", "S1", "S2", "S3"])
    report_df = pd.DataFrame(
        {"cluster_id": [0, 0, 1, 1]},
        index=sample_names,
    )
    y_true = np.array([0, 0, 1, 1])

    metrics = _calculate_ari_nmi_purity_metrics(report_df, sample_names, y_true, {})

    assert np.isnan(metrics.outlier_precision)
    assert np.isnan(metrics.outlier_recall)
    assert np.isnan(metrics.outlier_f1)
    assert np.isnan(metrics.singleton_outlier_isolated)
    assert np.isnan(metrics.grouped_outlier_cluster_recovered)
