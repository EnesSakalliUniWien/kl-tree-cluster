"""DataFrame adapters for typed benchmark result rows."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from benchmarks.shared.results.models import BenchmarkResultRow

RESULT_COLUMNS = [
    "test_case",
    "case_id",
    "case_category",
    "method",
    "params",
    "true_clusters",
    "found_clusters",
    "samples",
    "features",
    "noise",
    "ari",
    "nmi",
    "purity",
    "macro_recall",
    "macro_f1",
    "worst_cluster_recall",
    "outlier_precision",
    "outlier_recall",
    "outlier_f1",
    "singleton_outlier_isolated",
    "grouped_outlier_cluster_recovered",
    "cluster_count_abs_error",
    "over_split",
    "under_split",
    "status",
    "skip_reason",
    "labels_length",
]


def benchmark_rows_to_dataframe(rows: Iterable[BenchmarkResultRow]) -> pd.DataFrame:
    """Convert typed benchmark rows to the stable output DataFrame schema."""
    records = [
        {
            "test_case": row.test_case,
            "case_id": row.case_id,
            "case_category": row.case_category,
            "method": row.method,
            "params": row.params_display,
            "true_clusters": row.true_clusters,
            "found_clusters": row.found_clusters,
            "samples": row.samples,
            "features": row.features,
            "noise": row.noise,
            "ari": row.ari,
            "nmi": row.nmi,
            "purity": row.purity,
            "macro_recall": row.macro_recall,
            "macro_f1": row.macro_f1,
            "worst_cluster_recall": row.worst_cluster_recall,
            "outlier_precision": row.outlier_precision,
            "outlier_recall": row.outlier_recall,
            "outlier_f1": row.outlier_f1,
            "singleton_outlier_isolated": row.singleton_outlier_isolated,
            "grouped_outlier_cluster_recovered": row.grouped_outlier_cluster_recovered,
            "cluster_count_abs_error": row.cluster_count_abs_error,
            "over_split": row.over_split,
            "under_split": row.under_split,
            "status": row.status.value,
            "skip_reason": row.skip_reason,
            "labels_length": row.labels_length,
        }
        for row in rows
    ]
    return pd.DataFrame(records, columns=RESULT_COLUMNS)


__all__ = ["RESULT_COLUMNS", "benchmark_rows_to_dataframe"]
