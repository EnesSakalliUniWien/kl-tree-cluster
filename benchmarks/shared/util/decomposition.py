"""Decomposition helpers for benchmark runners and pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)

from benchmarks.shared.types.method_run_result import MethodRunResult


def _empty_report_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["cluster_id", "cluster_size"],
        index=pd.Index([], name="sample_id"),
    )


def _report_dataframe_from_assignments(assignments_table: pd.DataFrame) -> pd.DataFrame:
    if assignments_table.empty:
        return _empty_report_dataframe()
    return assignments_table[["cluster_id", "cluster_size"]].copy()


def _labels_from_assignment_table(
    assignments_table: pd.DataFrame,
    sample_index: Sequence[object] | pd.Index,
) -> np.ndarray:
    ordered_index = _sample_index(sample_index).rename("sample_id")
    if assignments_table.empty:
        return np.full(len(ordered_index), -1, dtype=int)
    return (
        assignments_table["cluster_id"]
        .reindex(ordered_index)
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )


def _sample_index(sample_names: Sequence[object] | pd.Index) -> pd.Index:
    return sample_names if isinstance(sample_names, pd.Index) else pd.Index(sample_names)


def _labels_and_report_from_decomposition(
    decomposition: dict,
    sample_index: Sequence[object] | pd.Index,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert decomposition output once into ordered labels and a report table."""
    assignments_table = build_sample_cluster_assignments(decomposition)
    labels = _labels_from_assignment_table(assignments_table, sample_index)
    report_df = _report_dataframe_from_assignments(assignments_table)
    return labels, report_df


def _labels_from_decomposition(
    decomposition: dict, sample_index: Sequence[object] | pd.Index
) -> list[int]:
    """Extract cluster labels for each sample from a decomposition result."""
    labels, _ = _labels_and_report_from_decomposition(decomposition, sample_index)
    return labels.tolist()


def _create_report_dataframe_from_labels(
    labels: list[int] | np.ndarray, sample_names: Sequence[object] | pd.Index
) -> pd.DataFrame:
    """Create a report dataframe from flat cluster labels."""
    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.size == 0:
        return _empty_report_dataframe()
    series = pd.Series(labels_arr, index=_sample_index(sample_names), name="cluster_id")
    sizes = series.value_counts()
    report = pd.DataFrame(
        {
            "cluster_id": series,
            "cluster_size": series.map(sizes),
        }
    )
    report.index.name = "sample_id"
    return report


def _count_non_noise_clusters(labels: np.ndarray) -> int:
    return int(len({int(label) for label in labels if int(label) >= 0}))


def _ok_result_from_labels(
    labels: list[int] | np.ndarray,
    sample_names: Sequence[object] | pd.Index,
    *,
    extra: dict | None = None,
) -> MethodRunResult:
    """Build a successful runner result from flat labels."""
    labels_array = np.asarray(labels, dtype=int)
    return MethodRunResult(
        labels=labels_array,
        found_clusters=_count_non_noise_clusters(labels_array),
        report_df=_create_report_dataframe_from_labels(labels_array, sample_names),
        status="ok",
        skip_reason=None,
        extra=extra,
    )


__all__ = [
    "_labels_and_report_from_decomposition",
    "_labels_from_decomposition",
    "_create_report_dataframe_from_labels",
    "_ok_result_from_labels",
]
