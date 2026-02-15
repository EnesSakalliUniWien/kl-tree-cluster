"""Decomposition helpers for benchmark runners and pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _labels_from_decomposition(
    decomposition: dict, sample_index: list[str]
) -> list[int]:
    """Extract cluster labels for each sample from a decomposition result."""
    assignments = {sample: -1 for sample in sample_index}
    for cluster_id, info in decomposition.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            assignments[leaf] = cluster_id
    return [assignments[sample] for sample in sample_index]


def _create_report_dataframe(cluster_assignments: dict) -> pd.DataFrame:
    """Create a report dataframe from cluster assignments."""
    if cluster_assignments:
        rows = []
        for cid, info in cluster_assignments.items():
            for leaf in info["leaves"]:
                rows.append(
                    {
                        "sample_id": leaf,
                        "cluster_id": cid,
                        "cluster_size": info["size"],
                    }
                )
        return pd.DataFrame(rows).set_index("sample_id")

    return pd.DataFrame(
        columns=["cluster_id", "cluster_size"],
        index=pd.Index([], name="sample_id"),
    )


def _create_report_dataframe_from_labels(
    labels: list[int] | np.ndarray, sample_names: pd.Index
) -> pd.DataFrame:
    """Create a report dataframe from flat cluster labels."""
    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.size == 0:
        return pd.DataFrame(
            columns=["cluster_id", "cluster_size"],
            index=pd.Index([], name="sample_id"),
        )
    series = pd.Series(labels_arr, index=sample_names, name="cluster_id")
    sizes = series.value_counts()
    report = pd.DataFrame(
        {
            "cluster_id": series,
            "cluster_size": series.map(sizes),
        }
    )
    report.index.name = "sample_id"
    return report


__all__ = [
    "_labels_from_decomposition",
    "_create_report_dataframe",
    "_create_report_dataframe_from_labels",
]
