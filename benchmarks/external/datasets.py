"""Common normalized object for external clustering benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalClusteringDataset:
    """A normalized external dataset with reference clustering labels.

    External benchmark libraries use different conventions.  This type preserves
    the original reference labels while making the data and selected reference
    partition explicit enough for the local benchmark runner to consume later.
    """

    case_id: str
    data: pd.DataFrame
    labels: np.ndarray
    source: str
    benchmark_family: str
    n_clusters: int
    metadata: dict[str, Any]

    def as_case_metadata(self) -> dict[str, Any]:
        """Return portable metadata for benchmark result rows and reports."""
        return {
            "name": self.case_id,
            "source_family": self.source,
            "feature_representation": "external_continuous",
            "simulation_model": self.benchmark_family,
            "observation_model": "external_feature_matrix",
            "benchmark_intent": "external_clustering_benchmark",
            "scientific_caution": (
                "External benchmark labels are reference partitions; they may encode "
                "noise or multiple valid clusterings depending on the source."
            ),
            "recommended_simulation_family": self.benchmark_family,
            "n_samples": int(self.data.shape[0]),
            "n_features": int(self.data.shape[1]),
            "n_clusters": int(self.n_clusters),
            **self.metadata,
        }


def labels_to_numpy(labels: object) -> np.ndarray:
    """Return a one-dimensional numpy label vector without relabeling semantics."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError(f"Reference labels must be one-dimensional, got shape={values.shape!r}.")
    return values


def matrix_to_dataframe(data: object, *, sample_prefix: str = "S") -> pd.DataFrame:
    """Return a stable numeric DataFrame for an external feature matrix."""
    frame = pd.DataFrame(np.asarray(data))
    frame.index = [f"{sample_prefix}{idx}" for idx in range(frame.shape[0])]
    frame.columns = [f"F{idx}" for idx in range(frame.shape[1])]
    return frame


__all__ = [
    "ExternalClusteringDataset",
    "labels_to_numpy",
    "matrix_to_dataframe",
]
