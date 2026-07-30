"""Optional adapter for the clustering-benchmarks/clustbench suite."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.external.datasets import (
    ExternalClusteringDataset,
    labels_to_numpy,
    matrix_to_dataframe,
)

CLUSTBENCH_DATA_URL = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"


def _load_clustbench_module() -> Any:
    try:
        return importlib.import_module("clustbench")
    except ImportError as exc:
        raise RuntimeError(
            "clustbench support requires the optional benchmark infrastructure extra: "
            "`uv sync --extra benchmark-infra`."
        ) from exc


def list_clustbench_batteries(*, path: Path | str | None = None) -> list[str]:
    """List available clustbench benchmark batteries from a local suite path."""
    clustbench = _load_clustbench_module()
    return list(clustbench.get_battery_names(path=str(path) if path is not None else None))


def list_clustbench_datasets(
    battery: str,
    *,
    path: Path | str | None = None,
) -> list[str]:
    """List datasets in one clustbench battery from a local suite path."""
    clustbench = _load_clustbench_module()
    return list(
        clustbench.get_dataset_names(
            battery,
            path=str(path) if path is not None else None,
        )
    )


def load_clustbench_dataset(
    battery: str,
    dataset: str,
    *,
    path: Path | str | None = None,
    url: str | None = CLUSTBENCH_DATA_URL,
    label_index: int = 0,
) -> ExternalClusteringDataset:
    """Load a clustbench dataset and select one reference partition.

    clustbench can expose multiple valid reference partitions for the same data.
    The adapter preserves the number of reference partitions in metadata and
    selects one partition explicitly via ``label_index`` for local metrics.
    """
    clustbench = _load_clustbench_module()
    kwargs: dict[str, object] = {}
    if path is not None:
        kwargs["path"] = str(path)
    elif url is not None:
        kwargs["url"] = url
    loaded = clustbench.load_dataset(battery, dataset, **kwargs)

    reference_partitions = list(loaded.labels)
    if not reference_partitions:
        raise ValueError(f"clustbench dataset {battery}/{dataset} has no reference labels.")
    if label_index < 0 or label_index >= len(reference_partitions):
        raise ValueError(
            f"label_index must select one of {len(reference_partitions)} reference partitions; "
            f"got {label_index}."
        )

    labels = labels_to_numpy(reference_partitions[label_index])
    data = matrix_to_dataframe(loaded.data)
    n_clusters_values = np.asarray(loaded.n_clusters)
    n_clusters = int(n_clusters_values[label_index])
    return ExternalClusteringDataset(
        case_id=f"clustbench_{battery}_{dataset}_labels{label_index}",
        data=data,
        labels=labels,
        source="clustbench",
        benchmark_family=f"clustbench:{battery}",
        n_clusters=n_clusters,
        metadata={
            "battery": battery,
            "dataset": dataset,
            "label_index": int(label_index),
            "reference_partition_count": len(reference_partitions),
            "source_url": url or "",
        },
    )


__all__ = [
    "CLUSTBENCH_DATA_URL",
    "list_clustbench_batteries",
    "list_clustbench_datasets",
    "load_clustbench_dataset",
]
