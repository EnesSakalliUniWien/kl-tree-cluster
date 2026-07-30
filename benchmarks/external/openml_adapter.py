"""Optional OpenML adapters for reproducible external benchmark tasks."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from benchmarks.external.datasets import ExternalClusteringDataset, labels_to_numpy


def _load_openml_module() -> Any:
    try:
        return importlib.import_module("openml")
    except ImportError as exc:
        raise RuntimeError(
            "OpenML support requires the optional benchmark infrastructure extra: "
            "`uv sync --extra benchmark-infra`."
        ) from exc


def list_openml_suite_task_ids(suite: int | str) -> list[int]:
    """Return task IDs from an OpenML benchmark suite."""
    openml = _load_openml_module()
    loaded_suite = openml.study.get_suite(suite)
    return [int(task_id) for task_id in loaded_suite.tasks]


def load_openml_classification_task(
    task_id: int,
    *,
    categorical_as: str = "one-hot",
) -> ExternalClusteringDataset:
    """Load an OpenML supervised task as an external clustering benchmark.

    The target labels are used only as reference partitions for clustering
    evaluation; no train/test split is applied because clustering is unsupervised.
    """
    if categorical_as != "one-hot":
        raise ValueError("Only categorical_as='one-hot' is currently supported.")
    openml = _load_openml_module()
    task = openml.tasks.get_task(int(task_id))
    dataset = task.get_dataset()
    target_name = task.target_name
    frame, labels, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name,
        dataset_format="dataframe",
    )
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame, columns=attribute_names)
    data = pd.get_dummies(frame, dummy_na=True).reset_index(drop=True)
    data.index = [f"S{idx}" for idx in range(data.shape[0])]
    data.columns = [str(column) for column in data.columns]
    label_values = labels_to_numpy(labels)
    n_clusters = int(pd.Series(label_values).nunique(dropna=True))
    return ExternalClusteringDataset(
        case_id=f"openml_task_{int(task_id)}",
        data=data,
        labels=label_values,
        source="openml",
        benchmark_family="openml_classification_as_clustering_reference",
        n_clusters=n_clusters,
        metadata={
            "task_id": int(task_id),
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": str(dataset.name),
            "target_name": str(target_name),
            "categorical_as": categorical_as,
            "categorical_feature_count": int(sum(bool(item) for item in categorical_indicator)),
        },
    )


__all__ = [
    "list_openml_suite_task_ids",
    "load_openml_classification_task",
]
