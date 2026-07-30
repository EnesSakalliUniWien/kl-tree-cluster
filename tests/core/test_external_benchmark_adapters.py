from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from benchmarks.external.clustbench_adapter import load_clustbench_dataset
from benchmarks.external.openml_adapter import (
    list_openml_suite_task_ids,
    load_openml_classification_task,
)


def test_clustbench_adapter_normalizes_multiple_reference_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dataset = SimpleNamespace(
        data=np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]),
        labels=[np.array([1, 1, 2]), np.array([1, 2, 2])],
        n_clusters=np.array([2, 2]),
    )
    fake_clustbench = SimpleNamespace(
        load_dataset=lambda battery, dataset, **kwargs: fake_dataset,
    )
    monkeypatch.setitem(sys.modules, "clustbench", fake_clustbench)

    external = load_clustbench_dataset(
        "wut",
        "x2",
        url="https://example.test/clustering-data",
        label_index=1,
    )

    assert external.case_id == "clustbench_wut_x2_labels1"
    assert external.data.shape == (3, 2)
    assert external.labels.tolist() == [1, 2, 2]
    assert external.n_clusters == 2
    assert external.metadata["reference_partition_count"] == 2
    assert external.as_case_metadata()["benchmark_intent"] == "external_clustering_benchmark"


def test_clustbench_adapter_rejects_invalid_reference_partition_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dataset = SimpleNamespace(
        data=np.array([[0.0], [1.0]]),
        labels=[np.array([1, 2])],
        n_clusters=np.array([2]),
    )
    fake_clustbench = SimpleNamespace(
        load_dataset=lambda battery, dataset, **kwargs: fake_dataset,
    )
    monkeypatch.setitem(sys.modules, "clustbench", fake_clustbench)

    with pytest.raises(ValueError, match="label_index"):
        load_clustbench_dataset("wut", "x2", label_index=3)


def test_openml_adapter_lists_suite_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_openml = SimpleNamespace(
        study=SimpleNamespace(get_suite=lambda suite: SimpleNamespace(tasks=[11, "12"])),
    )
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    assert list_openml_suite_task_ids("OpenML-CC18") == [11, 12]


def test_openml_adapter_loads_classification_task_as_reference_clustering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        dataset_id = 42
        name = "toy"

        def get_data(self, *, target: str, dataset_format: str):
            assert target == "class"
            assert dataset_format == "dataframe"
            return (
                pd.DataFrame({"x": [0.0, 1.0, 2.0], "cat": ["a", "b", "a"]}),
                pd.Series(["left", "right", "left"]),
                [False, True],
                ["x", "cat"],
            )

    fake_task = SimpleNamespace(
        target_name="class",
        get_dataset=lambda: FakeDataset(),
    )
    fake_openml = SimpleNamespace(
        tasks=SimpleNamespace(get_task=lambda task_id: fake_task),
    )
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    external = load_openml_classification_task(7)

    assert external.case_id == "openml_task_7"
    assert external.data.shape[0] == 3
    assert external.n_clusters == 2
    assert external.metadata["dataset_id"] == 42
    assert external.metadata["categorical_feature_count"] == 1


def test_openml_adapter_rejects_non_one_hot_categorical_mode() -> None:
    with pytest.raises(ValueError, match="one-hot"):
        load_openml_classification_task(7, categorical_as="raw")
