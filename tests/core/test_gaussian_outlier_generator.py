from __future__ import annotations

import numpy as np

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.generators.generate_gaussian_outliers import (
    GaussianOutlierConfig,
    generate_gaussian_outliers,
)


def test_singleton_outlier_generator_creates_unique_outlier_label() -> None:
    config = GaussianOutlierConfig(
        n_samples=121,
        n_features=20,
        n_inlier_clusters=4,
        outlier_count=1,
        outlier_distance=10.0,
        outlier_std=0.0,
        spatial_mode="clustered",
        label_mode="singleton",
        random_seed=42,
    )

    data, labels, metadata = generate_gaussian_outliers(config)

    assert data.shape == (121, 20)
    assert len(np.unique(labels)) == 5
    assert metadata["outlier_count"] == 1
    assert len(metadata["outlier_indices"]) == 1
    outlier_index = metadata["outlier_indices"][0]
    assert labels[outlier_index] == 4


def test_grouped_outlier_case_generation_returns_binary_matrix_and_metadata() -> None:
    case = {
        "name": "grouped_outlier_case",
        "generator": "gaussian_outliers",
        "n_samples": 166,
        "n_features": 30,
        "n_inlier_clusters": 4,
        "n_clusters": 5,
        "cluster_std": 0.7,
        "outlier_count": 6,
        "outlier_distance": 9.0,
        "outlier_std": 0.2,
        "outlier_spatial_mode": "clustered",
        "outlier_label_mode": "grouped",
        "seed": 7,
    }

    data_df, labels, x_original, metadata = generate_case_data(case)

    assert data_df.shape == (166, 30)
    assert x_original.shape == (166, 30)
    assert set(np.unique(data_df.values)).issubset({0, 1})
    assert len(np.unique(labels)) == 5
    assert metadata["generator"] == "gaussian_outliers"
    assert metadata["outlier_count"] == 6
    assert metadata["n_inlier_clusters"] == 4
    assert metadata["n_clusters"] == 5


def test_default_cases_include_outlier_family() -> None:
    all_cases = get_default_test_cases()
    family = [case for case in all_cases if case.get("generator") == "gaussian_outliers"]

    assert len(family) == 3
    assert {case["category"] for case in family} == {
        "gaussian_outlier_singleton",
        "gaussian_outlier_contamination",
    }
