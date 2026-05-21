from __future__ import annotations

import pytest

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data


def test_default_benchmark_cases_declare_generator_and_canonical_geometry() -> None:
    for case in get_default_test_cases():
        assert "generator" in case
        assert "n_rows" not in case
        assert "n_cols" not in case


def test_case_data_requires_explicit_generator() -> None:
    with pytest.raises(ValueError, match="Benchmark case generator requires 'generator'"):
        generate_case_data(
            {
                "name": "implicit_blobs_is_not_a_contract",
                "n_samples": 12,
                "n_features": 4,
                "n_clusters": 3,
                "cluster_std": 0.5,
            }
        )


def test_binary_case_data_rejects_legacy_geometry_names() -> None:
    with pytest.raises(ValueError, match="Binary generator requires 'n_samples'"):
        generate_case_data(
            {
                "name": "legacy_binary_shape",
                "generator": "binary",
                "n_rows": 12,
                "n_cols": 4,
                "n_clusters": 3,
            }
        )


def test_categorical_case_data_requires_category_count() -> None:
    with pytest.raises(ValueError, match="Categorical generator requires 'n_categories'"):
        generate_case_data(
            {
                "name": "missing_category_count",
                "generator": "categorical",
                "n_samples": 12,
                "n_features": 4,
                "n_clusters": 3,
            }
        )
