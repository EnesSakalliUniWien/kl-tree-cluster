from __future__ import annotations

import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis.tree.feature_space import FeatureSpace


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


def test_case_data_requires_explicit_name() -> None:
    with pytest.raises(ValueError, match="Benchmark case generator requires 'name'"):
        generate_case_data(
            {
                "generator": "blobs",
                "n_samples": 12,
                "n_features": 4,
                "n_clusters": 3,
                "cluster_std": 0.5,
            }
        )


def test_binary_case_data_rejects_old_geometry_names() -> None:
    with pytest.raises(ValueError, match="Binary generator requires 'n_samples'"):
        generate_case_data(
            {
                "name": "old_binary_shape",
                "generator": "binary",
                "n_rows": 12,
                "n_cols": 4,
                "n_clusters": 3,
                "seed": 1,
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
                "seed": 1,
            }
        )


def test_continuous_gaussian_ab_cases_keep_old_binary_cases() -> None:
    cases_by_name = {case["name"]: case for case in get_default_test_cases()}

    assert cases_by_name["gauss_clear_medium"]["generator"] == "blobs"
    assert cases_by_name["gauss_clear_medium_continuous"]["generator"] == "blobs_continuous"
    assert (
        cases_by_name["gauss_clear_medium_continuous"]["baseline_case_name"]
        == "gauss_clear_medium"
    )
    assert cases_by_name["dim_consolidated_4c_24f"]["generator"] == "dimensional_gaussian"
    assert (
        cases_by_name["dim_consolidated_4c_24f_continuous"]["generator"]
        == "dimensional_gaussian_continuous"
    )
    assert (
        cases_by_name["dim_consolidated_4c_24f_continuous"]["baseline_case_name"]
        == "dim_consolidated_4c_24f"
    )
    assert cases_by_name["gauss_single_outlier_4c"]["generator"] == "gaussian_outliers"
    assert (
        cases_by_name["gauss_single_outlier_4c_continuous"]["generator"]
        == "gaussian_outliers_continuous"
    )
    assert (
        cases_by_name["gauss_single_outlier_4c_continuous"]["baseline_case_name"]
        == "gauss_single_outlier_4c"
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "gauss_clear_medium_continuous",
        "dim_consolidated_4c_24f_continuous",
        "gauss_single_outlier_4c_continuous",
    ],
)
def test_continuous_case_data_carries_feature_space_and_euclidean_distance(
    case_name: str,
) -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == case_name)

    data_df, _labels, x_original, metadata = generate_case_data(case)

    feature_space = metadata["feature_space"]
    assert isinstance(feature_space, FeatureSpace)
    assert feature_space.family_label == "continuous"
    assert feature_space.raw_dimension == data_df.shape[1]
    assert data_df.attrs == {}
    assert x_original.shape == data_df.shape
    assert metadata["feature_representation"] == "continuous"
    assert metadata["distance_metric"] == "euclidean"
    assert metadata["requires_precomputed_kl_distance"] is True
    assert metadata["precomputed_distance_condensed"].shape[0] == (
        data_df.shape[0] * (data_df.shape[0] - 1) // 2
    )


def test_sbm_case_data_names_precomputed_kl_distance_metric() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "sbm_moderate")

    data_df, _labels, _x_original, metadata = generate_case_data(case)

    assert metadata["requires_precomputed_kl_distance"] is True
    assert metadata["distance_metric"] == "sbm_shifted_modularity"
    assert metadata["precomputed_distance_condensed"].shape[0] == (
        data_df.shape[0] * (data_df.shape[0] - 1) // 2
    )
