from __future__ import annotations

import pytest
from benchmarks.shared.cases import get_default_test_cases, get_test_cases_by_suite
from benchmarks.shared.cases.geometry import case_recipe_geometry
from benchmarks.shared.generators.case_data_contracts import case_metadata
from benchmarks.shared.generators.generate_case_data import generate_case_data
from tree_break_selection.tree.feature_space import FeatureSpace


def test_default_benchmark_cases_declare_generator_and_canonical_geometry() -> None:
    for case in get_default_test_cases():
        assert "generator" in case
        assert "n_rows" not in case
        assert "n_cols" not in case


def test_generated_case_metadata_names_source_family_and_representation() -> None:
    for case in get_default_test_cases():
        _data_df, _labels, _x_original, metadata = generate_case_data(case)
        assert isinstance(metadata["source_family"], str)
        assert metadata["source_family"]
        assert isinstance(metadata["feature_representation"], str)
        assert metadata["feature_representation"]


def test_case_recipe_geometry_matches_generated_metadata() -> None:
    for case in get_default_test_cases():
        _data_df, _labels, _x_original, metadata = generate_case_data(case)
        assert case_recipe_geometry(case) == (
            int(metadata["n_samples"]),
            int(metadata["n_features"]),
        )


def test_benchmark_case_suites_separate_input_contracts() -> None:
    binary_cases = get_test_cases_by_suite("binary")
    continuous_cases = get_test_cases_by_suite("continuous")
    categorical_cases = get_test_cases_by_suite("categorical")
    discretized_gaussian_cases = get_test_cases_by_suite("discretized_gaussian")
    graph_cases = get_test_cases_by_suite("graph")

    assert binary_cases
    assert continuous_cases
    assert categorical_cases
    assert discretized_gaussian_cases
    assert graph_cases

    assert {case["generator"] for case in binary_cases} == {"binary"}
    assert {case["generator"] for case in continuous_cases} == {
        "blobs_continuous",
        "dimensional_gaussian_continuous",
        "gaussian_outliers_continuous",
    }
    assert len(continuous_cases) == 9
    assert {case["generator"] for case in graph_cases} == {"sbm"}

    binary_names = {case["name"] for case in binary_cases}
    continuous_names = {case["name"] for case in continuous_cases}
    assert binary_names.isdisjoint(continuous_names)
    assert continuous_names == {
        "gauss_clear_medium_continuous",
        "gauss_moderate_3c_continuous",
        "gauss_null_large_continuous",
        "gauss_extreme_noise_highd_continuous",
        "dim_consolidated_4c_24f_continuous",
        "dim_consolidated_4c_72f_continuous",
        "dim_diffuse_6c_136f_continuous",
        "gauss_single_outlier_4c_continuous",
        "gauss_outlier_cluster_4c_continuous",
    }


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


def test_precomputed_tbs_distance_requires_explicit_metadata_flag() -> None:
    with pytest.raises(ValueError, match="requires_precomputed_tbs_distance=True"):
        case_metadata(
            test_case={"name": "broken_distance_contract"},
            n_samples=2,
            n_features=1,
            n_clusters=1,
            noise=0.0,
            generator="test",
            source_family="test",
            feature_representation="test",
            requires_precomputed_tbs_distance=False,
            precomputed_distance_condensed=[1.0],
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


def test_selected_continuous_examples_forward_baseline_case_contracts() -> None:
    cases_by_name = {case["name"]: case for case in get_default_test_cases()}

    assert cases_by_name["gauss_clear_medium"]["generator"] == "blobs"
    assert cases_by_name["gauss_clear_medium_continuous"]["generator"] == "blobs_continuous"
    assert (
        cases_by_name["gauss_clear_medium_continuous"]["baseline_case_name"] == "gauss_clear_medium"
    )
    assert (
        cases_by_name["gauss_clear_medium_continuous"]["representation_role"]
        == "continuous_example"
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
    assert (
        cases_by_name["dim_consolidated_4c_24f_continuous"]["representation_role"]
        == "continuous_dimensional_example"
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
    assert (
        cases_by_name["gauss_single_outlier_4c_continuous"]["representation_role"]
        == "continuous_outlier_example"
    )


def test_gaussian_source_cases_name_discretized_or_continuous_representation() -> None:
    cases_by_name = {case["name"]: case for case in get_default_test_cases()}

    _data_df, _labels, _x_original, binary_metadata = generate_case_data(
        cases_by_name["gauss_clear_medium"]
    )
    assert binary_metadata["source_family"] == "gaussian_blobs"
    assert binary_metadata["feature_representation"] == "median_binary"

    _data_df, _labels, _x_original, continuous_metadata = generate_case_data(
        cases_by_name["gauss_clear_medium_continuous"]
    )
    assert continuous_metadata["source_family"] == "gaussian_blobs"
    assert continuous_metadata["feature_representation"] == "continuous"


@pytest.mark.parametrize(
    "case_name",
    [
        "gauss_clear_medium_continuous",
        "dim_consolidated_4c_24f_continuous",
        "gauss_single_outlier_4c_continuous",
    ],
)
def test_gaussian_continuous_case_data_uses_standardized_euclidean_tree_distance(
    case_name: str,
) -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == case_name)

    data_df, _labels, x_original, metadata = generate_case_data(case)

    feature_space = metadata["feature_space"]
    assert isinstance(feature_space, FeatureSpace)
    assert feature_space.family_label == "continuous"
    assert feature_space.raw_dimension == data_df.shape[1]
    assert len(feature_space.blocks) == 1
    assert feature_space.blocks[0].name == "continuous"
    assert feature_space.blocks[0].column_indices == tuple(range(data_df.shape[1]))
    assert data_df.attrs == {}
    assert x_original.shape == data_df.shape
    assert metadata["feature_representation"] == "continuous"
    assert metadata["distance_metric"] == "standardized_euclidean"
    assert metadata["requires_precomputed_tbs_distance"] is True
    assert metadata["precomputed_distance_condensed"].shape[0] == (
        data_df.shape[0] * (data_df.shape[0] - 1) // 2
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "cont_lowrank_pggn_shrinkage",
        "mp_spike_below_bbp_continuous",
        "phylo_brownian_null_16taxa",
    ],
)
def test_diagnostic_continuous_case_data_keeps_mahalanobis_time_distance(
    case_name: str,
) -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == case_name)

    data_df, _labels, x_original, metadata = generate_case_data(case)

    feature_space = metadata["feature_space"]
    assert isinstance(feature_space, FeatureSpace)
    assert feature_space.family_label == "continuous"
    assert data_df.attrs == {}
    assert x_original.shape == data_df.shape
    assert metadata["feature_representation"] == "continuous"
    assert metadata["distance_metric"] == "mahalanobis_time"
    assert metadata["requires_precomputed_tbs_distance"] is True
    assert metadata["precomputed_distance_condensed"].shape[0] == (
        data_df.shape[0] * (data_df.shape[0] - 1) // 2
    )


def test_sbm_case_data_names_precomputed_tbs_distance_metric() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "sbm_moderate")

    data_df, _labels, _x_original, metadata = generate_case_data(case)

    assert metadata["requires_precomputed_tbs_distance"] is True
    assert metadata["distance_metric"] == "sbm_shifted_modularity"
    assert metadata["precomputed_distance_condensed"].shape[0] == (
        data_df.shape[0] * (data_df.shape[0] - 1) // 2
    )
