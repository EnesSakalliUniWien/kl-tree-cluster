"""Canonical contrast-covariance object for projected Wald tests."""

from __future__ import annotations

import numpy as np
import pytest
import tree_break_selection.hierarchy_analysis.statistics.contrast_covariance as contrast_covariance_module
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
    build_null_whitened_tangent_matrix,
    compute_whitened_wald_contrast,
)
from tree_break_selection.tree.continuous_distance import (
    continuous_time_distance_condensed,
    standardized_euclidean_distance_condensed,
)
from tree_break_selection.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)


def _block_diagonal_entries(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.array([block[0, 0] for block in blocks], dtype=np.float64)


def _make_categorical_space(n_features: int, n_categories: int) -> FeatureSpace:
    columns = tuple(f"F{i}_c{k}" for i in range(n_features) for k in range(n_categories))
    return infer_feature_space_from_columns(columns)


def _make_continuous_space() -> FeatureSpace:
    return FeatureSpace(
        column_names=("X0", "X1"),
        blocks=(
            FeatureBlock(
                name="X",
                family="continuous",
                column_indices=(0, 1),
                chart="identity",
                covariance="empirical_gaussian",
                contrast_dimension=2,
            ),
        ),
    )


def test_sibling_bernoulli_contrast_covariance_matches_null_variance() -> None:
    left = np.array([0.2, 0.7], dtype=np.float64)
    right = np.array([0.5, 0.4], dtype=np.float64)
    n_left = 40.0
    n_right = 60.0

    contrast_covariance = build_contrast_covariance(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        ridge=0.0,
    )

    pooled = (n_left * left + n_right * right) / (n_left + n_right)
    expected_variance = pooled * (1.0 - pooled) * (1.0 / n_left + 1.0 / n_right)

    assert contrast_covariance.feature_family == "bernoulli"
    assert contrast_covariance.degrees_of_freedom == left.size
    np.testing.assert_allclose(contrast_covariance.contrast_vector, left - right)
    np.testing.assert_allclose(
        _block_diagonal_entries(contrast_covariance.covariance_blocks),
        expected_variance,
    )


def test_child_parent_bernoulli_contrast_covariance_uses_nested_variance() -> None:
    child = np.array([0.3, 0.8], dtype=np.float64)
    parent = np.array([0.5, 0.6], dtype=np.float64)
    n_child = 25.0
    n_parent = 100.0

    contrast_covariance = build_contrast_covariance(
        child,
        parent,
        n_child,
        n_parent,
        comparison="child_parent",
        ridge=0.0,
    )

    nested_factor = 1.0 / n_child - 1.0 / n_parent
    expected_variance = parent * (1.0 - parent) * nested_factor

    assert contrast_covariance.feature_family == "bernoulli"
    assert contrast_covariance.degrees_of_freedom == child.size
    np.testing.assert_allclose(contrast_covariance.contrast_vector, child - parent)
    np.testing.assert_allclose(
        _block_diagonal_entries(contrast_covariance.covariance_blocks),
        expected_variance,
    )


def test_sibling_bernoulli_contrast_covariance_inflates_by_tree_time() -> None:
    left = np.array([0.2, 0.7], dtype=np.float64)
    right = np.array([0.5, 0.4], dtype=np.float64)
    n_left = 40.0
    n_right = 60.0

    contrast_covariance = build_contrast_covariance(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        tree_time=0.4,
        tree_time_normalizer=0.2,
        ridge=0.0,
    )

    pooled = (n_left * left + n_right * right) / (n_left + n_right)
    sampling_scale = 1.0 / n_left + 1.0 / n_right
    expected_variance = pooled * (1.0 - pooled) * sampling_scale * 3.0

    np.testing.assert_allclose(
        _block_diagonal_entries(contrast_covariance.covariance_blocks),
        expected_variance,
    )


def test_child_parent_bernoulli_contrast_covariance_inflates_by_branch_time() -> None:
    child = np.array([0.3, 0.8], dtype=np.float64)
    parent = np.array([0.5, 0.6], dtype=np.float64)

    contrast_covariance = build_contrast_covariance(
        child,
        parent,
        25.0,
        100.0,
        comparison="child_parent",
        tree_time=0.25,
        tree_time_normalizer=0.5,
        ridge=0.0,
    )

    nested_factor = 1.0 / 25.0 - 1.0 / 100.0
    expected_variance = parent * (1.0 - parent) * nested_factor * 1.5

    np.testing.assert_allclose(
        _block_diagonal_entries(contrast_covariance.covariance_blocks),
        expected_variance,
    )


def test_positive_tree_time_requires_positive_normalizer() -> None:
    with pytest.raises(ValueError, match="tree_time_normalizer"):
        build_contrast_covariance(
            np.array([0.2], dtype=np.float64),
            np.array([0.4], dtype=np.float64),
            20.0,
            30.0,
            comparison="sibling",
            tree_time=0.2,
        )


def test_sibling_categorical_contrast_covariance_uses_multinomial_blocks() -> None:
    left_blocks = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    right_blocks = np.array([[0.4, 0.2, 0.4], [0.3, 0.3, 0.4]], dtype=np.float64)
    left = left_blocks.ravel()
    right = right_blocks.ravel()
    feature_space = _make_categorical_space(n_features=2, n_categories=3)
    n_left = 50.0
    n_right = 150.0

    contrast_covariance = build_contrast_covariance(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        feature_space=feature_space,
        ridge=0.0,
    )

    pooled = (n_left * left_blocks + n_right * right_blocks) / (n_left + n_right)
    variance_scale = 1.0 / n_left + 1.0 / n_right
    expected_first_block = (
        np.diag(pooled[0, :-1]) - np.outer(pooled[0, :-1], pooled[0, :-1])
    ) * variance_scale

    assert contrast_covariance.feature_family == "categorical"
    assert contrast_covariance.degrees_of_freedom == 4
    np.testing.assert_allclose(
        contrast_covariance.contrast_vector,
        (left_blocks[:, :-1] - right_blocks[:, :-1]).ravel(),
    )
    np.testing.assert_allclose(contrast_covariance.covariance_blocks[0], expected_first_block)


def test_child_parent_categorical_contrast_covariance_uses_parent_multinomial_blocks() -> None:
    child_blocks = np.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2]], dtype=np.float64)
    parent_blocks = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2]], dtype=np.float64)
    child = child_blocks.ravel()
    parent = parent_blocks.ravel()
    feature_space = _make_categorical_space(n_features=2, n_categories=3)
    n_child = 30.0
    n_parent = 120.0

    contrast_covariance = build_contrast_covariance(
        child,
        parent,
        n_child,
        n_parent,
        comparison="child_parent",
        feature_space=feature_space,
        ridge=0.0,
    )

    nested_factor = 1.0 / n_child - 1.0 / n_parent
    expected_second_block = (
        np.diag(parent_blocks[1, :-1]) - np.outer(parent_blocks[1, :-1], parent_blocks[1, :-1])
    ) * nested_factor

    assert contrast_covariance.feature_family == "categorical"
    assert contrast_covariance.degrees_of_freedom == 4
    np.testing.assert_allclose(
        contrast_covariance.contrast_vector,
        (child_blocks[:, :-1] - parent_blocks[:, :-1]).ravel(),
    )
    np.testing.assert_allclose(contrast_covariance.covariance_blocks[1], expected_second_block)


def test_mixed_feature_space_contrast_covariance_combines_block_charts() -> None:
    feature_space = infer_feature_space_from_columns(
        ("B0", "F0_c0", "F0_c1", "F0_c2", "B1")
    )
    left = np.array([0.2, 0.1, 0.3, 0.6, 0.8], dtype=np.float64)
    right = np.array([0.5, 0.2, 0.2, 0.6, 0.4], dtype=np.float64)

    contrast_covariance = build_contrast_covariance(
        left,
        right,
        25.0,
        50.0,
        comparison="sibling",
        feature_space=feature_space,
    )

    assert contrast_covariance.feature_family == "mixed"
    assert contrast_covariance.degrees_of_freedom == 4
    np.testing.assert_allclose(
        contrast_covariance.contrast_vector,
        np.array([-0.3, -0.1, 0.1, 0.4], dtype=np.float64),
    )
    assert [block.shape for block in contrast_covariance.covariance_blocks] == [
        (1, 1),
        (2, 2),
        (1, 1),
    ]


def test_continuous_contrast_covariance_uses_empirical_gaussian_block() -> None:
    feature_space = _make_continuous_space()
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]], dtype=np.float64)
    left = np.array([1.5, -0.5], dtype=np.float64)
    right = np.array([0.5, 2.0], dtype=np.float64)
    n_left = 20.0
    n_right = 30.0
    ridge = 1e-6

    contrast_covariance = build_contrast_covariance(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block={"X": covariance},
        ridge=ridge,
    )

    expected_covariance = covariance * (1.0 / n_left + 1.0 / n_right) + ridge * np.eye(2)
    assert contrast_covariance.feature_family == "continuous"
    assert contrast_covariance.degrees_of_freedom == 2
    np.testing.assert_allclose(contrast_covariance.contrast_vector, left - right)
    np.testing.assert_allclose(contrast_covariance.covariance_blocks[0], expected_covariance)


def test_continuous_contrast_requires_covariance_block() -> None:
    feature_space = _make_continuous_space()

    with pytest.raises(ValueError, match="continuous_covariance_by_block"):
        build_contrast_covariance(
            np.array([0.2, 0.1], dtype=np.float64),
            np.array([0.4, 0.3], dtype=np.float64),
            20.0,
            30.0,
            comparison="sibling",
            feature_space=feature_space,
        )


def test_continuous_null_whitened_tangent_uses_empirical_covariance() -> None:
    feature_space = _make_continuous_space()
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]], dtype=np.float64)
    observations = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=np.float64)
    null_distribution = np.array([1.0, 2.0], dtype=np.float64)
    ridge = 1e-6

    tangent_rows = build_null_whitened_tangent_matrix(
        observations,
        null_distribution,
        feature_space=feature_space,
        continuous_covariance_by_block={"X": covariance},
        ridge=ridge,
    )

    cholesky = np.linalg.cholesky(covariance + ridge * np.eye(2))
    expected_rows = np.linalg.solve(cholesky, (observations - null_distribution).T).T
    np.testing.assert_allclose(tangent_rows, expected_rows)


def test_continuous_diagonal_covariance_whitening_uses_variance_vector() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1"))
    covariance_by_block = {
        "continuous": np.array([4.0, 9.0], dtype=np.float64),
    }
    left = np.array([2.0, -1.0], dtype=np.float64)
    right = np.array([0.0, 2.0], dtype=np.float64)

    z_scores = compute_whitened_wald_contrast(
        left,
        right,
        20.0,
        30.0,
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=covariance_by_block,
        ridge=0.0,
    )

    variance_scale = 1.0 / 20.0 + 1.0 / 30.0
    expected = (left - right) / np.sqrt(covariance_by_block["continuous"] * variance_scale)
    np.testing.assert_allclose(z_scores, expected)


def test_continuous_time_distance_matches_scaled_mahalanobis_formula() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1"))
    observations = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    covariance_by_block = {
        "continuous": np.diag(np.array([4.0, 9.0], dtype=np.float64)),
    }

    distances = continuous_time_distance_condensed(
        observations,
        feature_space,
        covariance_by_block=covariance_by_block,
    )

    np.testing.assert_allclose(
        distances,
        np.array([0.5, 0.5, 1.0], dtype=np.float64),
    )


def test_standardized_euclidean_distance_matches_zscore_formula() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1", "X2"))
    observations = np.array(
        [
            [0.0, 0.0, 7.0],
            [2.0, 0.0, 7.0],
            [0.0, 3.0, 7.0],
        ],
        dtype=np.float64,
    )

    distances = standardized_euclidean_distance_condensed(
        observations,
        feature_space,
    )

    centered = observations - np.mean(observations, axis=0, keepdims=True)
    scale = np.std(observations, axis=0, ddof=1, keepdims=True)
    scale[scale <= 0.0] = 1.0
    standardized = centered / scale
    expected = np.array(
        [
            np.linalg.norm(standardized[0] - standardized[1]),
            np.linalg.norm(standardized[0] - standardized[2]),
            np.linalg.norm(standardized[1] - standardized[2]),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(distances, expected)


def test_diagonal_bernoulli_null_whitening_matches_block_formula() -> None:
    observations = np.array(
        [
            [0.0, 1.0, 0.2, 0.8],
            [1.0, 0.0, 0.4, 0.6],
            [0.5, 0.5, 0.7, 0.3],
        ],
        dtype=np.float64,
    )
    null_distribution = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)

    assert hasattr(
        contrast_covariance_module,
        "_build_bernoulli_null_whitened_tangent_matrix",
    )
    vectorized = contrast_covariance_module._build_bernoulli_null_whitened_tangent_matrix(
        observations,
        null_distribution,
        contrast_covariance_module._resolve_distribution_feature_space(
            null_distribution,
            None,
        ),
        ridge=1e-12,
    )

    expected = np.column_stack(
        [
            contrast_covariance_module._block_null_whitened_tangent_matrix(
                observations,
                null_distribution,
                block,
                continuous_covariance_by_block={},
                ridge=1e-12,
            )
            for block in contrast_covariance_module._resolve_distribution_feature_space(
                null_distribution,
                None,
            ).blocks
        ]
    )
    np.testing.assert_allclose(vectorized, expected, rtol=0.0, atol=0.0)


def test_continuous_feature_space_from_columns_builds_one_full_covariance_block() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1", "X2"))

    assert feature_space.family_label == "continuous"
    assert len(feature_space.blocks) == 1
    block = feature_space.blocks[0]
    assert block.name == "continuous"
    assert block.column_indices == (0, 1, 2)
    assert block.contrast_dimension == 3
    assert block.covariance == "empirical_gaussian"


def test_continuous_null_whitening_uses_full_empirical_gaussian_block() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1", "X2"))
    observations = np.array(
        [
            [2.0, -1.0, 5.0],
            [1.5, 0.0, 4.0],
            [3.0, 1.0, 7.0],
        ],
        dtype=np.float64,
    )
    null_distribution = np.array([1.0, -0.5, 6.0], dtype=np.float64)
    covariance_by_block = {
        "continuous": np.array(
            [[4.0, 1.0, 0.5], [1.0, 9.0, 0.25], [0.5, 0.25, 16.0]],
            dtype=np.float64,
        ),
    }

    tangent_rows = build_null_whitened_tangent_matrix(
        observations,
        null_distribution,
        feature_space=feature_space,
        continuous_covariance_by_block=covariance_by_block,
        ridge=1e-12,
    )

    cholesky = np.linalg.cholesky(covariance_by_block["continuous"] + 1e-12 * np.eye(3))
    expected = np.linalg.solve(
        cholesky,
        (observations - null_distribution).T,
    )
    np.testing.assert_allclose(tangent_rows, expected.T, rtol=1e-14, atol=1e-14)


def test_continuous_covariance_rejects_non_psd_blocks() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1"))

    with pytest.raises(ValueError, match="positive semidefinite"):
        build_contrast_covariance(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            10.0,
            12.0,
            comparison="sibling",
            feature_space=feature_space,
            continuous_covariance_by_block={
                "continuous": np.array([[1.0, 3.0], [3.0, 1.0]], dtype=np.float64)
            },
        )


def test_grouped_categorical_null_whitening_matches_block_formula() -> None:
    feature_space = _make_categorical_space(n_features=4, n_categories=3)
    observations_blocks = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.6, 0.1, 0.3]],
            [[0.0, 0.0, 1.0], [0.3, 0.3, 0.4], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]],
        ],
        dtype=np.float64,
    )
    null_blocks = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.2, 0.4],
            [0.5, 0.25, 0.25],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float64,
    )
    observations = observations_blocks.reshape(observations_blocks.shape[0], -1)
    null_distribution = null_blocks.ravel()

    assert hasattr(
        contrast_covariance_module,
        "_build_grouped_categorical_null_whitened_tangent_matrix",
    )
    grouped = (
        contrast_covariance_module._build_grouped_categorical_null_whitened_tangent_matrix(
            observations,
            null_distribution,
            feature_space,
            ridge=1e-12,
        )
    )

    expected = np.column_stack(
        [
            contrast_covariance_module._block_null_whitened_tangent_matrix(
                observations,
                null_distribution,
                block,
                continuous_covariance_by_block={},
                ridge=1e-12,
            )
            for block in feature_space.blocks
        ]
    )
    np.testing.assert_allclose(grouped, expected, rtol=1e-13, atol=1e-13)


def test_grouped_categorical_null_whitening_supports_mixed_category_counts() -> None:
    columns = (
        "F0_c0",
        "F0_c1",
        "F0_c2",
        "F1_c0",
        "F1_c1",
        "F1_c2",
        "F1_c3",
        "F2_c0",
        "F2_c1",
    )
    feature_space = infer_feature_space_from_columns(columns)
    observations = np.array(
        [
            [0.2, 0.3, 0.5, 0.1, 0.2, 0.3, 0.4, 1.0, 0.0],
            [0.4, 0.4, 0.2, 0.3, 0.3, 0.2, 0.2, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    null_distribution = np.array(
        [0.3, 0.2, 0.5, 0.25, 0.25, 0.25, 0.25, 0.6, 0.4],
        dtype=np.float64,
    )

    assert hasattr(
        contrast_covariance_module,
        "_build_grouped_categorical_null_whitened_tangent_matrix",
    )
    grouped = (
        contrast_covariance_module._build_grouped_categorical_null_whitened_tangent_matrix(
            observations,
            null_distribution,
            feature_space,
            ridge=1e-12,
        )
    )

    expected = np.column_stack(
        [
            contrast_covariance_module._block_null_whitened_tangent_matrix(
                observations,
                null_distribution,
                block,
                continuous_covariance_by_block={},
                ridge=1e-12,
            )
            for block in feature_space.blocks
        ]
    )
    np.testing.assert_allclose(grouped, expected, rtol=1e-13, atol=1e-13)


def test_bernoulli_is_two_category_multinomial_for_sibling_whitening() -> None:
    left = np.array([0.2, 0.7], dtype=np.float64)
    right = np.array([0.5, 0.4], dtype=np.float64)
    n_left = 40.0
    n_right = 60.0

    bernoulli_z = compute_whitened_wald_contrast(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        ridge=0.0,
    )
    categorical_z = compute_whitened_wald_contrast(
        np.column_stack([left, 1.0 - left]).ravel(),
        np.column_stack([right, 1.0 - right]).ravel(),
        n_left,
        n_right,
        comparison="sibling",
        feature_space=_make_categorical_space(n_features=2, n_categories=2),
        ridge=0.0,
    )

    np.testing.assert_allclose(bernoulli_z, categorical_z)


def test_vectorized_bernoulli_wald_whitening_matches_block_formula() -> None:
    left = np.array([0.2, 0.7, 0.3, 0.8], dtype=np.float64)
    right = np.array([0.5, 0.4, 0.6, 0.1], dtype=np.float64)

    vectorized_z = compute_whitened_wald_contrast(
        left,
        right,
        40.0,
        60.0,
        comparison="sibling",
        ridge=1e-12,
    )
    block_z = build_contrast_covariance(
        left,
        right,
        40.0,
        60.0,
        comparison="sibling",
        ridge=1e-12,
    ).whitened_vector()

    np.testing.assert_allclose(vectorized_z, block_z, rtol=1e-14, atol=1e-14)


def test_vectorized_categorical_wald_whitening_matches_block_formula() -> None:
    columns = (
        "F0_c0",
        "F0_c1",
        "F0_c2",
        "F1_c0",
        "F1_c1",
        "F1_c2",
        "F1_c3",
        "F2_c0",
        "F2_c1",
    )
    feature_space = infer_feature_space_from_columns(columns)
    left = np.array(
        [0.2, 0.3, 0.5, 0.1, 0.2, 0.3, 0.4, 0.7, 0.3],
        dtype=np.float64,
    )
    right = np.array(
        [0.4, 0.4, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 0.8],
        dtype=np.float64,
    )

    vectorized_z = compute_whitened_wald_contrast(
        left,
        right,
        25.0,
        80.0,
        comparison="sibling",
        feature_space=feature_space,
        ridge=1e-12,
    )
    block_z = build_contrast_covariance(
        left,
        right,
        25.0,
        80.0,
        comparison="sibling",
        feature_space=feature_space,
        ridge=1e-12,
    ).whitened_vector()

    np.testing.assert_allclose(vectorized_z, block_z, rtol=1e-13, atol=1e-13)


def test_continuous_wald_whitening_matches_full_block_formula() -> None:
    feature_space = continuous_feature_space_from_columns(("X0", "X1", "X2"))
    covariance_by_block = {
        "continuous": np.array(
            [[4.0, 1.0, 0.5], [1.0, 9.0, 0.25], [0.5, 0.25, 16.0]],
            dtype=np.float64,
        ),
    }
    left = np.array([2.0, -1.0, 5.0], dtype=np.float64)
    right = np.array([1.5, 0.0, 4.0], dtype=np.float64)

    public_z = compute_whitened_wald_contrast(
        left,
        right,
        20.0,
        30.0,
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=covariance_by_block,
        ridge=1e-12,
    )
    block_z = build_contrast_covariance(
        left,
        right,
        20.0,
        30.0,
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=covariance_by_block,
        ridge=1e-12,
    ).whitened_vector()

    np.testing.assert_allclose(public_z, block_z, rtol=1e-14, atol=1e-14)


def test_sibling_contrast_rejects_child_parent_branch_length_parameter() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'branch_length'"):
        build_contrast_covariance(
            np.array([0.2], dtype=np.float64),
            np.array([0.4], dtype=np.float64),
            20.0,
            30.0,
            comparison="sibling",
            branch_length=1.0,
        )


def test_child_parent_contrast_rejects_sibling_branch_length_sum_parameter() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'branch_length_sum'"):
        build_contrast_covariance(
            np.array([0.2], dtype=np.float64),
            np.array([0.4], dtype=np.float64),
            20.0,
            30.0,
            comparison="child_parent",
            branch_length_sum=1.0,
        )


def test_bernoulli_contrast_rejects_values_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="Bernoulli block 'x1' values must lie"):
        build_contrast_covariance(
            np.array([0.2, 1.1], dtype=np.float64),
            np.array([0.4, 0.6], dtype=np.float64),
            20.0,
            30.0,
            comparison="sibling",
        )


def test_null_whitened_categorical_tangent_matches_child_parent_wald_coordinates() -> None:
    child_blocks = np.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2]], dtype=np.float64)
    parent_blocks = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2]], dtype=np.float64)
    child = child_blocks.ravel()
    parent = parent_blocks.ravel()
    feature_space = _make_categorical_space(n_features=2, n_categories=3)
    n_child = 30.0
    n_parent = 120.0
    nested_factor = 1.0 / n_child - 1.0 / n_parent

    tangent_row = build_null_whitened_tangent_matrix(
        child[None, :],
        parent,
        feature_space=feature_space,
    )[0]
    wald_z = compute_whitened_wald_contrast(
        child,
        parent,
        n_child,
        n_parent,
        comparison="child_parent",
        feature_space=feature_space,
    )

    np.testing.assert_allclose(wald_z, tangent_row / np.sqrt(nested_factor))
