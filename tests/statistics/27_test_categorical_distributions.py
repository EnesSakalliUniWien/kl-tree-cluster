"""Sibling Wald tests for Bernoulli and categorical distributions."""

from __future__ import annotations

import numpy as np
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.sibling_divergence_test import (
    sibling_divergence_test,
)
from tree_break_selection.tree.feature_space import FeatureSpace, infer_feature_space_from_columns


def _make_categorical_space(n_features: int, n_categories: int) -> FeatureSpace:
    columns = tuple(f"F{i}_c{k}" for i in range(n_features) for k in range(n_categories))
    return infer_feature_space_from_columns(columns)


def test_sibling_test_binary_identical() -> None:
    theta = np.array([0.3, 0.5, 0.7], dtype=np.float64)
    d = theta.shape[0]

    _, _, _, p_value = sibling_divergence_test(
        theta,
        theta,
        100.0,
        100.0,
        projection_dimension_from_edge_comparisons=d,
        parent_principal_component_projection=np.eye(d),
        parent_principal_component_eigenvalues=np.ones(d),
    )

    assert p_value > 0.05


def test_sibling_test_binary_different() -> None:
    theta_left = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    theta_right = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64)
    d = theta_left.shape[0]

    _, _, _, p_value = sibling_divergence_test(
        theta_left,
        theta_right,
        500.0,
        500.0,
        projection_dimension_from_edge_comparisons=d,
        parent_principal_component_projection=np.eye(d),
        parent_principal_component_eigenvalues=np.ones(d),
    )

    assert p_value < 0.05


def test_sibling_test_categorical_identical() -> None:
    theta = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    feature_space = _make_categorical_space(theta.shape[0], theta.shape[1])
    z_dim = theta.shape[0] * (theta.shape[1] - 1)

    _, _, _, p_value = sibling_divergence_test(
        theta.ravel(),
        theta.ravel(),
        100.0,
        100.0,
        projection_dimension_from_edge_comparisons=z_dim,
        parent_principal_component_projection=np.eye(z_dim),
        parent_principal_component_eigenvalues=np.ones(z_dim),
        feature_space=feature_space,
    )

    assert p_value > 0.05


def test_sibling_test_categorical_different() -> None:
    theta_left = np.array(
        [
            [0.9, 0.05, 0.025, 0.025],
            [0.9, 0.05, 0.025, 0.025],
            [0.9, 0.05, 0.025, 0.025],
        ],
        dtype=np.float64,
    )
    theta_right = np.array(
        [
            [0.025, 0.025, 0.05, 0.9],
            [0.025, 0.025, 0.05, 0.9],
            [0.025, 0.025, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    feature_space = _make_categorical_space(theta_left.shape[0], theta_left.shape[1])
    z_dim = theta_left.shape[0] * (theta_left.shape[1] - 1)

    _, _, _, p_value = sibling_divergence_test(
        theta_left.ravel(),
        theta_right.ravel(),
        500.0,
        500.0,
        projection_dimension_from_edge_comparisons=z_dim,
        parent_principal_component_projection=np.eye(z_dim),
        parent_principal_component_eigenvalues=np.ones(z_dim),
        feature_space=feature_space,
    )

    assert p_value < 0.05


def test_sibling_categorical_contrast_dimension_drops_simplex_category() -> None:
    theta_left = np.array(
        [
            [0.3, 0.3, 0.2, 0.2],
            [0.1, 0.4, 0.4, 0.1],
        ],
        dtype=np.float64,
    )
    theta_right = np.array(
        [
            [0.2, 0.2, 0.3, 0.3],
            [0.4, 0.1, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    feature_space = _make_categorical_space(theta_left.shape[0], theta_left.shape[1])

    z_scores = compute_whitened_wald_contrast(
        theta_left.ravel(),
        theta_right.ravel(),
        100.0,
        100.0,
        comparison="sibling",
        feature_space=feature_space,
    )

    assert z_scores.shape == (theta_left.shape[0] * (theta_left.shape[1] - 1),)
