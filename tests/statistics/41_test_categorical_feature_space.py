"""Contract tests for categorical one-hot feature spaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.cases.categorical import CATEGORICAL_CASES
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_projected_wald.child_parent_standardized_z_scores import (
    compute_child_parent_standardized_z_scores,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.tree.distributions import (
    CONTINUOUS_COVARIANCE_BY_BLOCK,
    MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
    validate_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _expected_null_whitened_categorical_rows(
    categorical_rows: np.ndarray,
    null_distribution: np.ndarray,
    feature_space: FeatureSpace,
    *,
    ridge: float = 1e-12,
) -> np.ndarray:
    expected_rows: list[np.ndarray] = []
    for categorical_row in categorical_rows:
        blocks: list[np.ndarray] = []
        for block in feature_space.blocks:
            block_row = categorical_row[list(block.column_indices)]
            block_null = null_distribution[list(block.column_indices)]
            if block.family == "bernoulli":
                variance = block_null[0] * (1.0 - block_null[0]) + ridge
                blocks.append((block_row - block_null) / np.sqrt(variance))
            elif block.family == "categorical":
                row_probabilities = block_row[:-1]
                null_probabilities = block_null[:-1]
                covariance = (
                    np.diag(null_probabilities)
                    - np.outer(null_probabilities, null_probabilities)
                    + ridge * np.eye(null_probabilities.shape[0])
                )
                cholesky = np.linalg.cholesky(covariance)
                blocks.append(np.linalg.solve(cholesky, row_probabilities - null_probabilities))
        expected_rows.append(np.concatenate(blocks))
    return np.vstack(expected_rows)


def _categorical_leaf_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        index=["L0", "L1", "L2", "L3"],
        columns=["F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2"],
    )


def _simple_binary_tree() -> PosetTree:
    tree = PosetTree()
    tree.add_node("root", is_leaf=False)
    for leaf in ["L0", "L1", "L2", "L3"]:
        tree.add_node(leaf, is_leaf=True, label=leaf)
        tree.add_edge("root", leaf)
    return tree


def _continuous_feature_space() -> FeatureSpace:
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


def _continuous_leaf_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        index=["L0", "L1", "L2", "L3"],
        columns=["X0", "X1"],
    )


def test_populate_node_divergences_stores_categorical_blocks_for_one_hot_columns() -> None:
    """Flat one-hot columns are one categorical block contract, not Bernoulli aliases."""
    tree = _simple_binary_tree()
    leaf_data = _categorical_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    assert isinstance(feature_space, FeatureSpace)

    tree.populate_node_divergences(leaf_data, feature_space=feature_space)

    assert tree.nodes["L0"]["distribution"].shape == (6,)
    np.testing.assert_array_equal(
        tree.nodes["L0"]["distribution"],
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    assert tree.nodes["root"]["distribution"].shape == (6,)
    for block in feature_space.blocks:
        np.testing.assert_allclose(
            tree.nodes["root"]["distribution"][list(block.column_indices)].sum(),
            1.0,
        )


def test_populate_node_divergences_rejects_one_hot_columns_without_explicit_contract() -> None:
    """One-hot category names alone must not silently select categorical statistics."""
    tree = _simple_binary_tree()
    leaf_data = _categorical_leaf_data()

    with pytest.raises(ValueError, match="feature_space"):
        tree.populate_node_divergences(leaf_data)


def test_populate_node_divergences_rejects_bernoulli_values_outside_unit_interval() -> None:
    """Bernoulli leaf data must be probabilities, not unvalidated continuous features."""
    tree = _simple_binary_tree()
    leaf_data = pd.DataFrame(
        [
            [0.0, 0.2],
            [0.4, 0.8],
            [1.1, 0.6],
            [1.0, 0.0],
        ],
        index=["L0", "L1", "L2", "L3"],
        columns=["F0", "F1"],
    )

    with pytest.raises(ValueError, match="Bernoulli block 'F0' values must lie"):
        tree.populate_node_divergences(leaf_data)


def test_feature_space_rejects_duplicate_column_and_block_names() -> None:
    with pytest.raises(ValueError, match="column_names must be unique"):
        FeatureSpace(
            column_names=("X", "X"),
            blocks=(
                FeatureBlock(
                    name="X0",
                    family="bernoulli",
                    column_indices=(0,),
                    chart="identity",
                    covariance="bernoulli",
                    contrast_dimension=1,
                ),
                FeatureBlock(
                    name="X1",
                    family="bernoulli",
                    column_indices=(1,),
                    chart="identity",
                    covariance="bernoulli",
                    contrast_dimension=1,
                ),
            ),
        )

    with pytest.raises(ValueError, match="block names must be unique"):
        FeatureSpace(
            column_names=("X0", "X1"),
            blocks=(
                FeatureBlock(
                    name="X",
                    family="bernoulli",
                    column_indices=(0,),
                    chart="identity",
                    covariance="bernoulli",
                    contrast_dimension=1,
                ),
                FeatureBlock(
                    name="X",
                    family="bernoulli",
                    column_indices=(1,),
                    chart="identity",
                    covariance="bernoulli",
                    contrast_dimension=1,
                ),
            ),
        )


def test_populate_node_divergences_stores_continuous_means_and_covariances() -> None:
    tree = _simple_binary_tree()
    leaf_data = _continuous_leaf_data()
    feature_space = _continuous_feature_space()

    tree.populate_node_divergences(leaf_data, feature_space=feature_space)

    expected_mean = leaf_data.to_numpy(dtype=np.float64).mean(axis=0)
    expected_covariance = np.cov(leaf_data.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
    np.testing.assert_allclose(tree.nodes["root"]["distribution"], expected_mean)
    np.testing.assert_allclose(
        tree.nodes["root"][CONTINUOUS_COVARIANCE_BY_BLOCK]["X"],
        expected_covariance,
    )
    np.testing.assert_allclose(
        tree.nodes["L0"][CONTINUOUS_COVARIANCE_BY_BLOCK]["X"],
        np.zeros((2, 2), dtype=np.float64),
    )


def test_populate_node_divergences_rejects_oversized_dense_continuous_covariance() -> None:
    tree = PosetTree()
    tree.add_node("root", is_leaf=False)
    tree.add_node("L0", is_leaf=True, label="L0")
    tree.add_node("L1", is_leaf=True, label="L1")
    tree.add_edge("root", "L0")
    tree.add_edge("root", "L1")
    columns = tuple(f"X{i}" for i in range(4097))
    leaf_data = pd.DataFrame(
        np.zeros((2, len(columns)), dtype=np.float64),
        index=["L0", "L1"],
        columns=columns,
    )
    feature_space = continuous_feature_space_from_columns(columns)

    with pytest.raises(ValueError, match="Dense empirical-Gaussian covariance"):
        tree.populate_node_divergences(leaf_data, feature_space=feature_space)


def test_populate_node_divergences_rejects_incomplete_one_hot_category_blocks() -> None:
    """A malformed one-hot schema should fail at the data boundary."""
    leaf_data = _categorical_leaf_data().drop(columns=["F1_c1"])

    with pytest.raises(ValueError, match="contiguous category ids"):
        infer_feature_space_from_columns(tuple(leaf_data.columns))


def test_child_parent_categorical_z_scores_use_drop_last_multinomial_dimension() -> None:
    """Child-parent categorical covariance must remove the simplex-null direction."""
    child_distribution = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.25, 0.50, 0.25],
        ],
        dtype=np.float64,
    ).ravel()
    parent_distribution = np.array(
        [
            [0.50, 0.25, 0.25],
            [0.25, 0.25, 0.50],
        ],
        dtype=np.float64,
    ).ravel()
    feature_space = infer_feature_space_from_columns(
        ("F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2")
    )

    z_scores = compute_child_parent_standardized_z_scores(
        child_distribution,
        parent_distribution,
        n_child=10,
        n_parent=40,
        feature_space=feature_space,
    )

    assert z_scores.shape == (4,)
    assert np.all(np.isfinite(z_scores))


def test_categorical_spectral_decomposition_uses_drop_last_projection_width() -> None:
    """Categorical projected-Wald z vectors and leaf-only PCA use the same coordinates."""
    tree = _simple_binary_tree()
    leaf_data = _categorical_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    assert feature_space is not None
    tree.populate_node_divergences(leaf_data, feature_space=feature_space)

    spectral_decomposition = compute_spectral_decomposition(
        tree,
        leaf_data,
        feature_space=feature_space,
        minimum_projection_dimension=2,
    )

    assert spectral_decomposition.test_projection_dimensions_by_node["root"] <= 4
    assert spectral_decomposition.raw_mp_signal_counts_by_node["root"] >= 0
    assert spectral_decomposition.effective_independent_rows_by_node["root"] == len(
        leaf_data
    )
    assert spectral_decomposition.mp_threshold_rows_by_node["root"] == len(leaf_data)
    root_projection = spectral_decomposition.principal_component_projections_by_node["root"]
    root_eigenvalues = spectral_decomposition.principal_component_eigenvalues_by_node["root"]
    assert root_projection.shape[1] == 4
    assert root_eigenvalues.shape[0] == root_projection.shape[0]


def test_categorical_spectral_decomposition_uses_parent_null_whitened_tangent_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Categorical PCA must use the same null-whitened tangent space as Wald."""
    tree = _simple_binary_tree()
    leaf_data = _categorical_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    assert feature_space is not None
    tree.populate_node_divergences(leaf_data, feature_space=feature_space)
    captured_matrices: list[np.ndarray] = []

    def _capture_eigendecompose_covariance(
        data_matrix: np.ndarray,
        *,
        compute_eigenvectors: bool,
    ):
        captured_matrices.append(np.asarray(data_matrix, dtype=np.float64).copy())
        return None

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral."
        "marchenko_pastur.eigendecompose_covariance",
        _capture_eigendecompose_covariance,
    )

    compute_spectral_decomposition(
        tree,
        leaf_data,
        feature_space=feature_space,
        minimum_projection_dimension=2,
    )

    categorical_rows = validate_feature_matrix(
        leaf_data.to_numpy(dtype=np.float64, copy=False),
        feature_space,
    )
    expected_matrix = _expected_null_whitened_categorical_rows(
        categorical_rows,
        tree.nodes["root"]["distribution"],
        feature_space,
    )
    assert len(captured_matrices) == 1
    np.testing.assert_allclose(captured_matrices[0], expected_matrix, atol=1e-10)


def test_continuous_spectral_decomposition_uses_parent_null_whitened_tangent_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _simple_binary_tree()
    leaf_data = _continuous_leaf_data()
    feature_space = _continuous_feature_space()
    tree.populate_node_divergences(leaf_data, feature_space=feature_space)
    captured_matrices: list[np.ndarray] = []

    def _capture_eigendecompose_covariance(
        data_matrix: np.ndarray,
        *,
        compute_eigenvectors: bool,
    ):
        captured_matrices.append(np.asarray(data_matrix, dtype=np.float64).copy())
        return None

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral."
        "marchenko_pastur.eigendecompose_covariance",
        _capture_eigendecompose_covariance,
    )

    compute_spectral_decomposition(
        tree,
        leaf_data,
        feature_space=feature_space,
        minimum_projection_dimension=1,
    )

    covariance = tree.nodes["root"][CONTINUOUS_COVARIANCE_BY_BLOCK]["X"]
    cholesky = np.linalg.cholesky(covariance + 1e-12 * np.eye(2))
    expected_matrix = np.linalg.solve(
        cholesky,
        (leaf_data.to_numpy(dtype=np.float64) - tree.nodes["root"]["distribution"]).T,
    ).T
    assert len(captured_matrices) == 1
    np.testing.assert_allclose(captured_matrices[0], expected_matrix, atol=1e-10)


def test_categorical_generator_returns_explicit_feature_space_without_dataframe_attrs() -> None:
    """Benchmark categorical cases carry the schema as typed metadata, not attrs."""
    case = CATEGORICAL_CASES["categorical_clear"][0]

    data_df, _labels, _x_original, metadata = generate_case_data(case)

    feature_space = metadata["feature_space"]
    assert isinstance(feature_space, FeatureSpace)
    assert feature_space.raw_dimension == data_df.shape[1]
    categorical_blocks = [
        block for block in feature_space.blocks if block.family == "categorical"
    ]
    assert len(categorical_blocks) == metadata["n_features_original"]
    assert all(block.raw_dimension == metadata["n_categories"] for block in categorical_blocks)
    assert data_df.attrs == {}


def test_feature_space_supports_mixed_bernoulli_and_unequal_categorical_blocks() -> None:
    columns = ("B0", "F0_c0", "F0_c1", "F0_c2", "B1", "F1_c0", "F1_c1")

    feature_space = infer_feature_space_from_columns(columns)

    assert [block.family for block in feature_space.blocks] == [
        "bernoulli",
        "categorical",
        "bernoulli",
        "categorical",
    ]
    assert feature_space.raw_dimension == 7
    assert feature_space.contrast_dimension == 5
    assert feature_space.family_label == "mixed"


def test_continuous_covariance_memory_contract_is_adjustable(monkeypatch) -> None:
    """Dense continuous covariance memory cap can be raised explicitly."""
    columns = [f"X{j}" for j in range(400)]
    leaf_data = pd.DataFrame(
        np.arange(4 * 400, dtype=np.float64).reshape(4, 400),
        index=["L0", "L1", "L2", "L3"],
        columns=columns,
    )
    feature_space = continuous_feature_space_from_columns(tuple(columns))

    monkeypatch.setenv(MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV, "1")
    with pytest.raises(ValueError, match="memory contract"):
        _simple_binary_tree().populate_node_divergences(
            leaf_data,
            feature_space=feature_space,
        )

    monkeypatch.setenv(MAX_EXACT_CONTINUOUS_COVARIANCE_WORK_MIB_ENV, "8")
    tree = _simple_binary_tree()
    tree.populate_node_divergences(leaf_data, feature_space=feature_space)
    assert CONTINUOUS_COVARIANCE_BY_BLOCK in tree.nodes["root"]
