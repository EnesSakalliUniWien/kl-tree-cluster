from __future__ import annotations

import pytest
from benchmarks.shared.generators.common import calculate_cluster_sizes
from benchmarks.shared.generators.generate_categorical_matrix import (
    generate_categorical_feature_matrix,
)


def test_categorical_generator_rejects_too_many_clusters() -> None:
    with pytest.raises(ValueError, match="n_clusters must be <= n_rows"):
        generate_categorical_feature_matrix(
            n_rows=2,
            n_cols=4,
            n_categories=3,
            n_clusters=3,
            random_seed=1,
        )


def test_categorical_generator_rejects_single_category() -> None:
    with pytest.raises(ValueError, match="n_categories must be >= 2"):
        generate_categorical_feature_matrix(
            n_rows=4,
            n_cols=4,
            n_categories=1,
            n_clusters=2,
            random_seed=1,
        )


def test_categorical_generator_uses_explicit_rng_for_unbalanced_sizes() -> None:
    sample_dict, cluster_assignments, distributions = generate_categorical_feature_matrix(
        n_rows=8,
        n_cols=3,
        n_categories=3,
        n_clusters=3,
        random_seed=17,
        balanced_clusters=False,
    )

    assert len(sample_dict) == 8
    assert set(sample_dict) == set(cluster_assignments)
    assert distributions.shape == (8, 3, 3)


def test_cluster_size_helper_requires_rng_for_unbalanced_sizes() -> None:
    with pytest.raises(ValueError, match="rng is required"):
        calculate_cluster_sizes(8, 3, balanced=False)
