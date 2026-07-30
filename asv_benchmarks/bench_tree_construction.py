"""ASV benchmarks for core TBS tree construction and annotation costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tree_break_selection.tree.construction.build import build_tree
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)


def _clustered_gaussian(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_features: int,
    n_clusters: int,
) -> np.ndarray:
    labels = np.arange(n_samples) % n_clusters
    centers = rng.normal(loc=0.0, scale=2.0, size=(n_clusters, n_features))
    return centers[labels] + rng.normal(loc=0.0, scale=0.35, size=(n_samples, n_features))


def _clustered_bernoulli(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_features: int,
    n_clusters: int,
) -> np.ndarray:
    labels = np.arange(n_samples) % n_clusters
    prototypes = rng.binomial(1, 0.5, size=(n_clusters, n_features))
    noise = rng.binomial(1, 0.05, size=(n_samples, n_features))
    return np.logical_xor(prototypes[labels], noise).astype(float)


def _one_hot_categorical(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_categories: int,
    n_blocks: int,
    n_clusters: int,
) -> np.ndarray:
    labels = np.arange(n_samples) % n_clusters
    dominant = (labels[:, None] + np.arange(n_blocks)[None, :]) % n_categories
    noise_mask = rng.random(size=(n_samples, n_blocks)) < 0.08
    noisy = rng.integers(0, n_categories, size=(n_samples, n_blocks))
    categories = np.where(noise_mask, noisy, dominant)
    one_hot = np.zeros((n_samples, n_blocks * n_categories), dtype=float)
    row = np.repeat(np.arange(n_samples), n_blocks)
    col = np.arange(n_blocks)[None, :] * n_categories + categories
    one_hot[row, col.ravel()] = 1.0
    return one_hot


def _make_case(case_id: str) -> tuple[pd.DataFrame, str, FeatureSpace]:
    rng = np.random.default_rng(20260729)
    if case_id == "bernoulli_160x64":
        matrix = _clustered_bernoulli(
            rng,
            n_samples=160,
            n_features=64,
            n_clusters=4,
        )
        columns = [f"x{j}" for j in range(matrix.shape[1])]
        metric = "hamming"
        feature_space_factory = bernoulli_feature_space_from_columns
    elif case_id == "gaussian_120x48":
        matrix = _clustered_gaussian(
            rng,
            n_samples=120,
            n_features=48,
            n_clusters=4,
        )
        columns = [f"x{j}" for j in range(matrix.shape[1])]
        metric = "euclidean"
        feature_space_factory = continuous_feature_space_from_columns
    elif case_id == "categorical_one_hot_120x80":
        n_categories = 4
        n_blocks = 20
        matrix = _one_hot_categorical(
            rng,
            n_samples=120,
            n_categories=n_categories,
            n_blocks=n_blocks,
            n_clusters=4,
        )
        columns = [
            f"F{block_id}_c{category_id}"
            for block_id in range(n_blocks)
            for category_id in range(n_categories)
        ]
        metric = "hamming"
        feature_space_factory = infer_feature_space_from_columns
    else:
        raise ValueError(f"Unknown ASV benchmark case {case_id!r}.")

    data = pd.DataFrame(
        matrix,
        index=[f"s{i}" for i in range(matrix.shape[0])],
        columns=columns,
    )
    return data, metric, feature_space_factory(data.columns)


class TreeConstructionSuite:
    """Track linkage/tree-build and distribution-population runtime."""

    params = ["bernoulli_160x64", "gaussian_120x48", "categorical_one_hot_120x80"]
    param_names = ["case_id"]
    timeout = 120

    def setup(self, case_id: str) -> None:
        data, metric, feature_space = _make_case(case_id)
        self.data = data
        self.feature_space = feature_space
        self.distance_metric = metric
        self.distance_condensed = pdist(data.to_numpy(dtype=float), metric=metric)
        self.tree_template = build_tree(
            self.data,
            self.distance_condensed,
            builder="linkage",
            rooting="linkage_root",
            linkage_method="average",
        ).tree

    def time_average_linkage_tree_build(self, case_id: str) -> None:
        build_tree(
            self.data,
            self.distance_condensed,
            builder="linkage",
            rooting="linkage_root",
            linkage_method="average",
        )

    def time_populate_node_divergences(self, case_id: str) -> None:
        tree = self.tree_template.copy(as_view=False)
        tree.populate_node_divergences(
            self.data,
            feature_space=self.feature_space,
        )

    def peakmem_distance_condensed(self, case_id: str) -> np.ndarray:
        return pdist(self.data.to_numpy(dtype=float), metric=self.distance_metric)
