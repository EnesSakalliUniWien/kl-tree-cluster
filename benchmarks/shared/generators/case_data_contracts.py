"""Shared contracts for benchmark case-data generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)
from scipy.spatial.distance import pdist

CaseDataResult = tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]
CaseGenerator = Callable[[dict, int | None], CaseDataResult]

CORE_METADATA_KEYS = frozenset(
    {
        "n_samples",
        "n_features",
        "n_clusters",
        "noise",
        "name",
        "generator",
        "source_family",
        "feature_representation",
        "requires_precomputed_kl_distance",
        "precomputed_distance_matrix",
        "precomputed_distance_condensed",
    }
)


def require_case_value(test_case: dict, key: str, generator_name: str) -> Any:
    if key not in test_case:
        raise ValueError(f"{generator_name} generator requires '{key}'.")
    return test_case[key]


def metadata_extras(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return generator-specific metadata without duplicating core fields."""
    return {key: value for key, value in metadata.items() if key not in CORE_METADATA_KEYS}


def case_metadata(
    *,
    test_case: dict,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    noise: float,
    generator: str,
    source_family: str,
    feature_representation: str,
    requires_precomputed_kl_distance: bool,
    precomputed_distance_matrix: np.ndarray | None = None,
    precomputed_distance_condensed: np.ndarray | None = None,
    distance_metric: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical benchmark metadata shape for generated cases."""
    has_precomputed_distance = (
        precomputed_distance_matrix is not None or precomputed_distance_condensed is not None
    )
    if has_precomputed_distance and not requires_precomputed_kl_distance:
        raise ValueError(
            "Precomputed KL tree distances require "
            "requires_precomputed_kl_distance=True."
        )
    if requires_precomputed_kl_distance and precomputed_distance_condensed is None:
        raise ValueError(
            "Cases requiring a precomputed KL tree distance must provide "
            "precomputed_distance_condensed."
        )
    metadata: dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": int(n_clusters),
        "noise": float(noise),
        "name": str(test_case["name"]),
        "generator": generator,
        "source_family": source_family,
        "feature_representation": feature_representation,
        "requires_precomputed_kl_distance": bool(requires_precomputed_kl_distance),
        "precomputed_distance_matrix": precomputed_distance_matrix,
        "precomputed_distance_condensed": precomputed_distance_condensed,
    }
    if distance_metric is not None:
        if not distance_metric:
            raise ValueError("distance_metric must be non-empty when provided.")
        metadata["distance_metric"] = distance_metric
    if extra:
        metadata.update(extra)
    return metadata


def one_hot_encode_categorical(
    matrix: np.ndarray,
    n_categories: int,
    sample_names: list[str],
) -> tuple[pd.DataFrame, int, FeatureSpace]:
    """Encode category-index columns into explicit one-hot feature blocks."""
    n_rows, n_cols = matrix.shape
    n_binary = n_cols * n_categories
    binary = np.zeros((n_rows, n_binary), dtype=int)
    for j in range(n_cols):
        for k in range(n_categories):
            binary[:, j * n_categories + k] = (matrix[:, j] == k).astype(int)
    feature_names = [f"F{j}_c{k}" for j in range(n_cols) for k in range(n_categories)]
    data_df = pd.DataFrame(binary, index=sample_names, columns=feature_names)
    feature_space = infer_feature_space_from_columns(tuple(data_df.columns))
    return data_df, n_binary, feature_space


def continuous_dataframe_and_metadata(
    matrix: np.ndarray,
    sample_names: list[str],
    feature_names: list[str],
) -> tuple[pd.DataFrame, FeatureSpace, np.ndarray]:
    """Return continuous benchmark data with its explicit diagonal Gaussian contract."""
    continuous_matrix = np.asarray(matrix, dtype=np.float64)
    data_df = pd.DataFrame(
        continuous_matrix,
        index=sample_names,
        columns=feature_names,
    )
    feature_space = continuous_feature_space_from_columns(tuple(data_df.columns))
    distance_condensed = pdist(continuous_matrix, metric="euclidean")
    return data_df, feature_space, distance_condensed
