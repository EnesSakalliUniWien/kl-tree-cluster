"""Case input preparation helpers for benchmark pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tree_break_selection.tree.construction import DEFAULT_BINARY_TREE_DISTANCE_METRIC
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    contains_categorical_feature_columns,
    validate_feature_space,
)

from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.util.method_sets import (
    DISTANCE_MATRIX_METHODS as _DISTANCE_MATRIX_METHODS,
)
from benchmarks.shared.util.method_sets import (
    TBS_DISTANCE_TREE_METHODS as _TBS_DISTANCE_TREE_METHODS,
)


@dataclass(frozen=True)
class PreparedCaseInputs:
    """Generated benchmark inputs after matrix and distance-contract validation."""

    data: pd.DataFrame
    labels: np.ndarray
    original_features: object
    metadata: dict[str, object]
    distance_condensed: np.ndarray | None
    distance_matrix: np.ndarray | None


def prepare_case_inputs(
    tc: dict[str, object],
    selected_methods: list[str],
) -> PreparedCaseInputs:
    """Generate case data and resolve shared distance representations."""
    data_t, y_t, x_original, meta = generate_case_data(tc)
    feature_space = meta.get("feature_space")
    if feature_space is not None:
        if not isinstance(feature_space, FeatureSpace):
            raise ValueError("Benchmark feature_space metadata must be a FeatureSpace.")
        validate_feature_space(tuple(data_t.columns), feature_space)
    elif contains_categorical_feature_columns(tuple(data_t.columns)):
        raise ValueError("Categorical benchmark data must carry explicit feature_space metadata.")

    needs_distance_matrix = any(
        method_id in _DISTANCE_MATRIX_METHODS for method_id in selected_methods
    )
    needs_tbs_tree_distance = any(
        method_id in _TBS_DISTANCE_TREE_METHODS for method_id in selected_methods
    )
    requires_precomputed_tbs_distance = bool(meta["requires_precomputed_tbs_distance"])
    needs_distance_condensed = needs_distance_matrix or (
        needs_tbs_tree_distance and requires_precomputed_tbs_distance
    )

    distance_condensed = None
    distance_matrix = None
    precomputed_distance_condensed = meta["precomputed_distance_condensed"]
    if needs_tbs_tree_distance and requires_precomputed_tbs_distance:
        if precomputed_distance_condensed is None:
            raise ValueError(
                f"Case '{meta['name']}' requires precomputed TBS tree distance but "
                "metadata does not provide precomputed_distance_condensed."
            )
    if precomputed_distance_condensed is not None and needs_distance_condensed:
        distance_condensed = np.asarray(precomputed_distance_condensed, dtype=float)

    precomputed_distance_matrix = meta["precomputed_distance_matrix"]
    if precomputed_distance_matrix is not None and needs_distance_matrix:
        distance_matrix = np.asarray(precomputed_distance_matrix, dtype=float)
        np.fill_diagonal(distance_matrix, 0.0)

    if needs_distance_condensed and distance_condensed is None and distance_matrix is not None:
        distance_condensed = squareform(distance_matrix)

    if needs_distance_matrix and distance_condensed is None:
        distance_condensed = pdist(data_t.values, metric=DEFAULT_BINARY_TREE_DISTANCE_METRIC)
    if needs_distance_matrix and distance_matrix is None:
        if distance_condensed is None:
            distance_condensed = pdist(data_t.values, metric=DEFAULT_BINARY_TREE_DISTANCE_METRIC)
        distance_matrix = squareform(distance_condensed)

    return PreparedCaseInputs(
        data=data_t,
        labels=np.asarray(y_t),
        original_features=x_original,
        metadata=meta,
        distance_condensed=distance_condensed,
        distance_matrix=distance_matrix,
    )


__all__ = ["PreparedCaseInputs", "prepare_case_inputs"]
