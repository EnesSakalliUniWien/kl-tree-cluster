"""Build the canonical KL hierarchy context for benchmark diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import (
    KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
    KL_TREE_DISTANCE_SOURCE_PRECOMPUTED,
    _require_precomputed_kl_distance_metric,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class KlTreeContext:
    """Benchmark case data plus the exact hierarchy used by the KL method."""

    data: pd.DataFrame
    true_labels: np.ndarray
    original_features: object
    metadata: dict[str, object]
    feature_space: FeatureSpace | None
    distance_condensed: np.ndarray
    tree_distance_metric: str
    tree_distance_source: str
    tree_linkage_method: str
    linkage_matrix: np.ndarray
    tree: PosetTree


def _resolve_kl_tree_distance(
    *,
    data: pd.DataFrame,
    metadata: dict[str, object],
    distance_condensed: np.ndarray | None,
    precomputed_distance_condensed: object,
) -> tuple[np.ndarray, str, str, str]:
    params = METHOD_SPECS["kl"].param_grid[0]
    tree_linkage_method = str(params["tree_linkage_method"])
    configured_metric = str(params["tree_distance_metric"])

    requires_precomputed = bool(metadata["requires_precomputed_kl_distance"])
    if requires_precomputed or precomputed_distance_condensed is not None:
        if distance_condensed is None:
            raise ValueError(
                f"Case '{metadata['name']}' requires/provides precomputed KL distance, "
                "but no condensed distance was prepared."
            )
        return (
            np.asarray(distance_condensed, dtype=float),
            _require_precomputed_kl_distance_metric(
                meta=metadata,
                case_name=str(metadata["name"]),
            ),
            KL_TREE_DISTANCE_SOURCE_PRECOMPUTED,
            tree_linkage_method,
        )

    return (
        pdist(data.values, metric=configured_metric),
        configured_metric,
        KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
        tree_linkage_method,
    )


def build_kl_tree_context(
    case: dict[str, object],
    *,
    populate_node_distributions: bool,
) -> KlTreeContext:
    """Generate one benchmark case and build the exact KL hierarchy."""
    (
        data,
        true_labels,
        original_features,
        metadata,
        prepared_distance_condensed,
        _distance_matrix,
        precomputed_distance_condensed,
    ) = prepare_case_inputs(case, ["kl"])
    (
        distance_for_tree,
        tree_distance_metric,
        tree_distance_source,
        tree_linkage_method,
    ) = _resolve_kl_tree_distance(
        data=data,
        metadata=metadata,
        distance_condensed=prepared_distance_condensed,
        precomputed_distance_condensed=precomputed_distance_condensed,
    )
    linkage_matrix = linkage(distance_for_tree, method=tree_linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data.index.tolist())
    feature_space = metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Benchmark feature_space metadata must be a FeatureSpace.")
    if populate_node_distributions:
        tree.populate_node_divergences(data, feature_space=feature_space)

    return KlTreeContext(
        data=data,
        true_labels=np.asarray(true_labels),
        original_features=original_features,
        metadata=metadata,
        feature_space=feature_space,
        distance_condensed=distance_for_tree,
        tree_distance_metric=tree_distance_metric,
        tree_distance_source=tree_distance_source,
        tree_linkage_method=tree_linkage_method,
        linkage_matrix=linkage_matrix,
        tree=tree,
    )


__all__ = [
    "KlTreeContext",
    "build_kl_tree_context",
]
