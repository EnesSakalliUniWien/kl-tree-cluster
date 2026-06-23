"""Build the canonical TBS hierarchy context for benchmark diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import (
    TBS_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
    TBS_TREE_DISTANCE_SOURCE_PRECOMPUTED,
    _require_precomputed_tbs_distance_metric,
)


@dataclass(frozen=True)
class TbsTreeContext:
    """Benchmark case data plus the exact hierarchy used by the TBS method."""

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


def _resolve_tbs_tree_distance(
    *,
    data: pd.DataFrame,
    metadata: dict[str, object],
    distance_condensed: np.ndarray | None,
) -> tuple[np.ndarray, str, str, str]:
    params = METHOD_SPECS["tbs"].param_grid[0]
    tree_linkage_method = str(params["tree_linkage_method"])
    configured_metric = str(params["tree_distance_metric"])

    requires_precomputed = bool(metadata["requires_precomputed_tbs_distance"])
    if requires_precomputed:
        if distance_condensed is None:
            raise ValueError(
                f"Case '{metadata['name']}' requires precomputed TBS tree distance, "
                "but no condensed distance was prepared."
            )
        return (
            np.asarray(distance_condensed, dtype=float),
            _require_precomputed_tbs_distance_metric(
                meta=metadata,
                case_name=str(metadata["name"]),
            ),
            TBS_TREE_DISTANCE_SOURCE_PRECOMPUTED,
            tree_linkage_method,
        )

    return (
        pdist(data.values, metric=configured_metric),
        configured_metric,
        TBS_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
        tree_linkage_method,
    )


def build_tbs_tree_context(
    case: dict[str, object],
    *,
    populate_node_distributions: bool,
) -> TbsTreeContext:
    """Generate one benchmark case and build the exact TBS hierarchy."""
    inputs = prepare_case_inputs(case, ["tbs"])
    (
        distance_for_tree,
        tree_distance_metric,
        tree_distance_source,
        tree_linkage_method,
    ) = _resolve_tbs_tree_distance(
        data=inputs.data,
        metadata=inputs.metadata,
        distance_condensed=inputs.distance_condensed,
    )
    linkage_matrix = linkage(distance_for_tree, method=tree_linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=inputs.data.index.tolist())
    feature_space = inputs.metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Benchmark feature_space metadata must be a FeatureSpace.")
    if populate_node_distributions:
        tree.populate_node_divergences(inputs.data, feature_space=feature_space)

    return TbsTreeContext(
        data=inputs.data,
        true_labels=inputs.labels,
        original_features=inputs.original_features,
        metadata=inputs.metadata,
        feature_space=feature_space,
        distance_condensed=distance_for_tree,
        tree_distance_metric=tree_distance_metric,
        tree_distance_source=tree_distance_source,
        tree_linkage_method=tree_linkage_method,
        linkage_matrix=linkage_matrix,
        tree=tree,
    )


__all__ = [
    "TbsTreeContext",
    "build_tbs_tree_context",
]
