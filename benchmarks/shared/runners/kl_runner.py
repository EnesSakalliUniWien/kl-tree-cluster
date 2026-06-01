"""KL method runner.

Builds a PosetTree using standard hierarchical clustering linkage and performs KL decomposition.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.feature_space import FeatureSpace
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import (
    _labels_and_report_from_decomposition,
)
from benchmarks.shared.util.time import elapsed_since


def _run_kl_on_distance(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    sibling_significance_level: float,
    *,
    tree_linkage_method: str,
    feature_space: FeatureSpace | None = None,
    extra: dict[str, object] | None = None,
) -> MethodRunResult:
    stage_timings: dict[str, float] = {}

    tree_build_start_sec = perf_counter()
    linkage_matrix = linkage(distance_condensed, method=tree_linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    stage_timings["tree_build_sec"] = elapsed_since(tree_build_start_sec)

    populate_start_sec = perf_counter()
    tree.populate_node_divergences(
        data_df,
        feature_space=feature_space,
    )
    stage_timings["populate_divergences_sec"] = elapsed_since(populate_start_sec)

    gate_annotation_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df,
        edge_alpha=config.EDGE_ALPHA,
        sibling_alpha=sibling_significance_level,
        leaf_data=data_df,
        feature_space=feature_space,
    )
    stage_timings.update(gate_annotation_bundle.stage_timings)

    decomposer = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_annotation_bundle,
        leaf_data=data_df,
        feature_space=feature_space,
        edge_alpha=config.EDGE_ALPHA,
        sibling_alpha=sibling_significance_level,
    )
    traversal_start_sec = perf_counter()
    decomposition = decomposer.decompose_tree()
    stage_timings["traversal_sec"] = elapsed_since(traversal_start_sec)
    tree.annotations_df = decomposer.annotations_df

    labels, report_df = _labels_and_report_from_decomposition(
        decomposition,
        data_df.index.tolist(),
    )
    result_extra = {
        "tree": tree,
        "decomposition": decomposition,
        "annotations": tree.annotations_df,
        "gate_bundle": gate_annotation_bundle,
        "linkage_matrix": linkage_matrix,
        "stage_timings": stage_timings,
    }
    if extra:
        duplicate_extra_keys = sorted(set(result_extra).intersection(extra))
        if duplicate_extra_keys:
            raise ValueError(
                "KL runner extra metadata must not override canonical result artifacts; "
                f"duplicate key(s): {duplicate_extra_keys!r}."
            )
        result_extra.update(extra)

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomposition["num_clusters"]),
        report_df=report_df,
        status="ok",
        skip_reason=None,
        extra=result_extra,
    )


def _run_kl_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    sibling_significance_level: float,
    tree_linkage_method: str = config.TREE_LINKAGE_METHOD,
    *,
    feature_space: FeatureSpace | None = None,
) -> MethodRunResult:
    return _run_kl_on_distance(
        data_df,
        distance_condensed,
        sibling_significance_level,
        tree_linkage_method=tree_linkage_method,
        feature_space=feature_space,
    )
