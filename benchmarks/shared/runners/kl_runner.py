"""KL method runner.

Builds a PosetTree and performs KL decomposition.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.feature_space import FeatureSpace
from kl_clustering_analysis.tree.phylogenetic import (
    iqtree3_tree_from_alignment,
    neighbor_joining_tree_from_distance,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import (
    _labels_and_report_from_decomposition,
)
from benchmarks.shared.util.time import elapsed_since


def _run_kl_on_distance(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray | None,
    sibling_significance_level: float,
    *,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    tree_linkage_method: str,
    iqtree_executable: str = "iqtree3",
    iqtree_model: str = "JC2",
    iqtree_threads: int = 1,
    iqtree_work_dir: str | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    spectral_include_internal_barycenters: bool = False,
    sibling_gate_profile: str | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
    passthrough: bool = config.PASSTHROUGH,
    extra: dict[str, object] | None = None,
) -> MethodRunResult:
    stage_timings: dict[str, float] = {}

    tree_build_start_sec = perf_counter()
    linkage_matrix = None
    phylogenetic_rooting = None
    iqtree_metadata = None
    if tree_builder == "linkage":
        if distance_condensed is None:
            raise ValueError("Linkage KL tree construction requires distance_condensed.")
        linkage_matrix = linkage(distance_condensed, method=tree_linkage_method)
        tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    elif tree_builder == "neighbor_joining":
        if distance_condensed is None:
            raise ValueError(
                "Neighbor-joining KL tree construction requires distance_condensed."
            )
        tree, phylogenetic_rooting = neighbor_joining_tree_from_distance(
            distance_condensed,
            data_df.index.astype(str).tolist(),
            rooting=tree_rooting,
        )
    elif tree_builder == "iqtree3":
        tree, phylogenetic_rooting, iqtree_metadata = iqtree3_tree_from_alignment(
            data_df,
            executable=iqtree_executable,
            model=iqtree_model,
            threads=iqtree_threads,
            rooting=tree_rooting,
            work_dir=iqtree_work_dir,
        )
    else:
        raise ValueError(f"Unsupported KL tree_builder: {tree_builder!r}.")
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
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_significance_level,
        leaf_data=data_df,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        spectral_include_internal_barycenters=(
            spectral_include_internal_barycenters
        ),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_stability_tree_distance_metric="hamming",
        root_stability_tree_linkage_method=tree_linkage_method,
        root_selective_permutation_guard_replicates=(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        root_selective_permutation_guard_tree_distance_metric="hamming",
        root_selective_permutation_guard_tree_linkage_method=tree_linkage_method,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(
            spectral_transport_unmatched_mode_penalty
        ),
    )
    stage_timings.update(gate_annotation_bundle.stage_timings)
    resolved_gate_config = gate_annotation_bundle.metadata.config

    decomposer = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_annotation_bundle,
        leaf_data=data_df,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        spectral_include_internal_barycenters=(
            spectral_include_internal_barycenters
        ),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_stability_tree_distance_metric="hamming",
        root_stability_tree_linkage_method=tree_linkage_method,
        root_selective_permutation_guard_replicates=(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        root_selective_permutation_guard_tree_distance_metric="hamming",
        root_selective_permutation_guard_tree_linkage_method=tree_linkage_method,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(
            spectral_transport_unmatched_mode_penalty
        ),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_significance_level,
        passthrough=passthrough,
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
        "tree_builder": str(tree_builder),
        "tree_rooting": str(tree_rooting),
        "phylogenetic_rooting": phylogenetic_rooting,
        "iqtree_metadata": iqtree_metadata,
        "stage_timings": stage_timings,
        "spectral_minimum_dimension": int(spectral_minimum_dimension),
        "spectral_include_internal_barycenters": bool(
            spectral_include_internal_barycenters
        ),
        "passthrough": bool(passthrough),
        "sibling_gate_profile": resolved_gate_config.sibling_gate_profile_id,
        "sibling_gate_method": str(resolved_gate_config.sibling_gate_method),
        "sibling_gate_alpha_penalty": float(
            resolved_gate_config.sibling_gate_alpha_penalty
        ),
        "root_stability_guard_threshold": (
            resolved_gate_config.root_stability_guard_threshold
        ),
        "root_stability_subsample_replicates": int(
            resolved_gate_config.root_stability_subsample_replicates
        ),
        "root_stability_feature_fraction": float(
            resolved_gate_config.root_stability_feature_fraction
        ),
        "root_stability_seed": int(resolved_gate_config.root_stability_seed),
        "root_stability_tree_distance_metric": str(
            resolved_gate_config.root_stability_tree_distance_metric
        ),
        "root_stability_tree_linkage_method": str(
            resolved_gate_config.root_stability_tree_linkage_method
        ),
        "root_selective_permutation_guard_replicates": int(
            resolved_gate_config.root_selective_permutation_guard_replicates
        ),
        "root_selective_permutation_guard_seed": int(
            resolved_gate_config.root_selective_permutation_guard_seed
        ),
        "root_selective_permutation_guard_alpha": (
            None
            if resolved_gate_config.root_selective_permutation_guard_alpha is None
            else float(resolved_gate_config.root_selective_permutation_guard_alpha)
        ),
        "root_selective_permutation_guard_scope": str(
            resolved_gate_config.root_selective_permutation_guard_scope
        ),
        "root_selective_permutation_guard_tree_distance_metric": str(
            resolved_gate_config.root_selective_permutation_guard_tree_distance_metric
        ),
        "root_selective_permutation_guard_tree_linkage_method": str(
            resolved_gate_config.root_selective_permutation_guard_tree_linkage_method
        ),
        "spectral_transport_passthrough_guard": bool(
            resolved_gate_config.spectral_transport_passthrough_guard
        ),
        "spectral_transport_max_cost": float(
            resolved_gate_config.spectral_transport_max_cost
        ),
        "spectral_transport_require_mp_blocks": bool(
            resolved_gate_config.spectral_transport_require_mp_blocks
        ),
        "spectral_transport_block_log_tolerance": float(
            resolved_gate_config.spectral_transport_block_log_tolerance
        ),
        "spectral_transport_unmatched_mode_penalty": float(
            resolved_gate_config.spectral_transport_unmatched_mode_penalty
        ),
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
    distance_condensed: np.ndarray | None,
    sibling_significance_level: float,
    tree_linkage_method: str = config.TREE_LINKAGE_METHOD,
    *,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    iqtree_executable: str = "iqtree3",
    iqtree_model: str = "JC2",
    iqtree_threads: int = 1,
    iqtree_work_dir: str | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    spectral_include_internal_barycenters: bool = False,
    sibling_gate_profile: str | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
    passthrough: bool = config.PASSTHROUGH,
) -> MethodRunResult:
    return _run_kl_on_distance(
        data_df,
        distance_condensed,
        sibling_significance_level,
        tree_builder=tree_builder,
        tree_rooting=tree_rooting,
        tree_linkage_method=tree_linkage_method,
        iqtree_executable=iqtree_executable,
        iqtree_model=iqtree_model,
        iqtree_threads=iqtree_threads,
        iqtree_work_dir=iqtree_work_dir,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        spectral_include_internal_barycenters=(
            spectral_include_internal_barycenters
        ),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_selective_permutation_guard_replicates=(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(
            spectral_transport_unmatched_mode_penalty
        ),
        passthrough=passthrough,
    )
