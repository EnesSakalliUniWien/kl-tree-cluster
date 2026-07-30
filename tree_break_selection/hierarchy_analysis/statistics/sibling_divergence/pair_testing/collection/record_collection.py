"""Collection of sibling pair records for divergence testing."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from tree_break_selection.core_utils.data_utils import extract_node_sample_size
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    resolve_node_continuous_covariance_by_block,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from ...projection.pair_testing.parent_projection_resolution import (
    resolve_parent_projection_inputs_for_sibling_test,
)
from ..types.sibling_pair_record import SiblingPairRecord
from ..wald_statistic.sibling_divergence_test import sibling_divergence_test
from .child_parent_edge_metadata import (
    determine_whether_sibling_pair_is_edge_blocked,
    determine_whether_sibling_pair_is_null_like,
    estimate_sibling_null_weight_from_child_parent_edges,
    extract_child_parent_edge_p_values_by_node,
    extract_child_parent_edge_significance_by_node,
    extract_child_parent_edge_testing_status_by_node,
    validate_child_parent_edge_annotation_requirements,
)
from .pair_observations import (
    compute_sibling_branch_length_sum,
    extract_sibling_pair_observations,
    identify_binary_sibling_children,
)
from .sibling_pair_record_building import build_sibling_pair_record


def _parent_spectral_summary(
    eigenvalues: np.ndarray,
    *,
    projection_dimension: float,
) -> dict[str, float]:
    """Return compact parent-spectrum summaries for selected-law diagnostics."""
    values = np.asarray(eigenvalues, dtype=float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return {
            "parent_spectral_eigenvalue_count": float(finite.size),
            "parent_positive_eigenvalue_count": 0.0,
            "parent_spectral_rank": 0.0,
            "parent_eigenvalue_sum": 0.0,
            "parent_top_eigenvalue": 0.0,
            "parent_top_eigenvalue_share": 0.0,
            "parent_spectral_entropy": 0.0,
            "parent_effective_rank": 0.0,
            "parent_retained_eigenvalue_sum": 0.0,
            "parent_retained_eigenvalue_share": 0.0,
            "parent_top_spectral_gap": 0.0,
            "parent_eigengap_at_projection_dimension": 0.0,
            "parent_spectral_gap_at_projection_dimension": 0.0,
            "parent_spectral_pseudodeterminant": 0.0,
            "parent_spectral_log_pseudodeterminant": 0.0,
            "parent_spectral_geometric_mean": 0.0,
        }

    positive = np.sort(positive)[::-1]
    eigenvalue_sum = float(np.sum(positive))
    log_pseudodeterminant = float(np.sum(np.log(positive)))
    pseudodeterminant = float(np.prod(positive))
    geometric_mean = float(np.exp(log_pseudodeterminant / positive.size))
    weights = positive / eigenvalue_sum
    entropy = float(-np.sum(weights * np.log(weights)))
    retained_count = 0
    if np.isfinite(projection_dimension):
        retained_count = min(
            max(int(np.floor(float(projection_dimension))), 0),
            int(positive.size),
        )
    retained_sum = float(np.sum(positive[:retained_count])) if retained_count else 0.0
    top_spectral_gap = float(positive[0] - positive[1]) if positive.size > 1 else 0.0
    eigengap = 0.0
    spectral_gap = 0.0
    if retained_count > 0 and retained_count < positive.size:
        next_value = float(positive[retained_count])
        spectral_gap = float(max(positive[retained_count - 1] - next_value, 0.0))
        if next_value > 0.0:
            eigengap = float(positive[retained_count - 1] / next_value)

    return {
        "parent_spectral_eigenvalue_count": float(finite.size),
        "parent_positive_eigenvalue_count": float(positive.size),
        "parent_spectral_rank": float(positive.size),
        "parent_eigenvalue_sum": eigenvalue_sum,
        "parent_top_eigenvalue": float(positive[0]),
        "parent_top_eigenvalue_share": float(positive[0] / eigenvalue_sum),
        "parent_spectral_entropy": entropy,
        "parent_effective_rank": float(np.exp(entropy)),
        "parent_retained_eigenvalue_sum": retained_sum,
        "parent_retained_eigenvalue_share": float(retained_sum / eigenvalue_sum),
        "parent_top_spectral_gap": top_spectral_gap,
        "parent_eigengap_at_projection_dimension": eigengap,
        "parent_spectral_gap_at_projection_dimension": spectral_gap,
        "parent_spectral_pseudodeterminant": pseudodeterminant,
        "parent_spectral_log_pseudodeterminant": log_pseudodeterminant,
        "parent_spectral_geometric_mean": geometric_mean,
    }


def collect_sibling_pair_records(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[object, int],
    parent_principal_component_projections: dict[object, np.ndarray],
    parent_principal_component_eigenvalues: dict[object, np.ndarray],
    feature_space: FeatureSpace | None = None,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
    adaptive_projection_dimension_energy_fraction: float | None = None,
) -> tuple[list[SiblingPairRecord], list[object]]:
    """Collect raw sibling-test records for every binary-child parent node."""
    validate_child_parent_edge_annotation_requirements(annotations_df)
    child_parent_edge_significance_by_node = extract_child_parent_edge_significance_by_node(
        annotations_df
    )
    child_parent_edge_p_values_by_node = extract_child_parent_edge_p_values_by_node(annotations_df)
    (
        child_parent_edge_tested_by_node,
        child_parent_edge_ancestor_blocked_by_node,
    ) = extract_child_parent_edge_testing_status_by_node(annotations_df)

    records: list[SiblingPairRecord] = []
    non_binary_nodes: list[object] = []
    mean_branch_length = compute_mean_branch_length(tree)

    for parent_node_id in tree.nodes:
        sibling_children = identify_binary_sibling_children(tree, parent_node_id)
        if sibling_children is None:
            non_binary_nodes.append(parent_node_id)
            continue

        left_child_id, right_child_id = sibling_children
        (
            left_distribution,
            right_distribution,
            left_sample_size,
            right_sample_size,
            branch_length_left,
            branch_length_right,
        ) = extract_sibling_pair_observations(
            tree,
            parent_node_id,
            left_child_id,
            right_child_id,
        )
        branch_length_sum = compute_sibling_branch_length_sum(
            branch_length_left,
            branch_length_right,
        )

        (
            projection_dimension_from_edge_comparisons,
            parent_principal_component_projection,
            parent_principal_component_eigenvalues_for_parent,
        ) = resolve_parent_projection_inputs_for_sibling_test(
            parent_node_id,
            sibling_projection_dimensions_from_edge_comparisons=(
                sibling_projection_dimensions_from_edge_comparisons
            ),
            parent_principal_component_projections=parent_principal_component_projections,
            parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
        )

        test_statistic, reference_scale, degrees_of_freedom, p_value = sibling_divergence_test(
            left_distribution,
            right_distribution,
            float(left_sample_size),
            float(right_sample_size),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
            projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
            parent_principal_component_projection=parent_principal_component_projection,
            parent_principal_component_eigenvalues=(
                parent_principal_component_eigenvalues_for_parent
            ),
            feature_space=feature_space,
            continuous_covariance_by_block=resolve_node_continuous_covariance_by_block(
                tree,
                parent_node_id,
                feature_space,
                continuous_covariance_policy=continuous_covariance_policy,
                continuous_covariance_min_child_leaf_count=(
                    continuous_covariance_min_child_leaf_count
                ),
            ),
            adaptive_projection_dimension_energy_fraction=(
                adaptive_projection_dimension_energy_fraction
            ),
        )
        parent_spectral_summary = _parent_spectral_summary(
            parent_principal_component_eigenvalues_for_parent,
            projection_dimension=float(degrees_of_freedom),
        )

        is_edge_blocked = determine_whether_sibling_pair_is_edge_blocked(
            left_child_id,
            right_child_id,
            child_parent_edge_tested_by_node=child_parent_edge_tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=(child_parent_edge_ancestor_blocked_by_node),
        )
        is_null_like = determine_whether_sibling_pair_is_null_like(
            left_child_id,
            right_child_id,
            child_parent_edge_significance_by_node=child_parent_edge_significance_by_node,
        )
        sibling_null_weight = estimate_sibling_null_weight_from_child_parent_edges(
            left_child_id,
            right_child_id,
            child_parent_edge_p_values_by_node=child_parent_edge_p_values_by_node,
            child_parent_edge_significance_by_node=child_parent_edge_significance_by_node,
            child_parent_edge_tested_by_node=child_parent_edge_tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=(child_parent_edge_ancestor_blocked_by_node),
        )
        records.append(
            build_sibling_pair_record(
                parent_node_id=parent_node_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                test_statistic=test_statistic,
                reference_scale=reference_scale,
                degrees_of_freedom=float(degrees_of_freedom),
                p_value=p_value,
                branch_length_sum=branch_length_sum,
                parent_sample_size=extract_node_sample_size(tree, parent_node_id),
                is_null_like=is_null_like,
                is_edge_blocked=is_edge_blocked,
                sibling_null_weight=sibling_null_weight,
                sibling_projection_dimension=float(degrees_of_freedom),
                **parent_spectral_summary,
                feature_family=(
                    "bernoulli" if feature_space is None else feature_space.family_label
                ),
            )
        )

    return records, non_binary_nodes


__all__ = ["collect_sibling_pair_records"]
