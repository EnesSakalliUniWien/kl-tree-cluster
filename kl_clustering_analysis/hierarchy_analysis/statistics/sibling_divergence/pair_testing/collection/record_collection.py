"""Collection of sibling pair records for divergence testing."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.core_utils.data_utils import extract_node_sample_size

from ...projection.pair_testing.parent_projection_resolution import (
    resolve_parent_projection_inputs_for_sibling_test,
)
from ...projection.pair_testing.projection_record_metadata import (
    determine_projection_metadata_for_sibling_test,
    resolve_sibling_test_calibration_scale,
)
from ..types.sibling_pair_record import SiblingPairRecord
from .child_parent_edge_metadata import (
    determine_whether_sibling_pair_is_gate2_blocked,
    determine_whether_sibling_pair_is_null_like,
    estimate_sibling_null_prior_from_child_parent_edges,
    extract_child_parent_edge_pvalues_by_node,
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
from .sibling_test_execution import run_sibling_divergence_wald_test_with_diagnostics


def collect_sibling_pair_records(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[object, int] | None = None,
    parent_principal_component_projections: dict[object, np.ndarray] | None = None,
    parent_principal_component_eigenvalues: dict[object, np.ndarray] | None = None,
) -> tuple[list[SiblingPairRecord], list[object]]:
    """Collect raw sibling-test records for every binary-child parent node."""
    validate_child_parent_edge_annotation_requirements(annotations_df)
    child_parent_edge_significance_by_node = extract_child_parent_edge_significance_by_node(
        annotations_df
    )
    child_parent_edge_pvalues_by_node = extract_child_parent_edge_pvalues_by_node(annotations_df)
    (
        child_parent_edge_tested_by_node,
        child_parent_edge_ancestor_blocked_by_node,
    ) = extract_child_parent_edge_testing_status_by_node(annotations_df)

    records: list[SiblingPairRecord] = []
    non_binary_nodes: list[object] = []

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

        (
            test_statistic,
            degrees_of_freedom,
            p_value,
            projection_diagnostics,
        ) = run_sibling_divergence_wald_test_with_diagnostics(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            left_sample_size=float(left_sample_size),
            right_sample_size=float(right_sample_size),
            branch_length_left=branch_length_left,
            branch_length_right=branch_length_right,
            mean_branch_length=mean_branch_length,
            parent_node_id=parent_node_id,
            projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
            parent_principal_component_projection=parent_principal_component_projection,
            parent_principal_component_eigenvalues=(
                parent_principal_component_eigenvalues_for_parent
            ),
        )

        (
            projection_dimension_source,
            resolved_projection_dimension,
            used_parent_principal_component_basis,
        ) = determine_projection_metadata_for_sibling_test(
            projection_diagnostics=projection_diagnostics,
            projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
            parent_principal_component_projection=parent_principal_component_projection,
        )

        is_gate2_blocked = determine_whether_sibling_pair_is_gate2_blocked(
            left_child_id,
            right_child_id,
            child_parent_edge_tested_by_node=child_parent_edge_tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=(
                child_parent_edge_ancestor_blocked_by_node
            ),
        )
        is_null_like = determine_whether_sibling_pair_is_null_like(
            left_child_id,
            right_child_id,
            child_parent_edge_significance_by_node=child_parent_edge_significance_by_node,
        )
        sibling_null_prior_from_edge_pvalue = (
            estimate_sibling_null_prior_from_child_parent_edges(
                left_child_id,
                right_child_id,
                child_parent_edge_pvalues_by_node=child_parent_edge_pvalues_by_node,
            )
        )
        sibling_test_calibration_scale = resolve_sibling_test_calibration_scale(
            projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
            degrees_of_freedom=float(degrees_of_freedom),
        )

        records.append(
            build_sibling_pair_record(
                parent_node_id=parent_node_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                test_statistic=test_statistic,
                degrees_of_freedom=float(degrees_of_freedom),
                p_value=p_value,
                branch_length_sum=compute_sibling_branch_length_sum(
                    branch_length_left,
                    branch_length_right,
                ),
                parent_sample_size=extract_node_sample_size(tree, parent_node_id),
                is_null_like=is_null_like,
                is_gate2_blocked=is_gate2_blocked,
                sibling_null_prior_from_edge_pvalue=sibling_null_prior_from_edge_pvalue,
                sibling_test_calibration_scale=sibling_test_calibration_scale,
                projection_dimension_source=projection_dimension_source,
                resolved_projection_dimension=resolved_projection_dimension,
                used_parent_principal_component_basis=used_parent_principal_component_basis,
            )
        )

    return records, non_binary_nodes


__all__ = ["collect_sibling_pair_records"]
