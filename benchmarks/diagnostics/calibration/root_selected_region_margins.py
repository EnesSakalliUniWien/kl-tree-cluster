#!/usr/bin/env python3
"""Root selected-region margin diagnostic.

This diagnostic replays the agglomerative hierarchy used by KL-TE and records
the merge-selection inequalities that create the two root child clusters. It
then joins those margins to the observed root edge, sibling, and spectral
quantities.

The output is descriptive geometry for the selected-region proof program. It is
not an external calibration model and it does not change production inference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    effective_rank,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from scipy import stats
from scipy.spatial.distance import squareform

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.kl_tree_context import KlTreeContext, build_kl_tree_context

SCHEMA_VERSION = "root_selected_region_margins/v4"
GENERATED_BY = "benchmarks.diagnostics.calibration.root_selected_region_margins"
DIAGNOSTIC_ROLE = "descriptive_root_selected_region_geometry_not_calibration"
DEFAULT_CASE_NAMES = (
    "binary_low_noise_2c",
    "cat_mod_4cat_6c",
    "dim_diffuse_6c_136f",
    "dim_diffuse_6c_136f_continuous",
    "cat_highcard_20cat_4c",
)
NEAR_ACTIVE_ABSOLUTE_TOLERANCE = 1e-12
SMOOTH_SELECTED_REGION_METRIC = "euclidean"
RELATIONSHIP_TARGET_COLUMN = "root_sibling_selected_ratio"
RELATIONSHIP_COVARIATES = (
    "root_child_min_merge_margin",
    "root_child_min_first_order_signed_distance",
    "root_child_min_null_whitened_first_order_signed_distance",
    "root_edge_path_radial_distance",
    "root_edge_path_statistic_margin",
    "root_edge_path_bh_action",
    "root_selected_eigenvalue_over_mp_upper_bound",
)


@dataclass(frozen=True)
class LinkageReplayResult:
    """Average-linkage replay tables for one hierarchy."""

    merge_margins: pd.DataFrame
    root_children: tuple[int, int]


def _validate_linkage_inputs(
    linkage_matrix: np.ndarray,
    distance_condensed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    linkage_array = np.asarray(linkage_matrix, dtype=np.float64)
    distance_array = np.asarray(distance_condensed, dtype=np.float64)
    if linkage_array.ndim != 2 or linkage_array.shape[1] != 4:
        raise ValueError(
            "linkage_matrix must have shape (n_leaves - 1, 4); "
            f"got {linkage_array.shape}."
        )
    n_leaves = int(linkage_array.shape[0] + 1)
    expected_distances = n_leaves * (n_leaves - 1) // 2
    if distance_array.ndim != 1 or distance_array.shape[0] != expected_distances:
        raise ValueError(
            "distance_condensed length does not match linkage leaf count: "
            f"expected {expected_distances}, got {distance_array.shape}."
        )
    if not np.isfinite(linkage_array).all():
        raise ValueError("linkage_matrix contains non-finite values.")
    if not np.isfinite(distance_array).all():
        raise ValueError("distance_condensed contains non-finite values.")
    return linkage_array, distance_array, n_leaves


def _pair_key(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise ValueError(f"Cluster pair cannot repeat cluster id {left!r}.")
    return (left, right) if left < right else (right, left)


def _cluster_distance(
    distances: dict[tuple[int, int], float],
    left: int,
    right: int,
) -> float:
    return float(distances[_pair_key(left, right)])


def _format_cluster_members(members: tuple[int, ...]) -> str:
    return "|".join(str(member) for member in members)


def _average_euclidean_linkage_gradient(
    leaf_matrix: np.ndarray,
    left_members: tuple[int, ...],
    right_members: tuple[int, ...],
) -> np.ndarray:
    """Gradient of average pairwise Euclidean linkage score."""
    matrix = np.asarray(leaf_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"leaf_matrix must be 2-D; got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("leaf_matrix contains non-finite values.")
    if not left_members or not right_members:
        raise ValueError("Average-linkage gradient requires non-empty clusters.")
    gradient = np.zeros_like(matrix)
    weight = 1.0 / float(len(left_members) * len(right_members))
    for left_index in left_members:
        for right_index in right_members:
            difference = matrix[left_index] - matrix[right_index]
            norm = float(np.linalg.norm(difference))
            if norm <= 0.0:
                raise ValueError(
                    "Euclidean average-linkage selected-region geometry is "
                    "undefined for coincident observations."
                )
            contribution = weight * difference / norm
            gradient[left_index] += contribution
            gradient[right_index] -= contribution
    return gradient


def euclidean_average_linkage_inequality_geometry(
    leaf_matrix: np.ndarray,
    *,
    selected_left_members: tuple[int, ...],
    selected_right_members: tuple[int, ...],
    competitor_left_members: tuple[int, ...],
    competitor_right_members: tuple[int, ...],
    margin: float,
    null_feature_covariance: np.ndarray | None = None,
) -> dict[str, object]:
    r"""First-order geometry for one smooth average-linkage inequality.

    The selected merge condition is written as
    \(g(X)=D_X(A_t,B_t)-D_X(C,D)\le 0\). The observed interior margin is
    \(-g(X)=D_X(C,D)-D_X(A_t,B_t)\). The first-order signed distance in the
    ambient Euclidean data metric is therefore ``margin / ||grad g||``.
    """
    margin_value = float(margin)
    if not np.isfinite(margin_value):
        raise ValueError(f"margin must be finite; got {margin!r}.")
    matrix = np.asarray(leaf_matrix, dtype=np.float64)
    selected_gradient = _average_euclidean_linkage_gradient(
        matrix,
        selected_left_members,
        selected_right_members,
    )
    competitor_gradient = _average_euclidean_linkage_gradient(
        matrix,
        competitor_left_members,
        competitor_right_members,
    )
    inequality_gradient = selected_gradient - competitor_gradient
    gradient_norm = float(np.linalg.norm(inequality_gradient))
    if gradient_norm <= 0.0:
        raise ValueError(
            "Selected-region merge inequality has zero gradient; first-order "
            "signed distance is undefined."
        )
    null_scale = np.nan
    null_whitened_distance = np.nan
    null_whitened_status = "not_requested"
    if null_feature_covariance is not None:
        feature_covariance = np.asarray(null_feature_covariance, dtype=np.float64)
        if feature_covariance.ndim != 2 or feature_covariance.shape != (
            matrix.shape[1],
            matrix.shape[1],
        ):
            raise ValueError(
                "null_feature_covariance must have shape "
                f"({matrix.shape[1]}, {matrix.shape[1]}); got {feature_covariance.shape}."
            )
        if not np.isfinite(feature_covariance).all():
            raise ValueError("null_feature_covariance contains non-finite values.")
        quadratic_scale = float(
            sum(
                row_gradient @ feature_covariance @ row_gradient
                for row_gradient in inequality_gradient
            )
        )
        if quadratic_scale <= 0.0 or not np.isfinite(quadratic_scale):
            raise ValueError(
                "Null-whitened selected-region distance requires a positive finite "
                f"gradient covariance scale; got {quadratic_scale!r}."
            )
        null_scale = float(np.sqrt(quadratic_scale))
        null_whitened_distance = float(margin_value / null_scale)
        null_whitened_status = "continuous_iid_empirical_gaussian_root_null"
    return {
        "selected_region_geometry_status": "smooth_first_order_geometry_defined",
        "merge_inequality_gradient_norm": gradient_norm,
        "merge_first_order_signed_distance": float(margin_value / gradient_norm),
        "merge_null_quadratic_gradient_scale": null_scale,
        "merge_null_whitened_first_order_signed_distance": null_whitened_distance,
        "null_whitened_geometry_status": null_whitened_status,
        "tangent_cone_status": "inactive_local_interior",
        "curvature_status": "not_materialized_high_dimensional_hessian_operator",
        "discrete_tie_cell_status": "not_discrete_tie_cell",
    }


def _default_selected_region_geometry_row(
    *,
    status: str,
    tied_minimum: bool,
    margin: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    if tied_minimum:
        tie_status = "selected_discrete_tie_cell"
        tangent_status = "active_discrete_or_nonsmooth_boundary"
    elif np.isfinite(margin) and margin <= absolute_tolerance:
        tie_status = "near_tie_without_recorded_multiple_minimum"
        tangent_status = "active_or_near_active_boundary"
    else:
        tie_status = "not_discrete_tie_cell"
        tangent_status = "inactive_local_interior"
    return {
        "selected_region_geometry_status": status,
        "merge_inequality_gradient_norm": np.nan,
        "merge_first_order_signed_distance": np.nan,
        "merge_null_quadratic_gradient_scale": np.nan,
        "merge_null_whitened_first_order_signed_distance": np.nan,
        "null_whitened_geometry_status": "undefined_without_smooth_continuous_null",
        "tangent_cone_status": tangent_status,
        "curvature_status": "undefined_without_smooth_euclidean_cell",
        "discrete_tie_cell_status": tie_status,
    }


def replay_average_linkage_margins(
    linkage_matrix: np.ndarray,
    distance_condensed: np.ndarray,
    *,
    leaf_matrix: np.ndarray | None = None,
    geometry_metric: str | None = None,
    null_feature_covariance: np.ndarray | None = None,
    absolute_tolerance: float = 1e-8,
) -> LinkageReplayResult:
    """Replay SciPy average linkage and expose selected-merge margins.

    For each agglomerative step, the selected merge must be a minimum-distance
    active pair up to ``absolute_tolerance``. The margin is the nearest
    competitor score minus the selected score. The final root merge has no
    competitor because only two active clusters remain.
    """
    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance cannot be negative.")
    linkage_array, distance_array, n_leaves = _validate_linkage_inputs(
        linkage_matrix,
        distance_condensed,
    )
    if geometry_metric == SMOOTH_SELECTED_REGION_METRIC and leaf_matrix is None:
        raise ValueError("Euclidean selected-region geometry requires leaf_matrix.")
    if leaf_matrix is not None:
        matrix = np.asarray(leaf_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != n_leaves:
            raise ValueError(
                "leaf_matrix must have shape (n_leaves, n_features); "
                f"got {matrix.shape} for {n_leaves} leaves."
            )
    else:
        matrix = None
    if null_feature_covariance is not None and matrix is None:
        raise ValueError(
            "null_feature_covariance requires Euclidean leaf_matrix geometry."
        )

    square_distances = squareform(distance_array)
    active_clusters: dict[int, tuple[int, ...]] = {
        leaf_id: (leaf_id,) for leaf_id in range(n_leaves)
    }
    distances: dict[tuple[int, int], float] = {}
    for left in range(n_leaves):
        for right in range(left + 1, n_leaves):
            distances[(left, right)] = float(square_distances[left, right])

    rows: list[dict[str, object]] = []
    for step_index, row in enumerate(linkage_array):
        left = int(row[0])
        right = int(row[1])
        selected_height = float(row[2])
        reported_count = int(row[3])
        if left not in active_clusters or right not in active_clusters:
            raise ValueError(
                "Linkage replay encountered an inactive selected cluster at "
                f"step {step_index}: left={left}, right={right}."
            )
        selected_key = _pair_key(left, right)
        selected_score = _cluster_distance(distances, left, right)
        if not np.isclose(
            selected_score,
            selected_height,
            rtol=0.0,
            atol=absolute_tolerance,
        ):
            raise ValueError(
                "Average-linkage replay score disagrees with linkage height at "
                f"step {step_index}: replay={selected_score}, linkage={selected_height}."
            )

        candidate_items = [
            (pair, score)
            for pair, score in distances.items()
            if pair[0] in active_clusters and pair[1] in active_clusters
        ]
        if not candidate_items:
            raise ValueError(f"No active candidate pairs at linkage step {step_index}.")
        minimum_score = min(score for _pair, score in candidate_items)
        if selected_score > minimum_score + absolute_tolerance:
            raise ValueError(
                "Average-linkage replay selected pair is not minimal at step "
                f"{step_index}: selected={selected_score}, minimum={minimum_score}."
            )
        tied_at_selected = [
            pair
            for pair, score in candidate_items
            if abs(float(score) - selected_score) <= absolute_tolerance
        ]
        competitors = [
            (pair, score) for pair, score in candidate_items if pair != selected_key
        ]
        if competitors:
            nearest_competitor_pair, nearest_competitor_score = min(
                competitors,
                key=lambda item: (float(item[1]), item[0]),
            )
            margin = float(nearest_competitor_score - selected_score)
            margin_status = "defined"
        else:
            nearest_competitor_pair = None
            nearest_competitor_score = np.nan
            margin = np.nan
            margin_status = "final_root_merge_no_competitor"

        new_cluster_id = n_leaves + step_index
        left_members = active_clusters[left]
        right_members = active_clusters[right]
        new_members = tuple(sorted((*left_members, *right_members)))
        if reported_count != len(new_members):
            raise ValueError(
                "Linkage replay cluster size disagrees with linkage count at "
                f"step {step_index}: replay={len(new_members)}, linkage={reported_count}."
            )
        selected_pair_tied = bool(len(tied_at_selected) > 1)
        if nearest_competitor_pair is None:
            competitor_left_members: tuple[int, ...] = ()
            competitor_right_members: tuple[int, ...] = ()
            geometry_row = _default_selected_region_geometry_row(
                status="final_root_merge_no_competitor",
                tied_minimum=False,
                margin=np.nan,
                absolute_tolerance=absolute_tolerance,
            )
            geometry_row["tangent_cone_status"] = "undefined_final_root_merge"
            geometry_row["discrete_tie_cell_status"] = "not_applicable_final_root_merge"
        else:
            competitor_left_members = active_clusters[nearest_competitor_pair[0]]
            competitor_right_members = active_clusters[nearest_competitor_pair[1]]
            if geometry_metric == SMOOTH_SELECTED_REGION_METRIC:
                geometry_row = euclidean_average_linkage_inequality_geometry(
                    matrix,
                    selected_left_members=left_members,
                    selected_right_members=right_members,
                    competitor_left_members=competitor_left_members,
                    competitor_right_members=competitor_right_members,
                    margin=margin,
                    null_feature_covariance=null_feature_covariance,
                )
                if margin <= absolute_tolerance:
                    geometry_row["tangent_cone_status"] = "active_smooth_boundary"
            elif geometry_metric is None:
                geometry_row = _default_selected_region_geometry_row(
                    status="geometry_not_requested",
                    tied_minimum=selected_pair_tied,
                    margin=margin,
                    absolute_tolerance=absolute_tolerance,
                )
            else:
                geometry_row = _default_selected_region_geometry_row(
                    status="nonsmooth_or_discrete_metric",
                    tied_minimum=selected_pair_tied,
                    margin=margin,
                    absolute_tolerance=absolute_tolerance,
                )
        rows.append(
            {
                "step_index": int(step_index),
                "new_cluster_id": int(new_cluster_id),
                "left_cluster_id": int(left),
                "right_cluster_id": int(right),
                "new_cluster_size": int(len(new_members)),
                "selected_left_members": _format_cluster_members(left_members),
                "selected_right_members": _format_cluster_members(right_members),
                "selected_score": selected_score,
                "linkage_height": selected_height,
                "nearest_competitor_pair": (
                    ""
                    if nearest_competitor_pair is None
                    else f"{nearest_competitor_pair[0]}|{nearest_competitor_pair[1]}"
                ),
                "nearest_competitor_left_members": _format_cluster_members(
                    competitor_left_members
                ),
                "nearest_competitor_right_members": _format_cluster_members(
                    competitor_right_members
                ),
                "nearest_competitor_score": float(nearest_competitor_score),
                "merge_margin_to_nearest_competitor": margin,
                "merge_margin_status": margin_status,
                "candidate_pair_count": int(len(candidate_items)),
                "tied_minimum_pair_count": int(len(tied_at_selected)),
                "selected_pair_tied_for_minimum": selected_pair_tied,
                **geometry_row,
            }
        )

        old_cluster_ids = (left, right)
        other_cluster_ids = [
            cluster_id for cluster_id in active_clusters if cluster_id not in old_cluster_ids
        ]
        for other in other_cluster_ids:
            left_distance = _cluster_distance(distances, left, other)
            right_distance = _cluster_distance(distances, right, other)
            updated_distance = (
                len(left_members) * left_distance + len(right_members) * right_distance
            ) / float(len(new_members))
            distances[_pair_key(new_cluster_id, other)] = float(updated_distance)

        for key in list(distances):
            if left in key or right in key:
                del distances[key]
        del active_clusters[left]
        del active_clusters[right]
        active_clusters[new_cluster_id] = new_members

    if len(active_clusters) != 1:
        raise ValueError(
            "Linkage replay should finish with one active root cluster; "
            f"got {sorted(active_clusters)}."
        )
    root_children = (int(linkage_array[-1, 0]), int(linkage_array[-1, 1]))
    merge_margins = pd.DataFrame.from_records(rows)
    return LinkageReplayResult(
        merge_margins=merge_margins,
        root_children=root_children,
    )


def annotate_root_child_construction_roles(
    merge_margins: pd.DataFrame,
    *,
    root_children: tuple[int, int],
) -> pd.DataFrame:
    """Mark which merge inequalities construct each root child cluster."""
    if merge_margins.empty:
        raise ValueError("merge_margins cannot be empty.")
    n_leaves = int(merge_margins.shape[0] + 1)
    descendant_internal_ids: dict[int, set[int]] = {
        leaf_id: set() for leaf_id in range(n_leaves)
    }
    for row in merge_margins.itertuples(index=False):
        new_cluster_id = int(row.new_cluster_id)
        left = int(row.left_cluster_id)
        right = int(row.right_cluster_id)
        descendant_internal_ids[new_cluster_id] = {
            new_cluster_id,
            *descendant_internal_ids[left],
            *descendant_internal_ids[right],
        }

    left_root_child, right_root_child = root_children
    left_descendants = descendant_internal_ids[left_root_child]
    right_descendants = descendant_internal_ids[right_root_child]
    root_cluster_id = int(merge_margins["new_cluster_id"].iloc[-1])
    if left_descendants & right_descendants:
        raise ValueError("Root-child internal descendant sets overlap.")

    roles: list[str] = []
    root_child_sides: list[str] = []
    for new_cluster_id in merge_margins["new_cluster_id"].astype(int):
        if new_cluster_id == root_cluster_id:
            roles.append("root_final_merge")
            root_child_sides.append("root")
        elif new_cluster_id in left_descendants:
            roles.append("root_child_construction")
            root_child_sides.append("left")
        elif new_cluster_id in right_descendants:
            roles.append("root_child_construction")
            root_child_sides.append("right")
        else:
            raise ValueError(
                "Merge is neither the root final merge nor a root-child "
                f"construction merge: {new_cluster_id}."
            )

    annotated = merge_margins.copy()
    annotated["root_region_role"] = roles
    annotated["root_child_side"] = root_child_sides
    return annotated


def summarize_root_child_margin_geometry(
    annotated_margins: pd.DataFrame,
    *,
    near_active_absolute_tolerance: float = NEAR_ACTIVE_ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    """Summarize merge-inequality margins for the root selected region."""
    if near_active_absolute_tolerance < 0.0:
        raise ValueError("near_active_absolute_tolerance cannot be negative.")
    construction = annotated_margins[
        annotated_margins["root_region_role"].eq("root_child_construction")
    ].copy()
    defined = construction[
        construction["merge_margin_status"].eq("defined")
    ].copy()
    if construction.empty:
        margin_status = "root_children_are_leaves"
        min_margin = np.nan
        median_margin = np.nan
        near_active_count = 0
        tied_minimum_count = 0
        smooth_count = 0
        discrete_tie_cell_count = 0
        min_signed_distance = np.nan
        median_signed_distance = np.nan
        null_whitened_count = 0
        min_null_whitened_distance = np.nan
        median_null_whitened_distance = np.nan
        selected_region_law_status = "root_children_are_leaves"
        curvature_status = "undefined_root_children_are_leaves"
    elif defined.empty:
        raise ValueError("Root-child construction rows have no defined margins.")
    else:
        margins = defined["merge_margin_to_nearest_competitor"].to_numpy(dtype=float)
        if not np.isfinite(margins).all():
            raise ValueError("Defined root-child construction margins must be finite.")
        min_margin = float(np.min(margins))
        median_margin = float(np.quantile(margins, 0.5))
        near_active_count = int(np.count_nonzero(margins <= near_active_absolute_tolerance))
        tied_minimum_count = int(
            defined["selected_pair_tied_for_minimum"].astype(bool).sum()
        )
        margin_status = "defined"
        smooth_constraints = defined[
            defined["selected_region_geometry_status"].eq(
                "smooth_first_order_geometry_defined"
            )
        ]
        smooth_count = int(smooth_constraints.shape[0])
        discrete_tie_cell_count = int(
            defined["discrete_tie_cell_status"].eq("selected_discrete_tie_cell").sum()
        )
        if smooth_count:
            signed_distances = smooth_constraints[
                "merge_first_order_signed_distance"
            ].to_numpy(dtype=float)
            if not np.isfinite(signed_distances).all():
                raise ValueError("Smooth selected-region signed distances must be finite.")
            min_signed_distance = float(np.min(signed_distances))
            median_signed_distance = float(np.quantile(signed_distances, 0.5))
            curvature_status = "not_materialized_high_dimensional_hessian_operator"
            null_whitened_constraints = smooth_constraints[
                smooth_constraints["null_whitened_geometry_status"].eq(
                    "continuous_iid_empirical_gaussian_root_null"
                )
            ]
            null_whitened_count = int(null_whitened_constraints.shape[0])
            if null_whitened_count:
                null_whitened_distances = null_whitened_constraints[
                    "merge_null_whitened_first_order_signed_distance"
                ].to_numpy(dtype=float)
                if not np.isfinite(null_whitened_distances).all():
                    raise ValueError(
                        "Null-whitened selected-region signed distances must be finite."
                    )
                min_null_whitened_distance = float(np.min(null_whitened_distances))
                median_null_whitened_distance = float(
                    np.quantile(null_whitened_distances, 0.5)
                )
            else:
                min_null_whitened_distance = np.nan
                median_null_whitened_distance = np.nan
        else:
            min_signed_distance = np.nan
            median_signed_distance = np.nan
            null_whitened_count = 0
            min_null_whitened_distance = np.nan
            median_null_whitened_distance = np.nan
            curvature_status = "undefined_without_smooth_euclidean_cell"
        if discrete_tie_cell_count:
            selected_region_law_status = "discrete_tie_cell_geometry_required"
        elif null_whitened_count == int(defined.shape[0]):
            selected_region_law_status = (
                "null_whitened_first_order_signed_distance_defined"
            )
        elif smooth_count == int(defined.shape[0]):
            selected_region_law_status = "smooth_first_order_signed_distance_defined"
        elif smooth_count:
            selected_region_law_status = "mixed_smooth_and_nonsmooth_geometry"
        else:
            selected_region_law_status = "nonsmooth_geometry_not_defined"

    side_rows: dict[str, object] = {}
    for side in ("left", "right"):
        side_defined = defined[defined["root_child_side"].eq(side)]
        if side_defined.empty:
            side_rows[f"{side}_root_child_construction_merge_count"] = 0
            side_rows[f"{side}_root_child_min_merge_margin"] = np.nan
        else:
            side_rows[f"{side}_root_child_construction_merge_count"] = int(
                side_defined.shape[0]
            )
            side_rows[f"{side}_root_child_min_merge_margin"] = float(
                side_defined["merge_margin_to_nearest_competitor"].min()
            )

    summary = {
        "root_selected_region_diagnostic_role": DIAGNOSTIC_ROLE,
        "root_child_margin_status": margin_status,
        "root_child_construction_merge_count": int(construction.shape[0]),
        "root_child_defined_margin_count": int(defined.shape[0]),
        "root_child_min_merge_margin": min_margin,
        "root_child_median_merge_margin": median_margin,
        "root_child_near_active_merge_count": near_active_count,
        "root_child_tied_minimum_merge_count": tied_minimum_count,
        "root_child_smooth_constraint_count": smooth_count,
        "root_child_discrete_tie_cell_count": discrete_tie_cell_count,
        "root_child_min_first_order_signed_distance": min_signed_distance,
        "root_child_median_first_order_signed_distance": median_signed_distance,
        "root_child_null_whitened_constraint_count": null_whitened_count,
        "root_child_min_null_whitened_first_order_signed_distance": (
            min_null_whitened_distance
        ),
        "root_child_median_null_whitened_first_order_signed_distance": (
            median_null_whitened_distance
        ),
        "root_selected_region_law_status": selected_region_law_status,
        "root_selected_region_curvature_status": curvature_status,
        "near_active_absolute_tolerance": float(near_active_absolute_tolerance),
    }
    summary.update(side_rows)
    return summary


def _negative_log10_probability(value: float) -> float:
    p_value = float(value)
    if not np.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
        raise ValueError(f"Probability must lie in [0, 1]; got {value!r}.")
    return float(-np.log10(max(p_value, float(np.nextafter(0.0, 1.0)))))


def projected_wald_edge_opening_geometry(
    *,
    statistic: float,
    degrees_of_freedom: float,
    alpha: float,
) -> dict[str, float | str]:
    r"""Boundary geometry for one child-parent projected-Wald edge.

    Conditional on the fixed projected subspace, the edge opens when
    \(Q \ge q_{1-\alpha,k}\) for \(Q=\|Pz\|_2^2\). The local radial signed
    distance in projected z-space is therefore
    \(\sqrt Q-\sqrt{q_{1-\alpha,k}}\).
    """
    statistic_value = float(statistic)
    degrees_value = float(degrees_of_freedom)
    alpha_value = float(alpha)
    if not np.isfinite(statistic_value) or statistic_value < 0.0:
        raise ValueError(
            f"Edge projected-Wald statistic must be finite and non-negative; got {statistic!r}."
        )
    if not np.isfinite(degrees_value) or degrees_value <= 0.0:
        raise ValueError(
            f"Edge projected-Wald degrees_of_freedom must be finite and positive; "
            f"got {degrees_of_freedom!r}."
        )
    if not np.isfinite(alpha_value) or not (0.0 < alpha_value <= 1.0):
        raise ValueError(f"Edge alpha must be finite and in (0, 1]; got {alpha!r}.")
    threshold = float(stats.chi2.isf(alpha_value, df=degrees_value))
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "Edge projected-Wald chi-square threshold must be finite and non-negative; "
            f"got {threshold!r}."
        )
    return {
        "edge_opening_boundary_status": "fixed_subspace_chi_square_radial_boundary",
        "edge_chi_square_threshold": threshold,
        "edge_statistic_margin": float(statistic_value - threshold),
        "edge_statistic_over_threshold": (
            float(statistic_value / threshold) if threshold > 0.0 else np.inf
        ),
        "edge_radial_distance": float(np.sqrt(statistic_value) - np.sqrt(threshold)),
    }


def _standardized_contrast_dimension(context: KlTreeContext) -> int:
    if context.feature_space is None:
        return int(context.data.shape[1])
    return int(context.feature_space.contrast_dimension)


def _root_continuous_null_feature_covariance(
    context: KlTreeContext,
    root: object,
) -> np.ndarray:
    """Return the root empirical-Gaussian feature covariance for iid leaves."""
    feature_space = context.feature_space
    if feature_space is None or feature_space.family_label != "continuous":
        raise ValueError(
            "Null-whitened Euclidean selected-region geometry currently requires "
            "an all-continuous FeatureSpace."
        )
    covariance_by_block = require_node_continuous_covariance_by_block(
        context.tree,
        root,
        feature_space,
    )
    if covariance_by_block is None:
        raise ValueError("Continuous FeatureSpace did not provide root covariance blocks.")
    covariance = np.zeros(
        (feature_space.raw_dimension, feature_space.raw_dimension),
        dtype=np.float64,
    )
    for block in feature_space.continuous_blocks:
        block_covariance = np.asarray(covariance_by_block[block.name], dtype=np.float64)
        expected_shape = (block.raw_dimension, block.raw_dimension)
        if block_covariance.shape != expected_shape:
            raise ValueError(
                f"Continuous block {block.name!r} covariance has shape "
                f"{block_covariance.shape}; expected {expected_shape}."
            )
        block_indices = list(block.column_indices)
        covariance[np.ix_(block_indices, block_indices)] = block_covariance
    return covariance


def _root_spectral_summary(
    *,
    root: object,
    sibling_projection_dimension: int,
    spectral_context,
) -> dict[str, object]:
    node_id = str(root)
    required_maps = (
        spectral_context.test_projection_dimensions_by_node,
        spectral_context.raw_mp_signal_counts_by_node,
        spectral_context.effective_independent_rows_by_node,
        spectral_context.mp_threshold_rows_by_node,
        spectral_context.principal_component_eigenvalues_by_node,
    )
    if any(node_id not in mapping for mapping in required_maps):
        raise KeyError(f"Missing root spectral context for node {node_id!r}.")
    eigenvalues = np.asarray(
        spectral_context.principal_component_eigenvalues_by_node[node_id],
        dtype=np.float64,
    )
    if eigenvalues.ndim != 1:
        raise ValueError(f"Root eigenvalues must be 1-D; got {eigenvalues.shape}.")
    if sibling_projection_dimension <= 0:
        raise ValueError("Root sibling projection dimension must be positive.")
    if sibling_projection_dimension > eigenvalues.shape[0]:
        raise ValueError(
            "Root sibling projection dimension exceeds root eigenvalue count: "
            f"{sibling_projection_dimension} > {eigenvalues.shape[0]}."
        )
    eigenvalue_sum = float(np.sum(eigenvalues))
    if eigenvalue_sum <= 0.0 or not np.isfinite(eigenvalue_sum):
        raise ValueError("Root eigenvalue sum must be positive and finite.")
    selected_eigenvalue_mass = float(
        np.sum(eigenvalues[:sibling_projection_dimension]) / eigenvalue_sum
    )
    mp_threshold_rows = int(spectral_context.mp_threshold_rows_by_node[node_id])
    active_spectrum_width = int(eigenvalues.shape[0])
    mp_upper_bound = float(
        (1.0 + np.sqrt(float(active_spectrum_width) / float(mp_threshold_rows))) ** 2
    )
    return {
        "root_edge_projection_dimension": int(
            spectral_context.test_projection_dimensions_by_node[node_id]
        ),
        "root_raw_mp_signal_count": int(
            spectral_context.raw_mp_signal_counts_by_node[node_id]
        ),
        "root_effective_independent_rows": int(
            spectral_context.effective_independent_rows_by_node[node_id]
        ),
        "root_mp_threshold_rows": mp_threshold_rows,
        "root_active_spectrum_width": active_spectrum_width,
        "root_eigenvalue_effective_rank": effective_rank(eigenvalues),
        "root_top_eigenvalue_share": float(eigenvalues[0] / eigenvalue_sum),
        "root_selected_eigenvalue_mass_fraction": selected_eigenvalue_mass,
        "root_mp_upper_bound": mp_upper_bound,
        "root_selected_eigenvalue_over_mp_upper_bound": float(
            eigenvalues[sibling_projection_dimension - 1] / mp_upper_bound
        ),
    }


def collect_observed_root_selected_region_row(
    case: dict[str, object],
    *,
    near_active_absolute_tolerance: float = NEAR_ACTIVE_ABSOLUTE_TOLERANCE,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Collect one observed root selected-region row and its merge margins."""
    context = build_kl_tree_context(case, populate_node_distributions=True)
    if context.tree_linkage_method != "average":
        raise ValueError(
            "Root selected-region margin replay currently supports average linkage; "
            f"got {context.tree_linkage_method!r}."
        )
    root = context.tree.root()
    leaf_matrix = (
        context.data.to_numpy(dtype=np.float64)
        if context.tree_distance_metric == SMOOTH_SELECTED_REGION_METRIC
        else None
    )
    null_feature_covariance = (
        _root_continuous_null_feature_covariance(context, root)
        if context.tree_distance_metric == SMOOTH_SELECTED_REGION_METRIC
        else None
    )
    replay_result = replay_average_linkage_margins(
        context.linkage_matrix,
        context.distance_condensed,
        leaf_matrix=leaf_matrix,
        geometry_metric=context.tree_distance_metric,
        null_feature_covariance=null_feature_covariance,
    )
    merge_margins = annotate_root_child_construction_roles(
        replay_result.merge_margins,
        root_children=replay_result.root_children,
    )
    margin_summary = summarize_root_child_margin_geometry(
        merge_margins,
        near_active_absolute_tolerance=near_active_absolute_tolerance,
    )

    edge_df, spectral_context = annotate_child_parent_divergence_with_context(
        context.tree,
        context.tree.annotations_df,
        significance_level_alpha=config.EDGE_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        context.tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    mean_branch_length = (
        compute_mean_branch_length(context.tree) if config.FELSENSTEIN_SCALING else None
    )
    records, non_binary_nodes = collect_sibling_pair_records(
        context.tree,
        edge_df,
        mean_branch_length,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=context.feature_space,
    )
    root_records = [record for record in records if record.parent == root]
    if len(root_records) != 1:
        raise ValueError(
            f"Expected exactly one root sibling record for {root!r}; got {len(root_records)}."
        )
    root_record = root_records[0]
    root_children = list(context.tree.successors(root))
    if len(root_children) != 2 or set(root_children) != {root_record.left, root_record.right}:
        raise ValueError("Root sibling record does not match the two root children.")

    left_child, right_child = root_record.left, root_record.right
    required_edge_columns = (
        "Child_Parent_Divergence_Test_Statistic",
        "Child_Parent_Divergence_df",
        "Child_Parent_Divergence_P_Value",
        "Child_Parent_Divergence_P_Value_BH",
        "Child_Parent_Divergence_Significant",
    )
    for child in (left_child, right_child):
        if child not in edge_df.index:
            raise KeyError(f"Missing edge row for root child {child!r}.")
        missing_columns = [column for column in required_edge_columns if column not in edge_df]
        if missing_columns:
            raise KeyError(f"Missing edge columns: {missing_columns}.")
    left_edge_bh = float(edge_df.loc[left_child, "Child_Parent_Divergence_P_Value_BH"])
    right_edge_bh = float(edge_df.loc[right_child, "Child_Parent_Divergence_P_Value_BH"])
    left_edge_raw = float(edge_df.loc[left_child, "Child_Parent_Divergence_P_Value"])
    right_edge_raw = float(edge_df.loc[right_child, "Child_Parent_Divergence_P_Value"])
    left_edge_statistic = float(
        edge_df.loc[left_child, "Child_Parent_Divergence_Test_Statistic"]
    )
    right_edge_statistic = float(
        edge_df.loc[right_child, "Child_Parent_Divergence_Test_Statistic"]
    )
    left_edge_degrees_of_freedom = float(
        edge_df.loc[left_child, "Child_Parent_Divergence_df"]
    )
    right_edge_degrees_of_freedom = float(
        edge_df.loc[right_child, "Child_Parent_Divergence_df"]
    )
    left_edge_geometry = projected_wald_edge_opening_geometry(
        statistic=left_edge_statistic,
        degrees_of_freedom=left_edge_degrees_of_freedom,
        alpha=config.EDGE_ALPHA,
    )
    right_edge_geometry = projected_wald_edge_opening_geometry(
        statistic=right_edge_statistic,
        degrees_of_freedom=right_edge_degrees_of_freedom,
        alpha=config.EDGE_ALPHA,
    )
    left_edge_raw_action = _negative_log10_probability(left_edge_raw)
    right_edge_raw_action = _negative_log10_probability(right_edge_raw)
    left_edge_bh_action = _negative_log10_probability(left_edge_bh)
    right_edge_bh_action = _negative_log10_probability(right_edge_bh)
    if not bool(edge_df.loc[left_child, "Child_Parent_Divergence_Significant"]):
        edge_status = "left_child_edge_closed"
    elif not bool(edge_df.loc[right_child, "Child_Parent_Divergence_Significant"]):
        edge_status = "right_child_edge_closed"
    else:
        edge_status = "both_root_child_edges_open"

    reference_expectation = float(root_record.reference_scale * root_record.degrees_of_freedom)
    if reference_expectation <= 0.0:
        raise ValueError("Root sibling reference expectation must be positive.")
    row = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "case_id": str(context.metadata["name"]),
        "category": str(context.metadata["category"]),
        "generator": str(context.metadata["generator"]),
        "feature_family": str(root_record.feature_family),
        "feature_representation": str(context.metadata["feature_representation"]),
        "n_samples": int(context.data.shape[0]),
        "feature_dimension": int(context.data.shape[1]),
        "standardized_contrast_dimension": _standardized_contrast_dimension(context),
        "tree_distance_metric": context.tree_distance_metric,
        "tree_distance_source": context.tree_distance_source,
        "tree_linkage_method": context.tree_linkage_method,
        "root": root,
        "root_left_child": left_child,
        "root_right_child": right_child,
        "root_left_child_sample_size": int(context.tree.nodes[left_child]["leaf_count"]),
        "root_right_child_sample_size": int(context.tree.nodes[right_child]["leaf_count"]),
        "root_child_balance": float(
            min(
                context.tree.nodes[left_child]["leaf_count"],
                context.tree.nodes[right_child]["leaf_count"],
            )
            / int(context.tree.nodes[root]["leaf_count"])
        ),
        "root_edge_status": edge_status,
        "root_edge_opening_boundary_status": left_edge_geometry[
            "edge_opening_boundary_status"
        ],
        "root_left_edge_statistic": left_edge_statistic,
        "root_right_edge_statistic": right_edge_statistic,
        "root_left_edge_degrees_of_freedom": left_edge_degrees_of_freedom,
        "root_right_edge_degrees_of_freedom": right_edge_degrees_of_freedom,
        "root_left_edge_chi_square_threshold": left_edge_geometry[
            "edge_chi_square_threshold"
        ],
        "root_right_edge_chi_square_threshold": right_edge_geometry[
            "edge_chi_square_threshold"
        ],
        "root_left_edge_statistic_margin": left_edge_geometry[
            "edge_statistic_margin"
        ],
        "root_right_edge_statistic_margin": right_edge_geometry[
            "edge_statistic_margin"
        ],
        "root_edge_path_statistic_margin": min(
            float(left_edge_geometry["edge_statistic_margin"]),
            float(right_edge_geometry["edge_statistic_margin"]),
        ),
        "root_left_edge_statistic_over_threshold": left_edge_geometry[
            "edge_statistic_over_threshold"
        ],
        "root_right_edge_statistic_over_threshold": right_edge_geometry[
            "edge_statistic_over_threshold"
        ],
        "root_edge_path_statistic_over_threshold": min(
            float(left_edge_geometry["edge_statistic_over_threshold"]),
            float(right_edge_geometry["edge_statistic_over_threshold"]),
        ),
        "root_left_edge_radial_distance": left_edge_geometry["edge_radial_distance"],
        "root_right_edge_radial_distance": right_edge_geometry["edge_radial_distance"],
        "root_edge_path_radial_distance": min(
            float(left_edge_geometry["edge_radial_distance"]),
            float(right_edge_geometry["edge_radial_distance"]),
        ),
        "root_left_edge_raw_p_value": left_edge_raw,
        "root_right_edge_raw_p_value": right_edge_raw,
        "root_edge_path_raw_p_value": max(left_edge_raw, right_edge_raw),
        "root_left_edge_raw_action": left_edge_raw_action,
        "root_right_edge_raw_action": right_edge_raw_action,
        "root_edge_path_raw_action": min(left_edge_raw_action, right_edge_raw_action),
        "root_left_edge_bh_p_value": left_edge_bh,
        "root_right_edge_bh_p_value": right_edge_bh,
        "root_edge_path_bh_p_value": max(left_edge_bh, right_edge_bh),
        "root_left_edge_bh_action": left_edge_bh_action,
        "root_right_edge_bh_action": right_edge_bh_action,
        "root_edge_path_bh_action": min(left_edge_bh_action, right_edge_bh_action),
        "root_sibling_statistic": float(root_record.stat),
        "root_sibling_reference_scale": float(root_record.reference_scale),
        "root_sibling_degrees_of_freedom": float(root_record.degrees_of_freedom),
        "root_sibling_reference_expectation": reference_expectation,
        "root_sibling_selected_ratio": float(root_record.stat / reference_expectation),
        "root_sibling_raw_p_value": float(root_record.p_value),
        "root_sibling_projection_dimension": int(root_record.sibling_projection_dimension),
        "root_sibling_null_weight": float(root_record.sibling_null_weight),
        "root_sibling_is_null_like": bool(root_record.is_null_like),
        "root_sibling_is_edge_blocked": bool(root_record.is_edge_blocked),
        "non_binary_sibling_node_count": int(len(non_binary_nodes)),
    }
    row.update(margin_summary)
    row.update(
        _root_spectral_summary(
            root=root,
            sibling_projection_dimension=int(root_record.sibling_projection_dimension),
            spectral_context=spectral_context,
        )
    )
    return row, merge_margins


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _select_cases(
    *,
    suite: str,
    case_names: Sequence[str],
) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    if not case_names:
        return cases
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing}.")
    return [by_name[name].copy() for name in case_names]


def summarize_root_selected_region_relationships(root_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize descriptive links between geometry variables and root ratio."""
    if RELATIONSHIP_TARGET_COLUMN not in root_table.columns:
        raise KeyError(f"Missing relationship target {RELATIONSHIP_TARGET_COLUMN!r}.")
    missing_covariates = [
        covariate
        for covariate in RELATIONSHIP_COVARIATES
        if covariate not in root_table.columns
    ]
    if missing_covariates:
        raise KeyError(f"Missing relationship covariates: {missing_covariates}.")

    target_values = pd.to_numeric(
        root_table[RELATIONSHIP_TARGET_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(target_values)) or np.any(target_values <= 0.0):
        raise ValueError("root_sibling_selected_ratio must be positive and finite.")
    log_target = np.log(target_values)

    rows: list[dict[str, object]] = []
    for covariate in RELATIONSHIP_COVARIATES:
        covariate_values = pd.to_numeric(root_table[covariate], errors="coerce").to_numpy(
            dtype=float
        )
        valid_mask = np.isfinite(covariate_values) & np.isfinite(log_target)
        valid_count = int(np.count_nonzero(valid_mask))
        unique_count = int(np.unique(covariate_values[valid_mask]).shape[0])
        if valid_count < 3:
            status = "insufficient_valid_pairs"
            spearman_r = np.nan
            spearman_p = np.nan
            log_pearson_r = np.nan
        elif unique_count < 2:
            status = "constant_covariate"
            spearman_r = np.nan
            spearman_p = np.nan
            log_pearson_r = np.nan
        else:
            status = "evaluated"
            spearman = stats.spearmanr(
                covariate_values[valid_mask],
                log_target[valid_mask],
            )
            spearman_r = float(spearman.statistic)
            spearman_p = float(spearman.pvalue)
            if np.all(covariate_values[valid_mask] > 0.0):
                log_pearson_r = float(
                    np.corrcoef(
                        np.log(covariate_values[valid_mask]),
                        log_target[valid_mask],
                    )[0, 1]
                )
            else:
                log_pearson_r = np.nan
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "diagnostic_role": DIAGNOSTIC_ROLE,
                "target": f"log_{RELATIONSHIP_TARGET_COLUMN}",
                "covariate": covariate,
                "relationship_status": status,
                "valid_pair_count": valid_count,
                "unique_covariate_count": unique_count,
                "spearman_r": spearman_r,
                "spearman_p_value": spearman_p,
                "log_log_pearson_r": log_pearson_r,
            }
        )
    return pd.DataFrame.from_records(rows)


def _make_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        resolved = (
            Path("benchmarks")
            / "results"
            / f"root_selected_region_margins_{stamp}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_root_selected_region_margin_diagnostic(
    *,
    suite: str = "full",
    case_names: Sequence[str] = DEFAULT_CASE_NAMES,
    output_dir: Path | None = None,
    near_active_absolute_tolerance: float = NEAR_ACTIVE_ABSOLUTE_TOLERANCE,
) -> dict[str, Path]:
    """Run the root selected-region margin diagnostic and write CSV outputs."""
    start = perf_counter()
    selected_cases = _select_cases(suite=suite, case_names=case_names)
    resolved_output_dir = _make_output_dir(output_dir)
    root_rows: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    for case in selected_cases:
        row, margins = collect_observed_root_selected_region_row(
            case,
            near_active_absolute_tolerance=near_active_absolute_tolerance,
        )
        root_rows.append(row)
        case_margins = margins.copy()
        case_margins.insert(0, "case_id", row["case_id"])
        case_margins.insert(1, "schema_version", SCHEMA_VERSION)
        case_margins.insert(2, "diagnostic_role", DIAGNOSTIC_ROLE)
        margin_tables.append(case_margins)

    root_table = pd.DataFrame.from_records(root_rows)
    merge_margin_table = pd.concat(margin_tables, ignore_index=True)
    relationship_table = summarize_root_selected_region_relationships(root_table)
    root_path = resolved_output_dir / "root_selected_region_summary.csv"
    merge_path = resolved_output_dir / "root_selected_region_merge_margins.csv"
    relationship_path = resolved_output_dir / "root_selected_region_relationships.csv"
    manifest_path = resolved_output_dir / "manifest.json"
    root_table.to_csv(root_path, index=False)
    merge_margin_table.to_csv(merge_path, index=False)
    relationship_table.to_csv(relationship_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "suite": suite,
        "case_names": [str(row["case_id"]) for row in root_rows],
        "near_active_absolute_tolerance": float(near_active_absolute_tolerance),
        "root_rows": int(root_table.shape[0]),
        "merge_margin_rows": int(merge_margin_table.shape[0]),
        "relationship_rows": int(relationship_table.shape[0]),
        "elapsed_seconds": float(perf_counter() - start),
        "outputs": {
            "root_selected_region_summary": str(root_path),
            "root_selected_region_merge_margins": str(merge_path),
            "root_selected_region_relationships": str(relationship_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "root_selected_region_summary": root_path,
        "root_selected_region_merge_margins": merge_path,
        "root_selected_region_relationships": relationship_path,
        "manifest": manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay average-linkage merge-selection margins for observed root "
            "contexts and join them to root edge/sibling/spectral quantities."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("full", "binary", "categorical", "continuous", "discretized_gaussian", "graph"),
        default="full",
    )
    parser.add_argument(
        "--case-names",
        default=",".join(DEFAULT_CASE_NAMES),
        help="Comma-separated case names from the selected suite.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--near-active-absolute-tolerance",
        type=float,
        default=NEAR_ACTIVE_ABSOLUTE_TOLERANCE,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_root_selected_region_margin_diagnostic(
        suite=args.suite,
        case_names=_parse_csv_list(args.case_names),
        output_dir=args.output_dir,
        near_active_absolute_tolerance=args.near_active_absolute_tolerance,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DIAGNOSTIC_ROLE",
    "LinkageReplayResult",
    "annotate_root_child_construction_roles",
    "collect_observed_root_selected_region_row",
    "projected_wald_edge_opening_geometry",
    "replay_average_linkage_margins",
    "run_root_selected_region_margin_diagnostic",
    "summarize_root_selected_region_relationships",
    "summarize_root_child_margin_geometry",
]
