"""Selected-hierarchy geometry covariate diagnostic.

This diagnostic records the tree, edge-selection, eigenvalue, and angular
geometry of selected sibling records under regenerated same-data null
hierarchies. It is descriptive evidence for deciding which variables belong in
an external selected-hierarchy null law. It is not a calibration fallback and
does not change production inference.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats
from tree_break_selection.core_utils.tree_utils import compute_node_depths
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    effective_rank,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    compute_sibling_branch_length_sum,
    extract_sibling_pair_observations,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from tree_break_selection.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from benchmarks.diagnostics.calibration.selected.hierarchy.selected_hierarchy_null_audit import (
    DEFAULT_CASE_NAMES,
    _build_tree,
    _parse_csv_list,
    _selected_cases,
    _simulate_null_data,
)
from benchmarks.diagnostics.calibration.selected.hierarchy.selected_hierarchy_stratification_diagnostic import (
    _parent_size_bin,
)
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "descriptive_selected_hierarchy_geometry_not_calibration"
RESPONSE_COLUMN = "log_selected_hierarchy_ratio"
SIMULATION_ID_COLUMN = "selected_hierarchy_simulation_id"
TAIL_LAW_ROLE = "descriptive_selected_ratio_tail_law_not_calibration"
TAIL_LAW_CONTEXT_COLUMNS = (
    "source_family",
    "feature_family",
    "parent_size_bin",
    "sibling_projection_dimension",
    "edge_action_bin",
)
EDGE_ACTION_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, np.inf)
EDGE_ACTION_BIN_LABELS = (
    "edge_action_0_2",
    "edge_action_2_4",
    "edge_action_4_6",
    "edge_action_6_8",
    "edge_action_ge8",
)

CORRELATION_COVARIATES: tuple[tuple[str, str], ...] = (
    ("tree_geometry", "parent_depth"),
    ("tree_geometry", "parent_size_fraction"),
    ("tree_geometry", "child_balance"),
    ("tree_geometry", "child_size_ratio"),
    ("tree_geometry", "branch_length_sum"),
    ("tree_geometry", "branch_length_asymmetry"),
    ("edge_selection", "left_edge_raw_p_value"),
    ("edge_selection", "right_edge_raw_p_value"),
    ("edge_selection", "min_child_edge_raw_p_value"),
    ("edge_selection", "max_child_edge_raw_p_value"),
    ("edge_selection", "left_edge_bh_p_value"),
    ("edge_selection", "right_edge_bh_p_value"),
    ("edge_selection", "min_child_edge_bh_p_value"),
    ("edge_selection", "max_child_edge_bh_p_value"),
    ("edge_selection", "negative_log10_min_child_edge_bh_p_value"),
    ("spectral", "raw_mp_signal_count"),
    ("spectral", "parent_test_projection_dimension"),
    ("spectral", "sibling_projection_dimension"),
    ("spectral", "effective_independent_rows"),
    ("spectral", "mp_threshold_rows"),
    ("spectral", "eigenvalue_effective_rank"),
    ("spectral", "top_eigenvalue_share"),
    ("spectral", "selected_eigenvalue_mass_fraction"),
    ("spectral", "eigengap_at_sibling_projection_dimension"),
    ("spectral", "selected_eigenvalue_over_mp_upper_bound"),
    ("angular", "selected_subspace_cos2"),
    ("angular", "selected_subspace_sin2"),
    ("angular", "selected_subspace_tan2"),
    ("angular", "top_component_cos2"),
    ("angular", "max_component_cos2"),
    ("angular", "component_cos2_entropy"),
)

MODEL_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tree_geometry",
        (
            "parent_depth",
            "parent_size_fraction",
            "child_balance",
            "branch_length_sum",
            "branch_length_asymmetry",
        ),
    ),
    (
        "edge_selection",
        ("negative_log10_min_child_edge_bh_p_value",),
    ),
    (
        "spectral",
        (
            "raw_mp_signal_count",
            "eigenvalue_effective_rank",
            "top_eigenvalue_share",
            "selected_eigenvalue_mass_fraction",
            "selected_eigenvalue_over_mp_upper_bound",
        ),
    ),
    (
        "angular",
        (
            "selected_subspace_cos2",
            "top_component_cos2",
            "component_cos2_entropy",
        ),
    ),
    (
        "spectrum_plus_angle",
        (
            "eigenvalue_effective_rank",
            "top_eigenvalue_share",
            "selected_eigenvalue_mass_fraction",
            "selected_eigenvalue_over_mp_upper_bound",
            "selected_subspace_cos2",
            "top_component_cos2",
            "max_component_cos2",
            "component_cos2_entropy",
        ),
    ),
)

CANDIDATE_EQUATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "edge_action",
        "log_R ~ A_edge",
        ("edge_action",),
    ),
    (
        "edge_sampling_geometry",
        "log_R ~ A_edge + log(p/n_parent) + log(1/n_L+1/n_R) + child_balance",
        (
            "edge_action",
            "log_feature_parent_aspect_ratio",
            "log_sampling_variance_scale",
            "child_balance",
        ),
    ),
    (
        "edge_spectral_modes",
        "log_R ~ A_edge + log(lambda_k/lambda_MP) + spectral_mass + effective_rank",
        (
            "edge_action",
            "log_selected_eigenvalue_over_mp_upper_bound",
            "selected_eigenvalue_mass_fraction",
            "eigenvalue_effective_rank",
        ),
    ),
    (
        "edge_angular_capture",
        "log_R ~ A_edge + rho + top_component_cos2 + component_cos2_entropy",
        (
            "edge_action",
            "selected_subspace_cos2",
            "top_component_cos2",
            "component_cos2_entropy",
        ),
    ),
    (
        "selected_energy_candidate",
        "log_R ~ A_edge + log(lambda_k/lambda_MP) + rho + log(p/n_parent)",
        (
            "edge_action",
            "log_selected_eigenvalue_over_mp_upper_bound",
            "selected_subspace_cos2",
            "log_feature_parent_aspect_ratio",
        ),
    ),
    (
        "full_descriptive_candidate",
        (
            "log_R ~ A_edge + log(p/n_parent) + log(1/n_L+1/n_R) + "
            "log(lambda_k/lambda_MP) + spectral_mass + rho + component_entropy"
        ),
        (
            "edge_action",
            "log_feature_parent_aspect_ratio",
            "log_sampling_variance_scale",
            "log_selected_eigenvalue_over_mp_upper_bound",
            "selected_eigenvalue_mass_fraction",
            "selected_subspace_cos2",
            "component_cos2_entropy",
        ),
    ),
)


@dataclass(frozen=True)
class GeometryStudySample:
    """Selected geometry rows from one regenerated hierarchy."""

    rows: tuple[dict[str, object], ...]
    candidate_records: int
    selected_records: int
    tested_edges: int
    significant_edges: int
    tree_metric: str


def _require_edge_value(
    edge_df: pd.DataFrame,
    node_id: object,
    column_name: str,
) -> object:
    if node_id not in edge_df.index:
        raise KeyError(f"Missing edge row for child node {node_id!r}.")
    if column_name not in edge_df.columns:
        raise KeyError(f"Missing edge column {column_name!r}.")
    return edge_df.loc[node_id, column_name]


def _edge_float(edge_df: pd.DataFrame, node_id: object, column_name: str) -> float:
    value = float(_require_edge_value(edge_df, node_id, column_name))
    if not np.isfinite(value):
        raise ValueError(
            f"Edge value {column_name!r} for child {node_id!r} must be finite; got {value!r}."
        )
    return value


def _edge_bool(edge_df: pd.DataFrame, node_id: object, column_name: str) -> bool:
    return bool(_require_edge_value(edge_df, node_id, column_name))


def _bounded_unit_interval(value: float, *, value_name: str, tolerance: float = 1e-8) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{value_name} must be finite; got {value!r}.")
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"{value_name} must lie in [0, 1]; got {value!r}.")
    return float(min(1.0, max(0.0, value)))


def _defined_ratio_or_nan(
    numerator: float,
    denominator: float,
    *,
    status_when_undefined: str,
) -> tuple[float, str]:
    if denominator > 0.0:
        return float(numerator / denominator), "defined"
    return np.nan, status_when_undefined


def _defined_log10_p_value(p_value: float) -> float:
    if not 0.0 <= p_value <= 1.0:
        raise ValueError(f"p_value must lie in [0, 1]; got {p_value!r}.")
    return float(-np.log10(max(p_value, float(np.nextafter(0.0, 1.0)))))


def _positive_log_array(values: pd.Series, *, value_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="raise").astype(float)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    finite_positive = np.isfinite(numeric_values) & (numeric_values > 0.0)
    output.loc[finite_positive] = np.log(numeric_values.loc[finite_positive])
    if bool(np.any(np.isfinite(numeric_values) & (numeric_values < 0.0))):
        raise ValueError(f"{value_name} must be non-negative for log transform.")
    return output


def _log_selected_hierarchy_ratio(ratio: float) -> tuple[float, str]:
    if ratio < 0.0 or not np.isfinite(ratio):
        raise ValueError(
            f"selected_hierarchy_ratio must be finite and non-negative; got {ratio!r}."
        )
    if ratio == 0.0:
        return np.nan, "undefined_zero_selected_hierarchy_ratio"
    return float(np.log(ratio)), "defined"


def _component_entropy(component_cos2: np.ndarray) -> float:
    total = float(np.sum(component_cos2))
    if total <= 0.0:
        return 0.0
    probabilities = component_cos2 / total
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _descendant_nodes_including_self(tree, node: object) -> tuple[object, ...]:
    nodes: list[object] = [node]
    stack = [node]
    while stack:
        current = stack.pop()
        children = list(tree.successors(current))
        nodes.extend(children)
        stack.extend(children)
    return tuple(nodes)


def _subtree_topology_geometry(
    *,
    tree,
    parent: object,
    node_depths: dict[object, int],
) -> dict[str, object]:
    parent_depth = int(node_depths[parent])
    nodes = _descendant_nodes_including_self(tree, parent)
    internal_nodes = tuple(node for node in nodes if not bool(tree.nodes[node]["is_leaf"]))
    leaf_nodes = tuple(node for node in nodes if bool(tree.nodes[node]["is_leaf"]))
    descendant_internal_node_count = max(len(internal_nodes) - 1, 0)
    subtree_height = max(int(node_depths[node]) - parent_depth for node in nodes)
    leaf_depths = np.asarray(
        [int(node_depths[node]) - parent_depth for node in leaf_nodes],
        dtype=float,
    )
    branch_lengths = np.asarray(
        [
            float(tree.edges[node, child]["branch_length"])
            for node in nodes
            for child in tree.successors(node)
        ],
        dtype=float,
    )
    positive_branch_lengths = branch_lengths[branch_lengths > 0.0]

    colless_sum = 0.0
    for node in internal_nodes:
        children = list(tree.successors(node))
        if len(children) != 2:
            continue
        left_count = int(tree.nodes[children[0]]["leaf_count"])
        right_count = int(tree.nodes[children[1]]["leaf_count"])
        colless_sum += abs(left_count - right_count)
    parent_leaf_count = int(tree.nodes[parent]["leaf_count"])
    max_colless = (parent_leaf_count - 1) * (parent_leaf_count - 2) / 2.0
    normalized_colless = float(colless_sum / max_colless) if max_colless > 0.0 else 0.0

    branch_length_mean = float(np.mean(branch_lengths)) if branch_lengths.size else np.nan
    branch_length_std = float(np.std(branch_lengths, ddof=0)) if branch_lengths.size else np.nan
    branch_length_cv = (
        float(branch_length_std / branch_length_mean)
        if np.isfinite(branch_length_mean) and branch_length_mean > 0.0
        else np.nan
    )
    branch_length_condition_ratio = (
        float(np.max(positive_branch_lengths) / np.min(positive_branch_lengths))
        if positive_branch_lengths.size
        else np.nan
    )

    return {
        "subtree_internal_node_count": int(len(internal_nodes)),
        "subtree_descendant_internal_node_count": int(descendant_internal_node_count),
        "subtree_leaf_count": int(len(leaf_nodes)),
        "subtree_height": int(subtree_height),
        "subtree_sackin_mean_depth": float(np.mean(leaf_depths)),
        "subtree_sackin_max_depth": float(np.max(leaf_depths)),
        "subtree_colless_imbalance": float(colless_sum),
        "subtree_colless_normalized": normalized_colless,
        "subtree_branch_length_total": (
            float(np.sum(branch_lengths)) if branch_lengths.size else np.nan
        ),
        "subtree_branch_length_mean": branch_length_mean,
        "subtree_branch_length_cv": branch_length_cv,
        "subtree_branch_length_condition_ratio": branch_length_condition_ratio,
    }


def _sibling_whitened_contrast(
    *,
    tree,
    record: SiblingPairRecord,
    feature_space: FeatureSpace | None,
) -> tuple[np.ndarray, int, int, float | None, float | None, float | None]:
    (
        left_distribution,
        right_distribution,
        left_sample_size,
        right_sample_size,
        branch_length_left,
        branch_length_right,
    ) = extract_sibling_pair_observations(
        tree,
        record.parent,
        record.left,
        record.right,
    )
    raw_branch_length_sum = compute_sibling_branch_length_sum(
        branch_length_left,
        branch_length_right,
    )
    z_scores = compute_whitened_wald_contrast(
        left_distribution,
        right_distribution,
        float(left_sample_size),
        float(right_sample_size),
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=require_node_continuous_covariance_by_block(
            tree,
            record.parent,
            feature_space,
        ),
    )
    return (
        z_scores,
        int(left_sample_size),
        int(right_sample_size),
        branch_length_left,
        branch_length_right,
        raw_branch_length_sum,
    )


def _spectral_geometry(
    *,
    parent: object,
    projection_dimension: int,
    z_scores: np.ndarray,
    spectral_context: SpectralContext,
) -> dict[str, object]:
    node_id = str(parent)
    if node_id not in spectral_context.principal_component_projections_by_node:
        raise KeyError(f"Missing PCA projection for selected parent {parent!r}.")
    if node_id not in spectral_context.principal_component_eigenvalues_by_node:
        raise KeyError(f"Missing PCA eigenvalues for selected parent {parent!r}.")

    projection_matrix = np.asarray(
        spectral_context.principal_component_projections_by_node[node_id],
        dtype=np.float64,
    )
    eigenvalues = np.asarray(
        spectral_context.principal_component_eigenvalues_by_node[node_id],
        dtype=np.float64,
    )
    if projection_matrix.ndim != 2:
        raise ValueError(
            f"Projection for parent {parent!r} must be 2-D; got {projection_matrix.shape}."
        )
    if eigenvalues.ndim != 1:
        raise ValueError(f"Eigenvalues for parent {parent!r} must be 1-D; got {eigenvalues.shape}.")
    if projection_matrix.shape[0] != eigenvalues.shape[0]:
        raise ValueError(
            f"Projection/eigenvalue row mismatch for parent {parent!r}: "
            f"{projection_matrix.shape[0]} vs {eigenvalues.shape[0]}."
        )
    if projection_matrix.shape[1] != z_scores.shape[0]:
        raise ValueError(
            f"Projection width for parent {parent!r} is {projection_matrix.shape[1]}, "
            f"but sibling z dimension is {z_scores.shape[0]}."
        )
    if projection_dimension <= 0:
        raise ValueError(
            "Selected sibling geometry requires a positive projection dimension; "
            f"parent={parent!r}, projection_dimension={projection_dimension!r}."
        )
    if projection_dimension > projection_matrix.shape[0]:
        raise ValueError(
            f"Requested projection dimension {projection_dimension} exceeds available "
            f"PCA rows {projection_matrix.shape[0]} for parent {parent!r}."
        )

    z_norm_sq = float(np.dot(z_scores, z_scores))
    if z_norm_sq <= 0.0 or not np.isfinite(z_norm_sq):
        raise ValueError(
            f"Selected sibling contrast for parent {parent!r} must have positive "
            f"finite squared norm; got {z_norm_sq!r}."
        )

    selected_projection = projection_matrix[:projection_dimension]
    projected_components = selected_projection @ z_scores
    component_cos2 = (projected_components * projected_components) / z_norm_sq
    selected_subspace_cos2 = _bounded_unit_interval(
        float(np.sum(component_cos2)),
        value_name="selected_subspace_cos2",
    )
    selected_subspace_sin2 = float(1.0 - selected_subspace_cos2)
    selected_subspace_tan2, tan2_status = _defined_ratio_or_nan(
        selected_subspace_sin2,
        selected_subspace_cos2,
        status_when_undefined="undefined_zero_selected_subspace_cos2",
    )
    selected_angle_degrees = float(np.degrees(np.arccos(np.sqrt(selected_subspace_cos2))))

    eigenvalue_sum = float(np.sum(eigenvalues))
    selected_eigenvalue_sum = float(np.sum(eigenvalues[:projection_dimension]))
    selected_eigenvalue_mass, selected_eigenvalue_mass_status = _defined_ratio_or_nan(
        selected_eigenvalue_sum,
        eigenvalue_sum,
        status_when_undefined="undefined_nonpositive_eigenvalue_sum",
    )
    top_eigenvalue_share, top_eigenvalue_share_status = _defined_ratio_or_nan(
        float(eigenvalues[0]),
        eigenvalue_sum,
        status_when_undefined="undefined_nonpositive_eigenvalue_sum",
    )

    if projection_dimension < eigenvalues.shape[0] and eigenvalues[projection_dimension] > 0.0:
        eigengap = float(eigenvalues[projection_dimension - 1] / eigenvalues[projection_dimension])
        eigengap_status = "defined"
    elif projection_dimension >= eigenvalues.shape[0]:
        eigengap = np.nan
        eigengap_status = "undefined_projection_reaches_spectrum_end"
    else:
        eigengap = np.nan
        eigengap_status = "undefined_next_eigenvalue_nonpositive"

    mp_threshold_rows = int(spectral_context.mp_threshold_rows_by_node[node_id])
    active_spectrum_width = int(eigenvalues.shape[0])
    mp_upper_bound = float(
        (1.0 + np.sqrt(float(active_spectrum_width) / float(mp_threshold_rows))) ** 2
    )
    selected_eigenvalue_over_mp, selected_eigenvalue_over_mp_status = _defined_ratio_or_nan(
        float(eigenvalues[projection_dimension - 1]),
        mp_upper_bound,
        status_when_undefined="undefined_nonpositive_mp_upper_bound",
    )

    return {
        "z_norm_sq": z_norm_sq,
        "selected_projected_norm_sq": float(np.sum(projected_components * projected_components)),
        "selected_subspace_cos2": selected_subspace_cos2,
        "selected_subspace_sin2": selected_subspace_sin2,
        "selected_subspace_tan2": selected_subspace_tan2,
        "selected_subspace_tan2_status": tan2_status,
        "selected_subspace_angle_degrees": selected_angle_degrees,
        "top_component_cos2": float(component_cos2[0]),
        "max_component_cos2": float(np.max(component_cos2)),
        "component_cos2_entropy": _component_entropy(component_cos2),
        "component_cos2_sum": float(np.sum(component_cos2)),
        "parent_test_projection_dimension": int(
            spectral_context.test_projection_dimensions_by_node[node_id]
        ),
        "raw_mp_signal_count": int(spectral_context.raw_mp_signal_counts_by_node[node_id]),
        "effective_independent_rows": int(
            spectral_context.effective_independent_rows_by_node[node_id]
        ),
        "mp_threshold_rows": mp_threshold_rows,
        "active_spectrum_width": active_spectrum_width,
        "eigenvalue_sum": eigenvalue_sum,
        "eigenvalue_effective_rank": effective_rank(eigenvalues),
        "top_eigenvalue_share": top_eigenvalue_share,
        "top_eigenvalue_share_status": top_eigenvalue_share_status,
        "selected_eigenvalue_mass_fraction": selected_eigenvalue_mass,
        "selected_eigenvalue_mass_status": selected_eigenvalue_mass_status,
        "eigengap_at_sibling_projection_dimension": eigengap,
        "eigengap_status": eigengap_status,
        "mp_upper_bound_from_recorded_spectrum_width": mp_upper_bound,
        "selected_eigenvalue_over_mp_upper_bound": selected_eigenvalue_over_mp,
        "selected_eigenvalue_over_mp_status": selected_eigenvalue_over_mp_status,
    }


def _selected_geometry_rows(
    *,
    tree,
    edge_df: pd.DataFrame,
    spectral_context: SpectralContext,
    records: Sequence[SiblingPairRecord],
    case_id: str,
    case_category: str,
    source_family: str,
    feature_representation: str,
    replicate_index: int,
    n_samples: int,
    n_features: int,
    feature_space: FeatureSpace | None,
) -> list[dict[str, object]]:
    node_depths = compute_node_depths(tree)
    rows: list[dict[str, object]] = []
    for record in records:
        if record.is_null_like or record.is_edge_blocked or record.degrees_of_freedom <= 0.0:
            continue

        (
            z_scores,
            left_sample_size,
            right_sample_size,
            branch_length_left,
            branch_length_right,
            branch_length_sum,
        ) = _sibling_whitened_contrast(
            tree=tree,
            record=record,
            feature_space=feature_space,
        )
        projection_dimension = int(record.sibling_projection_dimension)
        geometry = _spectral_geometry(
            parent=record.parent,
            projection_dimension=projection_dimension,
            z_scores=z_scores,
            spectral_context=spectral_context,
        )
        subtree_topology = _subtree_topology_geometry(
            tree=tree,
            parent=record.parent,
            node_depths=node_depths,
        )
        reference_expectation = float(record.reference_scale * record.degrees_of_freedom)
        if reference_expectation <= 0.0:
            raise ValueError(
                "Selected-hierarchy geometry requires positive reference expectation; "
                f"case={case_id!r}, parent={record.parent!r}."
            )
        selected_hierarchy_ratio = float(record.stat / reference_expectation)
        log_selected_hierarchy_ratio, log_selected_hierarchy_ratio_status = (
            _log_selected_hierarchy_ratio(selected_hierarchy_ratio)
        )

        left_edge_raw = _edge_float(edge_df, record.left, "Child_Parent_Divergence_P_Value")
        right_edge_raw = _edge_float(edge_df, record.right, "Child_Parent_Divergence_P_Value")
        left_edge_bh = _edge_float(edge_df, record.left, "Child_Parent_Divergence_P_Value_BH")
        right_edge_bh = _edge_float(edge_df, record.right, "Child_Parent_Divergence_P_Value_BH")
        min_edge_bh = min(left_edge_bh, right_edge_bh)
        max_edge_bh = max(left_edge_bh, right_edge_bh)
        parent_size_fraction = float(record.n_parent / n_samples)
        child_balance = float(min(left_sample_size, right_sample_size) / record.n_parent)
        child_size_ratio = float(
            max(left_sample_size, right_sample_size) / min(left_sample_size, right_sample_size)
        )
        branch_sum_value = float(branch_length_sum) if branch_length_sum is not None else np.nan
        branch_asymmetry = (
            float(abs(branch_length_left - branch_length_right) / branch_length_sum)
            if (
                branch_length_left is not None
                and branch_length_right is not None
                and branch_length_sum is not None
                and branch_length_sum > 0.0
            )
            else np.nan
        )

        row = {
            "case_id": case_id,
            "case_category": case_category,
            "source_family": source_family,
            "feature_representation": feature_representation,
            "replicate_index": int(replicate_index),
            SIMULATION_ID_COLUMN: f"{case_id}:{int(replicate_index)}",
            "parent": record.parent,
            "left_child": record.left,
            "right_child": record.right,
            "feature_family": record.feature_family,
            "n_samples": int(n_samples),
            "feature_dimension": int(n_features),
            "parent_depth": int(node_depths[record.parent]),
            "parent_sample_size": int(record.n_parent),
            "parent_size_fraction": parent_size_fraction,
            "parent_size_bin": _parent_size_bin(parent_size_fraction),
            "left_child_sample_size": left_sample_size,
            "right_child_sample_size": right_sample_size,
            "child_balance": child_balance,
            "child_size_ratio": child_size_ratio,
            "branch_length_left": branch_length_left,
            "branch_length_right": branch_length_right,
            "branch_length_sum": branch_sum_value,
            "branch_length_asymmetry": branch_asymmetry,
            "left_edge_raw_p_value": left_edge_raw,
            "right_edge_raw_p_value": right_edge_raw,
            "min_child_edge_raw_p_value": min(left_edge_raw, right_edge_raw),
            "max_child_edge_raw_p_value": max(left_edge_raw, right_edge_raw),
            "left_edge_bh_p_value": left_edge_bh,
            "right_edge_bh_p_value": right_edge_bh,
            "min_child_edge_bh_p_value": min_edge_bh,
            "max_child_edge_bh_p_value": max_edge_bh,
            "negative_log10_min_child_edge_bh_p_value": _defined_log10_p_value(min_edge_bh),
            "left_edge_tested": _edge_bool(edge_df, record.left, "Child_Parent_Divergence_Tested"),
            "right_edge_tested": _edge_bool(
                edge_df, record.right, "Child_Parent_Divergence_Tested"
            ),
            "left_edge_significant": _edge_bool(
                edge_df, record.left, "Child_Parent_Divergence_Significant"
            ),
            "right_edge_significant": _edge_bool(
                edge_df, record.right, "Child_Parent_Divergence_Significant"
            ),
            "statistic": float(record.stat),
            "reference_scale": float(record.reference_scale),
            "degrees_of_freedom": float(record.degrees_of_freedom),
            "reference_expectation": reference_expectation,
            "selected_hierarchy_ratio": selected_hierarchy_ratio,
            RESPONSE_COLUMN: log_selected_hierarchy_ratio,
            "log_selected_hierarchy_ratio_status": log_selected_hierarchy_ratio_status,
            "raw_p_value": float(record.p_value),
            "sibling_projection_dimension": projection_dimension,
            "study_role": STUDY_ROLE,
        }
        row.update(geometry)
        row.update(subtree_topology)
        rows.append(row)
    return rows


def _run_geometry_sample(
    *,
    data: pd.DataFrame,
    metadata: dict[str, object],
    feature_space: FeatureSpace | None,
    case_id: str,
    replicate_index: int,
) -> GeometryStudySample:
    tree, tree_metric = _build_tree(data, metadata)
    tree.populate_node_divergences(data, feature_space=feature_space)
    edge_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=data,
        feature_space=feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, _non_binary_nodes = collect_sibling_pair_records(
        tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    tested_edges = edge_df["Child_Parent_Divergence_Tested"].astype(bool)
    significant_edges = edge_df["Child_Parent_Divergence_Significant"].astype(bool)
    rows = _selected_geometry_rows(
        tree=tree,
        edge_df=edge_df,
        spectral_context=spectral_context,
        records=records,
        case_id=case_id,
        case_category=str(metadata["category"]),
        source_family=str(metadata["source_family"]),
        feature_representation=str(metadata["feature_representation"]),
        replicate_index=replicate_index,
        n_samples=int(data.shape[0]),
        n_features=int(data.shape[1]),
        feature_space=feature_space,
    )
    return GeometryStudySample(
        rows=tuple(rows),
        candidate_records=int(len(records)),
        selected_records=int(len(rows)),
        tested_edges=int(tested_edges.sum()),
        significant_edges=int(significant_edges.sum()),
        tree_metric=tree_metric,
    )


def _diagnose_case(
    case: dict[str, object],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    inputs = prepare_case_inputs(case, ["tbs"])
    feature_space = inputs.metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Prepared feature_space metadata must be a FeatureSpace.")
    if feature_space is not None and feature_space.family_label == "continuous":
        raise ValueError(
            "Continuous selected-hierarchy geometry requires a validated continuous "
            "null covariance generator before use."
        )

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    candidate_records = 0
    selected_records = 0
    tested_edges = 0
    significant_edges = 0
    tree_metric = ""

    for replicate_index in range(int(n_replicates)):
        simulated = _simulate_null_data(inputs.data, feature_space, rng=rng)
        sample = _run_geometry_sample(
            data=simulated,
            metadata=inputs.metadata,
            feature_space=feature_space,
            case_id=str(inputs.metadata["name"]),
            replicate_index=replicate_index,
        )
        rows.extend(sample.rows)
        candidate_records += sample.candidate_records
        selected_records += sample.selected_records
        tested_edges += sample.tested_edges
        significant_edges += sample.significant_edges
        tree_metric = sample.tree_metric

    case_summary = {
        "case_id": str(inputs.metadata["name"]),
        "case_category": str(inputs.metadata["category"]),
        "feature_family": "bernoulli" if feature_space is None else feature_space.family_label,
        "tree_distance_metric": tree_metric,
        "n_samples": int(inputs.data.shape[0]),
        "feature_dimension": int(inputs.data.shape[1]),
        "n_replicates": int(n_replicates),
        "n_candidate_records": int(candidate_records),
        "n_selected_records": int(selected_records),
        "selected_record_rate": (
            float(selected_records / candidate_records) if candidate_records else 0.0
        ),
        "tested_edges": int(tested_edges),
        "significant_edges": int(significant_edges),
        "edge_rejection_rate": (float(significant_edges / tested_edges) if tested_edges else 0.0),
        "study_role": STUDY_ROLE,
    }
    return pd.DataFrame.from_records(rows), case_summary


def _valid_numeric_pairs(
    table: pd.DataFrame,
    predictor: str,
    response: str,
) -> tuple[np.ndarray, np.ndarray]:
    if predictor not in table.columns:
        raise KeyError(f"Missing predictor column {predictor!r}.")
    if response not in table.columns:
        raise KeyError(f"Missing response column {response!r}.")
    values = table[[predictor, response]].to_numpy(dtype=float)
    finite_mask = np.isfinite(values).all(axis=1)
    return values[finite_mask, 0], values[finite_mask, 1]


def evaluate_covariate_relationships(
    records: pd.DataFrame,
    *,
    covariates: Sequence[tuple[str, str]] = CORRELATION_COVARIATES,
    response: str = RESPONSE_COLUMN,
    min_pairs: int = 8,
) -> pd.DataFrame:
    """Describe univariate covariate relationships with selected-ratio scale."""
    if records.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for block_name, covariate in covariates:
        x_values, y_values = _valid_numeric_pairs(records, covariate, response)
        if x_values.shape[0] < min_pairs:
            rows.append(
                {
                    "covariate_block": block_name,
                    "covariate": covariate,
                    "response": response,
                    "n_pairs": int(x_values.shape[0]),
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                    "relationship_status": "insufficient_finite_pairs",
                    "study_role": STUDY_ROLE,
                }
            )
            continue
        if np.unique(x_values).shape[0] <= 1:
            rows.append(
                {
                    "covariate_block": block_name,
                    "covariate": covariate,
                    "response": response,
                    "n_pairs": int(x_values.shape[0]),
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                    "relationship_status": "constant_covariate",
                    "study_role": STUDY_ROLE,
                }
            )
            continue

        pearson = stats.pearsonr(x_values, y_values)
        spearman = stats.spearmanr(x_values, y_values)
        rows.append(
            {
                "covariate_block": block_name,
                "covariate": covariate,
                "response": response,
                "n_pairs": int(x_values.shape[0]),
                "pearson_r": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
                "relationship_status": "descriptive_unadjusted",
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def _standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=0)
    if np.any(standard_deviations <= 0.0):
        zero_columns = np.flatnonzero(standard_deviations <= 0.0).tolist()
        raise ValueError(f"Cannot standardize constant predictor column(s): {zero_columns}.")
    return (matrix - means) / standard_deviations, means, standard_deviations


def _ols_diagnostic_fit(
    table: pd.DataFrame,
    *,
    predictors: Sequence[str],
    response: str,
    min_rows_per_predictor: int,
) -> dict[str, object]:
    required_columns = list(predictors) + [response]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise KeyError(f"Missing model column(s): {missing_columns!r}.")

    model_table = table[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
    predictor_count = len(predictors)
    minimum_rows = max(predictor_count + 2, predictor_count * min_rows_per_predictor)
    if model_table.shape[0] < minimum_rows:
        return {
            "model_table": model_table,
            "n_rows": int(model_table.shape[0]),
            "predictor_count": int(predictor_count),
            "r_squared": np.nan,
            "adjusted_r_squared": np.nan,
            "matrix_rank": np.nan,
            "condition_number": np.nan,
            "coefficients": np.array([], dtype=float),
            "fitted": np.array([], dtype=float),
            "model_status": "insufficient_rows_for_model",
        }

    x_raw = model_table[list(predictors)].to_numpy(dtype=float)
    y = model_table[response].to_numpy(dtype=float)
    constant_columns = [
        predictors[column_index]
        for column_index in range(x_raw.shape[1])
        if np.unique(x_raw[:, column_index]).shape[0] <= 1
    ]
    if constant_columns:
        return {
            "model_table": model_table,
            "n_rows": int(model_table.shape[0]),
            "predictor_count": int(predictor_count),
            "r_squared": np.nan,
            "adjusted_r_squared": np.nan,
            "matrix_rank": np.nan,
            "condition_number": np.nan,
            "coefficients": np.array([], dtype=float),
            "fitted": np.array([], dtype=float),
            "model_status": "constant_predictor_columns:" + ",".join(constant_columns),
        }

    x_standardized, _means, _stds = _standardize_matrix(x_raw)
    design = np.column_stack([np.ones(x_standardized.shape[0]), x_standardized])
    matrix_rank = int(np.linalg.matrix_rank(design))
    if matrix_rank < design.shape[1]:
        return {
            "model_table": model_table,
            "n_rows": int(model_table.shape[0]),
            "predictor_count": int(predictor_count),
            "r_squared": np.nan,
            "adjusted_r_squared": np.nan,
            "matrix_rank": matrix_rank,
            "condition_number": np.nan,
            "coefficients": np.array([], dtype=float),
            "fitted": np.array([], dtype=float),
            "model_status": "rank_deficient_design",
        }

    coefficients, _residuals, _rank, singular_values = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    fitted = design @ coefficients
    residual_sum_sq = float(np.sum((y - fitted) ** 2))
    total_sum_sq = float(np.sum((y - np.mean(y)) ** 2))
    r_squared, r2_status = _defined_ratio_or_nan(
        total_sum_sq - residual_sum_sq,
        total_sum_sq,
        status_when_undefined="undefined_zero_response_variance",
    )
    adjusted_r_squared = (
        float(
            1.0
            - (1.0 - r_squared)
            * (model_table.shape[0] - 1)
            / (model_table.shape[0] - predictor_count - 1)
        )
        if r2_status == "defined"
        else np.nan
    )
    return {
        "model_table": model_table,
        "n_rows": int(model_table.shape[0]),
        "predictor_count": int(predictor_count),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "matrix_rank": matrix_rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "coefficients": coefficients,
        "fitted": fitted,
        "model_status": "descriptive_in_sample",
    }


def _model_table(
    table: pd.DataFrame,
    *,
    predictors: Sequence[str],
    response: str,
) -> pd.DataFrame:
    required_columns = list(predictors) + [response]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise KeyError(f"Missing model column(s): {missing_columns!r}.")
    return table[required_columns].replace([np.inf, -np.inf], np.nan).dropna()


def _fit_standardized_linear_model(
    train_table: pd.DataFrame,
    *,
    predictors: Sequence[str],
    response: str,
    min_rows_per_predictor: int,
) -> dict[str, object]:
    predictor_count = len(predictors)
    minimum_rows = max(predictor_count + 2, predictor_count * min_rows_per_predictor)
    if train_table.shape[0] < minimum_rows:
        return {
            "model_status": "insufficient_rows_for_holdout_model",
            "n_rows": int(train_table.shape[0]),
        }

    x_raw = train_table[list(predictors)].to_numpy(dtype=float)
    y = train_table[response].to_numpy(dtype=float)
    constant_columns = [
        predictors[column_index]
        for column_index in range(x_raw.shape[1])
        if np.unique(x_raw[:, column_index]).shape[0] <= 1
    ]
    if constant_columns:
        return {
            "model_status": "constant_predictor_columns:" + ",".join(constant_columns),
            "n_rows": int(train_table.shape[0]),
        }

    x_standardized, means, stds = _standardize_matrix(x_raw)
    design = np.column_stack([np.ones(x_standardized.shape[0]), x_standardized])
    matrix_rank = int(np.linalg.matrix_rank(design))
    if matrix_rank < design.shape[1]:
        return {
            "model_status": "rank_deficient_design",
            "n_rows": int(train_table.shape[0]),
            "matrix_rank": matrix_rank,
        }

    coefficients, _residuals, _rank, singular_values = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    return {
        "model_status": "descriptive_holdout_fit",
        "n_rows": int(train_table.shape[0]),
        "matrix_rank": matrix_rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "coefficients": coefficients,
        "means": means,
        "standard_deviations": stds,
        "train_response_mean": float(np.mean(y)),
    }


def _predict_standardized_linear_model(
    model: dict[str, object],
    test_table: pd.DataFrame,
    *,
    predictors: Sequence[str],
) -> np.ndarray:
    coefficients = np.asarray(model["coefficients"], dtype=float)
    means = np.asarray(model["means"], dtype=float)
    standard_deviations = np.asarray(model["standard_deviations"], dtype=float)
    x_raw = test_table[list(predictors)].to_numpy(dtype=float)
    x_standardized = (x_raw - means) / standard_deviations
    design = np.column_stack([np.ones(x_standardized.shape[0]), x_standardized])
    return design @ coefficients


def _holdout_summary(
    *,
    equation_id: str,
    equation: str,
    predictors: Sequence[str],
    response: str,
    split_strategy: str,
    tail_quantile: float,
    predictions: list[np.ndarray],
    responses: list[np.ndarray],
    tail_labels: list[np.ndarray],
    train_means: list[float],
    n_train_rows: int,
    n_test_rows: int,
    n_folds: int,
    status: str,
    failure_reasons: list[str],
) -> dict[str, object]:
    if not predictions:
        return {
            "equation_id": equation_id,
            "equation": equation,
            "predictors": ",".join(predictors),
            "response": response,
            "split_strategy": split_strategy,
            "tail_quantile": float(tail_quantile),
            "n_folds": int(n_folds),
            "n_train_rows": int(n_train_rows),
            "n_test_rows": int(n_test_rows),
            "holdout_log_ratio_r_squared": np.nan,
            "holdout_mean_absolute_log_error": np.nan,
            "holdout_median_absolute_log_error": np.nan,
            "holdout_tail_event_rate": np.nan,
            "holdout_tail_auc_from_linear_score": np.nan,
            "holdout_tail_score_mean": np.nan,
            "holdout_non_tail_score_mean": np.nan,
            "holdout_tail_score_separation": np.nan,
            "equation_status": status,
            "failure_reasons": ";".join(failure_reasons),
            "study_role": STUDY_ROLE,
        }

    predicted = np.concatenate(predictions)
    observed = np.concatenate(responses)
    labels = np.concatenate(tail_labels)
    baseline = np.concatenate(
        [
            np.full(response_values.shape[0], train_mean, dtype=float)
            for response_values, train_mean in zip(responses, train_means, strict=True)
        ]
    )
    residual_sum_sq = float(np.sum((observed - predicted) ** 2))
    baseline_sum_sq = float(np.sum((observed - baseline) ** 2))
    holdout_r2, _r2_status = _defined_ratio_or_nan(
        baseline_sum_sq - residual_sum_sq,
        baseline_sum_sq,
        status_when_undefined="undefined_zero_holdout_response_variance",
    )
    absolute_error = np.abs(observed - predicted)
    tail_score_mean = float(np.mean(predicted[labels])) if np.any(labels) else np.nan
    non_tail_score_mean = float(np.mean(predicted[~labels])) if np.any(~labels) else np.nan
    return {
        "equation_id": equation_id,
        "equation": equation,
        "predictors": ",".join(predictors),
        "response": response,
        "split_strategy": split_strategy,
        "tail_quantile": float(tail_quantile),
        "n_folds": int(n_folds),
        "n_train_rows": int(n_train_rows),
        "n_test_rows": int(n_test_rows),
        "holdout_log_ratio_r_squared": holdout_r2,
        "holdout_mean_absolute_log_error": float(np.mean(absolute_error)),
        "holdout_median_absolute_log_error": float(np.median(absolute_error)),
        "holdout_tail_event_rate": float(np.mean(labels)),
        "holdout_tail_auc_from_linear_score": _binary_auc_score(predicted, labels),
        "holdout_tail_score_mean": tail_score_mean,
        "holdout_non_tail_score_mean": non_tail_score_mean,
        "holdout_tail_score_separation": float(tail_score_mean - non_tail_score_mean),
        "equation_status": status,
        "failure_reasons": ";".join(failure_reasons),
        "study_role": STUDY_ROLE,
    }


def _replicate_fold_ids(table: pd.DataFrame, *, n_folds: int) -> pd.Series:
    if "replicate_index" not in table.columns:
        raise KeyError("Replicate holdout requires replicate_index.")
    replicate_values = pd.to_numeric(table["replicate_index"], errors="raise").astype(int)
    return pd.Series(replicate_values % int(n_folds), index=table.index, dtype=int)


def _case_fold_ids(table: pd.DataFrame) -> pd.Series:
    if "case_id" not in table.columns:
        raise KeyError("Case holdout requires case_id.")
    case_values = pd.Series(table["case_id"], index=table.index).astype(str)
    unique_cases = tuple(sorted(case_values.unique()))
    case_to_fold = {case_id: fold_index for fold_index, case_id in enumerate(unique_cases)}
    return case_values.map(case_to_fold).astype(int)


def _source_family_fold_ids(table: pd.DataFrame) -> pd.Series:
    if "source_family" not in table.columns:
        raise KeyError("Family holdout requires source_family.")
    family_values = pd.Series(table["source_family"], index=table.index).astype(str)
    unique_families = tuple(sorted(family_values.unique()))
    family_to_fold = {family_id: fold_index for fold_index, family_id in enumerate(unique_families)}
    return family_values.map(family_to_fold).astype(int)


def _evaluate_holdout_strategy(
    table: pd.DataFrame,
    *,
    candidate_equations: Sequence[tuple[str, str, tuple[str, ...]]],
    response: str,
    tail_quantile: float,
    min_rows_per_predictor: int,
    split_strategy: str,
    fold_ids: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_values = tuple(sorted(int(value) for value in fold_ids.unique()))
    for equation_id, equation, predictors in candidate_equations:
        predictions: list[np.ndarray] = []
        responses: list[np.ndarray] = []
        tail_labels: list[np.ndarray] = []
        train_means: list[float] = []
        failure_reasons: list[str] = []
        n_train_rows = 0
        n_test_rows = 0

        model_table = _model_table(table, predictors=predictors, response=response)
        model_fold_ids = fold_ids.loc[model_table.index]
        for fold_id in fold_values:
            train_table = model_table.loc[model_fold_ids != fold_id]
            test_table = model_table.loc[model_fold_ids == fold_id]
            n_train_rows += int(train_table.shape[0])
            n_test_rows += int(test_table.shape[0])
            if test_table.empty:
                failure_reasons.append(f"fold_{fold_id}:empty_test")
                continue
            model = _fit_standardized_linear_model(
                train_table,
                predictors=predictors,
                response=response,
                min_rows_per_predictor=min_rows_per_predictor,
            )
            if model["model_status"] != "descriptive_holdout_fit":
                failure_reasons.append(f"fold_{fold_id}:{model['model_status']}")
                continue
            train_response = train_table[response].to_numpy(dtype=float)
            tail_threshold = float(np.quantile(train_response, tail_quantile))
            y_test = test_table[response].to_numpy(dtype=float)
            y_pred = _predict_standardized_linear_model(
                model,
                test_table,
                predictors=predictors,
            )
            predictions.append(y_pred)
            responses.append(y_test)
            tail_labels.append(y_test >= tail_threshold)
            train_means.append(float(model["train_response_mean"]))

        status = (
            f"descriptive_holdout_{split_strategy}"
            if predictions
            else f"no_valid_holdout_folds_{split_strategy}"
        )
        rows.append(
            _holdout_summary(
                equation_id=equation_id,
                equation=equation,
                predictors=predictors,
                response=response,
                split_strategy=split_strategy,
                tail_quantile=tail_quantile,
                predictions=predictions,
                responses=responses,
                tail_labels=tail_labels,
                train_means=train_means,
                n_train_rows=n_train_rows,
                n_test_rows=n_test_rows,
                n_folds=len(fold_values),
                status=status,
                failure_reasons=failure_reasons,
            )
        )
    return pd.DataFrame.from_records(rows)


def evaluate_candidate_equation_holdout(
    records: pd.DataFrame,
    *,
    candidate_equations: Sequence[tuple[str, str, tuple[str, ...]]] = (CANDIDATE_EQUATIONS),
    response: str = RESPONSE_COLUMN,
    tail_quantile: float = 0.9,
    n_replicate_folds: int = 5,
    min_rows_per_predictor: int = 5,
) -> pd.DataFrame:
    """Score candidate equations on held-out selected-hierarchy records.

    This remains descriptive: it estimates whether geometry equations transfer
    across regenerated hierarchy replicates or across case families. It does not
    define a production calibration law.
    """
    if records.empty:
        return pd.DataFrame()
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must lie in (0, 1).")
    if n_replicate_folds < 2:
        raise ValueError("n_replicate_folds must be at least 2.")

    table = _candidate_equation_table(records)
    replicate_holdout = _evaluate_holdout_strategy(
        table,
        candidate_equations=candidate_equations,
        response=response,
        tail_quantile=tail_quantile,
        min_rows_per_predictor=min_rows_per_predictor,
        split_strategy="replicate_modulo",
        fold_ids=_replicate_fold_ids(table, n_folds=n_replicate_folds),
    )
    case_holdout = _evaluate_holdout_strategy(
        table,
        candidate_equations=candidate_equations,
        response=response,
        tail_quantile=tail_quantile,
        min_rows_per_predictor=min_rows_per_predictor,
        split_strategy="leave_one_case_out",
        fold_ids=_case_fold_ids(table),
    )
    source_family_holdout = _evaluate_holdout_strategy(
        table,
        candidate_equations=candidate_equations,
        response=response,
        tail_quantile=tail_quantile,
        min_rows_per_predictor=min_rows_per_predictor,
        split_strategy="leave_one_source_family_out",
        fold_ids=_source_family_fold_ids(table),
    )
    return pd.concat(
        [replicate_holdout, case_holdout, source_family_holdout],
        ignore_index=True,
    )


def evaluate_covariate_block_models(
    records: pd.DataFrame,
    *,
    model_blocks: Sequence[tuple[str, tuple[str, ...]]] = MODEL_BLOCKS,
    response: str = RESPONSE_COLUMN,
    min_rows_per_predictor: int = 5,
) -> pd.DataFrame:
    """Fit descriptive linear models for covariate blocks.

    These are in-sample diagnostic summaries, not production calibration
    models. Rank-deficient or undersupported blocks are reported explicitly.
    """
    if records.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for block_name, predictors in model_blocks:
        fit = _ols_diagnostic_fit(
            records,
            predictors=predictors,
            response=response,
            min_rows_per_predictor=min_rows_per_predictor,
        )
        rows.append(
            {
                "covariate_block": block_name,
                "response": response,
                "predictors": ",".join(predictors),
                "n_rows": fit["n_rows"],
                "predictor_count": fit["predictor_count"],
                "r_squared": fit["r_squared"],
                "adjusted_r_squared": fit["adjusted_r_squared"],
                "matrix_rank": fit["matrix_rank"],
                "condition_number": fit["condition_number"],
                "model_status": fit["model_status"],
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def _candidate_equation_table(records: pd.DataFrame) -> pd.DataFrame:
    table = records.copy()
    table["edge_action"] = table["negative_log10_min_child_edge_bh_p_value"]
    table["feature_parent_aspect_ratio"] = table["feature_dimension"].astype(float) / table[
        "parent_sample_size"
    ].astype(float)
    table["sampling_variance_scale"] = 1.0 / table["left_child_sample_size"].astype(
        float
    ) + 1.0 / table["right_child_sample_size"].astype(float)
    table["log_feature_parent_aspect_ratio"] = _positive_log_array(
        table["feature_parent_aspect_ratio"],
        value_name="feature_parent_aspect_ratio",
    )
    table["log_sampling_variance_scale"] = _positive_log_array(
        table["sampling_variance_scale"],
        value_name="sampling_variance_scale",
    )
    table["log_selected_eigenvalue_over_mp_upper_bound"] = _positive_log_array(
        table["selected_eigenvalue_over_mp_upper_bound"],
        value_name="selected_eigenvalue_over_mp_upper_bound",
    )
    return table


def _edge_action_bin(edge_action: float) -> str:
    if not np.isfinite(edge_action) or edge_action < 0.0:
        raise ValueError(f"edge_action must be finite and non-negative; got {edge_action!r}.")
    for lower, upper, label in zip(
        EDGE_ACTION_BINS,
        EDGE_ACTION_BINS[1:],
        EDGE_ACTION_BIN_LABELS,
    ):
        if lower <= edge_action < upper:
            return label
    raise ValueError(f"edge_action did not match a bin: {edge_action!r}.")


def _selected_ratio_tail_law_table(records: pd.DataFrame) -> pd.DataFrame:
    table = _candidate_equation_table(records)
    if SIMULATION_ID_COLUMN not in table.columns:
        raise KeyError(
            "Selected-ratio tail-law evaluation requires explicit independent "
            f"simulation ids in {SIMULATION_ID_COLUMN!r}."
        )
    table["edge_action_bin"] = [_edge_action_bin(float(value)) for value in table["edge_action"]]
    return table


def _simulation_fold_ids(group: pd.DataFrame, *, n_folds: int) -> pd.Series:
    simulation_ids = pd.Series(group[SIMULATION_ID_COLUMN], index=group.index).astype(str)
    ordered_ids = tuple(sorted(simulation_ids.unique()))
    fold_by_simulation = {
        simulation_id: index % int(n_folds) for index, simulation_id in enumerate(ordered_ids)
    }
    return simulation_ids.map(fold_by_simulation).astype(int)


def _fold_tail_law_evaluation(
    group: pd.DataFrame,
    *,
    alpha: float,
    n_folds: int,
    min_train_simulations: int,
    min_train_records: int,
) -> dict[str, object]:
    predictions: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    thresholds: list[float] = []
    failures: list[str] = []
    n_train_rows = 0
    n_test_rows = 0
    used_folds = 0
    fold_ids = _simulation_fold_ids(group, n_folds=n_folds)

    for fold_id in sorted(int(value) for value in fold_ids.unique()):
        train = group.loc[fold_ids != fold_id]
        test = group.loc[fold_ids == fold_id]
        n_train_rows += int(train.shape[0])
        n_test_rows += int(test.shape[0])
        if test.empty:
            failures.append(f"fold_{fold_id}:empty_test")
            continue
        train_simulations = int(train[SIMULATION_ID_COLUMN].nunique())
        if train_simulations < min_train_simulations:
            failures.append(f"fold_{fold_id}:insufficient_train_simulations")
            continue
        if train.shape[0] < min_train_records:
            failures.append(f"fold_{fold_id}:insufficient_train_records")
            continue

        train_ratios = train["selected_hierarchy_ratio"].to_numpy(dtype=float)
        threshold = float(np.quantile(train_ratios, 1.0 - alpha))
        test_ratios = test["selected_hierarchy_ratio"].to_numpy(dtype=float)
        predictions.append(np.full(test_ratios.shape[0], threshold, dtype=float))
        observed.append(test_ratios)
        thresholds.append(threshold)
        used_folds += 1

    if not predictions:
        return {
            "n_train_rows_across_folds": int(n_train_rows),
            "n_test_rows_across_folds": int(n_test_rows),
            "n_used_folds": 0,
            "tail_threshold_mean": np.nan,
            "tail_threshold_median": np.nan,
            "heldout_exceedance_rate": np.nan,
            "heldout_exceedance_absolute_error": np.nan,
            "heldout_exceedance_standard_error": np.nan,
            "tail_law_status": "no_valid_tail_law_folds",
            "tail_law_failure_reasons": ";".join(failures),
        }

    predicted_thresholds = np.concatenate(predictions)
    observed_ratios = np.concatenate(observed)
    exceedances = observed_ratios > predicted_thresholds
    exceedance_rate = float(np.mean(exceedances))
    exceedance_se = float(np.sqrt(exceedance_rate * (1.0 - exceedance_rate) / exceedances.shape[0]))
    return {
        "n_train_rows_across_folds": int(n_train_rows),
        "n_test_rows_across_folds": int(n_test_rows),
        "n_used_folds": int(used_folds),
        "tail_threshold_mean": float(np.mean(thresholds)),
        "tail_threshold_median": float(np.median(thresholds)),
        "heldout_exceedance_rate": exceedance_rate,
        "heldout_exceedance_absolute_error": float(abs(exceedance_rate - alpha)),
        "heldout_exceedance_standard_error": exceedance_se,
        "tail_law_status": "descriptive_holdout_tail_law",
        "tail_law_failure_reasons": ";".join(failures),
    }


def _tail_law_admissibility_failures(
    *,
    n_matching_simulations: int,
    n_records: int,
    heldout_exceedance_se: float,
    required_min_matching_simulations: int,
    required_min_matched_records: int,
    max_exceedance_standard_error: float,
) -> list[str]:
    failures: list[str] = []
    if n_matching_simulations < required_min_matching_simulations:
        failures.append("matching_simulations_below_tail_resolution_contract")
    if n_records < required_min_matched_records:
        failures.append("matched_records_below_tail_resolution_contract")
    if (
        not np.isfinite(heldout_exceedance_se)
        or heldout_exceedance_se > max_exceedance_standard_error
    ):
        failures.append("heldout_exceedance_se_above_contract")
    return failures


def evaluate_selected_ratio_tail_law(
    records: pd.DataFrame,
    *,
    alpha: float = float(DEFAULT_SIBLING_ALPHA),
    n_folds: int = 5,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_exceedance_standard_error: float = 0.002,
    context_columns: Sequence[str] = TAIL_LAW_CONTEXT_COLUMNS,
) -> pd.DataFrame:
    r"""Evaluate selected-ratio tail quantiles inside explicit contexts.

    This diagnostic estimates \(P(R_u > r \mid M_u)\) with held-out replicate
    folds. It reports support and precision; it does not create an external
    calibration fallback.
    """
    if records.empty:
        return pd.DataFrame()
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if min_train_simulations <= 0 or min_train_records <= 0:
        raise ValueError("minimum training support values must be positive.")
    if required_min_matching_simulations <= 0 or required_min_matched_records <= 0:
        raise ValueError("required production support values must be positive.")
    if max_exceedance_standard_error <= 0.0:
        raise ValueError("max_exceedance_standard_error must be positive.")

    table = _selected_ratio_tail_law_table(records)
    missing_columns = [column for column in context_columns if column not in table.columns]
    if missing_columns:
        raise KeyError(f"Missing tail-law context column(s): {missing_columns!r}.")

    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(list(context_columns), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        ratios = group["selected_hierarchy_ratio"].to_numpy(dtype=float)
        n_matching_simulations = int(group[SIMULATION_ID_COLUMN].nunique())
        n_records = int(group.shape[0])
        fold_summary = _fold_tail_law_evaluation(
            group,
            alpha=alpha,
            n_folds=n_folds,
            min_train_simulations=min_train_simulations,
            min_train_records=min_train_records,
        )
        admissibility_failures = _tail_law_admissibility_failures(
            n_matching_simulations=n_matching_simulations,
            n_records=n_records,
            heldout_exceedance_se=float(fold_summary["heldout_exceedance_standard_error"]),
            required_min_matching_simulations=required_min_matching_simulations,
            required_min_matched_records=required_min_matched_records,
            max_exceedance_standard_error=max_exceedance_standard_error,
        )
        row = {column: value for column, value in zip(context_columns, group_values)}
        row.update(
            {
                "tail_law_role": TAIL_LAW_ROLE,
                "alpha": float(alpha),
                "n_records": n_records,
                "n_matching_simulations": n_matching_simulations,
                "record_tail_resolution": float(1.0 / (n_records + 1)),
                "matching_simulation_tail_resolution": float(1.0 / (n_matching_simulations + 1)),
                "required_min_matching_simulations": int(required_min_matching_simulations),
                "required_min_matched_records": int(required_min_matched_records),
                "max_exceedance_standard_error": float(max_exceedance_standard_error),
                "selected_ratio_mean": float(np.mean(ratios)),
                "selected_ratio_median": float(np.quantile(ratios, 0.5)),
                "selected_ratio_trainless_q90": float(np.quantile(ratios, 0.9)),
                "selected_ratio_trainless_q95": float(np.quantile(ratios, 0.95)),
                "selected_ratio_trainless_q99": float(np.quantile(ratios, 0.99)),
                "production_tail_law_admissible": not admissibility_failures,
                "tail_law_admissibility_failure_reasons": ";".join(admissibility_failures),
            }
        )
        row.update(fold_summary)
        rows.append(row)

    return pd.DataFrame.from_records(rows).sort_values(list(context_columns)).reset_index(drop=True)


def _binary_auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    finite_mask = np.isfinite(scores) & np.isfinite(labels)
    finite_scores = scores[finite_mask]
    finite_labels = labels[finite_mask].astype(bool)
    positive_count = int(np.count_nonzero(finite_labels))
    negative_count = int(finite_labels.shape[0] - positive_count)
    if positive_count == 0 or negative_count == 0:
        return np.nan
    ranks = stats.rankdata(finite_scores, method="average")
    positive_rank_sum = float(np.sum(ranks[finite_labels]))
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def evaluate_candidate_equations(
    records: pd.DataFrame,
    *,
    candidate_equations: Sequence[tuple[str, str, tuple[str, ...]]] = (CANDIDATE_EQUATIONS),
    response: str = RESPONSE_COLUMN,
    tail_quantile: float = 0.9,
    min_rows_per_predictor: int = 5,
) -> pd.DataFrame:
    """Score interpretable candidate equations for selected-ratio behavior.

    The score is descriptive and in-sample. It is intended to decide which
    equation families deserve a larger validation run, not to define a
    production external calibration model.
    """
    if records.empty:
        return pd.DataFrame()
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must lie in (0, 1).")

    table = _candidate_equation_table(records)
    finite_response = table[response].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_response.empty:
        raise ValueError("Candidate equation scoring requires finite log-ratio values.")
    tail_threshold = float(np.quantile(finite_response.to_numpy(dtype=float), tail_quantile))

    rows: list[dict[str, object]] = []
    for equation_id, equation, predictors in candidate_equations:
        fit = _ols_diagnostic_fit(
            table,
            predictors=predictors,
            response=response,
            min_rows_per_predictor=min_rows_per_predictor,
        )
        model_table = fit["model_table"]
        if fit["model_status"] != "descriptive_in_sample":
            rows.append(
                {
                    "equation_id": equation_id,
                    "equation": equation,
                    "predictors": ",".join(predictors),
                    "response": response,
                    "n_rows": fit["n_rows"],
                    "predictor_count": fit["predictor_count"],
                    "mean_log_ratio_r_squared": fit["r_squared"],
                    "mean_log_ratio_adjusted_r_squared": fit["adjusted_r_squared"],
                    "tail_quantile": float(tail_quantile),
                    "tail_threshold_log_ratio": tail_threshold,
                    "tail_event_rate": np.nan,
                    "tail_auc_from_linear_score": np.nan,
                    "tail_score_mean": np.nan,
                    "non_tail_score_mean": np.nan,
                    "tail_score_separation": np.nan,
                    "matrix_rank": fit["matrix_rank"],
                    "condition_number": fit["condition_number"],
                    "equation_status": fit["model_status"],
                    "study_role": STUDY_ROLE,
                }
            )
            continue

        response_values = model_table[response].to_numpy(dtype=float)
        tail_labels = response_values >= tail_threshold
        scores = np.asarray(fit["fitted"], dtype=float)
        tail_score_mean = float(np.mean(scores[tail_labels]))
        non_tail_score_mean = float(np.mean(scores[~tail_labels]))
        rows.append(
            {
                "equation_id": equation_id,
                "equation": equation,
                "predictors": ",".join(predictors),
                "response": response,
                "n_rows": fit["n_rows"],
                "predictor_count": fit["predictor_count"],
                "mean_log_ratio_r_squared": fit["r_squared"],
                "mean_log_ratio_adjusted_r_squared": fit["adjusted_r_squared"],
                "tail_quantile": float(tail_quantile),
                "tail_threshold_log_ratio": tail_threshold,
                "tail_event_rate": float(np.mean(tail_labels)),
                "tail_auc_from_linear_score": _binary_auc_score(scores, tail_labels),
                "tail_score_mean": tail_score_mean,
                "non_tail_score_mean": non_tail_score_mean,
                "tail_score_separation": float(tail_score_mean - non_tail_score_mean),
                "matrix_rank": fit["matrix_rank"],
                "condition_number": fit["condition_number"],
                "equation_status": "descriptive_in_sample",
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_selected_geometry_by_case(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-ratio and geometry scale by benchmark case."""
    if records.empty:
        return pd.DataFrame()
    if SIMULATION_ID_COLUMN not in records.columns:
        raise KeyError(
            f"Geometry summaries require explicit independent simulation ids in "
            f"{SIMULATION_ID_COLUMN!r}."
        )
    rows: list[dict[str, object]] = []
    for case_id, group in records.groupby("case_id", dropna=False):
        ratios = group["selected_hierarchy_ratio"].to_numpy(dtype=float)
        cos2 = group["selected_subspace_cos2"].to_numpy(dtype=float)
        eigen_mass = group["selected_eigenvalue_mass_fraction"].to_numpy(dtype=float)
        rows.append(
            {
                "case_id": case_id,
                "n_records": int(group.shape[0]),
                "n_matching_simulations": int(group[SIMULATION_ID_COLUMN].nunique()),
                "selected_hierarchy_ratio_mean": float(np.mean(ratios)),
                "selected_hierarchy_ratio_median": float(np.quantile(ratios, 0.5)),
                "selected_hierarchy_ratio_q95": float(np.quantile(ratios, 0.95)),
                "selected_hierarchy_ratio_q99": float(np.quantile(ratios, 0.99)),
                "selected_subspace_cos2_mean": float(np.mean(cos2)),
                "selected_subspace_cos2_q95": float(np.quantile(cos2, 0.95)),
                "selected_eigenvalue_mass_fraction_mean": float(np.mean(eigen_mass)),
                "selected_eigenvalue_mass_fraction_q95": float(np.quantile(eigen_mass, 0.95)),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("case_id").reset_index(drop=True)


def run_selected_hierarchy_geometry_covariate_study(
    *,
    case_names: list[str],
    output_dir: Path,
    n_replicates: int,
    seed: int,
    write_selected_records: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the selected-hierarchy geometry covariate diagnostic."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()

    record_tables: list[pd.DataFrame] = []
    case_summaries: list[dict[str, object]] = []
    for index, case in enumerate(_selected_cases(case_names), start=1):
        print(f"[{index}/{len(case_names)}] {case['name']}", flush=True)
        try:
            records, case_summary = _diagnose_case(
                case,
                n_replicates=int(n_replicates),
                seed=int(seed) + index * 1_000_000,
            )
            case_summary["status"] = "ok"
            case_summary["skip_reason"] = ""
            record_tables.append(records)
        except Exception as exc:
            case_summary = {
                "case_id": str(case["name"]),
                "case_category": str(case["category"]),
                "n_replicates": int(n_replicates),
                "status": "skip",
                "skip_reason": str(exc),
                "study_role": STUDY_ROLE,
            }
        case_summaries.append(case_summary)

    selected_records = (
        pd.concat(record_tables, ignore_index=True) if record_tables else pd.DataFrame()
    )
    case_summary = pd.DataFrame.from_records(case_summaries)
    geometry_summary_by_case = summarize_selected_geometry_by_case(selected_records)
    covariate_relationships = evaluate_covariate_relationships(selected_records)
    covariate_block_models = evaluate_covariate_block_models(selected_records)
    candidate_equations = evaluate_candidate_equations(selected_records)
    candidate_equation_holdout = evaluate_candidate_equation_holdout(selected_records)
    selected_ratio_tail_law = evaluate_selected_ratio_tail_law(selected_records)

    outputs = {
        "case_summary": case_summary,
        "geometry_summary_by_case": geometry_summary_by_case,
        "covariate_relationships": covariate_relationships,
        "covariate_block_models": covariate_block_models,
        "candidate_equations": candidate_equations,
        "candidate_equation_holdout": candidate_equation_holdout,
        "selected_ratio_tail_law": selected_ratio_tail_law,
    }
    if write_selected_records:
        outputs["selected_geometry_records"] = selected_records

    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "diagnostic": "selected_hierarchy_geometry_covariates",
        "study_role": STUDY_ROLE,
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "case_names": case_names,
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "write_selected_records": bool(write_selected_records),
        "response_column": RESPONSE_COLUMN,
        "covariate_blocks": {
            block_name: list(predictors) for block_name, predictors in MODEL_BLOCKS
        },
        "candidate_equations": {
            equation_id: {
                "equation": equation,
                "predictors": list(predictors),
            }
            for equation_id, equation, predictors in CANDIDATE_EQUATIONS
        },
        "selected_ratio_tail_law": {
            "role": TAIL_LAW_ROLE,
            "context_columns": list(TAIL_LAW_CONTEXT_COLUMNS),
            "edge_action_bins": [
                {
                    "label": label,
                    "lower": float(lower),
                    "upper": None if np.isinf(upper) else float(upper),
                }
                for lower, upper, label in zip(
                    EDGE_ACTION_BINS,
                    EDGE_ACTION_BINS[1:],
                    EDGE_ACTION_BIN_LABELS,
                )
            ],
            "alpha": float(DEFAULT_SIBLING_ALPHA),
            "independent_simulation_id_column": SIMULATION_ID_COLUMN,
            "production_min_matching_simulations": 499,
            "production_min_matched_records": 499,
            "production_max_exceedance_standard_error": 0.002,
        },
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "note": (
            "Diagnostic-only selected-hierarchy geometry study. Relationship "
            "tables are descriptive and unadjusted; holdout tables are "
            "descriptive transfer checks, not production calibration. The "
            "selected-ratio tail-law table reports context support and held-out "
            "tail exceedance only. These outputs do not define external "
            "calibration borrowing, scalar inflation, or production fallback."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record selected-hierarchy geometry covariates for external-null "
            "research. This is not a production calibration path."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--write-selected-records",
        action="store_true",
        help="Write the full row-level selected geometry table.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"selected_hierarchy_geometry_covariates_{format_timestamp_utc()}"
        )
    outputs = run_selected_hierarchy_geometry_covariate_study(
        case_names=_parse_csv_list(str(args.case_names)),
        output_dir=output_dir,
        n_replicates=int(args.n_replicates),
        seed=int(args.seed),
        write_selected_records=bool(args.write_selected_records),
    )
    print(outputs["case_summary"].to_string(index=False))
    print(outputs["geometry_summary_by_case"].to_string(index=False))
    print(f"Wrote selected-hierarchy geometry outputs to {output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "CORRELATION_COVARIATES",
    "MODEL_BLOCKS",
    "STUDY_ROLE",
    "evaluate_candidate_equation_holdout",
    "evaluate_candidate_equations",
    "evaluate_covariate_block_models",
    "evaluate_covariate_relationships",
    "evaluate_selected_ratio_tail_law",
    "run_selected_hierarchy_geometry_covariate_study",
    "summarize_selected_geometry_by_case",
]
