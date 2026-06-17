"""MP spectral-flow diagnostics for selected-neighborhood evidence.

This module tests whether neighboring internal nodes carry compatible
Marchenko-Pastur-supported eigenspaces. It is diagnostic-only: it does not
change traversal, gate calibration, or production p-values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.selected_family_traversal_panel import (
    _output_data_role,
)
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

SCHEMA_VERSION = "selected_neighborhood_spectral_flow/v1"
STUDY_ROLE = "diagnostic_selected_neighborhood_spectral_flow_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected_neighborhood_spectral_flow"

DEFAULT_CASE_NAMES = ("overlap_part_4c_small",)
DEFAULT_DATA_ROLES = ("null", "signal")
DEFAULT_METHOD_ID = "fixed_coordinate_global_passthrough_refined_v1"
DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE = 0.05
DEFAULT_UNMATCHED_MODE_PENALTY = 1.0

MODE_DISTANCE_WEIGHTS = {
    "projector": 1.0,
    "log_eigenvalue": 1.0,
    "multiplicity": 0.5,
    "polynomial": 0.5,
}

NODE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "depth",
    "n_children",
    "test_projection_dimension",
    "raw_mp_signal_count",
    "mp_threshold_rows",
    "effective_independent_rows",
    "mp_upper_edge",
    "top_eigenvalue",
    "top_eigenvalue_over_mp",
    "selected_eigenvalue_gap_ratio",
    "selected_eigenvalue_effective_rank",
    "sibling_p_value",
    "sibling_open",
)

EDGE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "parent_id",
    "child_id",
    "parent_depth",
    "child_depth",
    "parent_raw_mp_signal_count",
    "child_raw_mp_signal_count",
    "mp_common_dimension",
    "mp_dimension_gap",
    "mp_pair_supported",
    "floor_common_dimension",
    "mp_subspace_largest_cosine",
    "mp_subspace_mean_squared_cosine",
    "mp_subspace_chordal_distance",
    "floor_subspace_largest_cosine",
    "floor_subspace_mean_squared_cosine",
    "floor_subspace_chordal_distance",
    "mp_log_eigenvalue_delta",
    "floor_log_eigenvalue_delta",
    "parent_selected_eigenvalue_gap_ratio",
    "child_selected_eigenvalue_gap_ratio",
    "parent_top_eigenvalue_over_mp",
    "child_top_eigenvalue_over_mp",
    "spectral_barrier",
    "spectral_flow_affinity",
    "flow_status",
    "parent_sibling_p_value",
    "child_sibling_p_value",
    "parent_sibling_open",
    "child_sibling_open",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "flow_status",
    "edge_count",
    "mp_supported_count",
    "median_mp_subspace_chordal_distance",
    "median_mp_log_eigenvalue_delta",
    "median_spectral_barrier",
    "median_spectral_flow_affinity",
    "median_parent_top_eigenvalue_over_mp",
    "median_child_top_eigenvalue_over_mp",
)

SEPARATION_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "metric",
    "auc_signal_vs_selected_null",
    "finite_pair_count",
    "signal_median",
    "selected_null_median",
    "diagnostic_status",
)

BLOCK_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "block_id",
    "eigen_index_start",
    "eigen_index_stop",
    "multiplicity",
    "projector_rank",
    "eigenvalue_min",
    "eigenvalue_max",
    "eigenvalue_geometric_mean",
    "log_eigenvalue_center",
    "log_eigenvalue_spread",
    "normalized_characteristic_polynomial",
)

MODE_EDGE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "parent_id",
    "child_id",
    "parent_depth",
    "child_depth",
    "parent_mp_block_count",
    "child_mp_block_count",
    "parent_total_mp_multiplicity",
    "child_total_mp_multiplicity",
    "matched_mp_block_count",
    "unmatched_mp_block_count",
    "block_count_gap",
    "multiplicity_total_gap",
    "mean_block_projector_chordal_distance",
    "mean_block_log_eigenvalue_center_delta",
    "mean_block_multiplicity_distance",
    "mean_block_polynomial_distance",
    "mode_transport_cost",
    "mode_transport_affinity",
    "connection_laplacian_residual",
    "mode_flow_status",
    "parent_sibling_p_value",
    "child_sibling_p_value",
    "parent_sibling_open",
    "child_sibling_open",
)

MODE_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "mode_flow_status",
    "edge_count",
    "matched_edge_count",
    "median_mode_transport_cost",
    "median_mode_transport_affinity",
    "median_connection_laplacian_residual",
    "median_mean_block_projector_chordal_distance",
    "median_mean_block_log_eigenvalue_center_delta",
    "median_mean_block_polynomial_distance",
)

MODE_SEPARATION_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "metric",
    "auc_signal_vs_selected_null",
    "finite_pair_count",
    "signal_median",
    "selected_null_median",
    "diagnostic_status",
)


@dataclass(frozen=True)
class SubspaceComparison:
    """Sign-invariant comparison of two row-basis eigenspaces."""

    common_dimension: int
    largest_cosine: float
    mean_squared_cosine: float
    chordal_distance: float
    status: str


@dataclass(frozen=True)
class SpectralModeBlock:
    """MP-supported eigenvalue block with a stable projector signature."""

    block_id: int
    start: int
    stop: int
    multiplicity: int
    projector_rank: int
    eigenvalues: np.ndarray
    projector: np.ndarray
    polynomial_coefficients: np.ndarray
    log_center: float
    log_spread: float


@dataclass(frozen=True)
class SpectralModeDistance:
    """Distance between two multiplicity-aware spectral mode blocks."""

    projector_chordal_distance: float
    log_eigenvalue_center_delta: float
    multiplicity_distance: float
    polynomial_distance: float
    total_cost: float


@dataclass(frozen=True)
class SpectralModeMatching:
    """Optimal matching summary for two node spectral signatures."""

    matched_block_count: int
    unmatched_block_count: int
    mean_projector_chordal_distance: float
    mean_log_eigenvalue_center_delta: float
    mean_multiplicity_distance: float
    mean_polynomial_distance: float
    mode_transport_cost: float
    mode_transport_affinity: float
    connection_laplacian_residual: float
    status: str


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _annotation_value(annotations: pd.DataFrame, node: object, column: str) -> object:
    if column not in annotations or node not in annotations.index:
        return np.nan
    return annotations.at[node, column]


def _annotation_bool(annotations: pd.DataFrame, node: object, column: str) -> bool:
    value = _annotation_value(annotations, node, column)
    return False if pd.isna(value) else bool(value)


def _root(tree: nx.DiGraph) -> object:
    if hasattr(tree, "root"):
        return tree.root()
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {roots!r}.")
    return roots[0]


def _depths(tree: nx.DiGraph) -> dict[object, int]:
    return {
        node: int(depth)
        for node, depth in nx.single_source_shortest_path_length(tree, _root(tree)).items()
    }


def _parent_id(tree: nx.DiGraph, node: object) -> str:
    parents = list(tree.predecessors(node))
    return "" if not parents else str(parents[0])


def _positive_eigenvalues(eigenvalues: np.ndarray | None) -> np.ndarray:
    if eigenvalues is None:
        return np.zeros(0, dtype=float)
    values = np.asarray(eigenvalues, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def eigenvalue_geometry(eigenvalues: np.ndarray | None) -> dict[str, float]:
    """Return compact eigenvalue shape summaries."""
    values = _positive_eigenvalues(eigenvalues)
    if values.size == 0:
        return {
            "top_eigenvalue": math.nan,
            "selected_eigenvalue_gap_ratio": math.nan,
            "selected_eigenvalue_effective_rank": math.nan,
        }
    total = float(values.sum())
    weights = values / total
    entropy = float(-np.sum(weights * np.log(weights)))
    top = float(values[0])
    second = float(values[1]) if values.size > 1 else math.nan
    return {
        "top_eigenvalue": top,
        "selected_eigenvalue_gap_ratio": (
            float(top / second) if math.isfinite(second) and second > 0.0 else math.inf
        ),
        "selected_eigenvalue_effective_rank": float(math.exp(entropy)),
    }


def mp_upper_edge(*, feature_count: int, mp_threshold_rows: int) -> float:
    """Return the identity-population MP upper edge."""
    if int(feature_count) <= 0 or int(mp_threshold_rows) <= 0:
        return math.nan
    return float((1.0 + math.sqrt(float(feature_count) / float(mp_threshold_rows))) ** 2)


def compare_row_basis_subspaces(
    left_projection: np.ndarray | None,
    right_projection: np.ndarray | None,
    *,
    dimension: int | None = None,
) -> SubspaceComparison:
    """Compare two row-basis subspaces without sign sensitivity."""
    if left_projection is None or right_projection is None:
        return SubspaceComparison(0, math.nan, math.nan, math.nan, "projection_missing")
    left = np.asarray(left_projection, dtype=float)
    right = np.asarray(right_projection, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.size == 0 or right.size == 0:
        return SubspaceComparison(0, math.nan, math.nan, math.nan, "projection_empty")
    common_dimension = min(left.shape[0], right.shape[0])
    if dimension is not None:
        common_dimension = min(common_dimension, int(dimension))
    if common_dimension <= 0:
        return SubspaceComparison(0, math.nan, math.nan, math.nan, "no_common_dimension")
    singular_values = np.linalg.svd(
        left[:common_dimension] @ right[:common_dimension].T,
        compute_uv=False,
    )
    clipped = np.clip(singular_values[:common_dimension], 0.0, 1.0)
    sum_sq = float(np.sum(clipped**2))
    return SubspaceComparison(
        common_dimension=int(common_dimension),
        largest_cosine=float(clipped[0]) if clipped.size else math.nan,
        mean_squared_cosine=float(sum_sq / common_dimension),
        chordal_distance=float(
            math.sqrt(max(float(common_dimension) - sum_sq, 0.0) / common_dimension)
        ),
        status="subspace_compared",
    )


def log_eigenvalue_delta(
    left_eigenvalues: np.ndarray | None,
    right_eigenvalues: np.ndarray | None,
    *,
    dimension: int,
) -> float:
    """Return RMS log-eigenvalue drift over the shared leading dimensions."""
    if int(dimension) <= 0:
        return math.nan
    left = _positive_eigenvalues(left_eigenvalues)
    right = _positive_eigenvalues(right_eigenvalues)
    common = min(int(dimension), left.size, right.size)
    if common <= 0:
        return math.nan
    diff = np.log(left[:common]) - np.log(right[:common])
    return float(np.sqrt(np.mean(diff**2)))


def _row_space_projector(row_basis: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the orthogonal projector onto a row space."""
    basis = np.asarray(row_basis, dtype=float)
    if basis.ndim != 2 or basis.size == 0:
        return np.zeros((0, 0), dtype=float), 0
    _u, singular_values, vh = np.linalg.svd(basis, full_matrices=False)
    tolerance = np.finfo(float).eps * max(basis.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank <= 0:
        return np.zeros((basis.shape[1], basis.shape[1]), dtype=float), 0
    orthonormal_rows = vh[:rank]
    return orthonormal_rows.T @ orthonormal_rows, rank


def normalized_characteristic_polynomial(eigenvalues: np.ndarray) -> np.ndarray:
    """Return scale-normalized characteristic-polynomial coefficients."""
    values = _positive_eigenvalues(eigenvalues)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    scale = float(np.exp(np.mean(np.log(values))))
    normalized_roots = values / scale if scale > 0.0 else values
    coefficients = np.poly(normalized_roots).astype(float)
    norm = float(np.linalg.norm(coefficients))
    return coefficients / norm if norm > 0.0 else coefficients


def _polynomial_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return math.nan
    width = max(left.size, right.size)
    left_pad = np.pad(left, (0, width - left.size))
    right_pad = np.pad(right, (0, width - right.size))
    return float(np.linalg.norm(left_pad - right_pad) / math.sqrt(width))


def spectral_mode_blocks(
    eigenvalues: np.ndarray | None,
    projection: np.ndarray | None,
    *,
    raw_mp_signal_count: int,
    eigenvalue_block_log_tolerance: float = DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
) -> list[SpectralModeBlock]:
    """Group leading MP-supported eigenvalues into multiplicity blocks."""
    if projection is None:
        return []
    projected = np.asarray(projection, dtype=float)
    if projected.ndim != 2 or projected.size == 0:
        return []
    values = _positive_eigenvalues(eigenvalues)
    mp_count = min(int(raw_mp_signal_count), values.size, projected.shape[0])
    if mp_count <= 0:
        return []
    leading = values[:mp_count]
    blocks: list[SpectralModeBlock] = []
    start = 0
    tolerance = max(float(eigenvalue_block_log_tolerance), 0.0)
    for index in range(1, mp_count + 1):
        should_close = index == mp_count
        if not should_close:
            gap = abs(math.log(leading[index - 1]) - math.log(leading[index]))
            should_close = bool(gap > tolerance)
        if not should_close:
            continue
        block_values = leading[start:index]
        projector, projector_rank = _row_space_projector(projected[start:index])
        log_values = np.log(block_values)
        blocks.append(
            SpectralModeBlock(
                block_id=len(blocks),
                start=int(start),
                stop=int(index),
                multiplicity=int(index - start),
                projector_rank=int(projector_rank),
                eigenvalues=block_values,
                projector=projector,
                polynomial_coefficients=normalized_characteristic_polynomial(
                    block_values
                ),
                log_center=float(np.mean(log_values)),
                log_spread=float(np.std(log_values)),
            )
        )
        start = index
    return blocks


def _projector_chordal_distance(
    left_projector: np.ndarray,
    right_projector: np.ndarray,
    *,
    left_rank: int,
    right_rank: int,
) -> float:
    if left_rank <= 0 or right_rank <= 0:
        return math.nan
    if left_projector.shape != right_projector.shape:
        return math.nan
    overlap = float(np.trace(left_projector @ right_projector))
    denom = float(max(left_rank + right_rank, 1))
    squared = max((float(left_rank + right_rank) - 2.0 * overlap) / denom, 0.0)
    return float(math.sqrt(squared))


def spectral_mode_distance(
    left: SpectralModeBlock,
    right: SpectralModeBlock,
    *,
    weights: dict[str, float] | None = None,
) -> SpectralModeDistance:
    """Return multiplicity-aware spectral block distance."""
    active_weights = MODE_DISTANCE_WEIGHTS if weights is None else weights
    projector_distance = _projector_chordal_distance(
        left.projector,
        right.projector,
        left_rank=left.projector_rank,
        right_rank=right.projector_rank,
    )
    log_delta = abs(left.log_center - right.log_center)
    multiplicity_distance = abs(left.multiplicity - right.multiplicity) / max(
        left.multiplicity,
        right.multiplicity,
        1,
    )
    polynomial_distance = _polynomial_distance(
        left.polynomial_coefficients,
        right.polynomial_coefficients,
    )
    cost_terms = [
        active_weights.get("projector", 1.0) * _finite_float(projector_distance),
        active_weights.get("log_eigenvalue", 1.0) * _finite_float(log_delta),
        active_weights.get("multiplicity", 0.5)
        * _finite_float(multiplicity_distance),
        active_weights.get("polynomial", 0.5) * _finite_float(polynomial_distance),
    ]
    total_cost = float(sum(term for term in cost_terms if math.isfinite(term)))
    return SpectralModeDistance(
        projector_chordal_distance=projector_distance,
        log_eigenvalue_center_delta=float(log_delta),
        multiplicity_distance=float(multiplicity_distance),
        polynomial_distance=polynomial_distance,
        total_cost=total_cost,
    )


def match_spectral_mode_blocks(
    left_blocks: list[SpectralModeBlock],
    right_blocks: list[SpectralModeBlock],
    *,
    unmatched_mode_penalty: float = DEFAULT_UNMATCHED_MODE_PENALTY,
) -> SpectralModeMatching:
    """Optimally match MP spectral mode blocks across one tree edge."""
    left_count = len(left_blocks)
    right_count = len(right_blocks)
    if left_count == 0 and right_count == 0:
        return SpectralModeMatching(
            matched_block_count=0,
            unmatched_block_count=0,
            mean_projector_chordal_distance=math.nan,
            mean_log_eigenvalue_center_delta=math.nan,
            mean_multiplicity_distance=math.nan,
            mean_polynomial_distance=math.nan,
            mode_transport_cost=math.nan,
            mode_transport_affinity=math.nan,
            connection_laplacian_residual=math.nan,
            status="floor_only_no_mp_certified_block",
        )
    if left_count == 0 or right_count == 0:
        unmatched = max(left_count, right_count)
        residual = float(abs(unmatched_mode_penalty))
        return SpectralModeMatching(
            matched_block_count=0,
            unmatched_block_count=int(unmatched),
            mean_projector_chordal_distance=math.nan,
            mean_log_eigenvalue_center_delta=math.nan,
            mean_multiplicity_distance=math.nan,
            mean_polynomial_distance=math.nan,
            mode_transport_cost=residual,
            mode_transport_affinity=float(math.exp(-residual)),
            connection_laplacian_residual=residual,
            status="mp_block_missing_on_one_side",
        )

    distance_grid: list[list[SpectralModeDistance]] = []
    cost_matrix = np.zeros((left_count, right_count), dtype=float)
    for left_index, left_block in enumerate(left_blocks):
        row: list[SpectralModeDistance] = []
        for right_index, right_block in enumerate(right_blocks):
            distance = spectral_mode_distance(left_block, right_block)
            row.append(distance)
            cost_matrix[left_index, right_index] = distance.total_cost
        distance_grid.append(row)

    left_indices, right_indices = linear_sum_assignment(cost_matrix)
    matched_distances = [
        distance_grid[int(left_index)][int(right_index)]
        for left_index, right_index in zip(left_indices, right_indices, strict=True)
    ]
    matched = len(matched_distances)
    unmatched = left_count + right_count - 2 * matched
    denominator = max(left_count, right_count, 1)
    total_cost = (
        sum(distance.total_cost for distance in matched_distances)
        + unmatched * float(unmatched_mode_penalty)
    ) / denominator
    residual = math.sqrt(
        (
            sum(distance.total_cost**2 for distance in matched_distances)
            + unmatched * float(unmatched_mode_penalty) ** 2
        )
        / denominator
    )

    return SpectralModeMatching(
        matched_block_count=int(matched),
        unmatched_block_count=int(unmatched),
        mean_projector_chordal_distance=float(
            np.mean([d.projector_chordal_distance for d in matched_distances])
        ),
        mean_log_eigenvalue_center_delta=float(
            np.mean([d.log_eigenvalue_center_delta for d in matched_distances])
        ),
        mean_multiplicity_distance=float(
            np.mean([d.multiplicity_distance for d in matched_distances])
        ),
        mean_polynomial_distance=float(
            np.mean([d.polynomial_distance for d in matched_distances])
        ),
        mode_transport_cost=float(total_cost),
        mode_transport_affinity=float(math.exp(-total_cost)),
        connection_laplacian_residual=float(residual),
        status="mp_blocks_compared",
    )


def spectral_barrier(
    *,
    subspace_chordal_distance: float,
    log_eigenvalue_delta_value: float,
    mp_dimension_gap: float,
    parent_gap_ratio: float,
    child_gap_ratio: float,
) -> float:
    """Return a diagnostic spectral-flow barrier."""
    terms = [
        _finite_float(subspace_chordal_distance),
        _finite_float(log_eigenvalue_delta_value),
        _finite_float(mp_dimension_gap),
    ]
    positive_terms = [value for value in terms if math.isfinite(value)]
    if not positive_terms:
        return math.nan
    gap_values = [
        value
        for value in (_finite_float(parent_gap_ratio), _finite_float(child_gap_ratio))
        if math.isfinite(value) and value > 0.0
    ]
    gap_reward = 0.0
    if gap_values:
        min_gap = min(gap_values)
        gap_reward = math.log1p(min_gap) / (1.0 + math.log1p(min_gap))
    return float(max(sum(positive_terms) - 0.25 * gap_reward, 0.0))


def _flow_status(parent_raw_mp: int, child_raw_mp: int, common_dimension: int) -> str:
    if min(int(parent_raw_mp), int(child_raw_mp)) <= 0:
        return "floor_only_no_mp_certified_mode"
    if int(common_dimension) <= 0:
        return "mp_supported_but_no_common_subspace"
    return "mp_supported_subspace_compared"


def build_spectral_flow_panels(
    *,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    tree: nx.DiGraph,
    annotations: pd.DataFrame,
    spectral_context: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build node and parent-child spectral-flow rows from a real spectral context."""
    depths = _depths(tree)
    projections = spectral_context.principal_component_projections_by_node
    eigenvalues_by_node = spectral_context.principal_component_eigenvalues_by_node
    test_dimensions = spectral_context.test_projection_dimensions_by_node
    raw_mp_counts = spectral_context.raw_mp_signal_counts_by_node
    mp_rows = spectral_context.mp_threshold_rows_by_node
    effective_rows = spectral_context.effective_independent_rows_by_node

    node_records: list[dict[str, object]] = []
    for node in tree.nodes:
        node_key = str(node)
        projection = projections.get(node_key)
        feature_count = int(projection.shape[1]) if isinstance(projection, np.ndarray) and projection.ndim == 2 else 0
        threshold_rows = int(mp_rows.get(node_key, 0))
        upper_edge = mp_upper_edge(
            feature_count=feature_count,
            mp_threshold_rows=threshold_rows,
        )
        eig = eigenvalue_geometry(eigenvalues_by_node.get(node_key))
        top = eig["top_eigenvalue"]
        node_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "method_id": str(method_id),
                "replicate": int(replicate),
                "node_id": node_key,
                "parent_id": _parent_id(tree, node),
                "depth": int(depths.get(node, -1)),
                "n_children": int(tree.out_degree(node)),
                "test_projection_dimension": int(test_dimensions.get(node_key, 0)),
                "raw_mp_signal_count": int(raw_mp_counts.get(node_key, 0)),
                "mp_threshold_rows": threshold_rows,
                "effective_independent_rows": int(effective_rows.get(node_key, 0)),
                "mp_upper_edge": upper_edge,
                "top_eigenvalue": top,
                "top_eigenvalue_over_mp": (
                    float(top / upper_edge)
                    if math.isfinite(top) and math.isfinite(upper_edge) and upper_edge > 0.0
                    else math.nan
                ),
                **{k: v for k, v in eig.items() if k != "top_eigenvalue"},
                "sibling_p_value": _annotation_value(
                    annotations,
                    node,
                    "Sibling_Divergence_P_Value_Corrected",
                ),
                "sibling_open": _annotation_bool(
                    annotations,
                    node,
                    "Sibling_BH_Different",
                ),
            }
        )

    edge_records: list[dict[str, object]] = []
    node_table = pd.DataFrame.from_records(node_records).set_index("node_id", drop=False)
    for parent, child in tree.edges:
        parent_key = str(parent)
        child_key = str(child)
        parent_raw = int(raw_mp_counts.get(parent_key, 0))
        child_raw = int(raw_mp_counts.get(child_key, 0))
        mp_common_dimension = min(parent_raw, child_raw)
        parent_test = int(test_dimensions.get(parent_key, 0))
        child_test = int(test_dimensions.get(child_key, 0))
        floor_common_dimension = min(parent_test, child_test)
        mp_alignment = compare_row_basis_subspaces(
            projections.get(parent_key),
            projections.get(child_key),
            dimension=mp_common_dimension,
        )
        floor_alignment = compare_row_basis_subspaces(
            projections.get(parent_key),
            projections.get(child_key),
            dimension=floor_common_dimension,
        )
        mp_log_delta = log_eigenvalue_delta(
            eigenvalues_by_node.get(parent_key),
            eigenvalues_by_node.get(child_key),
            dimension=mp_common_dimension,
        )
        floor_log_delta = log_eigenvalue_delta(
            eigenvalues_by_node.get(parent_key),
            eigenvalues_by_node.get(child_key),
            dimension=floor_common_dimension,
        )
        dimension_gap = (
            abs(parent_raw - child_raw) / max(parent_raw, child_raw, 1)
            if max(parent_raw, child_raw, 1) > 0
            else 0.0
        )
        parent_gap = _finite_float(
            node_table.at[parent_key, "selected_eigenvalue_gap_ratio"]
            if parent_key in node_table.index
            else math.nan
        )
        child_gap = _finite_float(
            node_table.at[child_key, "selected_eigenvalue_gap_ratio"]
            if child_key in node_table.index
            else math.nan
        )
        barrier = spectral_barrier(
            subspace_chordal_distance=mp_alignment.chordal_distance,
            log_eigenvalue_delta_value=mp_log_delta,
            mp_dimension_gap=float(dimension_gap),
            parent_gap_ratio=parent_gap,
            child_gap_ratio=child_gap,
        )
        status = _flow_status(parent_raw, child_raw, mp_common_dimension)
        edge_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "method_id": str(method_id),
                "replicate": int(replicate),
                "parent_id": parent_key,
                "child_id": child_key,
                "parent_depth": int(depths.get(parent, -1)),
                "child_depth": int(depths.get(child, -1)),
                "parent_raw_mp_signal_count": parent_raw,
                "child_raw_mp_signal_count": child_raw,
                "mp_common_dimension": int(mp_common_dimension),
                "mp_dimension_gap": float(dimension_gap),
                "mp_pair_supported": bool(mp_common_dimension > 0),
                "floor_common_dimension": int(floor_common_dimension),
                "mp_subspace_largest_cosine": mp_alignment.largest_cosine,
                "mp_subspace_mean_squared_cosine": mp_alignment.mean_squared_cosine,
                "mp_subspace_chordal_distance": mp_alignment.chordal_distance,
                "floor_subspace_largest_cosine": floor_alignment.largest_cosine,
                "floor_subspace_mean_squared_cosine": floor_alignment.mean_squared_cosine,
                "floor_subspace_chordal_distance": floor_alignment.chordal_distance,
                "mp_log_eigenvalue_delta": mp_log_delta,
                "floor_log_eigenvalue_delta": floor_log_delta,
                "parent_selected_eigenvalue_gap_ratio": parent_gap,
                "child_selected_eigenvalue_gap_ratio": child_gap,
                "parent_top_eigenvalue_over_mp": (
                    node_table.at[parent_key, "top_eigenvalue_over_mp"]
                    if parent_key in node_table.index
                    else math.nan
                ),
                "child_top_eigenvalue_over_mp": (
                    node_table.at[child_key, "top_eigenvalue_over_mp"]
                    if child_key in node_table.index
                    else math.nan
                ),
                "spectral_barrier": barrier,
                "spectral_flow_affinity": (
                    float(math.exp(-barrier)) if math.isfinite(barrier) else math.nan
                ),
                "flow_status": status,
                "parent_sibling_p_value": _annotation_value(
                    annotations,
                    parent,
                    "Sibling_Divergence_P_Value_Corrected",
                ),
                "child_sibling_p_value": _annotation_value(
                    annotations,
                    child,
                    "Sibling_Divergence_P_Value_Corrected",
                ),
                "parent_sibling_open": _annotation_bool(
                    annotations,
                    parent,
                    "Sibling_BH_Different",
                ),
                "child_sibling_open": _annotation_bool(
                    annotations,
                    child,
                    "Sibling_BH_Different",
                ),
            }
        )

    return (
        pd.DataFrame.from_records(node_records, columns=NODE_COLUMNS),
        pd.DataFrame.from_records(edge_records, columns=EDGE_COLUMNS),
    )


def build_multiplicity_spectral_panels(
    *,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    tree: nx.DiGraph,
    annotations: pd.DataFrame,
    spectral_context: Any,
    eigenvalue_block_log_tolerance: float = DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
    unmatched_mode_penalty: float = DEFAULT_UNMATCHED_MODE_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build multiplicity-aware MP mode block and transport rows."""
    depths = _depths(tree)
    projections = spectral_context.principal_component_projections_by_node
    eigenvalues_by_node = spectral_context.principal_component_eigenvalues_by_node
    raw_mp_counts = spectral_context.raw_mp_signal_counts_by_node

    blocks_by_node: dict[str, list[SpectralModeBlock]] = {}
    block_records: list[dict[str, object]] = []
    for node in tree.nodes:
        node_key = str(node)
        blocks = spectral_mode_blocks(
            eigenvalues_by_node.get(node_key),
            projections.get(node_key),
            raw_mp_signal_count=int(raw_mp_counts.get(node_key, 0)),
            eigenvalue_block_log_tolerance=float(eigenvalue_block_log_tolerance),
        )
        blocks_by_node[node_key] = blocks
        for block in blocks:
            block_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "case_id": str(case_id),
                    "data_role": str(data_role),
                    "method_id": str(method_id),
                    "replicate": int(replicate),
                    "node_id": node_key,
                    "block_id": int(block.block_id),
                    "eigen_index_start": int(block.start),
                    "eigen_index_stop": int(block.stop),
                    "multiplicity": int(block.multiplicity),
                    "projector_rank": int(block.projector_rank),
                    "eigenvalue_min": float(np.min(block.eigenvalues)),
                    "eigenvalue_max": float(np.max(block.eigenvalues)),
                    "eigenvalue_geometric_mean": float(
                        np.exp(np.mean(np.log(block.eigenvalues)))
                    ),
                    "log_eigenvalue_center": float(block.log_center),
                    "log_eigenvalue_spread": float(block.log_spread),
                    "normalized_characteristic_polynomial": json.dumps(
                        [float(value) for value in block.polynomial_coefficients],
                        separators=(",", ":"),
                    ),
                }
            )

    mode_edge_records: list[dict[str, object]] = []
    for parent, child in tree.edges:
        parent_key = str(parent)
        child_key = str(child)
        parent_blocks = blocks_by_node.get(parent_key, [])
        child_blocks = blocks_by_node.get(child_key, [])
        parent_multiplicity = sum(block.multiplicity for block in parent_blocks)
        child_multiplicity = sum(block.multiplicity for block in child_blocks)
        matching = match_spectral_mode_blocks(
            parent_blocks,
            child_blocks,
            unmatched_mode_penalty=float(unmatched_mode_penalty),
        )
        block_count_gap = abs(len(parent_blocks) - len(child_blocks)) / max(
            len(parent_blocks),
            len(child_blocks),
            1,
        )
        multiplicity_total_gap = abs(parent_multiplicity - child_multiplicity) / max(
            parent_multiplicity,
            child_multiplicity,
            1,
        )
        mode_edge_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "method_id": str(method_id),
                "replicate": int(replicate),
                "parent_id": parent_key,
                "child_id": child_key,
                "parent_depth": int(depths.get(parent, -1)),
                "child_depth": int(depths.get(child, -1)),
                "parent_mp_block_count": int(len(parent_blocks)),
                "child_mp_block_count": int(len(child_blocks)),
                "parent_total_mp_multiplicity": int(parent_multiplicity),
                "child_total_mp_multiplicity": int(child_multiplicity),
                "matched_mp_block_count": int(matching.matched_block_count),
                "unmatched_mp_block_count": int(matching.unmatched_block_count),
                "block_count_gap": float(block_count_gap),
                "multiplicity_total_gap": float(multiplicity_total_gap),
                "mean_block_projector_chordal_distance": (
                    matching.mean_projector_chordal_distance
                ),
                "mean_block_log_eigenvalue_center_delta": (
                    matching.mean_log_eigenvalue_center_delta
                ),
                "mean_block_multiplicity_distance": (
                    matching.mean_multiplicity_distance
                ),
                "mean_block_polynomial_distance": matching.mean_polynomial_distance,
                "mode_transport_cost": matching.mode_transport_cost,
                "mode_transport_affinity": matching.mode_transport_affinity,
                "connection_laplacian_residual": (
                    matching.connection_laplacian_residual
                ),
                "mode_flow_status": matching.status,
                "parent_sibling_p_value": _annotation_value(
                    annotations,
                    parent,
                    "Sibling_Divergence_P_Value_Corrected",
                ),
                "child_sibling_p_value": _annotation_value(
                    annotations,
                    child,
                    "Sibling_Divergence_P_Value_Corrected",
                ),
                "parent_sibling_open": _annotation_bool(
                    annotations,
                    parent,
                    "Sibling_BH_Different",
                ),
                "child_sibling_open": _annotation_bool(
                    annotations,
                    child,
                    "Sibling_BH_Different",
                ),
            }
        )

    return (
        pd.DataFrame.from_records(block_records, columns=BLOCK_COLUMNS),
        pd.DataFrame.from_records(mode_edge_records, columns=MODE_EDGE_COLUMNS),
    )


def summarize_spectral_flow_edges(edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize spectral-flow rows by role, method, and support status."""
    if edge_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in edge_rows.groupby(["data_role", "method_id", "flow_status"], dropna=False):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": keys[0],
                "method_id": keys[1],
                "flow_status": keys[2],
                "edge_count": int(len(group)),
                "mp_supported_count": int(group["mp_pair_supported"].sum()),
                "median_mp_subspace_chordal_distance": _median(
                    group["mp_subspace_chordal_distance"]
                ),
                "median_mp_log_eigenvalue_delta": _median(
                    group["mp_log_eigenvalue_delta"]
                ),
                "median_spectral_barrier": _median(group["spectral_barrier"]),
                "median_spectral_flow_affinity": _median(
                    group["spectral_flow_affinity"]
                ),
                "median_parent_top_eigenvalue_over_mp": _median(
                    group["parent_top_eigenvalue_over_mp"]
                ),
                "median_child_top_eigenvalue_over_mp": _median(
                    group["child_top_eigenvalue_over_mp"]
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def summarize_mode_transport_edges(mode_edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize multiplicity-aware mode transport rows."""
    if mode_edge_rows.empty:
        return pd.DataFrame(columns=MODE_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in mode_edge_rows.groupby(
        ["data_role", "method_id", "mode_flow_status"],
        dropna=False,
    ):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": keys[0],
                "method_id": keys[1],
                "mode_flow_status": keys[2],
                "edge_count": int(len(group)),
                "matched_edge_count": int(group["matched_mp_block_count"].gt(0).sum()),
                "median_mode_transport_cost": _median(
                    group["mode_transport_cost"]
                ),
                "median_mode_transport_affinity": _median(
                    group["mode_transport_affinity"]
                ),
                "median_connection_laplacian_residual": _median(
                    group["connection_laplacian_residual"]
                ),
                "median_mean_block_projector_chordal_distance": _median(
                    group["mean_block_projector_chordal_distance"]
                ),
                "median_mean_block_log_eigenvalue_center_delta": _median(
                    group["mean_block_log_eigenvalue_center_delta"]
                ),
                "median_mean_block_polynomial_distance": _median(
                    group["mean_block_polynomial_distance"]
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=MODE_SUMMARY_COLUMNS)


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return math.nan if finite.empty else float(finite.median())


def _auc_for_metric(group: pd.DataFrame, metric: str) -> dict[str, object]:
    if metric not in group:
        return {
            "auc_signal_vs_selected_null": math.nan,
            "finite_pair_count": 0,
            "signal_median": math.nan,
            "selected_null_median": math.nan,
            "diagnostic_status": "metric_unavailable",
        }
    pair = group[["data_role", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    pair = pair[pair["data_role"].isin(["selected_null", "signal"])]
    if pair["data_role"].nunique() < 2 or pair[metric].nunique() < 2:
        return {
            "auc_signal_vs_selected_null": math.nan,
            "finite_pair_count": int(pair.shape[0]),
            "signal_median": _median(pair.loc[pair["data_role"].eq("signal"), metric]),
            "selected_null_median": _median(
                pair.loc[pair["data_role"].eq("selected_null"), metric]
            ),
            "diagnostic_status": "auc_unavailable",
        }
    labels = pair["data_role"].eq("signal").astype(int)
    values = pd.to_numeric(pair[metric], errors="coerce")
    return {
        "auc_signal_vs_selected_null": float(roc_auc_score(labels, values)),
        "finite_pair_count": int(pair.shape[0]),
        "signal_median": _median(pair.loc[pair["data_role"].eq("signal"), metric]),
        "selected_null_median": _median(
            pair.loc[pair["data_role"].eq("selected_null"), metric]
        ),
        "diagnostic_status": "auc_observed_diagnostic_only",
    }


def summarize_spectral_flow_separation(edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Return signal-vs-selected-null separation diagnostics."""
    metrics = (
        "spectral_flow_affinity",
        "mp_subspace_mean_squared_cosine",
        "mp_subspace_chordal_distance",
        "mp_log_eigenvalue_delta",
        "spectral_barrier",
        "parent_top_eigenvalue_over_mp",
        "child_top_eigenvalue_over_mp",
    )
    if edge_rows.empty:
        return pd.DataFrame(columns=SEPARATION_COLUMNS)
    records: list[dict[str, object]] = []
    for method_id, group in edge_rows.groupby("method_id", dropna=False):
        for metric in metrics:
            result = _auc_for_metric(group, metric)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "method_id": method_id,
                    "metric": metric,
                    **result,
                }
            )
    return pd.DataFrame.from_records(records, columns=SEPARATION_COLUMNS)


def summarize_mode_transport_separation(
    mode_edge_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return signal-vs-selected-null separation for mode transport metrics."""
    metrics = (
        "mode_transport_affinity",
        "mode_transport_cost",
        "connection_laplacian_residual",
        "mean_block_projector_chordal_distance",
        "mean_block_log_eigenvalue_center_delta",
        "mean_block_multiplicity_distance",
        "mean_block_polynomial_distance",
        "unmatched_mp_block_count",
        "block_count_gap",
        "multiplicity_total_gap",
    )
    if mode_edge_rows.empty:
        return pd.DataFrame(columns=MODE_SEPARATION_COLUMNS)
    records: list[dict[str, object]] = []
    for method_id, group in mode_edge_rows.groupby("method_id", dropna=False):
        for metric in metrics:
            result = _auc_for_metric(group, metric)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "method_id": method_id,
                    "metric": metric,
                    **result,
                }
            )
    return pd.DataFrame.from_records(records, columns=MODE_SEPARATION_COLUMNS)


def _run_one_spectral_flow(
    *,
    case: dict[str, object],
    data_role: str,
    method_id: str,
    replicate: int,
    data_seed: int,
    sibling_alpha: float,
    edge_alpha: float,
    spectral_minimum_dimension: int,
    eigenvalue_block_log_tolerance: float,
    unmatched_mode_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_id, source_family, feature_representation, n_samples, n_features, n_categories = (
        _case_contract(case)
    )
    data, feature_space, _truth_labels, _true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role=data_role,
        seed=data_seed,
    )
    distance = pdist(data.to_numpy(dtype=float), metric="hamming")
    profile_id = None if method_id == "baseline_projected_wald" else str(method_id)
    result = _run_kl_method(
        data,
        distance,
        sibling_significance_level=float(sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(edge_alpha),
        feature_space=feature_space,
        spectral_minimum_dimension=int(spectral_minimum_dimension),
        sibling_gate_profile=profile_id,
    )
    if result.status != "ok":
        raise RuntimeError(f"KL run failed for {case_id}/{data_role}: {result.skip_reason}")
    gate_bundle = result.extra["gate_bundle"]
    spectral_context = gate_bundle.edge_gate_result.spectral_context
    node_rows, edge_rows = build_spectral_flow_panels(
        case_id=case_id,
        data_role=_output_data_role(data_role),
        method_id=method_id,
        replicate=int(replicate),
        tree=result.extra["tree"],
        annotations=result.extra["annotations"],
        spectral_context=spectral_context,
    )
    block_rows, mode_edge_rows = build_multiplicity_spectral_panels(
        case_id=case_id,
        data_role=_output_data_role(data_role),
        method_id=method_id,
        replicate=int(replicate),
        tree=result.extra["tree"],
        annotations=result.extra["annotations"],
        spectral_context=spectral_context,
        eigenvalue_block_log_tolerance=float(eigenvalue_block_log_tolerance),
        unmatched_mode_penalty=float(unmatched_mode_penalty),
    )
    return node_rows, edge_rows, block_rows, mode_edge_rows


def run_selected_neighborhood_spectral_flow(
    *,
    output_dir: Path,
    suite: str = "full",
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES,
    data_roles: tuple[str, ...] = DEFAULT_DATA_ROLES,
    method_id: str = DEFAULT_METHOD_ID,
    replicates: int = 1,
    base_seed: int = 20260616,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    eigenvalue_block_log_tolerance: float = DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
    unmatched_mode_penalty: float = DEFAULT_UNMATCHED_MODE_PENALTY,
) -> dict[str, Path]:
    cases = _select_cases(suite=suite, case_names=case_names)
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    block_frames: list[pd.DataFrame] = []
    mode_edge_frames: list[pd.DataFrame] = []
    for case in cases:
        for replicate in range(int(replicates)):
            data_seed = int(base_seed) + replicate * 1009
            for data_role in data_roles:
                node_rows, edge_rows, block_rows, mode_edge_rows = _run_one_spectral_flow(
                    case=case,
                    data_role=str(data_role),
                    method_id=str(method_id),
                    replicate=replicate,
                    data_seed=data_seed,
                    sibling_alpha=float(sibling_alpha),
                    edge_alpha=float(edge_alpha),
                    spectral_minimum_dimension=int(spectral_minimum_dimension),
                    eigenvalue_block_log_tolerance=float(
                        eigenvalue_block_log_tolerance
                    ),
                    unmatched_mode_penalty=float(unmatched_mode_penalty),
                )
                node_frames.append(node_rows)
                edge_frames.append(edge_rows)
                block_frames.append(block_rows)
                mode_edge_frames.append(mode_edge_rows)

    node_rows = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame(columns=NODE_COLUMNS)
    edge_rows = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(columns=EDGE_COLUMNS)
    block_rows = (
        pd.concat(block_frames, ignore_index=True)
        if block_frames
        else pd.DataFrame(columns=BLOCK_COLUMNS)
    )
    mode_edge_rows = (
        pd.concat(mode_edge_frames, ignore_index=True)
        if mode_edge_frames
        else pd.DataFrame(columns=MODE_EDGE_COLUMNS)
    )
    summary = summarize_spectral_flow_edges(edge_rows)
    separation = summarize_spectral_flow_separation(edge_rows)
    mode_summary = summarize_mode_transport_edges(mode_edge_rows)
    mode_separation = summarize_mode_transport_separation(mode_edge_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    node_path = output_dir / "selected_neighborhood_spectral_flow_nodes.csv"
    edge_path = output_dir / "selected_neighborhood_spectral_flow_edges.csv"
    block_path = output_dir / "selected_neighborhood_spectral_flow_blocks.csv"
    mode_edge_path = output_dir / "selected_neighborhood_spectral_flow_mode_edges.csv"
    summary_path = output_dir / "selected_neighborhood_spectral_flow_summary.csv"
    separation_path = output_dir / "selected_neighborhood_spectral_flow_separation.csv"
    mode_summary_path = output_dir / "selected_neighborhood_spectral_flow_mode_summary.csv"
    mode_separation_path = output_dir / "selected_neighborhood_spectral_flow_mode_separation.csv"
    manifest_path = output_dir / "manifest.json"
    node_rows.to_csv(node_path, index=False)
    edge_rows.to_csv(edge_path, index=False)
    block_rows.to_csv(block_path, index=False)
    mode_edge_rows.to_csv(mode_edge_path, index=False)
    summary.to_csv(summary_path, index=False)
    separation.to_csv(separation_path, index=False)
    mode_summary.to_csv(mode_summary_path, index=False)
    mode_separation.to_csv(mode_separation_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "case_names": list(case_names),
        "data_roles": list(data_roles),
        "method_id": method_id,
        "replicates": int(replicates),
        "base_seed": int(base_seed),
        "sibling_alpha": float(sibling_alpha),
        "edge_alpha": float(edge_alpha),
        "spectral_minimum_dimension": int(spectral_minimum_dimension),
        "eigenvalue_block_log_tolerance": float(eigenvalue_block_log_tolerance),
        "unmatched_mode_penalty": float(unmatched_mode_penalty),
        "node_row_count": int(len(node_rows)),
        "edge_row_count": int(len(edge_rows)),
        "block_row_count": int(len(block_rows)),
        "mode_edge_row_count": int(len(mode_edge_rows)),
        "generated_at_compact": format_timestamp_utc(),
        "outputs": {
            "nodes": node_path,
            "edges": edge_path,
            "blocks": block_path,
            "mode_edges": mode_edge_path,
            "summary": summary_path,
            "separation": separation_path,
            "mode_summary": mode_summary_path,
            "mode_separation": mode_separation_path,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return {
        "nodes": node_path,
        "edges": edge_path,
        "blocks": block_path,
        "mode_edges": mode_edge_path,
        "summary": summary_path,
        "separation": separation_path,
        "mode_summary": mode_summary_path,
        "mode_separation": mode_separation_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MP eigenvector/eigenvalue spectral-flow diagnostics."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--data-roles", default=",".join(DEFAULT_DATA_ROLES))
    parser.add_argument("--method-id", default=DEFAULT_METHOD_ID)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260616)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument(
        "--spectral-minimum-dimension",
        type=int,
        default=EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    )
    parser.add_argument(
        "--eigenvalue-block-log-tolerance",
        type=float,
        default=DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
    )
    parser.add_argument(
        "--unmatched-mode-penalty",
        type=float,
        default=DEFAULT_UNMATCHED_MODE_PENALTY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_selected_neighborhood_spectral_flow(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        data_roles=parse_names(args.data_roles),
        method_id=str(args.method_id),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
        sibling_alpha=float(args.sibling_alpha),
        edge_alpha=float(args.edge_alpha),
        spectral_minimum_dimension=int(args.spectral_minimum_dimension),
        eigenvalue_block_log_tolerance=float(args.eigenvalue_block_log_tolerance),
        unmatched_mode_penalty=float(args.unmatched_mode_penalty),
    )


if __name__ == "__main__":
    main()
