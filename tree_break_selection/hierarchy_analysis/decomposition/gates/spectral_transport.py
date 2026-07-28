"""Spectral-transport support for conservative pass-through traversal.

The routines in this module are intentionally fail-closed. They annotate a
selected hierarchy with MP-mode transport diagnostics, then expose a boolean
that a traversal evaluator may use to allow or block pass-through. They do not
open sibling splits and they do not produce calibrated p-values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)

from .annotation_predicates import node_sibling_gate_open, node_split_prerequisites

DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE = 0.05
DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY = 1.0
DEFAULT_SPECTRAL_TRANSPORT_MAX_COST = 1.2

SPECTRAL_TRANSPORT_COLUMNS = (
    "Spectral_Transport_Node_MP_Block_Count",
    "Spectral_Transport_Node_MP_Total_Multiplicity",
    "Spectral_Transport_Child_Best_Mode_Cost",
    "Spectral_Transport_Child_Best_Mode_Affinity",
    "Spectral_Transport_Child_Best_Mode_Status",
    "Spectral_Transport_Best_Descendant_Split_Path_Cost",
    "Spectral_Transport_Best_Descendant_Split_Path_Has_MP_Evidence",
    "Spectral_Transport_Pass_Through_Supported",
    "Spectral_Transport_Pass_Through_Blocked",
    "Spectral_Transport_Bottleneck",
    "Spectral_Transport_Guard_Max_Cost",
    "Spectral_Transport_Guard_Require_MP_Blocks",
)


@dataclass(frozen=True)
class SpectralModeBlock:
    """Multiplicity-aware MP outlier block."""

    start: int
    stop: int
    multiplicity: int
    projector_rank: int
    eigenvalues: np.ndarray
    projector: np.ndarray
    polynomial_coefficients: np.ndarray
    log_center: float


@dataclass(frozen=True)
class ModeTransportEdge:
    """Parent-child mode transport summary."""

    cost: float
    affinity: float
    matched_block_count: int
    unmatched_block_count: int
    status: str


def _positive_eigenvalues(eigenvalues: np.ndarray | None) -> np.ndarray:
    if eigenvalues is None:
        return np.zeros(0, dtype=float)
    values = np.asarray(eigenvalues, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _row_space_projector(row_basis: np.ndarray) -> tuple[np.ndarray, int]:
    basis = np.asarray(row_basis, dtype=float)
    if basis.ndim != 2 or basis.size == 0:
        return np.zeros((0, 0), dtype=float), 0
    _u, singular_values, vh = np.linalg.svd(basis, full_matrices=False)
    tolerance = np.finfo(float).eps * max(basis.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank <= 0:
        return np.zeros((basis.shape[1], basis.shape[1]), dtype=float), 0
    rows = vh[:rank]
    return rows.T @ rows, rank


def normalized_characteristic_polynomial(eigenvalues: np.ndarray) -> np.ndarray:
    """Return a scale-normalized characteristic-polynomial fingerprint."""
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
    denominator = float(max(left_rank + right_rank, 1))
    squared = max((float(left_rank + right_rank) - 2.0 * overlap) / denominator, 0.0)
    return float(math.sqrt(squared))


def spectral_mode_blocks(
    eigenvalues: np.ndarray | None,
    projection: np.ndarray | None,
    *,
    raw_mp_signal_count: int,
    eigenvalue_block_log_tolerance: float = DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
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
    tolerance = max(float(eigenvalue_block_log_tolerance), 0.0)
    blocks: list[SpectralModeBlock] = []
    start = 0
    for index in range(1, mp_count + 1):
        should_close = index == mp_count
        if not should_close:
            log_gap = abs(math.log(leading[index - 1]) - math.log(leading[index]))
            should_close = bool(log_gap > tolerance)
        if not should_close:
            continue
        block_values = leading[start:index]
        projector, rank = _row_space_projector(projected[start:index])
        blocks.append(
            SpectralModeBlock(
                start=int(start),
                stop=int(index),
                multiplicity=int(index - start),
                projector_rank=int(rank),
                eigenvalues=block_values,
                projector=projector,
                polynomial_coefficients=normalized_characteristic_polynomial(block_values),
                log_center=float(np.mean(np.log(block_values))),
            )
        )
        start = index
    return blocks


def _mode_block_cost(left: SpectralModeBlock, right: SpectralModeBlock) -> float:
    projector = _projector_chordal_distance(
        left.projector,
        right.projector,
        left_rank=left.projector_rank,
        right_rank=right.projector_rank,
    )
    eigenvalue = abs(left.log_center - right.log_center)
    multiplicity = abs(left.multiplicity - right.multiplicity) / max(
        left.multiplicity,
        right.multiplicity,
        1,
    )
    polynomial = _polynomial_distance(
        left.polynomial_coefficients,
        right.polynomial_coefficients,
    )
    terms = (
        float(projector) if math.isfinite(float(projector)) else 0.0,
        float(eigenvalue) if math.isfinite(float(eigenvalue)) else 0.0,
        0.5 * float(multiplicity) if math.isfinite(float(multiplicity)) else 0.0,
        0.5 * float(polynomial) if math.isfinite(float(polynomial)) else 0.0,
    )
    return float(sum(terms))


def match_mode_transport_edge(
    parent_blocks: list[SpectralModeBlock],
    child_blocks: list[SpectralModeBlock],
    *,
    unmatched_mode_penalty: float = DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
) -> ModeTransportEdge:
    """Return optimal multiplicity-aware MP mode transport for one tree edge."""
    parent_count = len(parent_blocks)
    child_count = len(child_blocks)
    if parent_count == 0 and child_count == 0:
        return ModeTransportEdge(
            cost=math.nan,
            affinity=math.nan,
            matched_block_count=0,
            unmatched_block_count=0,
            status="floor_only_no_mp_certified_block",
        )
    if parent_count == 0 or child_count == 0:
        unmatched = max(parent_count, child_count)
        cost = float(unmatched_mode_penalty)
        return ModeTransportEdge(
            cost=cost,
            affinity=float(math.exp(-cost)),
            matched_block_count=0,
            unmatched_block_count=int(unmatched),
            status="mp_block_missing_on_one_side",
        )

    costs = np.zeros((parent_count, child_count), dtype=float)
    for parent_index, parent_block in enumerate(parent_blocks):
        for child_index, child_block in enumerate(child_blocks):
            costs[parent_index, child_index] = _mode_block_cost(
                parent_block,
                child_block,
            )
    parent_indices, child_indices = linear_sum_assignment(costs)
    matched_cost = float(
        sum(
            costs[int(parent_index), int(child_index)]
            for parent_index, child_index in zip(
                parent_indices,
                child_indices,
                strict=True,
            )
        )
    )
    matched = len(parent_indices)
    unmatched = parent_count + child_count - 2 * matched
    denominator = max(parent_count, child_count, 1)
    cost = (matched_cost + unmatched * float(unmatched_mode_penalty)) / denominator
    return ModeTransportEdge(
        cost=float(cost),
        affinity=float(math.exp(-cost)),
        matched_block_count=int(matched),
        unmatched_block_count=int(unmatched),
        status="mp_blocks_compared",
    )


def _empty_output(
    annotations_df: pd.DataFrame,
    *,
    max_cost: float,
    require_mp_blocks: bool,
) -> pd.DataFrame:
    out = annotations_df.copy()
    out["Spectral_Transport_Node_MP_Block_Count"] = 0
    out["Spectral_Transport_Node_MP_Total_Multiplicity"] = 0
    out["Spectral_Transport_Child_Best_Mode_Cost"] = np.nan
    out["Spectral_Transport_Child_Best_Mode_Affinity"] = np.nan
    out["Spectral_Transport_Child_Best_Mode_Status"] = ""
    out["Spectral_Transport_Best_Descendant_Split_Path_Cost"] = np.nan
    out["Spectral_Transport_Best_Descendant_Split_Path_Has_MP_Evidence"] = False
    out["Spectral_Transport_Pass_Through_Supported"] = False
    out["Spectral_Transport_Pass_Through_Blocked"] = False
    out["Spectral_Transport_Bottleneck"] = "spectral_transport_not_evaluated"
    out["Spectral_Transport_Guard_Max_Cost"] = float(max_cost)
    out["Spectral_Transport_Guard_Require_MP_Blocks"] = bool(require_mp_blocks)
    return out


def annotate_spectral_transport_passthrough_support(
    tree,
    annotations_df: pd.DataFrame,
    spectral_context: SpectralContext,
    *,
    max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    require_mp_blocks: bool = True,
    eigenvalue_block_log_tolerance: float = DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    unmatched_mode_penalty: float = DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
) -> pd.DataFrame:
    """Annotate pass-through support from MP mode transport.

    When ``require_mp_blocks`` is true, a pass-through path is supported only
    by measured MP-mode transport whose cost is at most ``max_cost``. Edges
    without matched MP blocks are treated as unmeasured bottlenecks rather than
    support. When ``require_mp_blocks`` is false, unmeasured edges remain
    pass-through neutral and only finite measured costs can veto a path.
    """
    threshold = float(max_cost)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"max_cost must be finite and nonnegative; got {max_cost!r}.")

    out = _empty_output(
        annotations_df,
        max_cost=threshold,
        require_mp_blocks=bool(require_mp_blocks),
    )
    projections = spectral_context.principal_component_projections_by_node
    eigenvalues = spectral_context.principal_component_eigenvalues_by_node
    raw_mp_counts = spectral_context.raw_mp_signal_counts_by_node

    blocks_by_node: dict[object, list[SpectralModeBlock]] = {}
    for node in tree.nodes:
        key = str(node)
        blocks = spectral_mode_blocks(
            eigenvalues.get(key),
            projections.get(key),
            raw_mp_signal_count=int(raw_mp_counts.get(key, 0)),
            eigenvalue_block_log_tolerance=float(eigenvalue_block_log_tolerance),
        )
        blocks_by_node[node] = blocks
        if node in out.index:
            out.loc[node, "Spectral_Transport_Node_MP_Block_Count"] = len(blocks)
            out.loc[node, "Spectral_Transport_Node_MP_Total_Multiplicity"] = sum(
                block.multiplicity for block in blocks
            )

    edge_transport: dict[tuple[object, object], ModeTransportEdge] = {}
    edge_supported: dict[tuple[object, object], bool] = {}
    edge_path_cost: dict[tuple[object, object], float] = {}
    edge_has_mp_evidence: dict[tuple[object, object], bool] = {}
    for parent, child in tree.edges:
        transport = match_mode_transport_edge(
            blocks_by_node.get(parent, []),
            blocks_by_node.get(child, []),
            unmatched_mode_penalty=float(unmatched_mode_penalty),
        )
        edge_transport[(parent, child)] = transport
        finite_cost = math.isfinite(transport.cost)
        has_mp_match = transport.matched_block_count > 0
        measured_transport = bool(has_mp_match or (finite_cost and not bool(require_mp_blocks)))
        edge_path_cost[(parent, child)] = float(transport.cost) if measured_transport else 0.0
        edge_has_mp_evidence[(parent, child)] = measured_transport
        if require_mp_blocks:
            edge_supported[(parent, child)] = bool(
                measured_transport and finite_cost and transport.cost <= threshold
            )
        else:
            edge_supported[(parent, child)] = bool(
                not measured_transport or (finite_cost and transport.cost <= threshold)
            )

    split_prerequisites = {node: node_split_prerequisites(tree, out, node) for node in tree.nodes}
    can_split = {
        node: bool(split_prerequisites[node] and node_sibling_gate_open(out, node))
        for node in tree.nodes
    }

    best_descendant_path_cost: dict[object, float] = {}
    best_descendant_path_has_mp_evidence: dict[object, bool] = {}
    supported_descendant_split: dict[object, bool] = {}
    for node in bottom_up_nodes(tree):
        best_cost = math.inf
        best_has_mp_evidence = False
        has_supported = False
        for child in tree.successors(node):
            transport = edge_transport.get((node, child))
            if transport is None:
                continue
            if child in out.index:
                current_best = out.loc[child, "Spectral_Transport_Child_Best_Mode_Cost"]
                if pd.isna(current_best) or (
                    math.isfinite(transport.cost) and transport.cost < float(current_best)
                ):
                    out.loc[child, "Spectral_Transport_Child_Best_Mode_Cost"] = transport.cost
                    out.loc[child, "Spectral_Transport_Child_Best_Mode_Affinity"] = (
                        transport.affinity
                    )
                    out.loc[child, "Spectral_Transport_Child_Best_Mode_Status"] = transport.status
            if not edge_supported.get((node, child), False):
                continue
            child_best = (
                0.0
                if can_split.get(child, False)
                else best_descendant_path_cost.get(child, math.inf)
            )
            if not math.isfinite(child_best):
                continue
            transport_path_cost = edge_path_cost.get((node, child), math.inf)
            if not math.isfinite(transport_path_cost):
                continue
            child_has_mp_evidence = best_descendant_path_has_mp_evidence.get(
                child,
                False,
            )
            path_has_mp_evidence = bool(
                edge_has_mp_evidence.get((node, child), False) or child_has_mp_evidence
            )
            path_cost = max(float(transport_path_cost), float(child_best))
            if path_cost < best_cost:
                best_cost = path_cost
                best_has_mp_evidence = path_has_mp_evidence
                has_supported = True
        best_descendant_path_cost[node] = best_cost
        best_descendant_path_has_mp_evidence[node] = best_has_mp_evidence
        supported_descendant_split[node] = has_supported
        if node in out.index and has_supported:
            out.loc[node, "Spectral_Transport_Best_Descendant_Split_Path_Cost"] = best_cost
            out.loc[
                node,
                "Spectral_Transport_Best_Descendant_Split_Path_Has_MP_Evidence",
            ] = best_has_mp_evidence

    for node in tree.nodes:
        if node not in out.index:
            continue
        pass_through_candidate = bool(
            split_prerequisites[node]
            and not can_split[node]
            and any(
                can_split.get(child, False) or supported_descendant_split.get(child, False)
                for child in tree.successors(node)
            )
        )
        supported = bool(pass_through_candidate and supported_descendant_split.get(node, False))
        out.loc[node, "Spectral_Transport_Pass_Through_Supported"] = supported
        out.loc[node, "Spectral_Transport_Pass_Through_Blocked"] = bool(
            pass_through_candidate and not supported
        )
        if not pass_through_candidate:
            out.loc[node, "Spectral_Transport_Bottleneck"] = "not_pass_through_candidate"
        elif supported:
            if best_descendant_path_has_mp_evidence.get(node, False):
                out.loc[node, "Spectral_Transport_Bottleneck"] = "supported_mp_mode_path"
            else:
                out.loc[node, "Spectral_Transport_Bottleneck"] = "unmeasured_no_matched_mp_path"
        else:
            out.loc[node, "Spectral_Transport_Bottleneck"] = "spectral_transport_bottleneck"

    return out


def spectral_mode_blocks_json(blocks: list[SpectralModeBlock]) -> str:
    """Return compact JSON useful for debug output."""
    return json.dumps(
        [
            {
                "start": block.start,
                "stop": block.stop,
                "multiplicity": block.multiplicity,
                "log_center": block.log_center,
            }
            for block in blocks
        ],
        separators=(",", ":"),
    )


__all__ = [
    "DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE",
    "DEFAULT_SPECTRAL_TRANSPORT_MAX_COST",
    "DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY",
    "SPECTRAL_TRANSPORT_COLUMNS",
    "ModeTransportEdge",
    "SpectralModeBlock",
    "annotate_spectral_transport_passthrough_support",
    "match_mode_transport_edge",
    "normalized_characteristic_polynomial",
    "spectral_mode_blocks",
    "spectral_mode_blocks_json",
]
