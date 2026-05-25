"""Gate-path tracing for oracle-recoverable benchmark failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.diagnostics.sibling_inflation_diagnostic import (
    collect_sibling_inflation_inputs,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import bottom_up_nodes, compute_node_depths
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.annotation_bundle import (
    GateAnnotationBundle,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    TraversalDecision,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    predict_empirical_inflation_factor,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace


@dataclass(frozen=True)
class SiblingInflationTrace:
    """Raw sibling-test and inflation quantities keyed by parent node."""

    parent: object
    raw_sibling_statistic: float
    raw_reference_scale: float
    raw_degrees_of_freedom: float
    raw_p_value: float
    empirical_inflation_factor: float
    inflation_applied: bool
    sibling_null_weight: float
    sibling_is_null_like: bool
    sibling_is_edge_blocked: bool
    branch_length_sum: float
    parent_sample_size: int
    feature_family: str


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _annotation_value(annotations_df: pd.DataFrame, node: object, column: str) -> Any:
    if node not in annotations_df.index:
        raise ValueError(f"Annotations are missing node {node!r}.")
    if column not in annotations_df.columns:
        raise ValueError(f"Annotations are missing required column {column!r}.")
    return annotations_df.at[node, column]


def _edge_prefix_values(
    annotations_df: pd.DataFrame,
    child: object | None,
    *,
    prefix: str,
) -> dict[str, object]:
    if child is None:
        return {
            f"{prefix}_edge_tested": False,
            f"{prefix}_edge_ancestor_blocked": False,
            f"{prefix}_edge_significant": False,
            f"{prefix}_edge_invalid": False,
            f"{prefix}_edge_p_value": np.nan,
            f"{prefix}_edge_p_value_bh": np.nan,
            f"{prefix}_edge_degrees_of_freedom": np.nan,
        }

    return {
        f"{prefix}_edge_tested": _as_bool(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_Tested",
            )
        ),
        f"{prefix}_edge_ancestor_blocked": _as_bool(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_Ancestor_Blocked",
            )
        ),
        f"{prefix}_edge_significant": _as_bool(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_Significant",
            )
        ),
        f"{prefix}_edge_invalid": _as_bool(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_Invalid",
            )
        ),
        f"{prefix}_edge_p_value": _as_float(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_P_Value",
            )
        ),
        f"{prefix}_edge_p_value_bh": _as_float(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_P_Value_BH",
            )
        ),
        f"{prefix}_edge_degrees_of_freedom": _as_float(
            _annotation_value(
                annotations_df,
                child,
                "Child_Parent_Divergence_df",
            )
        ),
    }


def _compute_gate_states(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
) -> tuple[dict[object, bool], dict[object, bool], dict[object, bool], dict[object, bool]]:
    children_by_node = {node: list(tree.successors(node)) for node in tree.nodes}
    split_prerequisites: dict[object, bool] = {}
    sibling_gate_open: dict[object, bool] = {}
    can_split: dict[object, bool] = {}

    for node, children in children_by_node.items():
        if len(children) == 2:
            left_child, right_child = children
            split_prerequisites[node] = bool(
                _annotation_value(
                    annotations_df,
                    left_child,
                    "Child_Parent_Divergence_Significant",
                )
                or _annotation_value(
                    annotations_df,
                    right_child,
                    "Child_Parent_Divergence_Significant",
                )
            )
        else:
            split_prerequisites[node] = False

        sibling_gate_open[node] = bool(
            not _annotation_value(annotations_df, node, "Sibling_Divergence_Skipped")
            and _annotation_value(annotations_df, node, "Sibling_BH_Different")
        )
        can_split[node] = bool(split_prerequisites[node] and sibling_gate_open[node])

    has_descendant_split: dict[object, bool] = {}
    for node in bottom_up_nodes(tree):
        has_descendant_split[node] = any(
            can_split[child] or has_descendant_split[child]
            for child in children_by_node[node]
        )

    return split_prerequisites, sibling_gate_open, can_split, has_descendant_split


def _decision_for_node(
    *,
    split_prerequisites_open: bool,
    sibling_gate_open: bool,
    has_descendant_split: bool,
    passthrough: bool,
) -> TraversalDecision:
    if passthrough:
        if not split_prerequisites_open:
            return TraversalDecision.BOUNDARY
        if sibling_gate_open:
            return TraversalDecision.SPLIT
        if has_descendant_split:
            return TraversalDecision.PASS_THROUGH
        return TraversalDecision.BOUNDARY

    if not split_prerequisites_open:
        return TraversalDecision.BOUNDARY
    if sibling_gate_open:
        return TraversalDecision.SPLIT
    return TraversalDecision.BOUNDARY


def _node_is_ancestor_of_any(
    tree: nx.DiGraph,
    node: object,
    targets: tuple[object, ...],
) -> bool:
    return any(node != target and nx.has_path(tree, node, target) for target in targets)


def _container_boundary(
    tree: nx.DiGraph,
    node: object,
    boundaries: tuple[object, ...],
) -> object | None:
    for boundary in boundaries:
        if node == boundary or nx.has_path(tree, boundary, node):
            return boundary
    return None


def _trace_relation(
    *,
    tree: nx.DiGraph,
    node: object,
    actual_boundary_nodes: tuple[object, ...],
    oracle_true_k_boundary_nodes: tuple[object, ...],
    decision: TraversalDecision,
) -> str:
    is_actual_boundary = node in actual_boundary_nodes
    is_oracle_boundary = node in oracle_true_k_boundary_nodes
    oracle_container = _container_boundary(tree, node, oracle_true_k_boundary_nodes)

    if is_actual_boundary and is_oracle_boundary:
        return "aligned_boundary"
    if is_actual_boundary and _node_is_ancestor_of_any(
        tree,
        node,
        oracle_true_k_boundary_nodes,
    ):
        return "actual_stops_above_oracle_boundary"
    if is_actual_boundary and oracle_container is not None and oracle_container != node:
        return "actual_fragment_inside_oracle_boundary"
    if is_oracle_boundary and decision in (
        TraversalDecision.SPLIT,
        TraversalDecision.PASS_THROUGH,
    ):
        return "actual_splits_oracle_boundary"
    if oracle_container is not None and decision in (
        TraversalDecision.SPLIT,
        TraversalDecision.PASS_THROUGH,
    ):
        return "actual_continues_inside_oracle_boundary"
    return "none"


def _actual_boundary_nodes(decomposition: dict[str, object]) -> tuple[object, ...]:
    assignments = decomposition["cluster_assignments"]
    if not isinstance(assignments, dict):
        raise TypeError("decomposition['cluster_assignments'] must be a dictionary.")
    return tuple(cluster["root_node"] for cluster in assignments.values())


def collect_sibling_inflation_trace(
    tree: nx.DiGraph,
    gate_annotation_bundle: GateAnnotationBundle,
    *,
    feature_space: FeatureSpace | None,
) -> dict[object, SiblingInflationTrace]:
    """Reconstruct raw sibling records and predicted inflation factors."""
    inputs = collect_sibling_inflation_inputs(
        tree,
        gate_annotation_bundle,
        feature_space=feature_space,
    )

    trace_by_parent: dict[object, SiblingInflationTrace] = {}
    for record in inputs.records:
        inflation_factor = np.nan
        inflation_applied = False
        if (
            inputs.model is not None
            and not record.is_null_like
            and record.degrees_of_freedom > 0
        ):
            inflation_factor = predict_empirical_inflation_factor(inputs.model, record)
            inflation_applied = True
        trace_by_parent[record.parent] = SiblingInflationTrace(
            parent=record.parent,
            raw_sibling_statistic=float(record.stat),
            raw_reference_scale=float(record.reference_scale),
            raw_degrees_of_freedom=float(record.degrees_of_freedom),
            raw_p_value=float(record.p_value),
            empirical_inflation_factor=float(inflation_factor),
            inflation_applied=inflation_applied,
            sibling_null_weight=float(record.sibling_null_weight),
            sibling_is_null_like=bool(record.is_null_like),
            sibling_is_edge_blocked=bool(record.is_edge_blocked),
            branch_length_sum=float(record.branch_length_sum),
            parent_sample_size=int(record.n_parent),
            feature_family=str(record.feature_family),
        )
    return trace_by_parent


def build_gate_path_trace_dataframe(
    *,
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    decomposition: dict[str, object],
    oracle_true_k_boundary_nodes: tuple[object, ...],
    oracle_any_k_boundary_nodes: tuple[object, ...],
    sibling_inflation_trace_by_parent: dict[object, SiblingInflationTrace],
    case_id: str,
    failure_class: str,
    kl_ari: float,
    oracle_true_k_ari: float,
    oracle_any_k_ari: float,
    passthrough: bool = config.PASSTHROUGH,
) -> pd.DataFrame:
    """Build one row per tree node with gate evidence and oracle comparison."""
    depths = compute_node_depths(tree)
    actual_boundaries = _actual_boundary_nodes(decomposition)
    (
        split_prerequisites,
        sibling_gate_open,
        _can_split,
        has_descendant_split,
    ) = _compute_gate_states(tree, annotations_df)

    rows: list[dict[str, object]] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        left_child = children[0] if len(children) > 0 else None
        right_child = children[1] if len(children) > 1 else None
        decision = _decision_for_node(
            split_prerequisites_open=split_prerequisites[node],
            sibling_gate_open=sibling_gate_open[node],
            has_descendant_split=has_descendant_split[node],
            passthrough=passthrough,
        )
        inflation_trace = sibling_inflation_trace_by_parent.get(node)

        row = {
            "case_id": case_id,
            "failure_class": failure_class,
            "kl_ari": float(kl_ari),
            "oracle_true_k_ari": float(oracle_true_k_ari),
            "oracle_any_k_ari": float(oracle_any_k_ari),
            "node_id": node,
            "depth": int(depths[node]),
            "is_leaf": bool(tree.nodes[node]["is_leaf"]),
            "leaf_count": int(tree.nodes[node]["leaf_count"]),
            "child_count": len(children),
            "left_child": left_child,
            "right_child": right_child,
            "actual_decision": decision.value,
            "trace_relation": _trace_relation(
                tree=tree,
                node=node,
                actual_boundary_nodes=actual_boundaries,
                oracle_true_k_boundary_nodes=oracle_true_k_boundary_nodes,
                decision=decision,
            ),
            "actual_boundary": node in actual_boundaries,
            "oracle_true_k_boundary": node in oracle_true_k_boundary_nodes,
            "oracle_any_k_boundary": node in oracle_any_k_boundary_nodes,
            "oracle_true_k_container": _container_boundary(
                tree,
                node,
                oracle_true_k_boundary_nodes,
            ),
            "split_prerequisites_open": split_prerequisites[node],
            "sibling_gate_open": sibling_gate_open[node],
            "has_descendant_split": has_descendant_split[node],
            "sibling_skipped": _as_bool(
                _annotation_value(annotations_df, node, "Sibling_Divergence_Skipped")
            ),
            "sibling_bh_different": _as_bool(
                _annotation_value(annotations_df, node, "Sibling_BH_Different")
            ),
            "sibling_bh_same": _as_bool(
                _annotation_value(annotations_df, node, "Sibling_BH_Same")
            ),
            "sibling_adjusted_statistic": _as_float(
                _annotation_value(annotations_df, node, "Sibling_Test_Statistic")
            ),
            "sibling_degrees_of_freedom": _as_float(
                _annotation_value(annotations_df, node, "Sibling_Degrees_of_Freedom")
            ),
            "sibling_adjusted_p_value": _as_float(
                _annotation_value(annotations_df, node, "Sibling_Divergence_P_Value")
            ),
            "sibling_corrected_p_value": _as_float(
                _annotation_value(
                    annotations_df,
                    node,
                    "Sibling_Divergence_P_Value_Corrected",
                )
            ),
            "sibling_invalid": _as_bool(
                _annotation_value(annotations_df, node, "Sibling_Divergence_Invalid")
            ),
            "sibling_test_method": str(
                _annotation_value(annotations_df, node, "Sibling_Test_Method")
            ),
            "sibling_projection_dimension": _as_float(
                _annotation_value(annotations_df, node, "Sibling_Projection_Dimension")
            ),
            "raw_sibling_statistic": (
                np.nan
                if inflation_trace is None
                else inflation_trace.raw_sibling_statistic
            ),
            "raw_reference_scale": (
                np.nan if inflation_trace is None else inflation_trace.raw_reference_scale
            ),
            "raw_sibling_p_value": (
                np.nan if inflation_trace is None else inflation_trace.raw_p_value
            ),
            "empirical_inflation_factor": (
                np.nan
                if inflation_trace is None
                else inflation_trace.empirical_inflation_factor
            ),
            "inflation_applied": (
                False if inflation_trace is None else inflation_trace.inflation_applied
            ),
            "sibling_null_weight": (
                np.nan if inflation_trace is None else inflation_trace.sibling_null_weight
            ),
            "sibling_is_null_like": (
                False if inflation_trace is None else inflation_trace.sibling_is_null_like
            ),
            "sibling_is_edge_blocked": (
                False if inflation_trace is None else inflation_trace.sibling_is_edge_blocked
            ),
            "sibling_branch_length_sum": (
                np.nan if inflation_trace is None else inflation_trace.branch_length_sum
            ),
            "sibling_feature_family": (
                "" if inflation_trace is None else inflation_trace.feature_family
            ),
        }
        row.update(_edge_prefix_values(annotations_df, left_child, prefix="left"))
        row.update(_edge_prefix_values(annotations_df, right_child, prefix="right"))
        rows.append(row)

    return pd.DataFrame.from_records(rows)


def summarize_gate_path_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per case from a node-level gate trace."""
    required = {"case_id", "failure_class", "trace_relation", "actual_boundary"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace dataframe is missing required columns: {sorted(missing)}.")
    rows: list[dict[str, object]] = []
    for case_id, group in trace_df.groupby("case_id", sort=False):
        relation_counts = group["trace_relation"].value_counts().to_dict()
        rows.append(
            {
                "case_id": case_id,
                "failure_class": str(group["failure_class"].iloc[0]),
                "kl_ari": float(group["kl_ari"].iloc[0]),
                "oracle_true_k_ari": float(group["oracle_true_k_ari"].iloc[0]),
                "oracle_any_k_ari": float(group["oracle_any_k_ari"].iloc[0]),
                "n_nodes": int(len(group)),
                "n_actual_boundaries": int(group["actual_boundary"].sum()),
                "n_oracle_true_k_boundaries": int(
                    group["oracle_true_k_boundary"].sum()
                ),
                "n_actual_splits_oracle_boundary": int(
                    relation_counts.get("actual_splits_oracle_boundary", 0)
                ),
                "n_actual_stops_above_oracle_boundary": int(
                    relation_counts.get("actual_stops_above_oracle_boundary", 0)
                ),
                "n_actual_fragments_inside_oracle_boundary": int(
                    relation_counts.get("actual_fragment_inside_oracle_boundary", 0)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "SiblingInflationTrace",
    "build_gate_path_trace_dataframe",
    "collect_sibling_inflation_trace",
    "summarize_gate_path_trace",
]
