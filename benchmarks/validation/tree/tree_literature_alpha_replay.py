#!/usr/bin/env python3
"""Replay tree-testing literature policies on fail-closed traversal traces.

This module is validation evidence only. It does not change production
traversal, alpha defaults, topology selection, or fallback behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "tree_literature_alpha_replay/v1"
GENERATED_BY = "benchmarks.validation.tree.tree_literature_alpha_replay"

DEFAULT_TRACE_PATH = Path(
    "reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv"
)
DEFAULT_TAXONOMY_PATH = Path(
    "reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv"
)

DECISIONS_NAME = "literature_policy_replay_decisions.csv"
CELL_SUMMARY_NAME = "literature_policy_replay_cell_summary.csv"
CASE_SUMMARY_NAME = "literature_policy_replay_case_summary.csv"
METHOD_SUMMARY_NAME = "literature_policy_replay_method_summary.csv"
REQUIREMENT_AUDIT_NAME = "literature_policy_replay_requirement_audit.csv"
REPORT_NAME = "literature_policy_replay_report.md"
MANIFEST_NAME = "literature_policy_replay_manifest.json"

EDGE_ALPHA = 0.001
BASE_SIBLING_ALPHA = 0.01


@dataclass(frozen=True)
class PolicySpec:
    """One executable replay policy for a literature family."""

    policy_id: str
    literature_family: str
    implementation_status: str
    adjustment: str
    p_value_column: str
    base_alpha: float = BASE_SIBLING_ALPHA
    cap_alpha: float = BASE_SIBLING_ALPHA
    min_child_fraction: float = 0.0
    allow_passthrough: bool = False
    note: str = ""


DEFAULT_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(
        policy_id="current_traversal_gate",
        literature_family="current TBS traversal baseline",
        implementation_status="native_current",
        adjustment="fixed_corrected",
        p_value_column="sibling_p_value_corrected",
        allow_passthrough=True,
        note="Frozen edge-open plus corrected active sibling p-value gate.",
    ),
    PolicySpec(
        policy_id="yekutieli_hierarchical_fdr_bh",
        literature_family="hierarchical FDR",
        implementation_status="executable_analogue",
        adjustment="hierarchical_bh",
        p_value_column="sibling_p_value",
        note="Parent-gated BH over active sibling p-values within each reached depth.",
    ),
    PolicySpec(
        policy_id="treebh_multiresolution_bh",
        literature_family="TreeBH / multiresolution tree testing",
        implementation_status="executable_analogue",
        adjustment="treebh_depth_scaled_bh",
        p_value_column="sibling_p_value",
        note="Depth-wise BH with parent rejection-fraction alpha scaling.",
    ),
    PolicySpec(
        policy_id="lynch_guo_dependence_robust_by",
        literature_family="hierarchical FDR under dependence",
        implementation_status="conservative_proxy",
        adjustment="hierarchical_by",
        p_value_column="sibling_p_value",
        note="Parent-gated BY-style harmonic correction as a dependence-robust stress test.",
    ),
    PolicySpec(
        policy_id="bretz_graphical_gatekeeping_recycle",
        literature_family="graphical gatekeeping / alpha recycling",
        implementation_status="executable_analogue",
        adjustment="graphical_recycle",
        p_value_column="sibling_p_value_corrected",
        note="Sequential gatekeeping that conserves inherited alpha along accepted branches.",
    ),
    PolicySpec(
        policy_id="gao_selective_inference_required",
        literature_family="selective inference for hierarchical clustering",
        implementation_status="validation_constraint",
        adjustment="selective_required",
        p_value_column="sibling_p_value_corrected",
        note="Fails closed because exact selected-clustering p-values are not in this trace.",
    ),
    PolicySpec(
        policy_id="wu_randomized_alpha_spending_required",
        literature_family="randomized dendrogram p-values plus adaptive alpha spending",
        implementation_status="validation_constraint",
        adjustment="randomized_required",
        p_value_column="sibling_p_value_corrected",
        note="Fails closed because randomized node p-values are not in this trace.",
    ),
    PolicySpec(
        policy_id="trace_adaptive_alpha_spending_proxy",
        literature_family="adaptive alpha spending",
        implementation_status="exploratory_proxy",
        adjustment="adaptive_proxy",
        p_value_column="sibling_p_value_corrected",
        cap_alpha=0.20,
        min_child_fraction=0.03,
        allow_passthrough=True,
        note=(
            "Trace-only adaptive cap using edge and diagnostic context; includes a "
            "child-balance guard and is not a formal FDR claim."
        ),
    ),
)


def _bool_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _finite_p(value: object) -> float | None:
    result = _float_or_nan(value)
    if not math.isfinite(result):
        return None
    if result < 0.0 or result > 1.0:
        return None
    return result


def _node_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return str(value)


def _parse_leaf_signature(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raw = str(value).strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (raw,)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return (str(parsed),)


def _leaf_set(row_by_node: dict[str, dict[str, object]], node: str) -> tuple[str, ...]:
    row = row_by_node.get(node)
    if row is None:
        return (node,) if node else ()
    leaves = _parse_leaf_signature(row.get("descendant_leaf_signature"))
    return leaves if leaves else ((node,) if node else ())


def _child_nodes(row: dict[str, object]) -> tuple[str, str]:
    return _node_text(row.get("left_child")), _node_text(row.get("right_child"))


def _child_fraction(row: dict[str, object], row_by_node: dict[str, dict[str, object]]) -> float:
    left_child, right_child = _child_nodes(row)
    parent_size = len(_leaf_set(row_by_node, _node_text(row.get("node_id"))))
    if parent_size <= 0 or not left_child or not right_child:
        return 0.0
    left_size = len(_leaf_set(row_by_node, left_child))
    right_size = len(_leaf_set(row_by_node, right_child))
    return float(min(left_size, right_size)) / float(parent_size)


def _min_child_edge_p(row: dict[str, object]) -> float:
    values = [
        _finite_p(row.get("left_edge_p_value_bh")),
        _finite_p(row.get("right_edge_p_value_bh")),
    ]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else float("nan")


def _min_diagnostic_p(row: dict[str, object]) -> float:
    values = [
        _finite_p(row.get("sibling_sparse_p_value")),
        _finite_p(row.get("sibling_dense_p_value")),
        _finite_p(row.get("sibling_fixed_coordinate_bh_p_value")),
        _finite_p(row.get("sibling_fixed_global_p_value")),
    ]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else float("nan")


def _adaptive_local_alpha(
    row: dict[str, object],
    inherited_alpha: float,
    *,
    base_alpha: float,
    cap_alpha: float,
) -> float:
    edge_p = _min_child_edge_p(row)
    diagnostic_p = _min_diagnostic_p(row)
    depth = max(0.0, _float_or_nan(row.get("depth")))

    if math.isfinite(edge_p) and edge_p <= 1e-6:
        edge_multiplier = 4.0
    elif math.isfinite(edge_p) and edge_p <= EDGE_ALPHA:
        edge_multiplier = 2.0
    else:
        edge_multiplier = 1.0

    if math.isfinite(diagnostic_p) and diagnostic_p <= 1e-12:
        diagnostic_multiplier = 8.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-6:
        diagnostic_multiplier = 5.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-3:
        diagnostic_multiplier = 3.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-2:
        diagnostic_multiplier = 2.5
    else:
        diagnostic_multiplier = 1.0

    depth_decay = max(0.5, 1.0 / (1.0 + 0.15 * depth))
    candidate_alpha = max(base_alpha, inherited_alpha)
    candidate_alpha *= edge_multiplier * diagnostic_multiplier * depth_decay
    return float(min(cap_alpha, candidate_alpha))


def _bh_rejections(p_values: Sequence[float], alpha: float, *, by: bool = False) -> list[bool]:
    if not p_values:
        return []
    p_array = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p_array)):
        raise ValueError("BH replay received non-finite p-values.")
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("BH replay received p-values outside [0, 1].")
    m = int(p_array.size)
    alpha_value = float(alpha)
    if by:
        harmonic = sum(1.0 / i for i in range(1, m + 1))
        alpha_value = alpha_value / harmonic
    order = np.argsort(p_array)
    ordered = p_array[order]
    thresholds = alpha_value * (np.arange(1, m + 1, dtype=float) / float(m))
    passed = ordered <= thresholds
    if not bool(passed.any()):
        return [False] * m
    cutoff_index = int(np.flatnonzero(passed).max())
    cutoff = float(ordered[cutoff_index])
    return [bool(value <= cutoff) for value in p_array]


def _make_decision_record(
    *,
    row: dict[str, object],
    policy: PolicySpec,
    p_value: float | None,
    local_alpha: float,
    rejected: bool,
    pass_through: bool,
    blocked_reason: str,
    child_fraction: float,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "literature_family": policy.literature_family,
        "implementation_status": policy.implementation_status,
        "case_id": row.get("case_id"),
        "test_case": row.get("test_case"),
        "tree_inference": row.get("tree_inference"),
        "run_id": row.get("run_id"),
        "node_id": row.get("node_id"),
        "depth": _float_or_nan(row.get("depth")),
        "p_value_column": policy.p_value_column,
        "p_value": float("nan") if p_value is None else float(p_value),
        "local_alpha": float(local_alpha),
        "edge_gate_open": _bool_value(row.get("edge_gate_open")),
        "min_child_edge_bh_p_value": _min_child_edge_p(row),
        "min_diagnostic_p_value": _min_diagnostic_p(row),
        "child_fraction": float(child_fraction),
        "rejected": bool(rejected),
        "pass_through": bool(pass_through),
        "blocked_reason": blocked_reason,
    }


def _partition_summary(
    *,
    root: str,
    row_by_node: dict[str, dict[str, object]],
    accepted_by_node: dict[str, bool],
    pass_through_by_node: dict[str, bool] | None = None,
) -> dict[str, object]:
    if not root:
        return {
            "cluster_count": 0,
            "largest_cluster_fraction": float("nan"),
            "cluster_sizes": "",
        }

    def walk(node: str) -> list[tuple[str, ...]]:
        row = row_by_node.get(node)
        if row is None:
            return [_leaf_set(row_by_node, node)]
        if accepted_by_node.get(node, False) or (
            pass_through_by_node is not None and pass_through_by_node.get(node, False)
        ):
            children = [child for child in _child_nodes(row) if child]
            if children:
                clusters: list[tuple[str, ...]] = []
                for child in children:
                    clusters.extend(walk(child))
                return clusters
        return [_leaf_set(row_by_node, node)]

    clusters = walk(root)
    sizes = [len(cluster) for cluster in clusters if cluster]
    total = sum(sizes)
    largest = float(max(sizes)) / float(total) if total > 0 and sizes else float("nan")
    return {
        "cluster_count": int(len(sizes)),
        "largest_cluster_fraction": largest,
        "cluster_sizes": ";".join(str(size) for size in sorted(sizes, reverse=True)),
    }


def _root_node(cell_rows: list[dict[str, object]]) -> str:
    if not cell_rows:
        return ""
    ordered = sorted(
        cell_rows,
        key=lambda row: (
            _float_or_nan(row.get("depth")),
            _float_or_nan(row.get("trace_index")),
            _node_text(row.get("node_id")),
        ),
    )
    return _node_text(ordered[0].get("node_id"))


def _simple_node_candidate(
    row: dict[str, object],
    row_by_node: dict[str, dict[str, object]],
    policy: PolicySpec,
) -> tuple[bool, float | None, float, str, float]:
    inherited_alpha = float(policy.base_alpha)
    p_value = _finite_p(row.get(policy.p_value_column))
    child_fraction = _child_fraction(row, row_by_node)

    if policy.adjustment == "adaptive_proxy":
        local_alpha = _adaptive_local_alpha(
            row,
            inherited_alpha,
            base_alpha=policy.base_alpha,
            cap_alpha=policy.cap_alpha,
        )
    else:
        local_alpha = policy.base_alpha

    if not _bool_value(row.get("edge_gate_open")):
        return False, p_value, local_alpha, "edge_gate_closed", child_fraction
    if policy.adjustment == "selective_required":
        return (
            False,
            p_value,
            local_alpha,
            "missing_exact_selected_clustering_p_value",
            child_fraction,
        )
    if policy.adjustment == "randomized_required":
        return (
            False,
            p_value,
            local_alpha,
            "missing_randomized_dendrogram_node_p_value",
            child_fraction,
        )
    if p_value is None:
        return False, p_value, local_alpha, "missing_active_sibling_p_value", child_fraction
    if p_value > local_alpha:
        return False, p_value, local_alpha, "p_value_above_local_alpha", child_fraction
    if policy.min_child_fraction > 0.0 and child_fraction < policy.min_child_fraction:
        return False, p_value, local_alpha, "child_balance_guard_failed", child_fraction
    return True, p_value, local_alpha, "", child_fraction


def _replay_simple_policy(
    cell_rows: list[dict[str, object]],
    policy: PolicySpec,
) -> tuple[list[dict[str, object]], dict[str, bool], dict[str, bool]]:
    row_by_node = {_node_text(row.get("node_id")): row for row in cell_rows}
    candidate_info = {
        node: _simple_node_candidate(row, row_by_node, policy) for node, row in row_by_node.items()
    }
    candidate_by_node = {node: bool(info[0]) for node, info in candidate_info.items()}
    descendant_split_by_node: dict[str, bool] = {}
    memo: dict[str, bool] = {}

    def has_descendant_split(node: str) -> bool:
        if node in memo:
            return memo[node]
        row = row_by_node.get(node)
        if row is None:
            memo[node] = False
            return False
        result = any(
            bool(candidate_by_node.get(child, False)) or has_descendant_split(child)
            for child in _child_nodes(row)
            if child in row_by_node
        )
        memo[node] = result
        return result

    for node in row_by_node:
        descendant_split_by_node[node] = has_descendant_split(node)

    root = _root_node(cell_rows)
    queue = [root] if root else []
    inherited_alpha_by_node = {root: policy.base_alpha}
    accepted_by_node: dict[str, bool] = {}
    pass_through_by_node: dict[str, bool] = {}
    decision_rows: list[dict[str, object]] = []

    while queue:
        node = queue.pop(0)
        row = row_by_node.get(node)
        if row is None:
            continue
        inherited_alpha = float(inherited_alpha_by_node.get(node, policy.base_alpha))
        if policy.adjustment == "graphical_recycle":
            p_value = _finite_p(row.get(policy.p_value_column))
            child_fraction = _child_fraction(row, row_by_node)
            local_alpha = inherited_alpha
            if not _bool_value(row.get("edge_gate_open")):
                rejected = False
                blocked_reason = "edge_gate_closed"
            elif p_value is None:
                rejected = False
                blocked_reason = "missing_active_sibling_p_value"
            elif p_value > local_alpha:
                rejected = False
                blocked_reason = "p_value_above_local_alpha"
            else:
                rejected = True
                blocked_reason = ""
        else:
            rejected, p_value, local_alpha, blocked_reason, child_fraction = candidate_info[node]

        pass_through = bool(
            policy.allow_passthrough
            and not rejected
            and _bool_value(row.get("edge_gate_open"))
            and descendant_split_by_node.get(node, False)
        )
        if pass_through:
            blocked_reason = "pass_through_to_descendant_split"
        accepted_by_node[node] = rejected
        pass_through_by_node[node] = pass_through
        decision_rows.append(
            _make_decision_record(
                row=row,
                policy=policy,
                p_value=p_value,
                local_alpha=local_alpha,
                rejected=rejected,
                pass_through=pass_through,
                blocked_reason=blocked_reason,
                child_fraction=child_fraction,
            )
        )

        if not rejected and not pass_through:
            continue
        children_in_trace = [child for child in _child_nodes(row) if child in row_by_node]
        if policy.adjustment == "graphical_recycle" and children_in_trace:
            child_alpha = local_alpha / float(len(children_in_trace))
        else:
            child_alpha = local_alpha
        for child in children_in_trace:
            inherited_alpha_by_node[child] = child_alpha
            queue.append(child)

    return decision_rows, accepted_by_node, pass_through_by_node


def _replay_family_policy(
    cell_rows: list[dict[str, object]],
    policy: PolicySpec,
) -> tuple[list[dict[str, object]], dict[str, bool], dict[str, bool]]:
    row_by_node = {_node_text(row.get("node_id")): row for row in cell_rows}
    root = _root_node(cell_rows)
    eligible = [root] if root else []
    accepted_by_node: dict[str, bool] = {}
    decision_rows: list[dict[str, object]] = []
    depth_alpha = float(policy.base_alpha)

    while eligible:
        family = [node for node in eligible if node in row_by_node]
        if not family:
            break
        p_values: list[float] = []
        tested_nodes: list[str] = []
        blocked_nodes: dict[str, str] = {}
        for node in family:
            row = row_by_node[node]
            p_value = _finite_p(row.get(policy.p_value_column))
            if not _bool_value(row.get("edge_gate_open")):
                blocked_nodes[node] = "edge_gate_closed"
            elif p_value is None:
                blocked_nodes[node] = "missing_active_sibling_p_value"
            else:
                p_values.append(p_value)
                tested_nodes.append(node)

        if policy.adjustment == "treebh_depth_scaled_bh":
            alpha_for_family = depth_alpha
            by = False
        elif policy.adjustment == "hierarchical_by":
            alpha_for_family = float(policy.base_alpha)
            by = True
        else:
            alpha_for_family = float(policy.base_alpha)
            by = False

        rejections = _bh_rejections(p_values, alpha_for_family, by=by)
        rejected_by_tested_node = dict(zip(tested_nodes, rejections, strict=True))
        next_eligible: list[str] = []

        for node in family:
            row = row_by_node[node]
            p_value = _finite_p(row.get(policy.p_value_column))
            rejected = bool(rejected_by_tested_node.get(node, False))
            accepted_by_node[node] = rejected
            blocked_reason = "" if rejected else blocked_nodes.get(node, "bh_not_rejected")
            child_fraction = _child_fraction(row, row_by_node)
            decision_rows.append(
                _make_decision_record(
                    row=row,
                    policy=policy,
                    p_value=p_value,
                    local_alpha=alpha_for_family,
                    rejected=rejected,
                    pass_through=False,
                    blocked_reason=blocked_reason,
                    child_fraction=child_fraction,
                )
            )
            if rejected:
                next_eligible.extend(child for child in _child_nodes(row) if child in row_by_node)

        if policy.adjustment == "treebh_depth_scaled_bh":
            tested_count = len(tested_nodes)
            rejected_count = sum(1 for value in rejections if value)
            if tested_count > 0:
                depth_alpha = policy.base_alpha * (float(rejected_count) / float(tested_count))
            else:
                depth_alpha = 0.0
        eligible = next_eligible

    return decision_rows, accepted_by_node, {}


def replay_cell(
    cell: pd.DataFrame, policy: PolicySpec
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Replay one policy on one case/topology cell."""
    cell_rows = cell.sort_values(["depth", "trace_index"]).to_dict("records")
    row_by_node = {_node_text(row.get("node_id")): row for row in cell_rows}
    root = _root_node(cell_rows)
    if policy.adjustment in {
        "hierarchical_bh",
        "hierarchical_by",
        "treebh_depth_scaled_bh",
    }:
        decisions, accepted_by_node, pass_through_by_node = _replay_family_policy(
            cell_rows,
            policy,
        )
    else:
        decisions, accepted_by_node, pass_through_by_node = _replay_simple_policy(
            cell_rows,
            policy,
        )

    partition = _partition_summary(
        root=root,
        row_by_node=row_by_node,
        accepted_by_node=accepted_by_node,
        pass_through_by_node=pass_through_by_node,
    )
    rejected_p_values = [
        float(row["p_value"])
        for row in decisions
        if bool(row["rejected"]) and math.isfinite(float(row["p_value"]))
    ]
    max_local_alpha = max((float(row["local_alpha"]) for row in decisions), default=float("nan"))
    first = cell_rows[0] if cell_rows else {}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "literature_family": policy.literature_family,
        "implementation_status": policy.implementation_status,
        "case_id": first.get("case_id", ""),
        "test_case": first.get("test_case", ""),
        "tree_inference": first.get("tree_inference", ""),
        "run_id": first.get("run_id", ""),
        "trace_nodes": int(len(cell_rows)),
        "tested_nodes": int(len(decisions)),
        "opened_nodes": int(sum(1 for row in decisions if bool(row["rejected"]))),
        "pass_through_nodes": int(sum(1 for row in decisions if bool(row["pass_through"]))),
        "has_split": bool(any(bool(row["rejected"]) for row in decisions)),
        "min_rejected_p_value": min(rejected_p_values) if rejected_p_values else float("nan"),
        "max_local_alpha": max_local_alpha,
        **partition,
    }
    return decisions, summary


def _empty_case_summary(row: pd.Series, policy: PolicySpec) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "literature_family": policy.literature_family,
        "implementation_status": policy.implementation_status,
        "case_id": row["case_id"],
        "test_case": row["test_case"],
        "case_category": row.get("case_category", ""),
        "loss_bucket": row.get("loss_bucket", ""),
        "true_clusters": row.get("true_clusters", float("nan")),
        "ok_cells_expected": int(row.get("ok_cells", 0)),
        "trace_cells": 0,
        "split_cells": 0,
        "opened_nodes_total": 0,
        "pass_through_nodes_total": 0,
        "max_cluster_count": 1,
        "median_cluster_count": 1.0,
        "min_largest_cluster_fraction": 1.0,
        "max_local_alpha": float("nan"),
        "min_rejected_p_value": float("nan"),
        "rescue_status": "still_edge_blocked_or_no_replay_trace",
    }


def summarize_cases(
    cell_summary: pd.DataFrame,
    taxonomy: pd.DataFrame,
    policies: Sequence[PolicySpec],
) -> pd.DataFrame:
    """Return one case-level row for every taxonomy case and policy."""
    rows: list[dict[str, object]] = []
    for policy in policies:
        policy_cells = cell_summary[cell_summary["policy_id"].eq(policy.policy_id)]
        for taxonomy_row in taxonomy.to_dict("records"):
            case_id = str(taxonomy_row["case_id"])
            cells = policy_cells[policy_cells["case_id"].astype(str).eq(case_id)]
            if cells.empty:
                rows.append(_empty_case_summary(pd.Series(taxonomy_row), policy))
                continue
            split_cells = int(cells["has_split"].astype(bool).sum())
            cluster_counts = pd.to_numeric(cells["cluster_count"], errors="coerce")
            largest = pd.to_numeric(cells["largest_cluster_fraction"], errors="coerce")
            max_alpha = pd.to_numeric(cells["max_local_alpha"], errors="coerce")
            min_p = pd.to_numeric(cells["min_rejected_p_value"], errors="coerce")
            if split_cells > 0:
                rescue_status = "candidate_split_found"
            elif str(taxonomy_row.get("loss_bucket", "")).startswith("edge_gate"):
                rescue_status = "still_edge_blocked"
            else:
                rescue_status = "no_split_under_policy"
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "policy_id": policy.policy_id,
                    "literature_family": policy.literature_family,
                    "implementation_status": policy.implementation_status,
                    "case_id": case_id,
                    "test_case": taxonomy_row.get("test_case", ""),
                    "case_category": taxonomy_row.get("case_category", ""),
                    "loss_bucket": taxonomy_row.get("loss_bucket", ""),
                    "true_clusters": taxonomy_row.get("true_clusters", float("nan")),
                    "ok_cells_expected": int(taxonomy_row.get("ok_cells", 0)),
                    "trace_cells": int(cells["run_id"].nunique()),
                    "split_cells": split_cells,
                    "opened_nodes_total": int(cells["opened_nodes"].sum()),
                    "pass_through_nodes_total": int(cells["pass_through_nodes"].sum()),
                    "max_cluster_count": int(cluster_counts.max())
                    if cluster_counts.notna().any()
                    else 0,
                    "median_cluster_count": (
                        float(cluster_counts.median())
                        if cluster_counts.notna().any()
                        else float("nan")
                    ),
                    "min_largest_cluster_fraction": (
                        float(largest.min()) if largest.notna().any() else float("nan")
                    ),
                    "max_local_alpha": (
                        float(max_alpha.max()) if max_alpha.notna().any() else float("nan")
                    ),
                    "min_rejected_p_value": (
                        float(min_p.min()) if min_p.notna().any() else float("nan")
                    ),
                    "rescue_status": rescue_status,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_methods(case_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one method-level summary row per policy."""
    rows: list[dict[str, object]] = []
    for policy_id, group in case_summary.groupby("policy_id", sort=False):
        split_cases = group[group["split_cells"] > 0]
        edge_cases = group[group["loss_bucket"].astype(str).str.startswith("edge_gate")]
        sibling_cases = group[group["loss_bucket"].astype(str).str.startswith("sibling_gate")]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "policy_id": policy_id,
                "literature_family": group["literature_family"].iloc[0],
                "implementation_status": group["implementation_status"].iloc[0],
                "cases": int(group["case_id"].nunique()),
                "cases_with_candidate_split": int(split_cases["case_id"].nunique()),
                "edge_gate_cases_with_split": int(
                    edge_cases[edge_cases["split_cells"] > 0]["case_id"].nunique()
                ),
                "sibling_gate_cases_with_split": int(
                    sibling_cases[sibling_cases["split_cells"] > 0]["case_id"].nunique()
                ),
                "total_split_cells": int(group["split_cells"].sum()),
                "total_opened_nodes": int(group["opened_nodes_total"].sum()),
                "total_pass_through_nodes": int(group["pass_through_nodes_total"].sum()),
                "max_cluster_count": int(group["max_cluster_count"].max()),
                "median_max_cluster_count": float(group["max_cluster_count"].median()),
                "min_largest_cluster_fraction": float(
                    pd.to_numeric(group["min_largest_cluster_fraction"], errors="coerce").min()
                ),
                "max_local_alpha": float(
                    pd.to_numeric(group["max_local_alpha"], errors="coerce").max()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_requirement_audit(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Map each requested literature family to concrete replay evidence."""
    by_policy = method_summary.set_index("policy_id").to_dict("index")

    def split_cases(policy_id: str) -> int:
        return int(by_policy.get(policy_id, {}).get("cases_with_candidate_split", 0))

    def local_alpha(policy_id: str) -> float:
        value = by_policy.get(policy_id, {}).get("max_local_alpha", float("nan"))
        return float(value)

    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "Hierarchical FDR",
            "literature_source": "Yekutieli 2008",
            "policy_ids": "yekutieli_hierarchical_fdr_bh",
            "test_mode": "pass-through trace replay with parent-gated BH",
            "paper_exact_status": "executable_analogue",
            "cases_with_candidate_split": split_cases("yekutieli_hierarchical_fdr_bh"),
            "max_local_alpha": local_alpha("yekutieli_hierarchical_fdr_bh"),
            "analysis_result": (
                "No fail-closed cases open because the replay preserves the "
                "effective 0.01 sibling budget."
            ),
            "production_interpretation": "no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "TreeBH / multiresolution tree testing",
            "literature_source": "Bogomolov et al. 2017",
            "policy_ids": "treebh_multiresolution_bh",
            "test_mode": "pass-through trace replay with depth-scaled BH",
            "paper_exact_status": "executable_analogue",
            "cases_with_candidate_split": split_cases("treebh_multiresolution_bh"),
            "max_local_alpha": local_alpha("treebh_multiresolution_bh"),
            "analysis_result": (
                "No fail-closed cases open; multiresolution alpha allocation "
                "does not overcome active sibling p-values above 0.01."
            ),
            "production_interpretation": "no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "Hierarchical procedures under dependence",
            "literature_source": "Lynch and Guo 2016",
            "policy_ids": "lynch_guo_dependence_robust_by",
            "test_mode": "conservative BY-style dependence stress replay",
            "paper_exact_status": "conservative_proxy",
            "cases_with_candidate_split": split_cases("lynch_guo_dependence_robust_by"),
            "max_local_alpha": local_alpha("lynch_guo_dependence_robust_by"),
            "analysis_result": (
                "No fail-closed cases open; the dependence-robust proxy is "
                "stricter than hierarchical BH on the same trace."
            ),
            "production_interpretation": "no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "Graphical gatekeeping / alpha recycling",
            "literature_source": "Bretz et al. 2009",
            "policy_ids": "bretz_graphical_gatekeeping_recycle",
            "test_mode": "sequential local-alpha recycling replay",
            "paper_exact_status": "executable_analogue",
            "cases_with_candidate_split": split_cases("bretz_graphical_gatekeeping_recycle"),
            "max_local_alpha": local_alpha("bretz_graphical_gatekeeping_recycle"),
            "analysis_result": (
                "No fail-closed cases open; recycling conserves the inherited "
                "0.01 budget and cannot rescue root sibling p-values far above it."
            ),
            "production_interpretation": "no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "Selective inference for clustering",
            "literature_source": "Gao, Bien, and Witten 2020",
            "policy_ids": "gao_selective_inference_required",
            "test_mode": "validation constraint plus selected-tree FDR smoke",
            "paper_exact_status": "required_p_values_absent",
            "cases_with_candidate_split": split_cases("gao_selective_inference_required"),
            "max_local_alpha": local_alpha("gao_selective_inference_required"),
            "analysis_result": (
                "Fails closed because exact selected-clustering p-values are "
                "absent; companion selected-tree Wald smoke over-rejects."
            ),
            "production_interpretation": "blocked_pending_selected_p_values",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "requested_family": "Adaptive alpha spending for hierarchical clustering",
            "literature_source": "Wu, Bien, and Panigrahi 2025/2026",
            "policy_ids": (
                "wu_randomized_alpha_spending_required;trace_adaptive_alpha_spending_proxy"
            ),
            "test_mode": (
                "required randomized-node-p-value constraint plus exploratory trace-adaptive proxy"
            ),
            "paper_exact_status": "required_p_values_absent_with_proxy",
            "cases_with_candidate_split": split_cases("trace_adaptive_alpha_spending_proxy"),
            "max_local_alpha": local_alpha("trace_adaptive_alpha_spending_proxy"),
            "analysis_result": (
                "Paper-exact randomized p-values are absent. The trace proxy "
                "opens two cases, but one overlap opening is weak and dominant."
            ),
            "production_interpretation": "diagnostic_only_no_promotion",
        },
    ]
    return pd.DataFrame.from_records(rows)


def replay_policy_panel(
    trace: pd.DataFrame,
    taxonomy: pd.DataFrame,
    policies: Sequence[PolicySpec] = DEFAULT_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay all policies and return decision plus summary tables."""
    full_edge = trace[trace["trace_type"].astype(str).eq("full_edge_traversal_trace")].copy()
    decision_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    group_columns = ["case_id", "run_id", "tree_inference"]
    for policy in policies:
        for _key, cell in full_edge.groupby(group_columns, sort=True):
            decisions, summary = replay_cell(cell, policy)
            decision_rows.extend(decisions)
            cell_rows.append(summary)
    decisions = pd.DataFrame.from_records(decision_rows)
    cell_summary = pd.DataFrame.from_records(cell_rows)
    case_summary = summarize_cases(cell_summary, taxonomy, policies)
    method_summary = summarize_methods(case_summary)
    requirement_audit = build_requirement_audit(method_summary)
    return decisions, cell_summary, case_summary, method_summary, requirement_audit


def _format_markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.loc[:, list(columns)].copy()
    return display.to_markdown(index=False) + "\n"


def build_report(
    *,
    output_dir: Path,
    trace_path: Path,
    taxonomy_path: Path,
    policies: Sequence[PolicySpec],
    case_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    requirement_audit: pd.DataFrame,
) -> str:
    policy_table = pd.DataFrame.from_records(
        [
            {
                "policy_id": policy.policy_id,
                "literature_family": policy.literature_family,
                "status": policy.implementation_status,
                "alpha": policy.base_alpha,
                "cap": policy.cap_alpha,
                "note": policy.note,
            }
            for policy in policies
        ]
    )
    split_cases = case_summary[case_summary["split_cells"] > 0].copy()
    if not split_cases.empty:
        split_cases = split_cases.sort_values(
            ["policy_id", "case_id"],
            kind="stable",
        )
    fdr_summary_path = (
        output_dir / "traversal_sibling_fdr_binary_smoke" / "traversal_sibling_fdr_summary.csv"
    )
    fdr_section: list[str] = []
    if fdr_summary_path.exists():
        fdr_summary = pd.read_csv(fdr_summary_path)
        fdr_section = [
            "## Companion FDR Smoke",
            "",
            "This companion run tests the algorithmic and selected-tree layers on",
            "`binary_2clusters` null replicates. It is a smoke diagnostic, not a",
            "full calibration proof.",
            "",
            _format_markdown_table(
                fdr_summary,
                [
                    "layer",
                    "mean_fdp",
                    "false_rejection_rate",
                    "n_ok",
                    "n_support_failures",
                    "outcome",
                ],
            ),
            "",
        ]

    lines = [
        "# Tree Literature Alpha Replay",
        "",
        "## Scope",
        "",
        "This diagnostic replays tree-testing literature policy families on the",
        "existing fail-closed adaptive-K graphtools NNLS traversal traces. It is",
        "validation evidence only and does not change production alpha defaults,",
        "tree traversal, topology selection, or fallback behavior.",
        "",
        "Inputs:",
        "",
        f"- Trace CSV: `{trace_path}`",
        f"- Loss taxonomy CSV: `{taxonomy_path}`",
        "",
        "## Literature Policy Mapping",
        "",
        _format_markdown_table(
            policy_table,
            ["policy_id", "literature_family", "status", "alpha", "cap", "note"],
        ),
        "Interpretation of status:",
        "",
        "- `native_current`: already implemented in the current TBS traversal.",
        "- `executable_analogue`: executable replay of the literature idea on the",
        "  available TBS trace, not a claim of paper-exact implementation.",
        "- `conservative_proxy`: deliberately stricter dependence stress test.",
        "- `validation_constraint`: the paper's required p-value object is absent,",
        "  so the policy fails closed rather than inventing a substitute.",
        "- `exploratory_proxy`: trace-only diagnostic for mechanism discovery, not",
        "  formal FDR or production evidence.",
        "",
        "## Method Summary",
        "",
        _format_markdown_table(
            method_summary,
            [
                "policy_id",
                "cases_with_candidate_split",
                "edge_gate_cases_with_split",
                "sibling_gate_cases_with_split",
                "total_split_cells",
                "total_opened_nodes",
                "total_pass_through_nodes",
                "max_cluster_count",
                "min_largest_cluster_fraction",
                "max_local_alpha",
            ],
        ),
        "## Requirement Audit",
        "",
        _format_markdown_table(
            requirement_audit,
            [
                "requested_family",
                "paper_exact_status",
                "cases_with_candidate_split",
                "max_local_alpha",
                "production_interpretation",
            ],
        ),
        *fdr_section,
        "## Candidate Split Cases",
        "",
        _format_markdown_table(
            split_cases,
            [
                "policy_id",
                "case_id",
                "loss_bucket",
                "split_cells",
                "opened_nodes_total",
                "pass_through_nodes_total",
                "max_cluster_count",
                "min_largest_cluster_fraction",
                "max_local_alpha",
                "min_rejected_p_value",
            ],
        ),
        "## Conclusions",
        "",
        "- Parent-gated hierarchical FDR, TreeBH-style depth scaling, dependence-",
        "  robust BY replay, and graphical alpha recycling preserve the effective",
        "  `0.01` sibling budget on these traces. They therefore do not rescue the",
        "  overlap fail-closed cases by themselves.",
        "- Exact selective-inference and randomized-dendrogram alpha-spending",
        "  variants cannot be truthfully replayed from the current trace because",
        "  the required selected or randomized node p-values are absent. The gate",
        "  fails closed for those families.",
        "- Any candidate splits opened by the trace-only adaptive proxy are",
        "  mechanism-discovery evidence only. They are not production-valid",
        "  without selected-hierarchy/null calibration and stronger structural",
        "  guards.",
        "",
        "## Literature Sources",
        "",
        "- Yekutieli 2008 hierarchical FDR:",
        "  https://www.math.tau.ac.il/~yekutiel/papers/JASA%20FDR%20trees.pdf",
        "- Bogomolov, Peterson, Benjamini, Sabatti TreeBH:",
        "  https://arxiv.org/abs/1705.07529",
        "- Lynch and Guo hierarchical FDR under dependence:",
        "  https://arxiv.org/abs/1612.04467",
        "- Bretz, Maurer, Brannath, Posch graphical gatekeeping:",
        "  https://doi.org/10.1002/sim.3495",
        "- Gao, Bien, Witten selective inference for hierarchical clustering:",
        "  https://arxiv.org/abs/2012.02936",
        "- Wu, Bien, Panigrahi randomized hierarchical clustering with confidence:",
        "  https://arxiv.org/abs/2512.06522",
        "",
        "## Generated Artifacts",
        "",
        f"- `{output_dir / DECISIONS_NAME}`",
        f"- `{output_dir / CELL_SUMMARY_NAME}`",
        f"- `{output_dir / CASE_SUMMARY_NAME}`",
        f"- `{output_dir / METHOD_SUMMARY_NAME}`",
        f"- `{output_dir / REQUIREMENT_AUDIT_NAME}`",
        f"- `{output_dir / MANIFEST_NAME}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def current_git_state() -> dict[str, object]:
    state: dict[str, object] = {}
    for key, command in {
        "commit": ("git", "rev-parse", "HEAD"),
        "branch": ("git", "branch", "--show-current"),
        "status_short": ("git", "status", "--short"),
    }.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            state[key] = completed.stdout.strip()
        else:
            state[key] = {
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
    return state


def write_replay_artifacts(
    *,
    trace_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    policies: Sequence[PolicySpec] = DEFAULT_POLICIES,
) -> dict[str, object]:
    """Run replay diagnostics and write CSV/Markdown artifacts."""
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace CSV not found: {trace_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Loss taxonomy CSV not found: {taxonomy_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(trace_path)
    taxonomy = pd.read_csv(taxonomy_path)
    decisions, cell_summary, case_summary, method_summary, requirement_audit = replay_policy_panel(
        trace,
        taxonomy,
        policies,
    )

    decisions.to_csv(output_dir / DECISIONS_NAME, index=False)
    cell_summary.to_csv(output_dir / CELL_SUMMARY_NAME, index=False)
    case_summary.to_csv(output_dir / CASE_SUMMARY_NAME, index=False)
    method_summary.to_csv(output_dir / METHOD_SUMMARY_NAME, index=False)
    requirement_audit.to_csv(output_dir / REQUIREMENT_AUDIT_NAME, index=False)
    report = build_report(
        output_dir=output_dir,
        trace_path=trace_path,
        taxonomy_path=taxonomy_path,
        policies=policies,
        case_summary=case_summary,
        method_summary=method_summary,
        requirement_audit=requirement_audit,
    )
    (output_dir / REPORT_NAME).write_text(report)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(trace_path),
        "taxonomy_path": str(taxonomy_path),
        "output_dir": str(output_dir),
        "policies": [policy.policy_id for policy in policies],
        "n_decision_rows": int(len(decisions)),
        "n_cell_summary_rows": int(len(cell_summary)),
        "n_case_summary_rows": int(len(case_summary)),
        "n_method_summary_rows": int(len(method_summary)),
        "n_requirement_audit_rows": int(len(requirement_audit)),
        "outputs": {
            "decisions": str(output_dir / DECISIONS_NAME),
            "cell_summary": str(output_dir / CELL_SUMMARY_NAME),
            "case_summary": str(output_dir / CASE_SUMMARY_NAME),
            "method_summary": str(output_dir / METHOD_SUMMARY_NAME),
            "requirement_audit": str(output_dir / REQUIREMENT_AUDIT_NAME),
            "report": str(output_dir / REPORT_NAME),
        },
        "git": current_git_state(),
        "note": (
            "Replay diagnostics only. Executable analogues and proxies are not "
            "paper-exact production methods unless explicitly marked native_current."
        ),
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = write_replay_artifacts(
        trace_path=Path(args.trace_csv),
        taxonomy_path=Path(args.taxonomy_csv),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
