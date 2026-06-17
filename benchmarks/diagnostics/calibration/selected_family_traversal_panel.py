"""Selected-family traversal diagnostics with multi-scale outputs.

This panel is diagnostic-only. It compares the default traversal with fixed
coordinate selected-family profiles and writes node, region, and sample-level
views so downstream analysis does not have to collapse the hierarchy to one
flat clustering.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    SIBLING_GATE_PROFILES,
    resolve_sibling_gate_profile,
)
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    DEFAULT_DATA_ROLES,
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
    _mean_lower_bound,
    _wilson_upper_bound,
)
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

STUDY_ROLE = "diagnostic_selected_family_traversal_not_calibration"
SCHEMA_VERSION = "selected_family_traversal_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected_family_traversal_panel"
BASELINE_METHOD_ID = "baseline_projected_wald"
NULL_OUTPUT_ROLE = "selected_null"
SIGNAL_OUTPUT_ROLE = "signal"
DEFAULT_METHODS = (
    BASELINE_METHOD_ID,
    "fixed_coordinate_guarded_v1",
    "fixed_coordinate_selective_root_v1",
    "fixed_coordinate_selective_passthrough_v1",
    "fixed_coordinate_global_passthrough_refined_v1",
)

TRAVERSAL_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "source_family",
    "feature_representation",
    "method_id",
    "profile_id",
    "sibling_gate_method",
    "edge_alpha",
    "sibling_alpha",
    "replicate",
    "data_seed",
    "true_clusters",
    "found_clusters",
    "ari",
    "exact_cluster_count",
    "false_split",
    "adaptive_projection_avoided",
    "stable_boundary_count",
    "selected_root_blocked_count",
    "selected_family_blocked_count",
    "unstable_passthrough_zone_count",
    "accepted_internal_split_count",
    "leaf_fragment_count",
    "selected_family_guard_tested_count",
    "selected_family_guard_block_count",
    "spectral_transport_blocked_count",
    "min_selected_family_p_value",
)

NODE_DECISION_COLUMNS = (
    "schema_version",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
    "node_id",
    "parent_id",
    "depth",
    "visited",
    "traversal_decision",
    "decision_class",
    "n_children",
    "n_descendant_leaves",
    "child_parent_edge_open",
    "sibling_open",
    "sibling_p_value",
    "sibling_projection_dimension",
    "root_stability_guard_blocked",
    "root_selective_guard_blocked",
    "selected_family_guard_blocked",
    "selected_family_p_value",
    "selected_family_base_p_value",
    "selected_family_guard_refined",
    "selected_family_scope",
    "topology_incidence_role",
    "topology_has_incoming_edge",
    "topology_has_outgoing_test",
    "topology_directed_degree",
    "topology_pass_through_candidate",
    "conditional_topology_status",
    "conditional_topology_log_odds",
    "conditional_topology_probability",
    "spectral_transport_node_mp_block_count",
    "spectral_transport_node_mp_total_multiplicity",
    "spectral_transport_child_best_mode_cost",
    "spectral_transport_child_best_mode_affinity",
    "spectral_transport_child_best_mode_status",
    "spectral_transport_best_descendant_split_path_cost",
    "spectral_transport_best_descendant_split_path_has_mp_evidence",
    "spectral_transport_passthrough_supported",
    "spectral_transport_passthrough_blocked",
    "spectral_transport_bottleneck",
    "study_role",
)

GUARD_COLUMNS = (
    "schema_version",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
    "node_id",
    "decision_class",
    "guard_scope",
    "observed_p_value",
    "selected_p_value",
    "base_selected_p_value",
    "null_min_p_value",
    "null_q05_p_value",
    "guard_alpha",
    "guard_replicates",
    "guard_seed",
    "guard_refined",
    "guard_would_block",
    "guard_blocked",
    "study_role",
)


@dataclass(frozen=True)
class SelectedFamilyTraversalPanelConfig:
    """Runtime contract for selected-family traversal diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    methods: tuple[str, ...]
    data_roles: tuple[str, ...]
    sibling_alpha: float
    edge_alpha: float
    replicates: int
    base_seed: int
    max_null_split_rate: float = 0.05
    min_signal_mean_ari: float = 0.75
    skip_unsupported_cases: bool = True
    resume_from_checkpoints: bool = False
    per_row_timeout_seconds: float | None = None

    @property
    def traversal_rows_path(self) -> Path:
        return self.output_dir / "selected_family_traversal_rows.csv"

    @property
    def checkpoint_rows_dir(self) -> Path:
        return self.output_dir / "checkpoint_rows"

    @property
    def guard_rows_path(self) -> Path:
        return self.output_dir / "selected_family_guard_rows.csv"

    @property
    def node_decisions_path(self) -> Path:
        return self.output_dir / "multiscale_node_decisions.csv"

    @property
    def regions_path(self) -> Path:
        return self.output_dir / "multiscale_regions.csv"

    @property
    def gene_assignments_path(self) -> Path:
        return self.output_dir / "multiscale_gene_assignments.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    """Validate selected-family traversal method identifiers."""
    values = tuple(str(method) for method in methods)
    if not values:
        raise ValueError("At least one traversal method is required.")
    allowed = {BASELINE_METHOD_ID, *SIBLING_GATE_PROFILES}
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(
            f"Unknown traversal method(s): {invalid!r}; "
            f"allowed={tuple(sorted(allowed))!r}."
        )
    return values


def _profile_id_for_method(method_id: str) -> str | None:
    return None if method_id == BASELINE_METHOD_ID else method_id


def _method_label(method_id: str) -> str:
    if method_id == BASELINE_METHOD_ID:
        return "projected_wald_inflation"
    profile = resolve_sibling_gate_profile(method_id)
    if profile is None:
        raise ValueError(f"Unknown profile: {method_id!r}.")
    return str(profile.sibling_gate_method)


def _safe_path_token(value: object) -> str:
    token = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in token)


def _checkpoint_prefix(
    *,
    checkpoint_dir: Path,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
) -> Path:
    return checkpoint_dir / (
        f"case={_safe_path_token(case_id)}"
        f"__role={_safe_path_token(data_role)}"
        f"__method={_safe_path_token(method_id)}"
        f"__replicate={int(replicate):04d}"
    )


class _SelectedFamilyRowTimeout(TimeoutError):
    """Raised when one selected-family panel row exceeds its time budget."""


@contextmanager
def _row_time_limit(seconds: float | None):
    if seconds is None or float(seconds) <= 0.0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise _SelectedFamilyRowTimeout(
            f"selected-family row exceeded {float(seconds):g} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _checkpoint_paths(prefix: Path) -> dict[str, Path]:
    return {
        "row": prefix.with_suffix(".row.csv"),
        "guard_rows": prefix.with_suffix(".guard.csv"),
        "node_decisions": prefix.with_suffix(".nodes.csv"),
        "regions": prefix.with_suffix(".regions.csv"),
        "gene_assignments": prefix.with_suffix(".genes.csv"),
    }


def _write_checkpoint(
    *,
    row: dict[str, object],
    guard_rows: pd.DataFrame,
    node_decisions: pd.DataFrame,
    regions: pd.DataFrame,
    gene_assignments: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    paths["row"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([row]).to_csv(paths["row"], index=False)
    guard_rows.to_csv(paths["guard_rows"], index=False)
    node_decisions.to_csv(paths["node_decisions"], index=False)
    regions.to_csv(paths["regions"], index=False)
    gene_assignments.to_csv(paths["gene_assignments"], index=False)


def _read_checkpoint(
    paths: dict[str, Path],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete selected-family checkpoint: {missing!r}")
    row_frame = pd.read_csv(paths["row"], keep_default_na=False)
    if row_frame.shape[0] != 1:
        raise ValueError(f"Checkpoint row must contain exactly one row: {paths['row']}")
    return (
        row_frame.iloc[0].to_dict(),
        pd.read_csv(paths["guard_rows"], keep_default_na=False),
        pd.read_csv(paths["node_decisions"], keep_default_na=False),
        pd.read_csv(paths["regions"], keep_default_na=False),
        pd.read_csv(paths["gene_assignments"], keep_default_na=False),
    )


def _output_data_role(data_role: str) -> str:
    """Return a CSV-stable role label.

    Pandas treats the literal string ``null`` as missing under default
    ``read_csv`` settings, so the public output uses ``selected_null`` while
    preserving ``null`` as the accepted CLI/input role.
    """
    return NULL_OUTPUT_ROLE if str(data_role) == "null" else str(data_role)


def _is_null_data_role(data_role: object) -> bool:
    return str(data_role) in {"null", NULL_OUTPUT_ROLE}


def _is_signal_data_role(data_role: object) -> bool:
    return str(data_role) == SIGNAL_OUTPUT_ROLE


def _annotation_bool(annotations: pd.DataFrame, node: object, column: str) -> bool:
    if column not in annotations or node not in annotations.index:
        return False
    value = annotations.at[node, column]
    return False if pd.isna(value) else bool(value)


def _annotation_float(annotations: pd.DataFrame, node: object, column: str) -> float:
    if column not in annotations or node not in annotations.index:
        return np.nan
    value = annotations.at[node, column]
    return np.nan if pd.isna(value) else float(value)


def _annotation_str(annotations: pd.DataFrame, node: object, column: str) -> str:
    if column not in annotations or node not in annotations.index:
        return ""
    value = annotations.at[node, column]
    return "" if pd.isna(value) else str(value)


def _parent_map(tree) -> dict[object, object | None]:
    parents: dict[object, object | None] = {}
    for node in tree.nodes:
        predecessors = list(tree.predecessors(node))
        parents[node] = predecessors[0] if predecessors else None
    return parents


def _depth_map(tree, root: object) -> dict[object, int]:
    depths = {root: 0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in tree.successors(node):
            depths[child] = depths[node] + 1
            stack.append(child)
    return depths


def _leaf_order(tree, root: object) -> list[object]:
    leaves: list[object] = []
    stack = [root]
    while stack:
        node = stack.pop()
        children = list(tree.successors(node))
        if not children:
            leaves.append(node)
        else:
            stack.extend(reversed(children))
    return leaves


def _node_label(tree, node: object) -> str:
    return str(tree.nodes[node].get("label", node))


def _ancestors_to_root(
    leaf: object,
    parents: dict[object, object | None],
) -> list[object]:
    path = [leaf]
    node = leaf
    while parents.get(node) is not None:
        node = parents[node]
        path.append(node)
    path.reverse()
    return path


def _decision_class(
    *,
    node: object,
    root: object,
    traversal_decision: str,
    is_leaf: bool,
    annotations: pd.DataFrame,
) -> str:
    root_selected_blocked = _annotation_bool(
        annotations,
        node,
        "Root_Selective_Permutation_Guard_Blocked",
    )
    root_stability_blocked = _annotation_bool(
        annotations,
        node,
        "Root_Stability_Guard_Blocked",
    )
    selected_family_blocked = _annotation_bool(
        annotations,
        node,
        "Selective_Permutation_Guard_Blocked",
    )
    if node == root and (root_selected_blocked or root_stability_blocked):
        return "selected_root_blocked"
    if selected_family_blocked:
        return "selected_family_blocked"
    if (
        traversal_decision == "boundary"
        and _annotation_bool(
            annotations,
            node,
            "Spectral_Transport_Pass_Through_Blocked",
        )
    ):
        return "spectral_transport_blocked"
    if traversal_decision == "split":
        return "accepted_internal_split"
    if traversal_decision == "pass_through":
        return "unstable_passthrough_zone"
    if is_leaf:
        return "leaf_fragment"
    return "stable_boundary"


def _topology_incidence_fields(
    *,
    node: object,
    root: object,
    child_count: int,
    is_leaf: bool,
    decision_class: str,
    traversal_decision: str,
) -> dict[str, object]:
    if node == root:
        incidence_role = "root"
    elif is_leaf:
        incidence_role = "leaf"
    else:
        incidence_role = "internal"
    has_incoming = incidence_role != "root"
    has_outgoing = incidence_role != "leaf" and int(child_count) >= 2
    pass_through = "pass" in str(traversal_decision) or (
        decision_class == "unstable_passthrough_zone"
    )
    if incidence_role == "leaf":
        status = "leaf_no_outgoing_test_fail_closed"
    else:
        status = "not_evaluated_missing_topology_features"
    return {
        "topology_incidence_role": incidence_role,
        "topology_has_incoming_edge": bool(has_incoming),
        "topology_has_outgoing_test": bool(has_outgoing),
        "topology_directed_degree": int(int(has_incoming) + int(child_count)),
        "topology_pass_through_candidate": bool(pass_through),
        "conditional_topology_status": status,
        "conditional_topology_log_odds": math.nan,
        "conditional_topology_probability": math.nan,
    }


def _build_node_decisions(
    *,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    data_seed: int,
    result,
) -> pd.DataFrame:
    tree = result.extra["tree"]
    annotations = result.extra["annotations"]
    decomposition = result.extra["decomposition"]
    root = tree.root() if hasattr(tree, "root") else tree.graph.get("root")
    parents = _parent_map(tree)
    depths = _depth_map(tree, root)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    trace = {
        row["node_id"]: row
        for row in decomposition.get("traversal_trace", [])
    }
    records: list[dict[str, object]] = []
    output_role = _output_data_role(data_role)
    for node in tree.nodes:
        children = list(tree.successors(node))
        trace_row = trace.get(node, {})
        traversal_decision = str(trace_row.get("decision", "not_visited"))
        is_leaf = len(children) == 0
        decision_class = _decision_class(
            node=node,
            root=root,
            traversal_decision=traversal_decision,
            is_leaf=is_leaf,
            annotations=annotations,
        )
        topology_fields = _topology_incidence_fields(
            node=node,
            root=root,
            child_count=len(children),
            is_leaf=is_leaf,
            decision_class=decision_class,
            traversal_decision=traversal_decision,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "data_role": output_role,
                "method_id": method_id,
                "replicate": int(replicate),
                "data_seed": int(data_seed),
                "node_id": str(node),
                "parent_id": "" if parents[node] is None else str(parents[node]),
                "depth": int(depths.get(node, -1)),
                "visited": bool(node in trace),
                "traversal_decision": traversal_decision,
                "decision_class": decision_class,
                "n_children": int(len(children)),
                "n_descendant_leaves": int(len(descendant_sets[node])),
                "child_parent_edge_open": _annotation_bool(
                    annotations,
                    node,
                    "Child_Parent_Divergence_Significant",
                ),
                "sibling_open": _annotation_bool(
                    annotations,
                    node,
                    "Sibling_BH_Different",
                ),
                "sibling_p_value": _annotation_float(
                    annotations,
                    node,
                    "Sibling_Divergence_P_Value",
                ),
                "sibling_projection_dimension": _annotation_float(
                    annotations,
                    node,
                    "Sibling_Projection_Dimension",
                ),
                "root_stability_guard_blocked": _annotation_bool(
                    annotations,
                    node,
                    "Root_Stability_Guard_Blocked",
                ),
                "root_selective_guard_blocked": _annotation_bool(
                    annotations,
                    node,
                    "Root_Selective_Permutation_Guard_Blocked",
                ),
                "selected_family_guard_blocked": _annotation_bool(
                    annotations,
                    node,
                    "Selective_Permutation_Guard_Blocked",
                ),
                "selected_family_p_value": _annotation_float(
                    annotations,
                    node,
                    "Selective_Permutation_P_Value",
                ),
                "selected_family_base_p_value": _annotation_float(
                    annotations,
                    node,
                    "Selective_Permutation_Base_P_Value",
                ),
                "selected_family_guard_refined": _annotation_bool(
                    annotations,
                    node,
                    "Selective_Permutation_Guard_Refined",
                ),
                "selected_family_scope": _annotation_str(
                    annotations,
                    node,
                    "Selective_Permutation_Guard_Scope",
                ),
                **topology_fields,
                "spectral_transport_node_mp_block_count": _annotation_float(
                    annotations,
                    node,
                    "Spectral_Transport_Node_MP_Block_Count",
                ),
                "spectral_transport_node_mp_total_multiplicity": _annotation_float(
                    annotations,
                    node,
                    "Spectral_Transport_Node_MP_Total_Multiplicity",
                ),
                "spectral_transport_child_best_mode_cost": _annotation_float(
                    annotations,
                    node,
                    "Spectral_Transport_Child_Best_Mode_Cost",
                ),
                "spectral_transport_child_best_mode_affinity": _annotation_float(
                    annotations,
                    node,
                    "Spectral_Transport_Child_Best_Mode_Affinity",
                ),
                "spectral_transport_child_best_mode_status": _annotation_str(
                    annotations,
                    node,
                    "Spectral_Transport_Child_Best_Mode_Status",
                ),
                "spectral_transport_best_descendant_split_path_cost": (
                    _annotation_float(
                        annotations,
                        node,
                        "Spectral_Transport_Best_Descendant_Split_Path_Cost",
                    )
                ),
                "spectral_transport_best_descendant_split_path_has_mp_evidence": (
                    _annotation_bool(
                        annotations,
                        node,
                        "Spectral_Transport_Best_Descendant_Split_Path_Has_MP_Evidence",
                    )
                ),
                "spectral_transport_passthrough_supported": _annotation_bool(
                    annotations,
                    node,
                    "Spectral_Transport_Pass_Through_Supported",
                ),
                "spectral_transport_passthrough_blocked": _annotation_bool(
                    annotations,
                    node,
                    "Spectral_Transport_Pass_Through_Blocked",
                ),
                "spectral_transport_bottleneck": _annotation_str(
                    annotations,
                    node,
                    "Spectral_Transport_Bottleneck",
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(records, columns=NODE_DECISION_COLUMNS)


def _build_guard_rows(node_decisions: pd.DataFrame, result) -> pd.DataFrame:
    annotations = result.extra["annotations"]
    records: list[dict[str, object]] = []
    for _, node_row in node_decisions.iterrows():
        node = node_row["node_id"]
        raw_index = next(
            (idx for idx in annotations.index if str(idx) == str(node)),
            None,
        )
        if raw_index is None:
            continue
        has_selected = pd.notna(
            _annotation_float(annotations, raw_index, "Selective_Permutation_P_Value")
        )
        has_root = pd.notna(
            _annotation_float(
                annotations,
                raw_index,
                "Root_Selective_Permutation_P_Value",
            )
        )
        blocked = bool(node_row["selected_family_guard_blocked"]) or bool(
            node_row["root_selective_guard_blocked"]
        )
        if not has_selected and not has_root and not blocked:
            continue
        scope = _annotation_str(
            annotations,
            raw_index,
            "Selective_Permutation_Guard_Scope",
        )
        if not scope and has_root:
            scope = "root"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": node_row["case_id"],
                "data_role": node_row["data_role"],
                "method_id": node_row["method_id"],
                "replicate": int(node_row["replicate"]),
                "data_seed": int(node_row["data_seed"]),
                "node_id": str(node),
                "decision_class": node_row["decision_class"],
                "guard_scope": scope,
                "observed_p_value": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Observed_P_Value",
                ),
                "selected_p_value": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_P_Value",
                ),
                "base_selected_p_value": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Base_P_Value",
                ),
                "null_min_p_value": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Null_Min_P_Value",
                ),
                "null_q05_p_value": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Null_Q05_P_Value",
                ),
                "guard_alpha": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Alpha",
                ),
                "guard_replicates": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Replicates",
                ),
                "guard_seed": _annotation_float(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Seed",
                ),
                "guard_refined": _annotation_bool(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Refined",
                ),
                "guard_would_block": _annotation_bool(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Would_Block",
                ),
                "guard_blocked": _annotation_bool(
                    annotations,
                    raw_index,
                    "Selective_Permutation_Guard_Blocked",
                )
                or _annotation_bool(
                    annotations,
                    raw_index,
                    "Root_Selective_Permutation_Guard_Blocked",
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(records, columns=GUARD_COLUMNS)


def _region_status(decision_class: str) -> str:
    if decision_class in {"selected_root_blocked", "selected_family_blocked"}:
        return "guard_blocked_zone"
    if decision_class == "spectral_transport_blocked":
        return "spectral_transport_blocked_zone"
    if decision_class == "unstable_passthrough_zone":
        return "unstable_passthrough_zone"
    if decision_class == "leaf_fragment":
        return "leaf_fragment_region"
    return "stable_internal_region"


def _build_regions_and_gene_assignments(
    *,
    node_decisions: pd.DataFrame,
    result,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = result.extra["tree"]
    decomposition = result.extra["decomposition"]
    root = tree.root() if hasattr(tree, "root") else tree.graph.get("root")
    parents = _parent_map(tree)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    raw_node_by_id = {str(node): node for node in tree.nodes}
    decision_by_node = {
        str(row["node_id"]): str(row["decision_class"])
        for _, row in node_decisions.iterrows()
    }
    zone_nodes = {
        str(row["node_id"])
        for _, row in node_decisions.iterrows()
        if row["decision_class"]
        in {
            "selected_root_blocked",
            "selected_family_blocked",
            "unstable_passthrough_zone",
        }
    }

    cluster_assignments = decomposition["cluster_assignments"]
    region_records: list[dict[str, object]] = []
    boundary_by_leaf: dict[str, dict[str, object]] = {}
    for cluster_id, cluster in cluster_assignments.items():
        root_node = str(cluster["root_node"])
        leaves = [str(leaf) for leaf in cluster["leaves"]]
        decision_class = decision_by_node.get(root_node, "stable_boundary")
        region_records.append(
            {
                "case_id": node_decisions["case_id"].iloc[0],
                "data_role": node_decisions["data_role"].iloc[0],
                "method_id": node_decisions["method_id"].iloc[0],
                "replicate": int(node_decisions["replicate"].iloc[0]),
                "region_id": f"flat_cluster_{int(cluster_id)}",
                "region_kind": "flat_boundary_projection",
                "root_node": root_node,
                "decision_class": decision_class,
                "region_status": _region_status(decision_class),
                "n_leaves": int(len(leaves)),
                "leaf_ids": ";".join(leaves),
                "study_role": STUDY_ROLE,
            }
        )
        for leaf in leaves:
            boundary_by_leaf[leaf] = {
                "flat_cluster_id": int(cluster_id),
                "stable_region_id": f"flat_cluster_{int(cluster_id)}",
                "boundary_root_node": root_node,
                "boundary_decision_class": decision_class,
            }

    for _, row in node_decisions.iterrows():
        node_id = str(row["node_id"])
        if row["decision_class"] != "unstable_passthrough_zone":
            continue
        raw_node = raw_node_by_id[node_id]
        leaves = sorted(str(leaf) for leaf in descendant_sets[raw_node])
        region_records.append(
            {
                "case_id": row["case_id"],
                "data_role": row["data_role"],
                "method_id": row["method_id"],
                "replicate": int(row["replicate"]),
                "region_id": f"zone_{node_id}",
                "region_kind": "unstable_or_passthrough_zone",
                "root_node": node_id,
                "decision_class": row["decision_class"],
                "region_status": "unstable_passthrough_zone",
                "n_leaves": int(len(leaves)),
                "leaf_ids": ";".join(leaves),
                "study_role": STUDY_ROLE,
            }
        )

    gene_records: list[dict[str, object]] = []
    for leaf in _leaf_order(tree, root):
        leaf_key = _node_label(tree, leaf)
        path = [str(node) for node in _ancestors_to_root(leaf, parents)]
        zone_region_id = ""
        for node in reversed(path):
            if node in zone_nodes:
                zone_region_id = f"zone_{node}"
                break
        boundary = boundary_by_leaf.get(
            leaf_key,
            {
                "flat_cluster_id": -1,
                "stable_region_id": "",
                "boundary_root_node": "",
                "boundary_decision_class": "",
            },
        )
        gene_records.append(
            {
                "case_id": node_decisions["case_id"].iloc[0],
                "data_role": node_decisions["data_role"].iloc[0],
                "method_id": node_decisions["method_id"].iloc[0],
                "replicate": int(node_decisions["replicate"].iloc[0]),
                "sample_id": leaf_key,
                "stable_region_id": boundary["stable_region_id"],
                "zone_region_id": zone_region_id,
                "final_fragment_cluster_id": int(boundary["flat_cluster_id"]),
                "boundary_root_node": boundary["boundary_root_node"],
                "path_decision_classes": ";".join(
                    decision_by_node.get(node, "not_visited") for node in path
                ),
                "path_node_ids": ";".join(path),
                "study_role": STUDY_ROLE,
            }
        )
    return (
        pd.DataFrame.from_records(region_records),
        pd.DataFrame.from_records(gene_records),
    )


def _adaptive_projection_avoided(method_id: str, result) -> bool:
    if method_id == BASELINE_METHOD_ID:
        return False
    profile = resolve_sibling_gate_profile(method_id)
    if profile is None:
        return False
    annotations = result.extra["annotations"]
    methods = (
        set(annotations["Sibling_Test_Method"].dropna().astype(str))
        if "Sibling_Test_Method" in annotations
        else set()
    )
    return (
        "projected_wald_inflation" not in methods
        and str(result.extra["sibling_gate_method"]) == profile.sibling_gate_method
    )


def _run_one(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    method_id: str,
    replicate: int,
    data_seed: int,
    sibling_alpha: float,
    edge_alpha: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, feature_space, truth_labels, true_clusters = _generate_data_with_truth(
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
    result = _run_kl_method(
        data,
        distance,
        sibling_significance_level=float(sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(edge_alpha),
        feature_space=feature_space,
        sibling_gate_profile=_profile_id_for_method(method_id),
    )
    predicted = np.asarray(result.labels)
    ari = float(adjusted_rand_score(np.asarray(truth_labels, dtype=int), predicted))
    node_decisions = _build_node_decisions(
        case_id=case_id,
        data_role=data_role,
        method_id=method_id,
        replicate=replicate,
        data_seed=data_seed,
        result=result,
    )
    guard_rows = _build_guard_rows(node_decisions, result)
    regions, gene_assignments = _build_regions_and_gene_assignments(
        node_decisions=node_decisions,
        result=result,
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": _output_data_role(data_role),
        "source_family": source_family,
        "feature_representation": feature_representation,
        "method_id": method_id,
        "profile_id": _profile_id_for_method(method_id) or "",
        "sibling_gate_method": _method_label(method_id),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "true_clusters": int(true_clusters),
        "found_clusters": int(result.found_clusters),
        "ari": ari,
        "exact_cluster_count": bool(int(result.found_clusters) == int(true_clusters)),
        "false_split": bool(_is_null_data_role(data_role) and int(result.found_clusters) > 1),
        "adaptive_projection_avoided": _adaptive_projection_avoided(method_id, result),
        "stable_boundary_count": int(
            node_decisions["decision_class"].eq("stable_boundary").sum()
        ),
        "selected_root_blocked_count": int(
            node_decisions["decision_class"].eq("selected_root_blocked").sum()
        ),
        "selected_family_blocked_count": int(
            node_decisions["decision_class"].eq("selected_family_blocked").sum()
        ),
        "unstable_passthrough_zone_count": int(
            node_decisions["decision_class"].eq("unstable_passthrough_zone").sum()
        ),
        "accepted_internal_split_count": int(
            node_decisions["decision_class"].eq("accepted_internal_split").sum()
        ),
        "leaf_fragment_count": int(
            node_decisions["decision_class"].eq("leaf_fragment").sum()
        ),
        "selected_family_guard_tested_count": int(guard_rows.shape[0]),
        "selected_family_guard_block_count": int(
            guard_rows["guard_blocked"].astype(bool).sum()
        )
        if not guard_rows.empty
        else 0,
        "spectral_transport_blocked_count": int(
            node_decisions["spectral_transport_passthrough_blocked"].astype(bool).sum()
        ),
        "min_selected_family_p_value": float(
            pd.to_numeric(guard_rows["selected_p_value"], errors="coerce").min()
        )
        if not guard_rows.empty
        else np.nan,
    }
    return row, guard_rows, node_decisions, regions, gene_assignments


def _status_for_transfer(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float,
    min_signal_mean_ari: float,
) -> str:
    if str(group["method_id"].iloc[0]) == BASELINE_METHOD_ID:
        return "diagnostic_only"
    if not bool(group["adaptive_projection_avoided"].astype(bool).all()):
        return "fixed_profile_adaptive_projection_detected"
    null_rows = group[group["data_role"].map(_is_null_data_role)]
    signal_rows = group[group["data_role"].map(_is_signal_data_role)]
    if null_rows.empty or signal_rows.empty:
        return "fixed_profile_insufficient_coverage"
    if float(null_rows["false_split"].mean()) > float(max_null_split_rate):
        return "fixed_profile_null_inflated"
    if float(signal_rows["ari"].mean()) < float(min_signal_mean_ari):
        return "fixed_profile_signal_weak"
    return "fixed_profile_transfer_candidate"


def _status_for_confidence(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float,
    min_signal_mean_ari: float,
) -> str:
    if str(group["method_id"].iloc[0]) == BASELINE_METHOD_ID:
        return "diagnostic_only"
    null_rows = group[group["data_role"].map(_is_null_data_role)]
    signal_rows = group[group["data_role"].map(_is_signal_data_role)]
    if null_rows.empty or signal_rows.empty:
        return "fixed_profile_insufficient_coverage"
    null_upper = _wilson_upper_bound(
        int(null_rows["false_split"].astype(bool).sum()),
        int(null_rows.shape[0]),
    )
    signal_lower = _mean_lower_bound(signal_rows["ari"])
    if float(null_upper) > float(max_null_split_rate):
        return "fixed_profile_confidence_null_uncertain"
    if float(signal_lower) < float(min_signal_mean_ari):
        return "fixed_profile_confidence_signal_uncertain"
    return "fixed_profile_confidence_candidate"


def build_selected_family_production_components(
    rows: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build conservative production-admissibility rows for selected traversal."""
    if rows.empty:
        components = pd.DataFrame.from_records(
            [
                {
                    "contract_id": "selected_family_traversal:empty",
                    "component_id": "selected_family_traversal:empty",
                    "component_type": "selected_family_traversal_validation",
                    "component_status": "fixed_profile_insufficient_coverage",
                    "required_for_production": True,
                    "notes": "No selected-family traversal rows were available.",
                }
            ]
        )
        evaluated = evaluate_production_admissibility_components(components)
        return evaluated, summarize_production_admissibility_contracts(evaluated)
    records: list[dict[str, object]] = []
    for (source_family, method_id), group in rows.groupby(
        ["source_family", "method_id"],
        sort=True,
    ):
        context = f"{source_family}:{method_id}"
        if method_id == BASELINE_METHOD_ID:
            records.append(
                {
                    "contract_id": f"selected_family_traversal:{context}",
                    "component_id": f"baseline_reference:{context}",
                    "component_type": "selected_family_baseline_reference",
                    "component_status": "diagnostic_only",
                    "required_for_production": True,
                    "notes": "Baseline projected-Wald traversal is reference-only.",
                }
            )
            continue
        adaptive_status = (
            "fixed_profile_adaptive_projection_avoided"
            if bool(group["adaptive_projection_avoided"].astype(bool).all())
            else "fixed_profile_adaptive_projection_detected"
        )
        records.extend(
            [
                {
                    "contract_id": f"selected_family_traversal:{context}",
                    "component_id": f"adaptive_projection:{context}",
                    "component_type": "fixed_profile_adaptive_projection_check",
                    "component_status": adaptive_status,
                    "required_for_production": True,
                    "notes": "Selected-family path must stay on fixed coordinates.",
                },
                {
                    "contract_id": f"selected_family_traversal:{context}",
                    "component_id": f"transfer:{context}",
                    "component_type": "selected_family_traversal_transfer",
                    "component_status": _status_for_transfer(
                        group,
                        max_null_split_rate=max_null_split_rate,
                        min_signal_mean_ari=min_signal_mean_ari,
                    ),
                    "required_for_production": True,
                    "notes": "Point null/signal traversal evidence.",
                },
                {
                    "contract_id": f"selected_family_traversal:{context}",
                    "component_id": f"confidence:{context}",
                    "component_type": "selected_family_traversal_confidence",
                    "component_status": _status_for_confidence(
                        group,
                        max_null_split_rate=max_null_split_rate,
                        min_signal_mean_ari=min_signal_mean_ari,
                    ),
                    "required_for_production": True,
                    "notes": "Confidence evidence keeps smoke runs non-production.",
                },
            ]
        )
    components = pd.DataFrame.from_records(records)
    evaluated = evaluate_production_admissibility_components(components)
    return evaluated, summarize_production_admissibility_contracts(evaluated)


def _concat_frames(frames: Iterable[pd.DataFrame], columns: Sequence[str] | None = None) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def run_selected_family_traversal_panel(
    config: SelectedFamilyTraversalPanelConfig,
) -> dict[str, Path]:
    """Run selected-family traversal diagnostics and write all outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if not math.isfinite(float(config.sibling_alpha)) or not 0.0 < float(
        config.sibling_alpha
    ) < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    if not math.isfinite(float(config.edge_alpha)) or not 0.0 < float(
        config.edge_alpha
    ) < 1.0:
        raise ValueError("edge_alpha must lie in (0, 1).")
    validate_methods(config.methods)
    validate_data_roles(config.data_roles)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    guard_frames: list[pd.DataFrame] = []
    node_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    gene_frames: list[pd.DataFrame] = []
    skipped_cases: list[dict[str, object]] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        try:
            (
                case_id,
                source_family,
                feature_representation,
                n_samples,
                n_features,
                n_categories,
            ) = _case_contract(case)
        except ValueError as exc:
            case_id = str(case.get("case_id", case.get("name", "unknown_case")))
            if not bool(config.skip_unsupported_cases):
                raise
            skipped_cases.append(
                {
                    "case_id": case_id,
                    "reason": str(exc),
                }
            )
            continue
        if source_family not in {"binary_template", "categorical_multinomial"}:
            if not bool(config.skip_unsupported_cases):
                raise ValueError(
                    "Selected-family traversal diagnostics support binary and direct "
                    f"categorical cases only; got {source_family!r}."
                )
            skipped_cases.append(
                {
                    "case_id": case_id,
                    "reason": (
                        "unsupported source_family for selected-family traversal: "
                        f"{source_family}"
                    ),
                }
            )
            continue
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate * 1009
            for data_role in config.data_roles:
                for method_id in config.methods:
                    checkpoint_paths = _checkpoint_paths(
                        _checkpoint_prefix(
                            checkpoint_dir=config.checkpoint_rows_dir,
                            case_id=case_id,
                            data_role=data_role,
                            method_id=method_id,
                            replicate=replicate,
                        )
                    )
                    if bool(config.resume_from_checkpoints) and checkpoint_paths[
                        "row"
                    ].exists():
                        (
                            row,
                            guard_rows,
                            node_decisions,
                            regions,
                            gene_assignments,
                        ) = _read_checkpoint(checkpoint_paths)
                    else:
                        try:
                            with _row_time_limit(config.per_row_timeout_seconds):
                                (
                                    row,
                                    guard_rows,
                                    node_decisions,
                                    regions,
                                    gene_assignments,
                                ) = _run_one(
                                    case=case,
                                    case_id=case_id,
                                    source_family=source_family,
                                    feature_representation=feature_representation,
                                    n_samples=n_samples,
                                    n_features=n_features,
                                    n_categories=n_categories,
                                    data_role=data_role,
                                    method_id=method_id,
                                    replicate=replicate,
                                    data_seed=data_seed,
                                    sibling_alpha=float(config.sibling_alpha),
                                    edge_alpha=float(config.edge_alpha),
                                )
                        except _SelectedFamilyRowTimeout as exc:
                            if not bool(config.skip_unsupported_cases):
                                raise
                            skipped_cases.append(
                                {
                                    "case_id": case_id,
                                    "data_role": data_role,
                                    "method_id": method_id,
                                    "replicate": int(replicate),
                                    "reason": str(exc),
                                }
                            )
                            continue
                        _write_checkpoint(
                            row=row,
                            guard_rows=guard_rows,
                            node_decisions=node_decisions,
                            regions=regions,
                            gene_assignments=gene_assignments,
                            paths=checkpoint_paths,
                        )
                    rows.append(row)
                    guard_frames.append(guard_rows)
                    node_frames.append(node_decisions)
                    region_frames.append(regions)
                    gene_frames.append(gene_assignments)

    traversal_rows = pd.DataFrame.from_records(rows, columns=TRAVERSAL_ROW_COLUMNS)
    guard_rows = _concat_frames(guard_frames, GUARD_COLUMNS)
    node_decisions = _concat_frames(node_frames, NODE_DECISION_COLUMNS)
    regions = _concat_frames(region_frames)
    gene_assignments = _concat_frames(gene_frames)
    production_components, production_summary = (
        build_selected_family_production_components(
            traversal_rows,
            max_null_split_rate=float(config.max_null_split_rate),
            min_signal_mean_ari=float(config.min_signal_mean_ari),
        )
    )

    traversal_rows.to_csv(config.traversal_rows_path, index=False)
    guard_rows.to_csv(config.guard_rows_path, index=False)
    node_decisions.to_csv(config.node_decisions_path, index=False)
    regions.to_csv(config.regions_path, index=False)
    gene_assignments.to_csv(config.gene_assignments_path, index=False)
    production_components.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "config": {
            "suite": config.suite,
            "case_names": list(config.case_names),
            "methods": list(config.methods),
            "data_roles": list(config.data_roles),
            "sibling_alpha": float(config.sibling_alpha),
            "edge_alpha": float(config.edge_alpha),
            "replicates": int(config.replicates),
            "base_seed": int(config.base_seed),
            "max_null_split_rate": float(config.max_null_split_rate),
            "min_signal_mean_ari": float(config.min_signal_mean_ari),
            "skip_unsupported_cases": bool(config.skip_unsupported_cases),
            "resume_from_checkpoints": bool(config.resume_from_checkpoints),
            "per_row_timeout_seconds": config.per_row_timeout_seconds,
        },
        "skipped_cases": skipped_cases,
        "outputs": {
            "selected_family_traversal_rows": str(config.traversal_rows_path),
            "checkpoint_rows_dir": str(config.checkpoint_rows_dir),
            "selected_family_guard_rows": str(config.guard_rows_path),
            "multiscale_node_decisions": str(config.node_decisions_path),
            "multiscale_regions": str(config.regions_path),
            "multiscale_gene_assignments": str(config.gene_assignments_path),
            "production_admissibility_components": str(
                config.production_components_path
            ),
            "production_admissibility_summary": str(config.production_summary_path),
        },
        "interpretation": (
            "Diagnostic-only selected traversal panel. Fixed-coordinate profiles "
            "avoid adaptive parent PCA, and the refined global pass-through profile "
            "surfaces the selected-family sibling-min null with Monte Carlo floor "
            "refinement. Production behavior remains fail-closed or diagnostic-only."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_family_traversal_rows": config.traversal_rows_path,
        "selected_family_guard_rows": config.guard_rows_path,
        "multiscale_node_decisions": config.node_decisions_path,
        "multiscale_regions": config.regions_path,
        "multiscale_gene_assignments": config.gene_assignments_path,
        "checkpoint_rows_dir": config.checkpoint_rows_dir,
        "production_admissibility_components": config.production_components_path,
        "production_admissibility_summary": config.production_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", type=parse_names, default=())
    parser.add_argument("--methods", type=parse_names, default=DEFAULT_METHODS)
    parser.add_argument("--data-roles", type=parse_names, default=DEFAULT_DATA_ROLES)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--max-null-split-rate", type=float, default=0.05)
    parser.add_argument("--min-signal-mean-ari", type=float, default=0.75)
    parser.add_argument(
        "--no-skip-unsupported-cases",
        action="store_false",
        dest="skip_unsupported_cases",
        default=True,
    )
    parser.add_argument("--resume-from-checkpoints", action="store_true")
    parser.add_argument("--per-row-timeout-seconds", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_family_traversal_panel(
        SelectedFamilyTraversalPanelConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=tuple(args.case_names),
            methods=validate_methods(args.methods),
            data_roles=validate_data_roles(args.data_roles),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            max_null_split_rate=float(args.max_null_split_rate),
            min_signal_mean_ari=float(args.min_signal_mean_ari),
            skip_unsupported_cases=bool(args.skip_unsupported_cases),
            resume_from_checkpoints=bool(args.resume_from_checkpoints),
            per_row_timeout_seconds=args.per_row_timeout_seconds,
        )
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
