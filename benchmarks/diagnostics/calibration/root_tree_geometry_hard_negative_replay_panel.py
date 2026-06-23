"""Replay a hard-negative root case across tree geometries.

This diagnostic tests whether a warning case remains fail-closed when the TBS
tree is rebuilt with alternative tree builders, distances, and linkage methods.
The motivating case is ``overlap_extreme_4c``: it can look selected-tail
calibrated under the observed root event, but that root fails validity replay.

Rows are diagnostic-only. A geometry is a hard-negative leak only if the root
becomes validity-supported and the root split is open under that geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)
from sklearn.metrics import adjusted_rand_score

SCHEMA_VERSION = "root_tree_geometry_hard_negative_replay_panel/v1"
STUDY_ROLE = "diagnostic_root_tree_geometry_hard_negative_replay_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "root_tree_geometry_hard_negative_replay_panel"
)
DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_TREE_GEOMETRIES = (
    "linkage:linkage_root:hamming:average",
    "linkage:linkage_root:hamming:complete",
    "linkage:linkage_root:jaccard:average",
    "linkage:linkage_root:rogerstanimoto:average",
    "neighbor_joining:mad:hamming:average",
    "neighbor_joining:mad:jaccard:average",
)
ROWS_OUTPUT = "root_tree_geometry_hard_negative_replay_rows.csv"
SUMMARY_OUTPUT = "root_tree_geometry_hard_negative_replay_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class TreeGeometrySpec:
    """One selected-tree construction and replay geometry."""

    tree_builder: str
    tree_rooting: str
    tree_distance_metric: str
    tree_linkage_method: str


@dataclass(frozen=True)
class RootTreeGeometryHardNegativeReplayConfig:
    """Runtime contract for hard-negative tree-geometry replay."""

    output_dir: Path
    suite: str = "binary"
    case_name: str = "overlap_extreme_4c"
    data_role: str = "signal"
    method_id: str = "tbs"
    profile_id: str = "fixed_coordinate_selective_root_v1"
    tree_geometries: tuple[TreeGeometrySpec, ...] = tuple(
        TreeGeometrySpec(*item.split(":")) for item in DEFAULT_TREE_GEOMETRIES
    )
    sibling_alpha: float = 0.01
    edge_alpha: float = 0.001
    replicates: int = 1
    base_seed: int = 20260617
    root_stability_threshold: float = 0.24
    root_selective_permutation_alpha: float = 0.01
    root_selective_permutation_replicates: int = 99
    root_selective_permutation_seed: int = 0

    @property
    def rows_path(self) -> Path:
        return self.output_dir / ROWS_OUTPUT

    @property
    def summary_path(self) -> Path:
        return self.output_dir / SUMMARY_OUTPUT

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_OUTPUT


def parse_tree_geometries(raw: str | None) -> tuple[TreeGeometrySpec, ...]:
    """Parse ``builder:rooting:distance:linkage`` geometry tokens."""
    tokens = DEFAULT_TREE_GEOMETRIES if raw is None else parse_names(raw)
    geometries: list[TreeGeometrySpec] = []
    for token in tokens:
        parts = tuple(part.strip() for part in str(token).split(":"))
        if len(parts) != 4 or any(not part for part in parts):
            raise ValueError(
                "Tree geometry tokens must have format "
                "'builder:rooting:distance:linkage'; got "
                f"{token!r}."
            )
        geometries.append(TreeGeometrySpec(*parts))
    if not geometries:
        raise ValueError("At least one tree geometry is required.")
    return tuple(geometries)


def _finite_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def classify_hard_negative_row(
    row: dict[str, object] | pd.Series,
    *,
    root_stability_threshold: float = 0.24,
    root_selective_permutation_alpha: float = 0.01,
) -> tuple[str, str, bool]:
    """Return status, method action, and root-validity support for one row."""
    if str(row.get("run_status", "ok")) != "ok":
        return (
            "hard_negative_geometry_skipped",
            "inspect_geometry_skip_before_using_this_replay",
            False,
        )

    mean_ari = _finite_float(row.get("root_stability_subsample_mean_ari"))
    stability_blocked = _bool_value(row.get("root_stability_guard_blocked"))
    root_stability_supported = (
        math.isfinite(mean_ari)
        and mean_ari >= float(root_stability_threshold)
        and not stability_blocked
    )

    selective_p_value = _finite_float(row.get("root_selective_permutation_p_value"))
    selective_blocked = _bool_value(
        row.get("root_selective_permutation_guard_blocked")
    )
    selective_would_block = _bool_value(
        row.get("root_selective_permutation_guard_would_block")
    )
    root_selective_supported = (
        math.isfinite(selective_p_value)
        and selective_p_value <= float(root_selective_permutation_alpha)
        and not selective_blocked
        and not selective_would_block
    )

    root_validity_supported = bool(
        root_stability_supported and root_selective_supported
    )
    root_open = _bool_value(row.get("root_sibling_open"))
    if root_validity_supported and root_open:
        return (
            "hard_negative_control_leaked_root_validity_supported",
            "block_promotion_and_inspect_geometry_rescue",
            True,
        )
    if root_validity_supported:
        return (
            "hard_negative_root_valid_but_root_split_closed",
            "keep_tail_rescue_disabled_and_inspect_root_family",
            True,
        )
    if not root_stability_supported:
        return (
            "hard_negative_control_blocked_root_unstable",
            "keep_fail_closed_root_unstable",
            False,
        )
    return (
        "hard_negative_control_blocked_selected_root_permutation",
        "keep_fail_closed_root_not_selectively_significant",
        False,
    )


def classify_legacy_hard_negative_row(
    row: dict[str, object] | pd.Series,
    *,
    root_truth_ari_tolerance: float = 0.05,
) -> tuple[str, str, bool]:
    """Return a comparator status for the old method without root-validity guards."""
    if str(row.get("run_status", "ok")) != "ok":
        return (
            "legacy_geometry_skipped",
            "legacy_geometry_unsupported_or_failed",
            False,
        )

    root_ari = _finite_float(row.get("root_partition_truth_ari"))
    found_clusters = _finite_float(row.get("found_clusters"))
    true_clusters = _finite_float(row.get("true_clusters"))
    root_misaligned = (
        not math.isfinite(root_ari)
        or abs(root_ari) <= float(root_truth_ari_tolerance)
    )
    if root_misaligned and found_clusters > true_clusters:
        return (
            "legacy_fragments_with_invalid_root_no_validity_guard",
            "treat_as_legacy_warning_fragmentation_without_root_validity",
            False,
        )
    if root_misaligned:
        return (
            "legacy_no_validity_guard_root_misaligned_but_not_fragmented",
            "treat_as_legacy_fail_closed_comparator",
            False,
        )
    return (
        "legacy_root_truth_aligned_without_validity_guard",
        "inspect_legacy_root_alignment_not_a_current_rescue_rule",
        False,
    )


def _root_node(tree: object) -> object:
    if hasattr(tree, "root"):
        return tree.root()
    graph = getattr(tree, "graph", {})
    return graph.get("root")


def _root_partition_summary(
    *,
    tree: object,
    sample_ids: Sequence[object],
    truth_labels: np.ndarray,
) -> dict[str, object]:
    root = _root_node(tree)
    if root is None:
        return {
            "root_node": "",
            "root_child_count": 0,
            "root_child_size_signature": "",
            "root_child_balance": math.nan,
            "root_partition_truth_ari": math.nan,
        }
    children = list(tree.successors(root))
    if not children:
        return {
            "root_node": str(root),
            "root_child_count": 0,
            "root_child_size_signature": "",
            "root_child_balance": math.nan,
            "root_partition_truth_ari": math.nan,
        }

    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    labels = np.zeros(len(sample_ids), dtype=int)
    child_sizes: list[int] = []
    for child_index, child in enumerate(children):
        descendants = set(descendant_sets.get(child, ()))
        child_sizes.append(len(descendants))
        for sample_index, sample_id in enumerate(sample_ids):
            if sample_id in descendants:
                labels[sample_index] = child_index

    min_size = min(child_sizes)
    max_size = max(child_sizes)
    balance = math.nan if max_size == 0 else float(min_size / max_size)
    truth = np.asarray(truth_labels, dtype=int)
    ari = (
        float(adjusted_rand_score(truth, labels))
        if truth.shape[0] == labels.shape[0]
        else math.nan
    )
    return {
        "root_node": str(root),
        "root_child_count": int(len(children)),
        "root_child_size_signature": ";".join(str(size) for size in sorted(child_sizes)),
        "root_child_balance": balance,
        "root_partition_truth_ari": ari,
    }


def _unrooted_edge_cut_summary(
    *,
    tree: object,
    sample_ids: Sequence[object],
    truth_labels: np.ndarray,
    truth_ari_floor: float = 0.05,
) -> dict[str, object]:
    """Summarize the best undirected edge bipartition in the selected tree.

    The tree object is rooted for traversal, but every directed edge still
    induces an undirected bipartition: descendants of the child side versus all
    remaining leaves. This benchmark-only diagnostic asks whether removing the
    privileged root makes a truth-aligned coarse split visible anywhere in the
    same geometry.
    """
    root = _root_node(tree)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    sample_list = list(sample_ids)
    all_samples = set(sample_list)
    truth = np.asarray(truth_labels, dtype=int)

    best: dict[str, object] | None = None
    edge_count = 0
    for parent, child in tree.edges():
        child_side = set(descendant_sets.get(child, ()))
        if not child_side or child_side == all_samples:
            continue
        other_side = all_samples - child_side
        if not other_side:
            continue
        labels = np.asarray(
            [1 if sample_id in child_side else 0 for sample_id in sample_list],
            dtype=int,
        )
        if truth.shape[0] != labels.shape[0]:
            continue
        edge_count += 1
        ari = float(adjusted_rand_score(truth, labels))
        min_size = min(len(child_side), len(other_side))
        max_size = max(len(child_side), len(other_side))
        balance = math.nan if max_size == 0 else float(min_size / max_size)
        candidate = {
            "best_unrooted_edge_cut_parent": str(parent),
            "best_unrooted_edge_cut_child": str(child),
            "best_unrooted_edge_cut_size_signature": ";".join(
                str(size) for size in sorted((len(child_side), len(other_side)))
            ),
            "best_unrooted_edge_cut_balance": balance,
            "best_unrooted_edge_cut_truth_ari": ari,
            "best_unrooted_edge_cut_is_root_edge": bool(parent == root),
        }
        if best is None or ari > float(best["best_unrooted_edge_cut_truth_ari"]):
            best = candidate

    if best is None:
        return {
            "unrooted_edge_count": int(edge_count),
            "best_unrooted_edge_cut_parent": "",
            "best_unrooted_edge_cut_child": "",
            "best_unrooted_edge_cut_size_signature": "",
            "best_unrooted_edge_cut_balance": math.nan,
            "best_unrooted_edge_cut_truth_ari": math.nan,
            "best_unrooted_edge_cut_is_root_edge": False,
            "unrooted_geometry_status": "unrooted_edge_cut_unmeasured",
            "rootless_method_action": "inspect_tree_before_rootless_rescue",
        }

    best_ari = float(best["best_unrooted_edge_cut_truth_ari"])
    if best_ari >= float(truth_ari_floor):
        best.update(
            {
                "unrooted_edge_count": int(edge_count),
                "unrooted_geometry_status": (
                    "unrooted_edge_cut_truth_aligned_bipartition_available"
                ),
                "rootless_method_action": (
                    "inspect_rootless_edge_tail_law_before_root_rescue"
                ),
            }
        )
    else:
        best.update(
            {
                "unrooted_edge_count": int(edge_count),
                "unrooted_geometry_status": (
                    "unrooted_edge_cut_no_truth_aligned_bipartition"
                ),
                "rootless_method_action": (
                    "rootless_rescue_not_supported_by_tree_geometry"
                ),
            }
        )
    return best


def _annotation_value(
    annotations: pd.DataFrame,
    root: object,
    column: str,
    default: object,
) -> object:
    if root in annotations.index and column in annotations:
        value = annotations.at[root, column]
        if not pd.isna(value):
            return value
    return default


def _row_for_geometry(
    *,
    config: RootTreeGeometryHardNegativeReplayConfig,
    geometry: TreeGeometrySpec,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    replicate: int,
) -> dict[str, object]:
    data_seed = int(config.base_seed) + int(replicate) * 1009
    data, feature_space, truth_labels, true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role=config.data_role,
        seed=data_seed,
    )
    params = {
        "tree_distance_metric": geometry.tree_distance_metric,
        "tree_linkage_method": geometry.tree_linkage_method,
        "tree_builder": geometry.tree_builder,
        "tree_rooting": geometry.tree_rooting,
        "root_stability_tree_distance_metric": geometry.tree_distance_metric,
        "root_stability_tree_linkage_method": geometry.tree_linkage_method,
        "root_selective_permutation_guard_tree_distance_metric": (
            geometry.tree_distance_metric
        ),
        "root_selective_permutation_guard_tree_linkage_method": (
            geometry.tree_linkage_method
        ),
        "root_selective_permutation_guard_replicates": int(
            config.root_selective_permutation_replicates
        ),
        "root_selective_permutation_guard_seed": int(
            config.root_selective_permutation_seed
        ),
        "root_selective_permutation_guard_alpha": float(
            config.root_selective_permutation_alpha
        ),
    }
    if config.method_id == "tbs":
        params["sibling_gate_profile"] = config.profile_id
    result = run_clustering_result(
        data_df=data,
        method_id=config.method_id,
        params=params,
        significance_level=float(config.sibling_alpha),
        edge_alpha=float(config.edge_alpha),
        feature_space=feature_space,
    )

    base_row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": config.data_role,
        "source_family": source_family,
        "feature_representation": feature_representation,
        "method_id": config.method_id,
        "profile_id": config.profile_id,
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "tree_builder": geometry.tree_builder,
        "tree_rooting": geometry.tree_rooting,
        "tree_distance_metric": geometry.tree_distance_metric,
        "tree_linkage_method": geometry.tree_linkage_method,
        "root_replay_distance_metric": geometry.tree_distance_metric,
        "root_replay_linkage_method": geometry.tree_linkage_method,
        "sibling_alpha": float(config.sibling_alpha),
        "edge_alpha": float(config.edge_alpha),
        "root_stability_threshold": float(config.root_stability_threshold),
        "root_selective_permutation_alpha": float(
            config.root_selective_permutation_alpha
        ),
        "root_selective_permutation_replicates": int(
            config.root_selective_permutation_replicates
        ),
        "true_clusters": int(true_clusters),
        "run_status": result.status,
        "skip_reason": "" if result.skip_reason is None else str(result.skip_reason),
        "found_clusters": math.nan,
        "ari": math.nan,
        "root_node": "",
        "root_child_count": 0,
        "root_child_size_signature": "",
        "root_child_balance": math.nan,
        "root_partition_truth_ari": math.nan,
        "root_sibling_p_value": math.nan,
        "root_sibling_open": False,
        "root_stability_guard_blocked": False,
        "root_stability_subsample_mean_ari": math.nan,
        "root_stability_subsample_median_ari": math.nan,
        "root_stability_subsample_q10_ari": math.nan,
        "root_selective_permutation_p_value": math.nan,
        "root_selective_permutation_guard_blocked": False,
        "root_selective_permutation_guard_would_block": False,
        "unrooted_edge_count": 0,
        "best_unrooted_edge_cut_parent": "",
        "best_unrooted_edge_cut_child": "",
        "best_unrooted_edge_cut_size_signature": "",
        "best_unrooted_edge_cut_balance": math.nan,
        "best_unrooted_edge_cut_truth_ari": math.nan,
        "best_unrooted_edge_cut_is_root_edge": False,
        "unrooted_geometry_status": "unrooted_edge_cut_unmeasured",
        "rootless_method_action": "inspect_tree_before_rootless_rescue",
    }
    if result.status != "ok":
        if config.method_id == "tbs_legacy_c2ef9a69":
            status, action, validity_supported = classify_legacy_hard_negative_row(
                base_row
            )
        else:
            status, action, validity_supported = classify_hard_negative_row(
                base_row,
                root_stability_threshold=float(config.root_stability_threshold),
                root_selective_permutation_alpha=float(
                    config.root_selective_permutation_alpha
                ),
            )
        base_row.update(
            {
                "root_validity_supported": bool(validity_supported),
                "hard_negative_control_status": status,
                "method_action": action,
            }
        )
        return base_row

    predicted = np.asarray(result.labels, dtype=int)
    truth = np.asarray(truth_labels, dtype=int)
    annotations = result.extra["annotations"]
    tree = result.extra["tree"]
    root = _root_node(tree)
    root_summary = _root_partition_summary(
        tree=tree,
        sample_ids=data.index.tolist(),
        truth_labels=truth,
    )
    unrooted_summary = _unrooted_edge_cut_summary(
        tree=tree,
        sample_ids=data.index.tolist(),
        truth_labels=truth,
    )
    base_row.update(root_summary)
    base_row.update(unrooted_summary)
    base_row.update(
        {
            "found_clusters": int(result.found_clusters),
            "ari": float(adjusted_rand_score(truth, predicted)),
            "root_sibling_p_value": _finite_float(
                _annotation_value(
                    annotations,
                    root,
                    "Sibling_Divergence_P_Value",
                    math.nan,
                )
            ),
            "root_sibling_open": _bool_value(
                _annotation_value(annotations, root, "Sibling_BH_Different", False)
            ),
            "root_stability_guard_blocked": _bool_value(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Stability_Guard_Blocked",
                    False,
                )
            ),
            "root_stability_subsample_mean_ari": _finite_float(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Stability_Subsample_Mean_ARI",
                    math.nan,
                )
            ),
            "root_stability_subsample_median_ari": _finite_float(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Stability_Subsample_Median_ARI",
                    math.nan,
                )
            ),
            "root_stability_subsample_q10_ari": _finite_float(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Stability_Subsample_Q10_ARI",
                    math.nan,
                )
            ),
            "root_selective_permutation_p_value": _finite_float(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Selective_Permutation_P_Value",
                    math.nan,
                )
            ),
            "root_selective_permutation_guard_blocked": _bool_value(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Selective_Permutation_Guard_Blocked",
                    False,
                )
            ),
            "root_selective_permutation_guard_would_block": _bool_value(
                _annotation_value(
                    annotations,
                    root,
                    "Root_Selective_Permutation_Guard_Would_Block",
                    False,
                )
            ),
        }
    )
    if config.method_id == "tbs_legacy_c2ef9a69":
        status, action, validity_supported = classify_legacy_hard_negative_row(
            base_row
        )
    else:
        status, action, validity_supported = classify_hard_negative_row(
            base_row,
            root_stability_threshold=float(config.root_stability_threshold),
            root_selective_permutation_alpha=(
                float(config.root_selective_permutation_alpha)
            ),
        )
    base_row.update(
        {
            "root_validity_supported": bool(validity_supported),
            "hard_negative_control_status": status,
            "method_action": action,
        }
    )
    return base_row


def build_root_tree_geometry_hard_negative_replay_rows(
    config: RootTreeGeometryHardNegativeReplayConfig,
) -> pd.DataFrame:
    """Run the configured tree-geometry replay rows."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if config.data_role not in {"signal", "null"}:
        raise ValueError("data_role must be 'signal' or 'null'.")
    if config.method_id not in {"tbs", "tbs_legacy_c2ef9a69"}:
        raise ValueError("method_id must be 'tbs' or 'tbs_legacy_c2ef9a69'.")
    if not 0.0 < float(config.sibling_alpha) < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    if not 0.0 < float(config.edge_alpha) < 1.0:
        raise ValueError("edge_alpha must lie in (0, 1).")
    if not 0.0 < float(config.root_selective_permutation_alpha) < 1.0:
        raise ValueError("root_selective_permutation_alpha must lie in (0, 1).")

    cases = _select_cases(suite=config.suite, case_names=(config.case_name,))
    if len(cases) != 1:
        raise ValueError(
            f"Expected exactly one case named {config.case_name!r}; got {len(cases)}."
        )
    case = cases[0]
    (
        case_id,
        source_family,
        feature_representation,
        n_samples,
        n_features,
        n_categories,
    ) = _case_contract(case)

    records: list[dict[str, object]] = []
    for replicate in range(int(config.replicates)):
        for geometry in config.tree_geometries:
            records.append(
                _row_for_geometry(
                    config=config,
                    geometry=geometry,
                    case=case,
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    replicate=replicate,
                )
            )
    return pd.DataFrame.from_records(records)


def summarize_root_tree_geometry_hard_negative_replay_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize hard-negative replay rows."""
    if rows.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "ok_geometry_count": 0,
                    "skipped_geometry_count": 0,
                    "root_validity_supported_count": 0,
                    "hard_negative_leak_count": 0,
                    "rootless_truth_aligned_geometry_count": 0,
                    "max_best_unrooted_edge_cut_truth_ari": math.nan,
                    "blocked_count": 0,
                    "summary_status": "hard_negative_control_unmeasured",
                }
            ]
        )
    leak_mask = rows["hard_negative_control_status"].eq(
        "hard_negative_control_leaked_root_validity_supported"
    )
    legacy_warning_mask = rows["hard_negative_control_status"].eq(
        "legacy_fragments_with_invalid_root_no_validity_guard"
    )
    rootless_supported_mask = (
        rows["unrooted_geometry_status"].eq(
            "unrooted_edge_cut_truth_aligned_bipartition_available"
        )
        if "unrooted_geometry_status" in rows
        else pd.Series(False, index=rows.index)
    )
    best_unrooted_edge_ari = (
        pd.to_numeric(
            rows["best_unrooted_edge_cut_truth_ari"],
            errors="coerce",
        )
        if "best_unrooted_edge_cut_truth_ari" in rows
        else pd.Series(dtype=float)
    )
    skipped_mask = rows["run_status"].ne("ok")
    valid_mask = rows["root_validity_supported"].astype(bool)
    summary_status = (
        "hard_negative_control_failed"
        if bool(leak_mask.any())
        else "hard_negative_control_supported"
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(rows["case_id"].iloc[0]),
                "data_role": str(rows["data_role"].iloc[0]),
                "row_count": int(rows.shape[0]),
                "ok_geometry_count": int((~skipped_mask).sum()),
                "skipped_geometry_count": int(skipped_mask.sum()),
                "root_validity_supported_count": int(valid_mask.sum()),
                "hard_negative_leak_count": int(leak_mask.sum()),
                "legacy_invalid_root_fragmentation_count": int(
                    legacy_warning_mask.sum()
                ),
                "rootless_truth_aligned_geometry_count": int(
                    rootless_supported_mask.sum()
                ),
                "max_best_unrooted_edge_cut_truth_ari": float(
                    best_unrooted_edge_ari.max()
                ),
                "blocked_count": int(
                    rows["hard_negative_control_status"]
                    .astype(str)
                    .str.startswith("hard_negative_control_blocked")
                    .sum()
                ),
                "min_root_stability_subsample_mean_ari": float(
                    pd.to_numeric(
                        rows["root_stability_subsample_mean_ari"],
                        errors="coerce",
                    ).min()
                ),
                "max_root_stability_subsample_mean_ari": float(
                    pd.to_numeric(
                        rows["root_stability_subsample_mean_ari"],
                        errors="coerce",
                    ).max()
                ),
                "min_root_selective_permutation_p_value": float(
                    pd.to_numeric(
                        rows["root_selective_permutation_p_value"],
                        errors="coerce",
                    ).min()
                ),
                "max_root_selective_permutation_p_value": float(
                    pd.to_numeric(
                        rows["root_selective_permutation_p_value"],
                        errors="coerce",
                    ).max()
                ),
                "geometry_statuses": ";".join(
                    sorted(set(rows["hard_negative_control_status"].astype(str)))
                ),
                "summary_status": summary_status,
            }
        ]
    )


def run_root_tree_geometry_hard_negative_replay(
    config: RootTreeGeometryHardNegativeReplayConfig,
) -> dict[str, Path]:
    """Run hard-negative replay and write rows, summary, and manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_root_tree_geometry_hard_negative_replay_rows(config)
    summary = summarize_root_tree_geometry_hard_negative_replay_rows(rows)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "tree_geometries": [asdict(geometry) for geometry in config.tree_geometries],
        },
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "interpretation": (
            "Diagnostic-only tree-geometry replay. The configured case is a "
            "hard negative for root-tail rescue unless the selected root is "
            "validity-supported and open under a replay geometry."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-name", default="overlap_extreme_4c")
    parser.add_argument("--data-role", default="signal", choices=("signal", "null"))
    parser.add_argument(
        "--method-id",
        default="tbs",
        choices=("tbs", "tbs_legacy_c2ef9a69"),
    )
    parser.add_argument(
        "--profile-id",
        default="fixed_coordinate_selective_root_v1",
    )
    parser.add_argument(
        "--tree-geometries",
        default=None,
        help=(
            "Comma-separated builder:rooting:distance:linkage tokens. "
            f"Defaults to {','.join(DEFAULT_TREE_GEOMETRIES)}."
        ),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260617)
    parser.add_argument("--root-stability-threshold", type=float, default=0.24)
    parser.add_argument("--root-selective-permutation-alpha", type=float, default=0.01)
    parser.add_argument("--root-selective-permutation-replicates", type=int, default=99)
    parser.add_argument("--root-selective-permutation-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_root_tree_geometry_hard_negative_replay(
        RootTreeGeometryHardNegativeReplayConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_name=str(args.case_name),
            data_role=str(args.data_role),
            method_id=str(args.method_id),
            profile_id=str(args.profile_id),
            tree_geometries=parse_tree_geometries(args.tree_geometries),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            root_stability_threshold=float(args.root_stability_threshold),
            root_selective_permutation_alpha=float(
                args.root_selective_permutation_alpha
            ),
            root_selective_permutation_replicates=int(
                args.root_selective_permutation_replicates
            ),
            root_selective_permutation_seed=int(
                args.root_selective_permutation_seed
            ),
        )
    )


if __name__ == "__main__":
    main()
