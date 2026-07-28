#!/usr/bin/env python3
"""Compare adaptive-K NNLS tree topologies, branch lengths, and cluster paths.

This diagnostic is evidence-only. It rebuilds the graphtools adaptive-K NNLS
tree-strategy cells for a target case set, compares complete tree objects, and
optionally calls the installed BranchArchitect extra or a local checkout via
``TBS_BRANCHARCHITECT_PATH``. It does not change production topology selection,
alpha policy, or fallback behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.result_records.models import BenchmarkRunStatus
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.tree_consensus import (
    TARGET_TREE_CONSENSUS_METHOD,
    tree_inference_from_run_id,
)
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "brancharchitect_tree_comparison/v1"
GENERATED_BY = "benchmarks.validation.tree.brancharchitect_tree_comparison"

DEFAULT_TAXONOMY_PATH = Path(
    "reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv"
)
DEFAULT_OUTPUT_DIR = Path("reports/tree_consensus_brancharchitect_tree_comparison_20260709")

TREE_CELLS_NAME = "brancharchitect_tree_cells.csv"
TREE_NEWICK_NAME = "brancharchitect_tree_newick.csv"
TREE_PAIRWISE_NAME = "brancharchitect_tree_pairwise.csv"
TREE_CASE_SUMMARY_NAME = "brancharchitect_tree_case_summary.csv"
TREE_REPORT_NAME = "brancharchitect_tree_comparison_report.md"
TREE_MANIFEST_NAME = "brancharchitect_tree_comparison_manifest.json"

BRANCHARCHITECT_PATH_ENV = "TBS_BRANCHARCHITECT_PATH"
BUNDLED_BRANCHARCHITECT_PATH = Path(__file__).resolve().parents[3] / "vendor" / "BranchArchitect"


@dataclass(frozen=True)
class TreeSnapshot:
    """Comparable representation of one rebuilt tree-strategy cell."""

    case_id: str
    test_case: int
    case_category: str
    tree_inference: str
    run_id: str
    status: str
    skip_reason: str
    true_clusters: int
    found_clusters: int
    ari: float
    nmi: float
    macro_f1: float
    purity: float
    leaf_labels: tuple[str, ...]
    label_by_leaf: dict[str, int]
    split_lengths: dict[frozenset[str], float]
    internal_splits: frozenset[frozenset[str]]
    edge_lengths: dict[tuple[str, str], float]
    root_path_edges_by_leaf: dict[str, frozenset[tuple[str, str]]]
    branch_length_missing_count: int
    branch_length_zero_count: int
    newick: str


@dataclass(frozen=True)
class BranchArchitectAdapter:
    """Optional BranchArchitect hooks loaded from an installation or checkout."""

    status: str
    error: str
    read_newick: Any = None
    relative_robinson_foulds_distance: Any = None
    weighted_robinson_foulds_distance: Any = None
    tree_interpolation_pipeline: Any = None
    pipeline_config: Any = None

    @property
    def has_distances(self) -> bool:
        return (
            self.status in {"distances_available", "interpolation_available"}
            and self.read_newick is not None
            and self.relative_robinson_foulds_distance is not None
            and self.weighted_robinson_foulds_distance is not None
        )

    @property
    def has_interpolation(self) -> bool:
        return (
            self.status == "interpolation_available"
            and self.tree_interpolation_pipeline is not None
            and self.pipeline_config is not None
        )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _slugify(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _parse_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _load_default_case_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(
            f"Default fail-closed taxonomy file does not exist: {path}. "
            "Pass --case-names explicitly or generate the fail-closed p-value report first."
        )
    frame = pd.read_csv(path)
    if "case_id" not in frame.columns:
        raise ValueError(f"{path} is missing required column case_id.")
    return tuple(frame["case_id"].dropna().astype(str).tolist())


def _brancharchitect_path(path: str | Path | None) -> Path | None:
    raw = str(path or os.environ.get(BRANCHARCHITECT_PATH_ENV, "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if BUNDLED_BRANCHARCHITECT_PATH.exists():
        return BUNDLED_BRANCHARCHITECT_PATH
    return None


def load_brancharchitect_adapter(
    brancharchitect_path: str | Path | None = None,
    *,
    require_interpolation: bool = False,
) -> BranchArchitectAdapter:
    """Load BranchArchitect from the installed extra or a configured checkout."""

    resolved_path = _brancharchitect_path(brancharchitect_path)
    if resolved_path is not None and not resolved_path.exists():
        return BranchArchitectAdapter(
            status="path_missing",
            error=f"BranchArchitect path does not exist: {resolved_path}",
        )

    if resolved_path is not None:
        path_text = str(resolved_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    try:
        from brancharchitect.distances.distances import (  # type: ignore[import-not-found]
            relative_robinson_foulds_distance,
            weighted_robinson_foulds_distance,
        )
        from brancharchitect.io import read_newick  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        if (
            resolved_path is None
            and (exc.name or "").split(".", maxsplit=1)[0] == "brancharchitect"
        ):
            return BranchArchitectAdapter(
                status="not_configured",
                error=(
                    "BranchArchitect is not installed and the bundled submodule is not initialized. "
                    "Run `git submodule update --init` and `uv sync --extra brancharchitect`, "
                    f"set {BRANCHARCHITECT_PATH_ENV}, or pass --brancharchitect-path."
                ),
            )
        install_hint = (
            f"Could not import a BranchArchitect dependency from {resolved_path}."
            if resolved_path is not None
            else "The installed BranchArchitect package has a missing dependency."
        )
        return BranchArchitectAdapter(
            status="distance_import_error",
            error=(
                f"{type(exc).__name__}: {exc}. Install with "
                f"`uv sync --extra brancharchitect`; {install_hint}"
            ),
        )
    except Exception as exc:  # pragma: no cover - exercised when optional path is bad
        install_hint = (
            "Initialize `vendor/BranchArchitect` with `git submodule update --init`, "
            f"set {BRANCHARCHITECT_PATH_ENV}, or pass --brancharchitect-path."
            if resolved_path is None
            else f"Could not import BranchArchitect from {resolved_path}."
        )
        return BranchArchitectAdapter(
            status="distance_import_error",
            error=(
                f"{type(exc).__name__}: {exc}. Install with "
                f"`uv sync --extra brancharchitect`; {install_hint}"
            ),
        )

    if not require_interpolation:
        return BranchArchitectAdapter(
            status="distances_available",
            error="",
            read_newick=read_newick,
            relative_robinson_foulds_distance=relative_robinson_foulds_distance,
            weighted_robinson_foulds_distance=weighted_robinson_foulds_distance,
        )

    try:
        from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (  # type: ignore[import-not-found]
            TreeInterpolationPipeline,
        )
        from brancharchitect.movie_pipeline.types import (  # type: ignore[import-not-found]
            PipelineConfig,
        )
    except Exception as exc:  # pragma: no cover - depends on optional extras
        return BranchArchitectAdapter(
            status="interpolation_import_error",
            error=f"{type(exc).__name__}: {exc}",
            read_newick=read_newick,
            relative_robinson_foulds_distance=relative_robinson_foulds_distance,
            weighted_robinson_foulds_distance=weighted_robinson_foulds_distance,
        )

    return BranchArchitectAdapter(
        status="interpolation_available",
        error="",
        read_newick=read_newick,
        relative_robinson_foulds_distance=relative_robinson_foulds_distance,
        weighted_robinson_foulds_distance=weighted_robinson_foulds_distance,
        tree_interpolation_pipeline=TreeInterpolationPipeline,
        pipeline_config=PipelineConfig,
    )


def _tree_root(tree: Any) -> object:
    root = getattr(tree, "graph", {}).get("root")
    if root is not None:
        return root
    roots = [node for node, degree in tree.in_degree() if int(degree) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one tree root, got {roots!r}.")
    return roots[0]


def _raise_recursion_limit_for_tree(tree: Any) -> None:
    """Allow serialization of ladder-like hierarchical trees."""
    try:
        node_count = len(tree.nodes)
    except TypeError:
        node_count = len(list(tree.nodes()))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), int(node_count) * 4 + 1000))


def _tree_children(tree: Any, node: object) -> list[object]:
    return list(tree.successors(node))


def _leaf_label(tree: Any, node: object) -> str:
    attrs = tree.nodes[node]
    return str(attrs.get("label", node))


def _newick_label(label: str) -> str:
    safe = all(ch.isalnum() or ch in "_-." for ch in label)
    if safe and label:
        return label
    return "'" + label.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _edge_branch_length(tree: Any, parent: object, child: object) -> tuple[float, bool]:
    value = _finite_or_nan(tree.edges[parent, child].get("branch_length"))
    if not math.isfinite(value) or value < 0.0:
        return 0.0, True
    return float(value), False


def _tree_descendants(tree: Any, root: object) -> dict[object, tuple[str, ...]]:
    cache: dict[object, tuple[str, ...]] = {}

    def visit(node: object) -> tuple[str, ...]:
        if node in cache:
            return cache[node]
        children = _tree_children(tree, node)
        if not children:
            leaves = (_leaf_label(tree, node),)
        else:
            merged: list[str] = []
            for child in children:
                merged.extend(visit(child))
            leaves = tuple(sorted(merged))
        cache[node] = leaves
        return leaves

    visit(root)
    return cache


def _ordered_children(
    tree: Any,
    node: object,
    descendants: Mapping[object, tuple[str, ...]],
) -> list[object]:
    return sorted(
        _tree_children(tree, node),
        key=lambda child: (len(descendants[child]), descendants[child], str(child)),
    )


def _tree_newick(tree: Any, root: object, descendants: Mapping[object, tuple[str, ...]]) -> str:
    def emit(node: object, incoming_length: float | None = None) -> str:
        children = _ordered_children(tree, node, descendants)
        if children:
            body = (
                "("
                + ",".join(
                    emit(child, _edge_branch_length(tree, node, child)[0]) for child in children
                )
                + ")"
            )
        else:
            body = _newick_label(_leaf_label(tree, node))
        if incoming_length is not None:
            body += f":{max(float(incoming_length), 0.0):.12g}"
        return body

    return emit(root) + ";"


def _root_path_edges_by_leaf(
    tree: Any,
    root: object,
    descendants: Mapping[object, tuple[str, ...]],
) -> tuple[dict[str, frozenset[tuple[str, str]]], dict[tuple[str, str], float], int, int]:
    paths: dict[str, frozenset[tuple[str, str]]] = {}
    edge_lengths: dict[tuple[str, str], float] = {}
    missing_count = 0
    zero_count = 0

    def walk(node: object, path: tuple[tuple[str, str], ...]) -> None:
        children = _ordered_children(tree, node, descendants)
        if not children:
            paths[_leaf_label(tree, node)] = frozenset(path)
            return
        for child in children:
            length, missing = _edge_branch_length(tree, node, child)
            edge = (str(node), str(child))
            edge_lengths[edge] = float(length)
            nonlocal missing_count, zero_count
            missing_count += int(missing)
            zero_count += int(length == 0.0)
            walk(child, (*path, edge))

    walk(root, ())
    return paths, edge_lengths, missing_count, zero_count


def _split_lengths_from_tree(
    tree: Any,
    root: object,
    descendants: Mapping[object, tuple[str, ...]],
) -> dict[frozenset[str], float]:
    split_lengths: dict[frozenset[str], float] = {}
    for parent, child in tree.edges():
        child_leaves = frozenset(descendants[child])
        length, _missing = _edge_branch_length(tree, parent, child)
        split_lengths[child_leaves] = split_lengths.get(child_leaves, 0.0) + float(length)
    return split_lengths


def _internal_splits(
    split_lengths: Mapping[frozenset[str], float],
    n_leaves: int,
) -> frozenset[frozenset[str]]:
    return frozenset(split for split in split_lengths if 1 < len(split) < n_leaves)


def _cluster_size_string(labels: Sequence[int]) -> str:
    counts = Counter(int(label) for label in labels)
    return ";".join(str(count) for _label, count in sorted(counts.items()))


def _tree_path_distance(snapshot: TreeSnapshot, left: str, right: str) -> float:
    left_edges = snapshot.root_path_edges_by_leaf[left]
    right_edges = snapshot.root_path_edges_by_leaf[right]
    path_edges = left_edges ^ right_edges
    return float(sum(snapshot.edge_lengths.get(edge, 0.0) for edge in path_edges))


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _sample_leaf_pairs(
    leaves: Sequence[str],
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    ordered = tuple(sorted(leaves))
    n = len(ordered)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return []
    if max_pairs <= 0 or total_pairs <= max_pairs:
        return [(ordered[i], ordered[j]) for i in range(n) for j in range(i + 1, n)]

    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    batch = max(4096, int(max_pairs) * 2)
    while len(pairs) < max_pairs:
        left = rng.integers(0, n, size=batch)
        right = rng.integers(0, n, size=batch)
        keep = left != right
        low = np.minimum(left[keep], right[keep])
        high = np.maximum(left[keep], right[keep])
        for pair in zip(low.tolist(), high.tolist(), strict=True):
            pairs.add((int(pair[0]), int(pair[1])))
            if len(pairs) >= max_pairs:
                break
    return [(ordered[i], ordered[j]) for i, j in sorted(pairs)]


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return math.nan
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if np.nanstd(left_arr) <= 0.0 or np.nanstd(right_arr) <= 0.0:
        return math.nan
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def _cluster_path_metrics(snapshot: TreeSnapshot, *, max_pairs: int) -> dict[str, object]:
    leaves = [leaf for leaf in snapshot.leaf_labels if leaf in snapshot.label_by_leaf]
    pairs = _sample_leaf_pairs(
        leaves,
        max_pairs=max_pairs,
        seed=_stable_seed(snapshot.case_id, snapshot.run_id, "cluster_paths"),
    )
    intra: list[float] = []
    inter: list[float] = []
    for left, right in pairs:
        distance = _tree_path_distance(snapshot, left, right)
        if snapshot.label_by_leaf[left] == snapshot.label_by_leaf[right]:
            intra.append(distance)
        else:
            inter.append(distance)

    intra_mean = float(np.mean(intra)) if intra else math.nan
    inter_mean = float(np.mean(inter)) if inter else math.nan
    separation = inter_mean - intra_mean if math.isfinite(intra_mean + inter_mean) else math.nan
    ratio = (
        inter_mean / intra_mean
        if math.isfinite(intra_mean) and math.isfinite(inter_mean) and intra_mean > 0.0
        else math.nan
    )
    return {
        "sampled_path_pair_count": len(pairs),
        "sampled_intra_cluster_pair_count": len(intra),
        "sampled_inter_cluster_pair_count": len(inter),
        "intra_cluster_path_mean": intra_mean,
        "inter_cluster_path_mean": inter_mean,
        "cluster_path_separation": separation,
        "cluster_path_ratio": ratio,
    }


def snapshot_from_computed(
    *,
    case_id: str,
    test_case: int,
    case_category: str,
    result_row: Any,
    computed: Any,
) -> TreeSnapshot:
    """Build a complete tree snapshot from one successful computed result."""

    if computed.tree is None:
        raise ValueError(f"Computed result for {case_id}/{computed.run_id} has no tree.")
    tree = computed.tree
    _raise_recursion_limit_for_tree(tree)
    root = _tree_root(tree)
    descendants = _tree_descendants(tree, root)
    leaf_labels = tuple(sorted(descendants[root]))
    root_paths, edge_lengths, missing_count, zero_count = _root_path_edges_by_leaf(
        tree,
        root,
        descendants,
    )
    split_lengths = _split_lengths_from_tree(tree, root, descendants)
    labels = [int(label) for label in np.asarray(computed.labels).tolist()]
    sample_ids = [str(sample_id) for sample_id in computed.data.index.tolist()]
    label_by_leaf = {
        sample_id: int(label)
        for sample_id, label in zip(sample_ids, labels, strict=True)
        if sample_id in set(leaf_labels)
    }

    return TreeSnapshot(
        case_id=str(case_id),
        test_case=int(test_case),
        case_category=str(case_category),
        tree_inference=tree_inference_from_run_id(str(computed.run_id)),
        run_id=str(computed.run_id),
        status="ok",
        skip_reason="",
        true_clusters=_safe_int(getattr(result_row, "true_clusters", 0)),
        found_clusters=_safe_int(getattr(result_row, "found_clusters", 0)),
        ari=_finite_or_nan(getattr(result_row, "ari", math.nan)),
        nmi=_finite_or_nan(getattr(result_row, "nmi", math.nan)),
        macro_f1=_finite_or_nan(getattr(result_row, "macro_f1", math.nan)),
        purity=_finite_or_nan(getattr(result_row, "purity", math.nan)),
        leaf_labels=leaf_labels,
        label_by_leaf=label_by_leaf,
        split_lengths=split_lengths,
        internal_splits=_internal_splits(split_lengths, len(leaf_labels)),
        edge_lengths=edge_lengths,
        root_path_edges_by_leaf=root_paths,
        branch_length_missing_count=missing_count,
        branch_length_zero_count=zero_count,
        newick=_tree_newick(tree, root, descendants),
    )


def _skip_cell_row(
    *,
    case_id: str,
    test_case: int,
    case_category: str,
    row: Any,
) -> dict[str, object]:
    run_id = str(getattr(row, "run_id", ""))
    status = getattr(row, "status", "")
    status_text = str(getattr(status, "value", status))
    try:
        tree_inference = tree_inference_from_run_id(run_id)
    except ValueError:
        tree_inference = ""
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case_id),
        "test_case": int(test_case),
        "case_category": str(case_category),
        "tree_inference": tree_inference,
        "run_id": run_id,
        "status": status_text,
        "skip_reason": str(getattr(row, "skip_reason", "")),
        "true_clusters": _safe_int(getattr(row, "true_clusters", 0)),
        "found_clusters": _safe_int(getattr(row, "found_clusters", 0)),
        "ari": _finite_or_nan(getattr(row, "ari", math.nan)),
        "nmi": _finite_or_nan(getattr(row, "nmi", math.nan)),
        "macro_f1": _finite_or_nan(getattr(row, "macro_f1", math.nan)),
        "purity": _finite_or_nan(getattr(row, "purity", math.nan)),
        "n_leaves": 0,
        "n_edges": 0,
        "internal_split_count": 0,
        "total_branch_length": math.nan,
        "mean_branch_length": math.nan,
        "max_branch_length": math.nan,
        "branch_length_missing_count": 0,
        "branch_length_zero_count": 0,
        "cluster_sizes": "",
        "largest_cluster_fraction": math.nan,
        "sampled_path_pair_count": 0,
        "sampled_intra_cluster_pair_count": 0,
        "sampled_inter_cluster_pair_count": 0,
        "intra_cluster_path_mean": math.nan,
        "inter_cluster_path_mean": math.nan,
        "cluster_path_separation": math.nan,
        "cluster_path_ratio": math.nan,
    }


def cell_row_from_snapshot(snapshot: TreeSnapshot, *, max_path_pairs: int) -> dict[str, object]:
    """Return the per-cell table row for one tree snapshot."""

    branch_lengths = list(snapshot.edge_lengths.values())
    cluster_sizes = _cluster_size_string(snapshot.label_by_leaf.values())
    largest_fraction = (
        max(Counter(snapshot.label_by_leaf.values()).values()) / len(snapshot.label_by_leaf)
        if snapshot.label_by_leaf
        else math.nan
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "case_id": snapshot.case_id,
        "test_case": snapshot.test_case,
        "case_category": snapshot.case_category,
        "tree_inference": snapshot.tree_inference,
        "run_id": snapshot.run_id,
        "status": snapshot.status,
        "skip_reason": snapshot.skip_reason,
        "true_clusters": snapshot.true_clusters,
        "found_clusters": snapshot.found_clusters,
        "ari": snapshot.ari,
        "nmi": snapshot.nmi,
        "macro_f1": snapshot.macro_f1,
        "purity": snapshot.purity,
        "n_leaves": len(snapshot.leaf_labels),
        "n_edges": len(snapshot.edge_lengths),
        "internal_split_count": len(snapshot.internal_splits),
        "total_branch_length": float(np.sum(branch_lengths)) if branch_lengths else math.nan,
        "mean_branch_length": float(np.mean(branch_lengths)) if branch_lengths else math.nan,
        "max_branch_length": float(np.max(branch_lengths)) if branch_lengths else math.nan,
        "branch_length_missing_count": snapshot.branch_length_missing_count,
        "branch_length_zero_count": snapshot.branch_length_zero_count,
        "cluster_sizes": cluster_sizes,
        "largest_cluster_fraction": largest_fraction,
    }
    row.update(_cluster_path_metrics(snapshot, max_pairs=max_path_pairs))
    return row


def newick_row_from_snapshot(snapshot: TreeSnapshot) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": snapshot.case_id,
        "test_case": snapshot.test_case,
        "case_category": snapshot.case_category,
        "tree_inference": snapshot.tree_inference,
        "run_id": snapshot.run_id,
        "n_leaves": len(snapshot.leaf_labels),
        "n_edges": len(snapshot.edge_lengths),
        "branch_length_missing_count": snapshot.branch_length_missing_count,
        "newick": snapshot.newick,
    }


def _branch_length_common_metrics(
    left: Mapping[frozenset[str], float],
    right: Mapping[frozenset[str], float],
) -> dict[str, object]:
    common = sorted(set(left).intersection(right), key=lambda split: (len(split), sorted(split)))
    diffs = [float(left[split]) - float(right[split]) for split in common]
    abs_diffs = [abs(diff) for diff in diffs]
    return {
        "common_weighted_split_count": len(common),
        "common_branch_length_mae": float(np.mean(abs_diffs)) if abs_diffs else math.nan,
        "common_branch_length_rmse": (
            float(math.sqrt(np.mean(np.square(diffs)))) if diffs else math.nan
        ),
        "common_branch_length_corr": _safe_corr(
            [float(left[split]) for split in common],
            [float(right[split]) for split in common],
        ),
    }


def _pairwise_path_metrics(
    left: TreeSnapshot,
    right: TreeSnapshot,
    *,
    max_pairs: int,
) -> dict[str, object]:
    common_leaves = sorted(set(left.leaf_labels).intersection(right.leaf_labels))
    pairs = _sample_leaf_pairs(
        common_leaves,
        max_pairs=max_pairs,
        seed=_stable_seed(left.case_id, left.run_id, right.run_id, "pairwise_paths"),
    )
    left_distances: list[float] = []
    right_distances: list[float] = []
    for left_leaf, right_leaf in pairs:
        left_distances.append(_tree_path_distance(left, left_leaf, right_leaf))
        right_distances.append(_tree_path_distance(right, left_leaf, right_leaf))
    diffs = np.asarray(left_distances, dtype=float) - np.asarray(right_distances, dtype=float)
    return {
        "sampled_leaf_path_pair_count": len(pairs),
        "leaf_path_mae": float(np.mean(np.abs(diffs))) if len(diffs) else math.nan,
        "leaf_path_rmse": (float(math.sqrt(np.mean(diffs * diffs))) if len(diffs) else math.nan),
        "leaf_path_corr": _safe_corr(left_distances, right_distances),
    }


def _label_agreement(left: TreeSnapshot, right: TreeSnapshot) -> float:
    common_leaves = sorted(set(left.label_by_leaf).intersection(right.label_by_leaf))
    if len(common_leaves) < 2:
        return math.nan
    return float(
        adjusted_rand_score(
            [left.label_by_leaf[leaf] for leaf in common_leaves],
            [right.label_by_leaf[leaf] for leaf in common_leaves],
        )
    )


def _read_brancharchitect_tree(
    adapter: BranchArchitectAdapter,
    newick: str,
    temp_dir: Path,
    name: str,
) -> Any:
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{_slugify(name)}.nwk"
    path.write_text(newick, encoding="utf-8")
    return adapter.read_newick(str(path), treat_zero_as_epsilon=True)


def _brancharchitect_pair_metrics(
    left: TreeSnapshot,
    right: TreeSnapshot,
    adapter: BranchArchitectAdapter,
    *,
    temp_dir: Path,
    enable_interpolation: bool,
    interpolation_max_leaves: int,
) -> dict[str, object]:
    base: dict[str, object] = {
        "brancharchitect_status": adapter.status,
        "brancharchitect_error": adapter.error,
        "brancharchitect_rf": math.nan,
        "brancharchitect_weighted_rf": math.nan,
        "brancharchitect_interpolation_status": "not_requested"
        if not enable_interpolation
        else "not_available",
        "brancharchitect_processing_time": math.nan,
        "brancharchitect_frame_count": math.nan,
        "brancharchitect_temporal_event_count": math.nan,
        "brancharchitect_spr_event_count": math.nan,
        "brancharchitect_spr_total_hops": math.nan,
        "brancharchitect_spr_total_branch_length": math.nan,
    }
    if not adapter.has_distances:
        return base
    try:
        left_tree = _read_brancharchitect_tree(adapter, left.newick, temp_dir, left.run_id)
        right_tree = _read_brancharchitect_tree(adapter, right.newick, temp_dir, right.run_id)
        if hasattr(right_tree, "initialize_split_indices"):
            right_tree.initialize_split_indices(left_tree.taxa_encoding)
        base["brancharchitect_rf"] = float(
            adapter.relative_robinson_foulds_distance(left_tree, right_tree)
        )
        base["brancharchitect_weighted_rf"] = float(
            adapter.weighted_robinson_foulds_distance(left_tree, right_tree)
        )
    except Exception as exc:
        base["brancharchitect_status"] = "distance_error"
        base["brancharchitect_error"] = f"{type(exc).__name__}: {exc}"
        return base

    if not enable_interpolation:
        return base
    if not adapter.has_interpolation:
        base["brancharchitect_interpolation_status"] = adapter.status
        return base
    if max(len(left.leaf_labels), len(right.leaf_labels)) > interpolation_max_leaves:
        base["brancharchitect_interpolation_status"] = "skipped_leaf_limit"
        return base

    try:
        config = adapter.pipeline_config(enable_rooting=False)
        pipeline = adapter.tree_interpolation_pipeline(config=config)
        result = pipeline.process_trees([left_tree, right_tree])
        events = list(result.get("temporal_events", []))
        spr_events = [event for event in events if event.get("event_type") == "spr_move"]
        base.update(
            {
                "brancharchitect_interpolation_status": "ok",
                "brancharchitect_processing_time": float(result.get("processing_time", math.nan)),
                "brancharchitect_frame_count": len(result.get("frames", [])),
                "brancharchitect_temporal_event_count": len(events),
                "brancharchitect_spr_event_count": len(spr_events),
                "brancharchitect_spr_total_hops": int(
                    sum(_safe_int(event.get("total_hops")) for event in spr_events)
                ),
                "brancharchitect_spr_total_branch_length": float(
                    sum(
                        _finite_or_nan(event.get("total_branch_length"))
                        for event in spr_events
                        if math.isfinite(_finite_or_nan(event.get("total_branch_length")))
                    )
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - optional BranchArchitect runtime
        base["brancharchitect_interpolation_status"] = "interpolation_error"
        base["brancharchitect_error"] = f"{type(exc).__name__}: {exc}"
    return base


def compare_snapshots(
    left: TreeSnapshot,
    right: TreeSnapshot,
    *,
    max_path_pairs: int,
    brancharchitect: BranchArchitectAdapter | None = None,
    temp_dir: Path | None = None,
    enable_brancharchitect_interpolation: bool = False,
    brancharchitect_interpolation_max_leaves: int = 120,
) -> dict[str, object]:
    """Compare two complete tree snapshots from the same case."""

    left_splits = set(left.internal_splits)
    right_splits = set(right.internal_splits)
    split_union = left_splits | right_splits
    split_intersection = left_splits & right_splits
    all_weighted_splits = set(left.split_lengths) | set(right.split_lengths)
    weighted_l1 = float(
        sum(
            abs(left.split_lengths.get(split, 0.0) - right.split_lengths.get(split, 0.0))
            for split in all_weighted_splits
        )
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "case_id": left.case_id,
        "test_case": left.test_case,
        "case_category": left.case_category,
        "left_tree_inference": left.tree_inference,
        "right_tree_inference": right.tree_inference,
        "left_run_id": left.run_id,
        "right_run_id": right.run_id,
        "leaf_set_equal": set(left.leaf_labels) == set(right.leaf_labels),
        "n_common_leaves": len(set(left.leaf_labels).intersection(right.leaf_labels)),
        "left_internal_split_count": len(left_splits),
        "right_internal_split_count": len(right_splits),
        "common_internal_split_count": len(split_intersection),
        "rooted_internal_rf": len(left_splits ^ right_splits),
        "rooted_internal_rf_relative": (
            len(left_splits ^ right_splits) / (len(left_splits) + len(right_splits))
            if len(left_splits) + len(right_splits)
            else 0.0
        ),
        "rooted_internal_split_jaccard": (
            len(split_intersection) / len(split_union) if split_union else 1.0
        ),
        "rooted_weighted_split_l1": weighted_l1,
        "left_found_clusters": left.found_clusters,
        "right_found_clusters": right.found_clusters,
        "predicted_label_ari_between_topologies": _label_agreement(left, right),
    }
    row.update(_branch_length_common_metrics(left.split_lengths, right.split_lengths))
    row.update(_pairwise_path_metrics(left, right, max_pairs=max_path_pairs))
    if brancharchitect is not None and temp_dir is not None:
        row.update(
            _brancharchitect_pair_metrics(
                left,
                right,
                brancharchitect,
                temp_dir=temp_dir,
                enable_interpolation=enable_brancharchitect_interpolation,
                interpolation_max_leaves=brancharchitect_interpolation_max_leaves,
            )
        )
    else:
        row.update(
            {
                "brancharchitect_status": "not_requested",
                "brancharchitect_error": "",
                "brancharchitect_rf": math.nan,
                "brancharchitect_weighted_rf": math.nan,
                "brancharchitect_interpolation_status": "not_requested",
                "brancharchitect_processing_time": math.nan,
                "brancharchitect_frame_count": math.nan,
                "brancharchitect_temporal_event_count": math.nan,
                "brancharchitect_spr_event_count": math.nan,
                "brancharchitect_spr_total_hops": math.nan,
                "brancharchitect_spr_total_branch_length": math.nan,
            }
        )
    return row


def _case_lookup(
    suite: str,
    case_names: Sequence[str],
) -> tuple[list[tuple[int, dict[str, object]]], list[str]]:
    all_cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): (idx, dict(case)) for idx, case in enumerate(all_cases, start=1)}
    missing = [name for name in case_names if name not in by_name]
    selected = [by_name[name] for name in case_names if name in by_name]
    return selected, missing


def rebuild_snapshots(
    *,
    suite: str,
    case_names: Sequence[str],
    significance_level: float = DEFAULT_SIBLING_ALPHA,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
) -> tuple[list[TreeSnapshot], list[dict[str, object]]]:
    """Rerun target adaptive-K NNLS cells and return complete tree snapshots."""

    spec = METHOD_SPECS[TARGET_TREE_CONSENSUS_METHOD]
    selected_cases, missing = _case_lookup(suite, case_names)
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing!r}")

    snapshots: list[TreeSnapshot] = []
    skip_rows: list[dict[str, object]] = []
    for case_idx, case in selected_cases:
        case_name = str(case["name"])
        inputs = prepare_case_inputs(case, [TARGET_TREE_CONSENSUS_METHOD])
        for params in spec.param_grid:
            row, computed, _method_audit = run_single_method_once(
                method_id=TARGET_TREE_CONSENSUS_METHOD,
                spec=spec,
                params=dict(params),
                case_idx=case_idx,
                case_name=case_name,
                tc_seed=case["seed"],
                significance_level=significance_level,
                edge_alpha=edge_alpha,
                data_t=inputs.data,
                y_t=inputs.labels,
                x_original=inputs.original_features,
                meta=inputs.metadata,
                distance_matrix=inputs.distance_matrix,
                distance_condensed=inputs.distance_condensed,
                matrix_audit=False,
            )
            if row.status == BenchmarkRunStatus.OK and computed is not None:
                snapshots.append(
                    snapshot_from_computed(
                        case_id=case_name,
                        test_case=case_idx,
                        case_category=str(case.get("category", "")),
                        result_row=row,
                        computed=computed,
                    )
                )
            else:
                skip_rows.append(
                    _skip_cell_row(
                        case_id=case_name,
                        test_case=case_idx,
                        case_category=str(case.get("category", "")),
                        row=row,
                    )
                )
    return snapshots, skip_rows


def _write_report(
    *,
    output_dir: Path,
    cell_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame,
    case_summary: pd.DataFrame,
    adapter: BranchArchitectAdapter,
    case_names: Sequence[str],
    brancharchitect_path: Path | None,
    enable_brancharchitect_interpolation: bool,
) -> Path:
    report_path = output_dir / TREE_REPORT_NAME
    ok_cells = (
        cell_rows[cell_rows["status"].astype(str).eq("ok")] if not cell_rows.empty else cell_rows
    )
    skip_cells = (
        cell_rows[~cell_rows["status"].astype(str).eq("ok")] if not cell_rows.empty else cell_rows
    )
    lines = [
        "# BranchArchitect Tree Comparison Gate",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- generated_by: `{GENERATED_BY}`",
        f"- generated_at_utc: `{format_timestamp_utc()}`",
        f"- target_method: `{TARGET_TREE_CONSENSUS_METHOD}`",
        f"- case_count: `{len(set(case_names))}`",
        f"- ok_tree_cells: `{len(ok_cells)}`",
        f"- skipped_cells: `{len(skip_cells)}`",
        f"- pairwise_comparisons: `{len(pairwise_rows)}`",
        f"- brancharchitect_path: `{brancharchitect_path or ''}`",
        f"- brancharchitect_status: `{adapter.status}`",
        f"- brancharchitect_interpolation_requested: `{enable_brancharchitect_interpolation}`",
        "",
        "## Interpretation",
        "",
        "This report compares complete rebuilt tree objects, not only stopped traversal traces. "
        "The cluster-path columns are label-free with respect to ground truth: they compare "
        "branch-length tree distances within and between the predicted clusters.",
        "",
        "BranchArchitect is optional. When configured, the pairwise table adds "
        "BranchArchitect RF and weighted RF metrics; interpolation movement-path metrics are "
        "filled only when BranchArchitect and its optional runtime dependencies are present.",
        "",
    ]
    if pairwise_rows.empty:
        lines.extend(["## Pairwise Summary", "", "No pairwise tree comparisons were produced.", ""])
    else:
        metrics = {
            "median_rooted_internal_rf_relative": pairwise_rows[
                "rooted_internal_rf_relative"
            ].median(),
            "median_rooted_weighted_split_l1": pairwise_rows["rooted_weighted_split_l1"].median(),
            "median_leaf_path_rmse": pairwise_rows["leaf_path_rmse"].median(),
            "median_predicted_label_ari_between_topologies": pairwise_rows[
                "predicted_label_ari_between_topologies"
            ].median(),
        }
        lines.extend(["## Pairwise Summary", ""])
        for key, value in metrics.items():
            lines.append(f"- {key}: `{_finite_or_nan(value):.6g}`")
        lines.append("")
        top = pairwise_rows.sort_values(
            ["rooted_internal_rf_relative", "leaf_path_rmse"],
            ascending=[False, False],
        ).head(10)
        lines.extend(["## Largest Topology Differences", ""])
        for row in top.itertuples(index=False):
            lines.append(
                "- "
                f"{row.case_id}: {row.left_tree_inference} vs {row.right_tree_inference}, "
                f"relative_rooted_RF={row.rooted_internal_rf_relative:.3g}, "
                f"leaf_path_RMSE={row.leaf_path_rmse:.3g}, "
                f"cluster_label_ARI={row.predicted_label_ari_between_topologies:.3g}"
            )
        lines.append("")
    if not case_summary.empty:
        lines.extend(["## Case Summary", ""])
        for row in case_summary.sort_values(
            ["min_predicted_label_ari_between_topologies", "max_rooted_internal_rf_relative"],
            ascending=[True, False],
        ).itertuples(index=False):
            lines.append(
                "- "
                f"{row.case_id}: ok_cells={row.ok_tree_cells}, "
                f"cluster_count_range={row.min_found_clusters}-{row.max_found_clusters}, "
                f"max_relative_rooted_RF={row.max_rooted_internal_rf_relative:.3g}, "
                f"max_leaf_path_RMSE={row.max_leaf_path_rmse:.3g}, "
                f"min_cluster_label_ARI={row.min_predicted_label_ari_between_topologies:.3g}"
            )
        lines.append("")
    if not skip_cells.empty:
        by_reason = skip_cells["skip_reason"].fillna("").astype(str).value_counts().head(10)
        lines.extend(["## Skipped Cells", ""])
        for reason, count in by_reason.items():
            lines.append(f"- `{count}`: {reason}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _case_summary(cell_rows: pd.DataFrame, pairwise_rows: pd.DataFrame) -> pd.DataFrame:
    if cell_rows.empty:
        return pd.DataFrame()

    ok_cells = cell_rows[cell_rows["status"].astype(str).eq("ok")].copy()
    skip_cells = cell_rows[~cell_rows["status"].astype(str).eq("ok")].copy()
    ok_summary = (
        ok_cells.groupby(["case_id", "test_case", "case_category"], dropna=False)
        .agg(
            ok_tree_cells=("case_id", "size"),
            min_found_clusters=("found_clusters", "min"),
            max_found_clusters=("found_clusters", "max"),
            median_found_clusters=("found_clusters", "median"),
            max_largest_cluster_fraction=("largest_cluster_fraction", "max"),
            median_total_branch_length=("total_branch_length", "median"),
            max_total_branch_length=("total_branch_length", "max"),
            finite_cluster_path_rows=(
                "cluster_path_separation",
                lambda values: int(values.notna().sum()),
            ),
        )
        .reset_index()
    )
    skip_summary = (
        skip_cells.groupby("case_id", dropna=False)
        .agg(skipped_tree_cells=("case_id", "size"))
        .reset_index()
        if not skip_cells.empty
        else pd.DataFrame(columns=["case_id", "skipped_tree_cells"])
    )
    if pairwise_rows.empty:
        pair_summary = pd.DataFrame(columns=["case_id"])
    else:
        pair_summary = (
            pairwise_rows.groupby("case_id", dropna=False)
            .agg(
                pairwise_comparisons=("case_id", "size"),
                median_rooted_internal_rf_relative=("rooted_internal_rf_relative", "median"),
                max_rooted_internal_rf_relative=("rooted_internal_rf_relative", "max"),
                median_rooted_weighted_split_l1=("rooted_weighted_split_l1", "median"),
                max_rooted_weighted_split_l1=("rooted_weighted_split_l1", "max"),
                median_leaf_path_rmse=("leaf_path_rmse", "median"),
                max_leaf_path_rmse=("leaf_path_rmse", "max"),
                min_predicted_label_ari_between_topologies=(
                    "predicted_label_ari_between_topologies",
                    "min",
                ),
                median_predicted_label_ari_between_topologies=(
                    "predicted_label_ari_between_topologies",
                    "median",
                ),
                brancharchitect_interpolation_ok_pairs=(
                    "brancharchitect_interpolation_status",
                    lambda values: int(values.astype(str).eq("ok").sum()),
                ),
                brancharchitect_interpolation_skipped_leaf_limit_pairs=(
                    "brancharchitect_interpolation_status",
                    lambda values: int(values.astype(str).eq("skipped_leaf_limit").sum()),
                ),
            )
            .reset_index()
        )

    summary = ok_summary.merge(skip_summary, on="case_id", how="left").merge(
        pair_summary,
        on="case_id",
        how="left",
    )
    summary["skipped_tree_cells"] = (
        pd.to_numeric(summary["skipped_tree_cells"], errors="coerce").fillna(0).astype(int)
    )
    summary.insert(0, "schema_version", SCHEMA_VERSION)
    return summary.sort_values(["test_case", "case_id"], kind="mergesort")


def write_brancharchitect_tree_comparison_artifacts(
    *,
    snapshots: Sequence[TreeSnapshot],
    skip_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    case_names: Sequence[str],
    max_path_pairs: int = 20_000,
    brancharchitect_path: str | Path | None = None,
    enable_brancharchitect_interpolation: bool = False,
    brancharchitect_interpolation_max_leaves: int = 120,
    source_paths: Sequence[Path] = (),
) -> dict[str, Path]:
    """Write comparison CSV/Markdown artifacts for rebuilt tree snapshots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = load_brancharchitect_adapter(
        brancharchitect_path,
        require_interpolation=enable_brancharchitect_interpolation,
    )
    resolved_brancharchitect_path = _brancharchitect_path(brancharchitect_path)

    cell_records = [
        cell_row_from_snapshot(snapshot, max_path_pairs=max_path_pairs) for snapshot in snapshots
    ]
    cell_records.extend(dict(row) for row in skip_rows)
    cell_rows = pd.DataFrame(cell_records)
    if not cell_rows.empty:
        cell_rows = cell_rows.sort_values(
            ["test_case", "case_id", "tree_inference", "run_id"],
            kind="mergesort",
        )

    newick_rows = pd.DataFrame([newick_row_from_snapshot(snapshot) for snapshot in snapshots])
    if not newick_rows.empty:
        newick_rows = newick_rows.sort_values(
            ["test_case", "case_id", "tree_inference", "run_id"],
            kind="mergesort",
        )

    pairwise_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="tbs_brancharchitect_") as temp:
        temp_dir = Path(temp)
        for case_id, case_snapshots_iter in _group_snapshots_by_case(snapshots).items():
            case_snapshots = sorted(
                case_snapshots_iter,
                key=lambda snapshot: (snapshot.tree_inference, snapshot.run_id),
            )
            for left, right in combinations(case_snapshots, 2):
                pairwise_records.append(
                    compare_snapshots(
                        left,
                        right,
                        max_path_pairs=max_path_pairs,
                        brancharchitect=adapter,
                        temp_dir=temp_dir / _slugify(case_id),
                        enable_brancharchitect_interpolation=(enable_brancharchitect_interpolation),
                        brancharchitect_interpolation_max_leaves=(
                            brancharchitect_interpolation_max_leaves
                        ),
                    )
                )
    pairwise_rows = pd.DataFrame(pairwise_records)
    if not pairwise_rows.empty:
        pairwise_rows = pairwise_rows.sort_values(
            ["test_case", "case_id", "left_tree_inference", "right_tree_inference"],
            kind="mergesort",
        )

    cell_path = output_dir / TREE_CELLS_NAME
    newick_path = output_dir / TREE_NEWICK_NAME
    pairwise_path = output_dir / TREE_PAIRWISE_NAME
    case_summary_path = output_dir / TREE_CASE_SUMMARY_NAME
    case_summary = _case_summary(cell_rows, pairwise_rows)
    cell_rows.to_csv(cell_path, index=False)
    newick_rows.to_csv(newick_path, index=False)
    pairwise_rows.to_csv(pairwise_path, index=False)
    case_summary.to_csv(case_summary_path, index=False)
    report_path = _write_report(
        output_dir=output_dir,
        cell_rows=cell_rows,
        pairwise_rows=pairwise_rows,
        case_summary=case_summary,
        adapter=adapter,
        case_names=case_names,
        brancharchitect_path=resolved_brancharchitect_path,
        enable_brancharchitect_interpolation=enable_brancharchitect_interpolation,
    )
    manifest_path = output_dir / TREE_MANIFEST_NAME
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "target_method": TARGET_TREE_CONSENSUS_METHOD,
        "case_names": list(case_names),
        "max_path_pairs": int(max_path_pairs),
        "brancharchitect_path": str(resolved_brancharchitect_path or ""),
        "brancharchitect_status": adapter.status,
        "brancharchitect_error": adapter.error,
        "enable_brancharchitect_interpolation": bool(enable_brancharchitect_interpolation),
        "brancharchitect_interpolation_max_leaves": int(brancharchitect_interpolation_max_leaves),
        "source_paths": [str(path) for path in source_paths],
        "outputs": {
            "tree_cells": str(cell_path),
            "tree_case_summary": str(case_summary_path),
            "tree_newick": str(newick_path),
            "tree_pairwise": str(pairwise_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return {
        "tree_cells": cell_path,
        "tree_case_summary": case_summary_path,
        "tree_newick": newick_path,
        "tree_pairwise": pairwise_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def _group_snapshots_by_case(
    snapshots: Sequence[TreeSnapshot],
) -> dict[str, list[TreeSnapshot]]:
    grouped: dict[str, list[TreeSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.case_id, []).append(snapshot)
    return grouped


def run_brancharchitect_tree_comparison(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    suite: str = "full",
    case_names: Sequence[str] = (),
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    max_cases: int | None = None,
    max_path_pairs: int = 20_000,
    significance_level: float = DEFAULT_SIBLING_ALPHA,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    brancharchitect_path: str | Path | None = None,
    enable_brancharchitect_interpolation: bool = False,
    brancharchitect_interpolation_max_leaves: int = 120,
) -> dict[str, Path]:
    """Rebuild target cells and write BranchArchitect/tree comparison artifacts."""

    selected_case_names = tuple(case_names) or _load_default_case_names(taxonomy_path)
    if max_cases is not None:
        selected_case_names = selected_case_names[: int(max_cases)]
    snapshots, skip_rows = rebuild_snapshots(
        suite=suite,
        case_names=selected_case_names,
        significance_level=significance_level,
        edge_alpha=edge_alpha,
    )
    return write_brancharchitect_tree_comparison_artifacts(
        snapshots=snapshots,
        skip_rows=skip_rows,
        output_dir=output_dir,
        case_names=selected_case_names,
        max_path_pairs=max_path_pairs,
        brancharchitect_path=brancharchitect_path,
        enable_brancharchitect_interpolation=enable_brancharchitect_interpolation,
        brancharchitect_interpolation_max_leaves=brancharchitect_interpolation_max_leaves,
        source_paths=(taxonomy_path,),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", type=_parse_names, default=())
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-path-pairs", type=int, default=20_000)
    parser.add_argument("--significance-level", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--brancharchitect-path", default=None)
    parser.add_argument(
        "--enable-brancharchitect-interpolation",
        action="store_true",
        help="Run BranchArchitect interpolation movement analysis for leaf-limited pairs.",
    )
    parser.add_argument("--brancharchitect-interpolation-max-leaves", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_brancharchitect_tree_comparison(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=args.case_names,
        taxonomy_path=args.taxonomy_path,
        max_cases=args.max_cases,
        max_path_pairs=args.max_path_pairs,
        significance_level=args.significance_level,
        edge_alpha=args.edge_alpha,
        brancharchitect_path=args.brancharchitect_path,
        enable_brancharchitect_interpolation=args.enable_brancharchitect_interpolation,
        brancharchitect_interpolation_max_leaves=(args.brancharchitect_interpolation_max_leaves),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
