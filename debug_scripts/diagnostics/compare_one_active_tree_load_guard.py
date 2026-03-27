"""Compare a tree-level low-leverage load guard against the current KL baseline.

Experiment-only. Production code is unchanged.

Workflow:
1. Promote all one-active nodes to deterministic 1D candidates.
2. After parent eigenvalues are available, identify low-leverage candidates
   using a data-driven global split on variance ratios.
3. Compute per-tree low-leverage load:
   - count
   - total n_rows
4. If the tree exceeds the learned load threshold, revert only the
   low-leverage one-active nodes back to baseline semantics.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from debug_scripts.diagnostics.compare_one_active_feature_1d_strength import (
    _run_kl_suite,
    _summarize,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.types import (
    NodeSpectralResult,
)


@dataclass(frozen=True)
class Thresholds:
    low_ratio_threshold: float
    heavy_count_threshold: float
    heavy_rows_threshold: float


def _canonical_one_feature_basis(pca_projection: np.ndarray | None) -> bool:
    if pca_projection is None:
        return False
    basis = np.asarray(pca_projection, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != 1:
        return False
    nonzero = np.flatnonzero(np.abs(basis[0]) > 1e-12)
    if nonzero.size != 1:
        return False
    return bool(np.isclose(np.linalg.norm(basis[0]), 1.0))


def _largest_gap_threshold(values: list[float]) -> float | None:
    values = sorted(float(v) for v in values if np.isfinite(v))
    if len(values) < 2:
        return None
    best_gap = None
    best_pair = None
    for left, right in zip(values, values[1:]):
        gap = right - left
        if best_gap is None or gap > best_gap:
            best_gap = gap
            best_pair = (left, right)
    if best_pair is None:
        return None
    left, right = best_pair
    return float((left + right) / 2.0)


def _fit_log_kmeans_split(values: list[float]) -> float | None:
    values = [float(v) for v in values if np.isfinite(v) and v > 0]
    if len(values) < 2:
        return None
    log_values = np.log(np.asarray(values, dtype=np.float64)).reshape(-1, 1)
    labels = KMeans(n_clusters=2, random_state=0, n_init=20).fit_predict(log_values)
    centers = np.asarray(
        KMeans(n_clusters=2, random_state=0, n_init=20).fit(log_values).cluster_centers_
    ).ravel()
    low_label = int(np.argmin(centers))
    high_label = 1 - low_label
    low_vals = np.exp(log_values[labels == low_label].ravel())
    high_vals = np.exp(log_values[labels == high_label].ravel())
    if low_vals.size == 0 or high_vals.size == 0:
        return None
    low_max = float(np.max(low_vals))
    high_min = float(np.min(high_vals))
    if low_max < high_min:
        return float(np.sqrt(low_max * high_min))
    return float(np.exp(np.mean([centers[low_label], centers[high_label]])))


def _fit_tree_load_thresholds() -> Thresholds:
    from benchmarks.shared.cases import get_default_test_cases
    from benchmarks.shared.runners.kl_runner import _run_kl_method
    from benchmarks.shared.util.case_inputs import prepare_case_inputs
    from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
        child_parent_projected_wald,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection import projected_wald
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
        marchenko_pastur,
        tree_estimator,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (
        wald_statistic,
    )

    original_tree_process_node = tree_estimator._process_node
    original_mp_process_node = marchenko_pastur._process_node
    original_projected_wald_kernel = projected_wald.run_projected_wald_kernel
    original_child_parent_projected_wald_kernel = (
        child_parent_projected_wald.run_projected_wald_kernel
    )
    original_sibling_projected_wald_kernel = wald_statistic.run_projected_wald_kernel

    ratio_rows: list[dict[str, Any]] = []

    def patched_run_projected_wald_kernel(
        z: np.ndarray,
        *,
        seed: int | None = None,
        minimum_projection_dimension: int | None = None,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
        child_pca_projections: list[np.ndarray] | None = None,
        whitening: str = "per_component",
    ):
        if _canonical_one_feature_basis(pca_projection):
            spectral_k = 1
        return original_projected_wald_kernel(
            z,
            seed=seed,
            minimum_projection_dimension=minimum_projection_dimension,
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=pca_eigenvalues,
            whitening=whitening,
        )

    def patched_process_node(
        spectral_task,
        full_feature_matrix,
        dimension_method,
        minimum_projection_dimension,
        feature_count,
        compute_eigendecomposition_outputs,
    ):
        del dimension_method
        descendant_leaf_row_indices = spectral_task.row_indices
        if len(descendant_leaf_row_indices) < 2:
            return original_tree_process_node(
                spectral_task,
                full_feature_matrix,
                "marchenko_pastur",
                minimum_projection_dimension,
                feature_count,
                compute_eigendecomposition_outputs,
            )

        descendant_leaf_feature_rows = full_feature_matrix[descendant_leaf_row_indices, :]
        if spectral_task.internal_distributions:
            descendant_feature_matrix = np.vstack(
                [
                    descendant_leaf_feature_rows,
                    np.asarray(spectral_task.internal_distributions, dtype=np.float64),
                ]
            )
        else:
            descendant_feature_matrix = descendant_leaf_feature_rows

        variances = np.var(descendant_feature_matrix, axis=0)
        active_indices = np.flatnonzero(variances > 0)
        if active_indices.size == 1:
            feature_index = int(active_indices[0])
            ratio_rows.append(
                {
                    "case_id": None,
                    "node_id": spectral_task.node_id,
                    "active_feature": feature_index,
                    "active_variance": float(variances[feature_index]),
                    "n_leaves": int(len(descendant_leaf_row_indices)),
                    "n_rows": int(descendant_feature_matrix.shape[0]),
                    "has_internal": bool(spectral_task.internal_distributions),
                }
            )
            if compute_eigendecomposition_outputs:
                projection = np.zeros((1, int(feature_count)), dtype=np.float64)
                projection[0, feature_index] = 1.0
                eigenvalues = np.array([1.0], dtype=np.float64)
            else:
                projection = None
                eigenvalues = None
            return NodeSpectralResult(
                node_id=spectral_task.node_id,
                projection_dimension=1,
                projection_matrix=projection,
                eigenvalues=eigenvalues,
            )

        return original_tree_process_node(
            spectral_task,
            full_feature_matrix,
            "marchenko_pastur",
            minimum_projection_dimension,
            feature_count,
            compute_eigendecomposition_outputs,
        )

    tree_estimator._process_node = patched_process_node
    marchenko_pastur._process_node = patched_process_node
    projected_wald.run_projected_wald_kernel = patched_run_projected_wald_kernel
    child_parent_projected_wald.run_projected_wald_kernel = patched_run_projected_wald_kernel
    wald_statistic.run_projected_wald_kernel = patched_run_projected_wald_kernel

    try:
        case_loads: list[dict[str, Any]] = []
        ratio_cursor = 0
        for case in get_default_test_cases():
            case_name = str(case["name"])
            data_df, _y, _x_original, _meta, distance_condensed, _dm, _pre = prepare_case_inputs(
                case,
                ["kl"],
            )
            tree = _run_kl_method(data_df, distance_condensed, 0.01).extra["tree"]
            case_rows = ratio_rows[ratio_cursor:]
            ratio_cursor = len(ratio_rows)
            pca_eigenvalues = tree.annotations_df.attrs.get("_pca_eigenvalues", {})
            for row in case_rows:
                row["case_id"] = case_name
                parent_node = next(iter(tree.predecessors(row["node_id"])), None)
                parent_lambda_max = None
                if parent_node is not None:
                    parent_eigs = pca_eigenvalues.get(parent_node)
                    if parent_eigs is not None:
                        parent_eigs = np.asarray(parent_eigs, dtype=np.float64)
                        if parent_eigs.size > 0:
                            parent_lambda_max = float(np.max(parent_eigs))
                row["parent_lambda_max"] = parent_lambda_max
                row["variance_ratio"] = (
                    float(row["active_variance"] / parent_lambda_max)
                    if parent_lambda_max is not None and parent_lambda_max > 0
                    else None
                )

        valid_ratios = [r["variance_ratio"] for r in ratio_rows if r.get("variance_ratio") is not None]
        low_ratio_threshold = _fit_log_kmeans_split(valid_ratios)
        if low_ratio_threshold is None:
            raise RuntimeError("Unable to fit low-ratio split for one-active nodes.")

        ratio_df = pd.DataFrame(ratio_rows)
        ratio_df["is_low_ratio"] = (
            ratio_df["variance_ratio"].notna() & (ratio_df["variance_ratio"] <= low_ratio_threshold)
        )
        load_df = (
            ratio_df.groupby("case_id", dropna=False)
            .agg(
                low_count=("is_low_ratio", "sum"),
                low_rows=("n_rows", lambda s: int(ratio_df.loc[s.index, "n_rows"][ratio_df.loc[s.index, "is_low_ratio"]].sum())),
            )
            .reset_index()
        )
        positive_counts = [int(v) for v in load_df["low_count"].tolist() if int(v) > 0]
        positive_rows = [float(v) for v in load_df["low_rows"].tolist() if float(v) > 0]
        heavy_count_threshold = _largest_gap_threshold(positive_counts)
        heavy_rows_threshold = _largest_gap_threshold(positive_rows)
        if heavy_count_threshold is None or heavy_rows_threshold is None:
            raise RuntimeError("Unable to fit heavy-load thresholds.")
        return Thresholds(
            low_ratio_threshold=float(low_ratio_threshold),
            heavy_count_threshold=float(heavy_count_threshold),
            heavy_rows_threshold=float(heavy_rows_threshold),
        )
    finally:
        tree_estimator._process_node = original_tree_process_node
        marchenko_pastur._process_node = original_mp_process_node
        projected_wald.run_projected_wald_kernel = original_projected_wald_kernel
        child_parent_projected_wald.run_projected_wald_kernel = (
            original_child_parent_projected_wald_kernel
        )
        wald_statistic.run_projected_wald_kernel = original_sibling_projected_wald_kernel


@contextmanager
def _patch_tree_load_guard(thresholds: Thresholds) -> dict[str, Any]:
    from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
        child_parent_divergence,
        child_parent_projected_wald,
        child_parent_spectral_decomposition,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection import projected_wald
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
        marchenko_pastur,
        tree_estimator,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (
        wald_statistic,
    )

    original_tree_process_node = tree_estimator._process_node
    original_mp_process_node = marchenko_pastur._process_node
    original_projected_wald_kernel = projected_wald.run_projected_wald_kernel
    original_child_parent_projected_wald_kernel = (
        child_parent_projected_wald.run_projected_wald_kernel
    )
    original_sibling_projected_wald_kernel = wald_statistic.run_projected_wald_kernel
    original_compute_child_parent_context = (
        child_parent_spectral_decomposition.compute_child_parent_spectral_context
    )
    original_compute_child_parent_context_alias = (
        child_parent_divergence.compute_child_parent_spectral_context
    )

    pending_one_active_meta: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "thresholds": {
            "low_ratio_threshold": thresholds.low_ratio_threshold,
            "heavy_count_threshold": thresholds.heavy_count_threshold,
            "heavy_rows_threshold": thresholds.heavy_rows_threshold,
        },
        "candidate_nodes": 0,
        "allowed_nodes": 0,
        "blocked_nodes": 0,
        "heavy_trees": [],
        "audit_rows": [],
    }

    def patched_run_projected_wald_kernel(
        z: np.ndarray,
        *,
        seed: int | None = None,
        minimum_projection_dimension: int | None = None,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
        child_pca_projections: list[np.ndarray] | None = None,
        whitening: str = "per_component",
    ):
        if _canonical_one_feature_basis(pca_projection):
            spectral_k = 1
        return original_projected_wald_kernel(
            z,
            seed=seed,
            minimum_projection_dimension=minimum_projection_dimension,
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=pca_eigenvalues,
            whitening=whitening,
        )

    def patched_process_node(
        spectral_task,
        full_feature_matrix,
        dimension_method,
        minimum_projection_dimension,
        feature_count,
        compute_eigendecomposition_outputs,
    ):
        del dimension_method
        descendant_leaf_row_indices = spectral_task.row_indices
        if len(descendant_leaf_row_indices) < 2:
            return original_tree_process_node(
                spectral_task,
                full_feature_matrix,
                "marchenko_pastur",
                minimum_projection_dimension,
                feature_count,
                compute_eigendecomposition_outputs,
            )

        descendant_leaf_feature_rows = full_feature_matrix[descendant_leaf_row_indices, :]
        if spectral_task.internal_distributions:
            descendant_feature_matrix = np.vstack(
                [
                    descendant_leaf_feature_rows,
                    np.asarray(spectral_task.internal_distributions, dtype=np.float64),
                ]
            )
        else:
            descendant_feature_matrix = descendant_leaf_feature_rows

        variances = np.var(descendant_feature_matrix, axis=0)
        active_indices = np.flatnonzero(variances > 0)
        if active_indices.size == 1:
            feature_index = int(active_indices[0])
            row_sums = descendant_feature_matrix.sum(axis=1)
            row_norms = np.linalg.norm(descendant_feature_matrix, axis=1)
            pending_one_active_meta[spectral_task.node_id] = {
                "active_feature": feature_index,
                "active_variance": float(variances[feature_index]),
                "n_leaves": int(len(descendant_leaf_row_indices)),
                "n_rows": int(descendant_feature_matrix.shape[0]),
                "has_internal": bool(spectral_task.internal_distributions),
                "baseline_projection_dimension": max(int(minimum_projection_dimension), 1),
                "row_sum_variance": float(np.var(row_sums)),
                "row_l2_norm_variance": float(np.var(row_norms)),
                "mean_within_row_variance": float(np.mean(np.var(descendant_feature_matrix, axis=1))),
            }
            stats["candidate_nodes"] += 1
            if compute_eigendecomposition_outputs:
                projection = np.zeros((1, int(feature_count)), dtype=np.float64)
                projection[0, feature_index] = 1.0
                eigenvalues = np.array([1.0], dtype=np.float64)
            else:
                projection = None
                eigenvalues = None
            return NodeSpectralResult(
                node_id=spectral_task.node_id,
                projection_dimension=1,
                projection_matrix=projection,
                eigenvalues=eigenvalues,
            )

        return original_tree_process_node(
            spectral_task,
            full_feature_matrix,
            "marchenko_pastur",
            minimum_projection_dimension,
            feature_count,
            compute_eigendecomposition_outputs,
        )

    def patched_compute_child_parent_spectral_context(tree, leaf_data):
        node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues = (
            original_compute_child_parent_context(tree, leaf_data)
        )

        if (
            node_spectral_dimensions is None
            or node_pca_projections is None
            or node_pca_eigenvalues is None
            or not pending_one_active_meta
        ):
            pending_one_active_meta.clear()
            return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues

        tree_rows: list[dict[str, Any]] = []
        for node_id, meta in list(pending_one_active_meta.items()):
            current_projection = node_pca_projections.get(node_id)
            if not _canonical_one_feature_basis(current_projection):
                continue
            parent_node = next(iter(tree.predecessors(node_id)), None)
            parent_lambda_max = None
            if parent_node is not None:
                parent_eigs = node_pca_eigenvalues.get(parent_node)
                if parent_eigs is not None:
                    parent_eigs = np.asarray(parent_eigs, dtype=np.float64)
                    if parent_eigs.size > 0:
                        parent_lambda_max = float(np.max(parent_eigs))
            ratio = (
                float(meta["active_variance"] / parent_lambda_max)
                if parent_lambda_max is not None and parent_lambda_max > 0
                else None
            )
            is_low = ratio is not None and ratio <= thresholds.low_ratio_threshold
            row = {
                "node_id": node_id,
                "parent_node": parent_node,
                "n_rows": int(meta["n_rows"]),
                "n_leaves": int(meta["n_leaves"]),
                "variance_ratio": ratio,
                "is_low": bool(is_low),
                "row_sum_variance": float(meta["row_sum_variance"]),
                "row_l2_norm_variance": float(meta["row_l2_norm_variance"]),
                "mean_within_row_variance": float(meta["mean_within_row_variance"]),
            }
            tree_rows.append(row)

        low_count = int(sum(r["is_low"] for r in tree_rows))
        low_rows = int(sum(r["n_rows"] for r in tree_rows if r["is_low"]))
        heavy_tree = (
            low_count > thresholds.heavy_count_threshold
            or low_rows > thresholds.heavy_rows_threshold
        )
        stats["heavy_trees"].append(
            {
                "tree_root": tree.root(),
                "low_count": low_count,
                "low_rows": low_rows,
                "heavy_tree": bool(heavy_tree),
            }
        )

        for row in tree_rows:
            row["low_count"] = low_count
            row["low_rows"] = low_rows
            row["heavy_tree"] = bool(heavy_tree)
            stats["audit_rows"].append(row)
            allow = not (heavy_tree and row["is_low"])
            if allow:
                stats["allowed_nodes"] += 1
                continue
            stats["blocked_nodes"] += 1
            node_id = row["node_id"]
            meta = pending_one_active_meta[node_id]
            node_spectral_dimensions[node_id] = int(meta["baseline_projection_dimension"])
            node_pca_projections.pop(node_id, None)
            node_pca_eigenvalues.pop(node_id, None)

        pending_one_active_meta.clear()
        return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues

    tree_estimator._process_node = patched_process_node
    marchenko_pastur._process_node = patched_process_node
    projected_wald.run_projected_wald_kernel = patched_run_projected_wald_kernel
    child_parent_projected_wald.run_projected_wald_kernel = patched_run_projected_wald_kernel
    wald_statistic.run_projected_wald_kernel = patched_run_projected_wald_kernel
    child_parent_spectral_decomposition.compute_child_parent_spectral_context = (
        patched_compute_child_parent_spectral_context
    )
    child_parent_divergence.compute_child_parent_spectral_context = (
        patched_compute_child_parent_spectral_context
    )

    try:
        yield stats
    finally:
        tree_estimator._process_node = original_tree_process_node
        marchenko_pastur._process_node = original_mp_process_node
        projected_wald.run_projected_wald_kernel = original_projected_wald_kernel
        child_parent_projected_wald.run_projected_wald_kernel = (
            original_child_parent_projected_wald_kernel
        )
        wald_statistic.run_projected_wald_kernel = original_sibling_projected_wald_kernel
        child_parent_spectral_decomposition.compute_child_parent_spectral_context = (
            original_compute_child_parent_context
        )
        child_parent_divergence.compute_child_parent_spectral_context = (
            original_compute_child_parent_context_alias
        )
        pending_one_active_meta.clear()


def _mean_or_none(series: pd.Series) -> float | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _build_learned_rule_summary(
    thresholds: Thresholds,
    audit_df: pd.DataFrame,
) -> dict[str, Any]:
    if audit_df.empty:
        return {
            "name": "estimated_tree_load_guard",
            "decision_logic": [],
            "estimation": {},
            "row_variance_role": "audit_only",
            "row_variance_summary": {},
        }

    low_df = audit_df.loc[audit_df["is_low"].fillna(False)].copy()
    blocked_low_df = low_df.loc[low_df["heavy_tree"].fillna(False)].copy()
    allowed_low_df = low_df.loc[~low_df["heavy_tree"].fillna(False)].copy()

    return {
        "name": "estimated_tree_load_guard",
        "decision_logic": [
            "Estimate a low-leverage split from the log( active_variance / lambda_max(parent) ) distribution over one-active nodes.",
            "Mark a one-active node as low-leverage when its variance ratio is at or below the estimated split.",
            "For each tree, compute low-leverage_count and low_leverage_total_rows across those marked nodes.",
            "Estimate heavy-tree cutoffs from the positive-gap structure of low-leverage_count and low_leverage_total_rows across trees.",
            "Allow deterministic 1D only when the node is not simultaneously low-leverage and inside a heavy tree.",
            "If a node is low-leverage inside a heavy tree, revert that node to baseline semantics instead of forcing 1D.",
        ],
        "estimation": {
            "low_ratio_threshold": {
                "value": float(thresholds.low_ratio_threshold),
                "method": "2-cluster KMeans on log(variance_ratio) over one-active nodes",
            },
            "heavy_count_threshold": {
                "value": float(thresholds.heavy_count_threshold),
                "method": "largest positive gap over per-tree low_leverage_count > 0",
            },
            "heavy_rows_threshold": {
                "value": float(thresholds.heavy_rows_threshold),
                "method": "largest positive gap over per-tree low_leverage_total_rows > 0",
            },
        },
        "features": {
            "node_level": [
                "active_variance",
                "parent_lambda_max",
                "variance_ratio",
            ],
            "tree_level": [
                "low_leverage_count",
                "low_leverage_total_rows",
            ],
        },
        "row_variance_role": "audit_only",
        "row_variance_summary": {
            "low_leverage_node_count": int(len(low_df)),
            "blocked_low_node_count": int(len(blocked_low_df)),
            "allowed_low_node_count": int(len(allowed_low_df)),
            "blocked_low_mean_row_sum_variance": _mean_or_none(
                blocked_low_df["row_sum_variance"]
            ),
            "allowed_low_mean_row_sum_variance": _mean_or_none(
                allowed_low_df["row_sum_variance"]
            ),
            "blocked_low_mean_row_l2_norm_variance": _mean_or_none(
                blocked_low_df["row_l2_norm_variance"]
            ),
            "allowed_low_mean_row_l2_norm_variance": _mean_or_none(
                allowed_low_df["row_l2_norm_variance"]
            ),
            "blocked_low_mean_within_row_variance": _mean_or_none(
                blocked_low_df["mean_within_row_variance"]
            ),
            "allowed_low_mean_within_row_variance": _mean_or_none(
                allowed_low_df["mean_within_row_variance"]
            ),
            "note": (
                "Row-variance metrics are retained for audit. "
                "The primary rule is still driven by variance-ratio leverage and "
                "tree-level low-leverage support load."
            ),
        },
    }


def main() -> None:
    output_dir = Path("/tmp/kl_one_active_tree_load_guard")
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _fit_tree_load_thresholds()
    baseline = _run_kl_suite()

    with _patch_tree_load_guard(thresholds) as guard_stats:
        candidate = _run_kl_suite()

    baseline_summary = _summarize(baseline)
    candidate_summary = _summarize(candidate)
    compare = baseline.merge(
        candidate,
        on=["test_case", "case_id"],
        how="outer",
        suffixes=("_baseline", "_candidate"),
    )
    compare["ari_delta"] = compare["ari_candidate"] - compare["ari_baseline"]
    compare["found_delta"] = (
        compare["found_clusters_candidate"] - compare["found_clusters_baseline"]
    )
    changed = compare.loc[
        (compare["found_delta"].fillna(0) != 0)
        | (compare["ari_delta"].abs().fillna(0) > 1e-12)
    ].copy()

    baseline.to_csv(output_dir / "baseline.csv", index=False)
    candidate.to_csv(output_dir / "tree_load_guard.csv", index=False)
    compare.to_csv(output_dir / "tree_load_guard_diff.csv", index=False)
    audit_df = pd.DataFrame(guard_stats["audit_rows"])
    audit_df.to_csv(output_dir / "tree_load_guard_audit.csv", index=False)
    pd.DataFrame(guard_stats["heavy_trees"]).to_csv(output_dir / "heavy_trees.csv", index=False)

    summary = {
        "thresholds": {
            "low_ratio_threshold": thresholds.low_ratio_threshold,
            "heavy_count_threshold": thresholds.heavy_count_threshold,
            "heavy_rows_threshold": thresholds.heavy_rows_threshold,
        },
        "learned_rule": _build_learned_rule_summary(thresholds, audit_df),
        "baseline": baseline_summary,
        "tree_load_guard": candidate_summary,
        "delta_vs_baseline": {
            "mean_ari": float(candidate_summary["mean_ari"] - baseline_summary["mean_ari"]),
            "exact_k": int(candidate_summary["exact_k"] - baseline_summary["exact_k"]),
            "k_equals_1": int(candidate_summary["k_equals_1"] - baseline_summary["k_equals_1"]),
            "changed_cases": int(len(changed)),
            "improved_cases": int((changed["ari_delta"] > 1e-12).sum()),
            "worsened_cases": int((changed["ari_delta"] < -1e-12).sum()),
        },
        "candidate_nodes": int(guard_stats["candidate_nodes"]),
        "allowed_nodes": int(guard_stats["allowed_nodes"]),
        "blocked_nodes": int(guard_stats["blocked_nodes"]),
        "changed_cases": changed[
            [
                "case_id",
                "ari_baseline",
                "ari_candidate",
                "ari_delta",
                "found_clusters_baseline",
                "found_clusters_candidate",
                "found_delta",
            ]
        ].to_dict(orient="records"),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
