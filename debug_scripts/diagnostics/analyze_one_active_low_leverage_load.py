"""Analyze one-active 1D leverage load on the full default KL suite.

This experiment compares the baseline KL run against the pooled one-active 1D
variant and extracts per-case tree-level diagnostics:

- number of one-active nodes
- total subtree support they carry
- data-driven low-vs-high leverage split based on variance ratios
- support carried by the low-leverage subset
- projected signal magnitude along the one-active axis

The production pipeline is not modified.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.types import (
    NodeSpectralResult,
)


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


@contextmanager
def _patch_one_active_feature_to_1d_recording() -> list[dict[str, Any]]:
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

    hits: list[dict[str, Any]] = []

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
            child_pca_projections=child_pca_projections,
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
            hits.append(
                {
                    "node_id": spectral_task.node_id,
                    "active_feature": feature_index,
                    "active_variance": float(variances[feature_index]),
                    "n_leaves": int(len(descendant_leaf_row_indices)),
                    "n_rows": int(descendant_feature_matrix.shape[0]),
                    "has_internal": bool(spectral_task.internal_distributions),
                    "basis_norm": 1.0,
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
        yield hits
    finally:
        tree_estimator._process_node = original_tree_process_node
        marchenko_pastur._process_node = original_mp_process_node
        projected_wald.run_projected_wald_kernel = original_projected_wald_kernel
        child_parent_projected_wald.run_projected_wald_kernel = (
            original_child_parent_projected_wald_kernel
        )
        wald_statistic.run_projected_wald_kernel = original_sibling_projected_wald_kernel


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_ari(y_true: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(adjusted_rand_score(y_true, labels))
    except ValueError:
        return None


def _prepare_case_inputs(case: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data_df, y_true, _x_original, _meta, distance_condensed, _dm, _pre = prepare_case_inputs(
        case,
        ["kl"],
    )
    if distance_condensed is None:
        raise ValueError(f"Missing condensed distance input for case {case.get('name')!r}.")
    return data_df, y_true, np.asarray(distance_condensed, dtype=np.float64)


def _augment_node_metrics(
    data_df: pd.DataFrame,
    y_true: np.ndarray,
    tree,
    node_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pca_eigenvalues = tree.stats_df.attrs.get("_pca_eigenvalues", {})
    root_node = tree.root()

    for row in node_hits:
        node_id = row["node_id"]
        parent_node = next(iter(tree.predecessors(node_id)), None)
        row["parent_node"] = parent_node
        row["depth"] = int(nx.shortest_path_length(tree, root_node, node_id))

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

        children = list(tree.successors(node_id))
        subtree_leaves = tree.get_leaves(node_id, return_labels=True, sort=True)
        subtree_leaf_matrix = data_df.loc[subtree_leaves].values.astype(np.float64)
        subtree_internal_rows = []
        for descendant_node in nx.descendants(tree, node_id):
            if tree.out_degree(descendant_node) == 0:
                continue
            distribution = tree.nodes[descendant_node].get("distribution")
            if distribution is None:
                continue
            distribution_array = np.asarray(distribution, dtype=np.float64)
            if distribution_array.shape == (data_df.shape[1],):
                subtree_internal_rows.append(distribution_array)
        if subtree_internal_rows:
            subtree_matrix = np.vstack([subtree_leaf_matrix, np.vstack(subtree_internal_rows)])
        else:
            subtree_matrix = subtree_leaf_matrix

        row_sums = subtree_matrix.sum(axis=1)
        row_norms = np.linalg.norm(subtree_matrix, axis=1)
        row["row_sum_variance"] = float(np.var(row_sums))
        row["row_l2_norm_variance"] = float(np.var(row_norms))
        row["mean_within_row_variance"] = float(np.mean(np.var(subtree_matrix, axis=1)))
        row["active_to_row_sum_variance_ratio"] = (
            float(row["active_variance"] / row["row_sum_variance"])
            if row["row_sum_variance"] > 0
            else None
        )
        row["active_to_row_l2_variance_ratio"] = (
            float(row["active_variance"] / row["row_l2_norm_variance"])
            if row["row_l2_norm_variance"] > 0
            else None
        )

        if len(children) == 2:
            left_node, right_node = children
            left_leaves = tree.get_leaves(left_node, return_labels=True, sort=True)
            right_leaves = tree.get_leaves(right_node, return_labels=True, sort=True)
            left_matrix = data_df.loc[left_leaves].values.astype(np.float64)
            right_matrix = data_df.loc[right_leaves].values.astype(np.float64)
            mean_diff = left_matrix.mean(axis=0) - right_matrix.mean(axis=0)
            feature_index = int(row["active_feature"])
            row["axis_signal"] = float(abs(mean_diff[feature_index]))
            row["axis_signal_weighted_rows"] = float(row["axis_signal"] * row["n_rows"])
            row["axis_signal_weighted_leaves"] = float(row["axis_signal"] * row["n_leaves"])
        else:
            row["axis_signal"] = None
            row["axis_signal_weighted_rows"] = None
            row["axis_signal_weighted_leaves"] = None

        subtree_true = y_true[[data_df.index.get_loc(label) for label in subtree_leaves]]
        row["unique_true_labels_in_subtree"] = int(len(set(subtree_true.tolist())))

    return node_hits


def _fit_global_low_leverage_split(ratios: list[float]) -> dict[str, Any]:
    ratios = [float(r) for r in ratios if r is not None and np.isfinite(r) and r > 0]
    if len(ratios) < 2:
        return {"threshold": None, "low_center": None, "high_center": None, "n_fit": len(ratios)}

    log_ratios = np.log(np.asarray(ratios, dtype=np.float64)).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=20)
    labels = kmeans.fit_predict(log_ratios)
    centers = kmeans.cluster_centers_.ravel()
    low_label = int(np.argmin(centers))
    high_label = 1 - low_label
    low_vals = np.exp(log_ratios[labels == low_label].ravel())
    high_vals = np.exp(log_ratios[labels == high_label].ravel())

    if low_vals.size == 0 or high_vals.size == 0:
        return {"threshold": None, "low_center": None, "high_center": None, "n_fit": len(ratios)}

    low_max = float(np.max(low_vals))
    high_min = float(np.min(high_vals))
    threshold = float(np.sqrt(low_max * high_min)) if low_max < high_min else float(
        np.exp(np.mean([centers[low_label], centers[high_label]]))
    )

    return {
        "threshold": threshold,
        "low_center": float(np.exp(centers[low_label])),
        "high_center": float(np.exp(centers[high_label])),
        "low_max": low_max,
        "high_min": high_min,
        "n_fit": len(ratios),
    }


def _sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(sum(values)) if values else 0.0


def _case_metrics(
    case_name: str,
    baseline_result,
    experiment_result,
    y_true: np.ndarray,
    node_rows: list[dict[str, Any]],
    global_split: dict[str, Any],
) -> dict[str, Any]:
    threshold = global_split["threshold"]
    low_rows = [
        r
        for r in node_rows
        if threshold is not None and r.get("variance_ratio") is not None and r["variance_ratio"] <= threshold
    ]
    baseline_ari = _safe_ari(y_true, baseline_result.labels)
    experiment_ari = _safe_ari(y_true, experiment_result.labels)

    return {
        "case_id": case_name,
        "baseline_found_clusters": int(baseline_result.found_clusters),
        "experiment_found_clusters": int(experiment_result.found_clusters),
        "baseline_ari": baseline_ari,
        "experiment_ari": experiment_ari,
        "ari_delta": (
            float(experiment_ari - baseline_ari)
            if baseline_ari is not None and experiment_ari is not None
            else None
        ),
        "found_delta": int(experiment_result.found_clusters - baseline_result.found_clusters),
        "one_active_count": int(len(node_rows)),
        "one_active_total_rows": int(sum(int(r["n_rows"]) for r in node_rows)),
        "one_active_total_leaves": int(sum(int(r["n_leaves"]) for r in node_rows)),
        "one_active_mean_rows": float(np.mean([r["n_rows"] for r in node_rows])) if node_rows else 0.0,
        "one_active_mean_depth": float(np.mean([r["depth"] for r in node_rows])) if node_rows else 0.0,
        "one_active_with_internal_count": int(sum(bool(r["has_internal"]) for r in node_rows)),
        "basis_norm_mean": float(np.mean([r["basis_norm"] for r in node_rows])) if node_rows else None,
        "axis_signal_sum": _sum_metric(node_rows, "axis_signal"),
        "axis_signal_weighted_rows_sum": _sum_metric(node_rows, "axis_signal_weighted_rows"),
        "mean_row_sum_variance": float(np.mean([r["row_sum_variance"] for r in node_rows]))
        if node_rows
        else None,
        "mean_row_l2_norm_variance": float(np.mean([r["row_l2_norm_variance"] for r in node_rows]))
        if node_rows
        else None,
        "mean_within_row_variance": float(np.mean([r["mean_within_row_variance"] for r in node_rows]))
        if node_rows
        else None,
        "low_leverage_count": int(len(low_rows)),
        "low_leverage_total_rows": int(sum(int(r["n_rows"]) for r in low_rows)),
        "low_leverage_total_leaves": int(sum(int(r["n_leaves"]) for r in low_rows)),
        "low_leverage_axis_signal_sum": _sum_metric(low_rows, "axis_signal"),
        "low_leverage_axis_signal_weighted_rows_sum": _sum_metric(
            low_rows, "axis_signal_weighted_rows"
        ),
        "mean_low_leverage_row_sum_variance": (
            float(np.mean([r["row_sum_variance"] for r in low_rows])) if low_rows else None
        ),
        "mean_low_leverage_row_l2_norm_variance": (
            float(np.mean([r["row_l2_norm_variance"] for r in low_rows])) if low_rows else None
        ),
        "mean_low_leverage_within_row_variance": (
            float(np.mean([r["mean_within_row_variance"] for r in low_rows])) if low_rows else None
        ),
        "min_variance_ratio": (
            float(min(r["variance_ratio"] for r in node_rows if r.get("variance_ratio") is not None))
            if any(r.get("variance_ratio") is not None for r in node_rows)
            else None
        ),
        "max_variance_ratio": (
            float(max(r["variance_ratio"] for r in node_rows if r.get("variance_ratio") is not None))
            if any(r.get("variance_ratio") is not None for r in node_rows)
            else None
        ),
    }


def main() -> None:
    output_dir = Path("/tmp/kl_one_active_low_leverage_load")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case_rows: list[dict[str, Any]] = []
    per_node_rows: list[dict[str, Any]] = []

    for case in get_default_test_cases():
        case_name = str(case["name"])
        data_df, y_true, distance_condensed = _prepare_case_inputs(case)
        baseline_result = _run_kl_method(data_df, distance_condensed, 0.01)
        with _patch_one_active_feature_to_1d_recording() as node_hits:
            experiment_result = _run_kl_method(data_df, distance_condensed, 0.01)

        node_rows = _augment_node_metrics(
            data_df,
            y_true,
            experiment_result.extra["tree"],
            node_hits,
        )
        for node_row in node_rows:
            node_row["case_id"] = case_name
        per_node_rows.extend(node_rows)
        per_case_rows.append(
            {
                "case_id": case_name,
                "baseline_result": baseline_result,
                "experiment_result": experiment_result,
                "y_true": y_true,
                "node_rows": node_rows,
            }
        )

    finite_ratios = [
        float(row["variance_ratio"])
        for row in per_node_rows
        if row.get("variance_ratio") is not None and np.isfinite(row["variance_ratio"])
    ]
    global_split = _fit_global_low_leverage_split(finite_ratios)

    case_metrics_rows = [
        _case_metrics(
            case_name=row["case_id"],
            baseline_result=row["baseline_result"],
            experiment_result=row["experiment_result"],
            y_true=row["y_true"],
            node_rows=row["node_rows"],
            global_split=global_split,
        )
        for row in per_case_rows
    ]

    case_metrics_df = pd.DataFrame(case_metrics_rows).sort_values("ari_delta")
    node_df = pd.DataFrame(per_node_rows)
    if not node_df.empty:
        node_df["is_low_leverage"] = (
            node_df["variance_ratio"].notna()
            & (global_split["threshold"] is not None)
            & (node_df["variance_ratio"] <= global_split["threshold"])
        )

    case_metrics_df.to_csv(output_dir / "case_metrics.csv", index=False)
    if not node_df.empty:
        node_df.to_csv(output_dir / "node_metrics.csv", index=False)

    ari_delta = pd.to_numeric(case_metrics_df["ari_delta"], errors="coerce")
    improved = case_metrics_df.loc[ari_delta > 1e-12]
    worsened = case_metrics_df.loc[ari_delta < -1e-12]
    unchanged = case_metrics_df.loc[ari_delta.abs() <= 1e-12]

    def _group_summary(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"n_cases": 0}
        return {
            "n_cases": int(len(df)),
            "mean_one_active_count": float(df["one_active_count"].mean()),
            "mean_one_active_total_rows": float(df["one_active_total_rows"].mean()),
            "mean_low_leverage_count": float(df["low_leverage_count"].mean()),
            "mean_low_leverage_total_rows": float(df["low_leverage_total_rows"].mean()),
            "mean_low_leverage_axis_signal_weighted_rows_sum": float(
                df["low_leverage_axis_signal_weighted_rows_sum"].mean()
            ),
            "mean_row_sum_variance": float(df["mean_row_sum_variance"].dropna().mean())
            if df["mean_row_sum_variance"].notna().any()
            else None,
            "mean_row_l2_norm_variance": float(df["mean_row_l2_norm_variance"].dropna().mean())
            if df["mean_row_l2_norm_variance"].notna().any()
            else None,
            "mean_within_row_variance": float(df["mean_within_row_variance"].dropna().mean())
            if df["mean_within_row_variance"].notna().any()
            else None,
            "mean_low_leverage_row_sum_variance": float(
                df["mean_low_leverage_row_sum_variance"].dropna().mean()
            )
            if df["mean_low_leverage_row_sum_variance"].notna().any()
            else None,
            "mean_min_variance_ratio": float(df["min_variance_ratio"].dropna().mean())
            if df["min_variance_ratio"].notna().any()
            else None,
        }

    summary = {
        "global_low_leverage_split": global_split,
        "basis_vector_note": "Canonical one-active basis vectors are unit vectors, so basis norm is constant 1.0; row-level variance metrics were added to check whether the row geometry adds signal beyond low-leverage support.",
        "group_summaries": {
            "improved": _group_summary(improved),
            "worsened": _group_summary(worsened),
            "unchanged": _group_summary(unchanged),
        },
        "worst_cases": case_metrics_df.head(10).to_dict(orient="records"),
        "best_cases": case_metrics_df.tail(10).iloc[::-1].to_dict(orient="records"),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
