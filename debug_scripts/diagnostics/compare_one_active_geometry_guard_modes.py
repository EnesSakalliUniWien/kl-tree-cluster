"""Compare guarded one-active 1D variants against the current KL baseline.

This is an experiment-only runner. It leaves production behavior unchanged and
monkeypatches the spectral worker plus Gate 2 spectral context assembly during
the experimental pass.

Guard families:

- ``small_subtree_proxy``: allow one-active 1D only for tiny subtrees.
- ``ratio_*``: allow one-active 1D only when the active-feature variance is
  large enough relative to the parent's leading eigenvalue.
- ``ratio_*_depth*`` / ``ratio_*_leaves*``: add a depth or subtree-mass guard.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from debug_scripts.diagnostics.compare_one_active_feature_1d_strength import (
    _run_kl_suite,
    _summarize,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.types import (
    NodeSpectralResult,
)


@dataclass(frozen=True)
class GuardSpec:
    name: str
    min_ratio: float | None = None
    min_depth: int | None = None
    max_leaves: int | None = None
    max_rows: int | None = None


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
def _patch_one_active_guard(spec: GuardSpec) -> dict[str, Any]:
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
    guard_audit_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "guard": asdict(spec),
        "candidate_nodes": 0,
        "allowed_nodes": 0,
        "blocked_nodes": 0,
        "candidates_with_internal_rows": 0,
        "guard_audit": guard_audit_rows,
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
            pending_one_active_meta[spectral_task.node_id] = {
                "active_feature": feature_index,
                "active_variance": float(variances[feature_index]),
                "n_leaves": int(len(descendant_leaf_row_indices)),
                "n_rows": int(descendant_feature_matrix.shape[0]),
                "has_internal": bool(spectral_task.internal_distributions),
                "baseline_projection_dimension": max(int(minimum_projection_dimension), 1),
            }
            stats["candidate_nodes"] += 1
            if spectral_task.internal_distributions:
                stats["candidates_with_internal_rows"] += 1
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

    def patched_compute_child_parent_spectral_context(tree, leaf_data, spectral_method):
        node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues = (
            original_compute_child_parent_context(tree, leaf_data, spectral_method)
        )

        if (
            node_spectral_dimensions is None
            or node_pca_projections is None
            or node_pca_eigenvalues is None
            or not pending_one_active_meta
        ):
            pending_one_active_meta.clear()
            return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues

        node_depths = nx.shortest_path_length(tree, source=tree.graph.get("root", tree.root()))

        for node_id, meta in list(pending_one_active_meta.items()):
            current_projection = node_pca_projections.get(node_id)
            if not _canonical_one_feature_basis(current_projection):
                continue

            parent_node = next(iter(tree.predecessors(node_id)), None)
            parent_lambda_max = None
            if parent_node is not None:
                parent_eigenvalues = node_pca_eigenvalues.get(parent_node)
                if parent_eigenvalues is not None:
                    parent_eigenvalues = np.asarray(parent_eigenvalues, dtype=np.float64)
                    if parent_eigenvalues.size > 0:
                        parent_lambda_max = float(np.max(parent_eigenvalues))

            active_variance = float(meta["active_variance"])
            ratio = (
                active_variance / parent_lambda_max
                if parent_lambda_max is not None and parent_lambda_max > 0
                else None
            )
            depth = int(node_depths[node_id])
            allow = True

            if spec.min_ratio is not None:
                allow = allow and ratio is not None and ratio >= spec.min_ratio
            if spec.min_depth is not None:
                allow = allow and depth >= spec.min_depth
            if spec.max_leaves is not None:
                allow = allow and int(meta["n_leaves"]) <= spec.max_leaves
            if spec.max_rows is not None:
                allow = allow and int(meta["n_rows"]) <= spec.max_rows

            guard_audit_rows.append(
                {
                    "guard_name": spec.name,
                    "node_id": node_id,
                    "parent_node": parent_node,
                    "depth": depth,
                    "n_leaves": int(meta["n_leaves"]),
                    "n_rows": int(meta["n_rows"]),
                    "has_internal": bool(meta["has_internal"]),
                    "active_feature": int(meta["active_feature"]),
                    "active_variance": active_variance,
                    "parent_lambda_max": parent_lambda_max,
                    "variance_ratio": ratio,
                    "allowed": bool(allow),
                }
            )

            if allow:
                stats["allowed_nodes"] += 1
                continue

            stats["blocked_nodes"] += 1
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


def _compare_against_baseline(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    changed_mask = (
        compare["found_delta"].fillna(0) != 0
    ) | (compare["ari_delta"].abs().fillna(0) > 1e-12)
    changed = compare.loc[changed_mask].copy()
    delta = {
        "mean_ari": float(_summarize(candidate)["mean_ari"] - _summarize(baseline)["mean_ari"]),
        "exact_k": int(_summarize(candidate)["exact_k"] - _summarize(baseline)["exact_k"]),
        "k_equals_1": int(_summarize(candidate)["k_equals_1"] - _summarize(baseline)["k_equals_1"]),
        "changed_cases": int(len(changed)),
        "improved_cases": int((changed["ari_delta"] > 1e-12).sum()),
        "worsened_cases": int((changed["ari_delta"] < -1e-12).sum()),
    }
    return compare, delta


def main() -> None:
    output_dir = Path("/tmp/kl_one_active_geometry_guard_modes")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_kl_suite()
    baseline_summary = _summarize(baseline)

    variants = [
        GuardSpec(name="small_subtree_proxy", max_leaves=4, max_rows=6),
        GuardSpec(name="ratio_0p03", min_ratio=0.03),
        GuardSpec(name="ratio_0p04", min_ratio=0.04),
        GuardSpec(name="ratio_0p05", min_ratio=0.05),
        GuardSpec(name="ratio_0p03_depth6", min_ratio=0.03, min_depth=6),
        GuardSpec(name="ratio_0p03_leaves15", min_ratio=0.03, max_leaves=15),
    ]

    summary: dict[str, Any] = {
        "baseline": baseline_summary,
        "variants": [],
    }

    baseline.to_csv(output_dir / "baseline.csv", index=False)

    for spec in variants:
        with _patch_one_active_guard(spec) as guard_stats:
            candidate = _run_kl_suite()
        candidate_summary = _summarize(candidate)
        compare_df, delta = _compare_against_baseline(baseline, candidate)

        candidate.to_csv(output_dir / f"{spec.name}.csv", index=False)
        compare_df.to_csv(output_dir / f"{spec.name}_diff.csv", index=False)

        guard_audit = pd.DataFrame(guard_stats["guard_audit"])
        if not guard_audit.empty:
            guard_audit.to_csv(output_dir / f"{spec.name}_guard_audit.csv", index=False)

        summary["variants"].append(
            {
                "name": spec.name,
                "guard": asdict(spec),
                "summary": candidate_summary,
                "delta_vs_baseline": delta,
                "candidate_nodes": int(guard_stats["candidate_nodes"]),
                "allowed_nodes": int(guard_stats["allowed_nodes"]),
                "blocked_nodes": int(guard_stats["blocked_nodes"]),
                "candidates_with_internal_rows": int(
                    guard_stats["candidates_with_internal_rows"]
                ),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
