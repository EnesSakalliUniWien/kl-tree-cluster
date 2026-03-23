"""Compare baseline KL against a deterministic 1D one-active-feature variant.

This is an experiment-only runner. It leaves production behavior unchanged and
monkeypatches the per-node spectral worker during the experimental pass.

The experimental variant changes only one case:

- If a node's subtree has exactly one active-variance feature, build a
  deterministic 1D projection basis on that feature and set k=1.

Everything else stays on the current production path.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.types import (
    NodeSpectralResult,
)


def _normalize_benchmark_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Test": "test_case",
        "Case_Name": "case_id",
        "Method": "method",
        "True": "true_clusters",
        "Found": "found_clusters",
        "ARI": "ari",
    }
    normalized = df.rename(columns=rename_map).copy()
    for column in ("test_case", "true_clusters", "found_clusters"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "ari" in normalized.columns:
        normalized["ari"] = pd.to_numeric(normalized["ari"], errors="coerce")
    return normalized


def _summarize(df: pd.DataFrame) -> dict[str, object]:
    ari = pd.to_numeric(df["ari"], errors="coerce")
    found = pd.to_numeric(df["found_clusters"], errors="coerce")
    true = pd.to_numeric(df["true_clusters"], errors="coerce")
    return {
        "n_cases": int(len(df)),
        "mean_ari": float(ari.mean(skipna=True)),
        "median_ari": float(ari.median(skipna=True)),
        "exact_k": int(((found == true) & found.notna() & true.notna()).sum()),
        "k_equals_1": int((found == 1).sum()),
        "ok_cases": int((df.get("Status", pd.Series(index=df.index)) == "ok").sum()),
        "ari_nan_cases": int(ari.isna().sum()),
    }


@contextmanager
def _patch_one_active_feature_to_1d() -> dict[str, int]:
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
        marchenko_pastur,
        tree_estimator,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
        child_parent_projected_wald,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection import projected_wald
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

    stats = {
        "one_active_hits": 0,
        "one_active_with_internal_rows": 0,
    }

    def _is_canonical_one_feature_basis(pca_projection: np.ndarray | None) -> bool:
        if pca_projection is None:
            return False
        basis = np.asarray(pca_projection, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != 1:
            return False
        nonzero = np.flatnonzero(np.abs(basis[0]) > 1e-12)
        if nonzero.size != 1:
            return False
        return bool(np.isclose(np.linalg.norm(basis[0]), 1.0))

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
        if _is_canonical_one_feature_basis(pca_projection):
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

        active_mask = np.var(descendant_feature_matrix, axis=0) > 0
        active_indices = np.flatnonzero(active_mask)

        if active_indices.size == 1:
            stats["one_active_hits"] += 1
            if spectral_task.internal_distributions:
                stats["one_active_with_internal_rows"] += 1
            if compute_eigendecomposition_outputs:
                projection = np.zeros((1, int(feature_count)), dtype=np.float64)
                projection[0, int(active_indices[0])] = 1.0
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
        yield stats
    finally:
        tree_estimator._process_node = original_tree_process_node
        marchenko_pastur._process_node = original_mp_process_node
        projected_wald.run_projected_wald_kernel = original_projected_wald_kernel
        child_parent_projected_wald.run_projected_wald_kernel = (
            original_child_parent_projected_wald_kernel
        )
        wald_statistic.run_projected_wald_kernel = original_sibling_projected_wald_kernel


def _run_kl_suite() -> pd.DataFrame:
    test_cases = get_default_test_cases()
    df, _ = benchmark_cluster_algorithm(
        test_cases=test_cases,
        methods=["kl"],
        verbose=False,
        plot_umap=False,
        plot_manifold=False,
        concat_plots_pdf=False,
    )
    return _normalize_benchmark_df(df)


def main() -> None:
    output_dir = Path("/tmp/kl_one_active_feature_1d_strength")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_kl_suite()
    baseline_summary = _summarize(baseline)

    with _patch_one_active_feature_to_1d() as patch_stats:
        experiment = _run_kl_suite()
    experiment_summary = _summarize(experiment)

    compare = baseline.merge(
        experiment,
        on=["test_case", "case_id"],
        how="outer",
        suffixes=("_baseline", "_one_active_1d"),
    )
    compare["ari_delta"] = compare["ari_one_active_1d"] - compare["ari_baseline"]
    compare["found_delta"] = compare["found_clusters_one_active_1d"] - compare["found_clusters_baseline"]

    category_compare = (
        compare.groupby("Case_Category_baseline", dropna=False)
        .agg(
            n_cases=("case_id", "count"),
            mean_ari_baseline=("ari_baseline", "mean"),
            mean_ari_one_active_1d=("ari_one_active_1d", "mean"),
            mean_ari_delta=("ari_delta", "mean"),
            mean_found_delta=("found_delta", "mean"),
            changed_cases=("found_delta", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) != 0).sum())),
        )
        .reset_index()
        .rename(columns={"Case_Category_baseline": "Case_Category"})
        .sort_values(["mean_ari_delta", "Case_Category"], ascending=[True, True])
    )

    changed = compare[
        (compare["found_delta"].fillna(0) != 0)
        | (compare["ari_delta"].abs().fillna(0) > 1e-12)
    ].copy()
    changed = changed.sort_values(["ari_delta", "case_id"], ascending=[True, True])

    summary = {
        "baseline": baseline_summary,
        "one_active_1d": experiment_summary,
        "patch_stats": patch_stats,
        "delta": {
            "mean_ari": experiment_summary["mean_ari"] - baseline_summary["mean_ari"],
            "median_ari": experiment_summary["median_ari"] - baseline_summary["median_ari"],
            "exact_k": experiment_summary["exact_k"] - baseline_summary["exact_k"],
            "k_equals_1": experiment_summary["k_equals_1"] - baseline_summary["k_equals_1"],
            "changed_cases": int(len(changed)),
            "improved_cases": int((changed["ari_delta"] > 1e-12).sum()),
            "worsened_cases": int((changed["ari_delta"] < -1e-12).sum()),
        },
        "top_improvements": changed.sort_values("ari_delta", ascending=False)
        .head(10)[["case_id", "ari_baseline", "ari_one_active_1d", "ari_delta", "found_clusters_baseline", "found_clusters_one_active_1d"]]
        .to_dict(orient="records"),
        "top_regressions": changed.sort_values("ari_delta", ascending=True)
        .head(10)[["case_id", "ari_baseline", "ari_one_active_1d", "ari_delta", "found_clusters_baseline", "found_clusters_one_active_1d"]]
        .to_dict(orient="records"),
    }

    baseline.to_csv(output_dir / "baseline.csv", index=False)
    experiment.to_csv(output_dir / "one_active_1d.csv", index=False)
    compare.to_csv(output_dir / "diff.csv", index=False)
    category_compare.to_csv(output_dir / "category_diff.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
