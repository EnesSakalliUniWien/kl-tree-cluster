"""Trace the PRODUCTION calibration path exactly.

Unlike trace_calibration.py (which uses a simplified manual path), this script
replicates every step of the production `run_gate_annotation_pipeline`:

1. Gate 2: annotate_child_parent_divergence  (identical)
2. Derive sibling spectral dims via geometric-mean-of-children
3. Derive parent + child PCA projections for orthogonal-complement padding
4. collect_sibling_pair_records  with child_pca_projections + whitening
5. interpolate_sibling_null_priors  (if blocked pairs exist)
6. fit_inflation_model  (global ĉ)
7. compute_pool_stats  (local kernel in log-k space)
8. _deflate_and_test  using predict_local_inflation_factor  (per-node ĉ)
9. Full decompose() comparison

This is a diagnostic-only script.  It does NOT modify any production code.
"""

from __future__ import annotations

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.conditional_deflation import (
    compute_pool_stats,
    predict_local_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    fit_inflation_model,
    predict_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_null_prior_interpolation import (
    interpolate_sibling_null_priors,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
    count_null_focal_pairs,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_child_pca_projections,
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    resolve_minimum_projection_dimension_backend,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def trace_case(
    n_samples: int, n_features: int, n_clusters: int, noise: float, seed: int = 42, label: str = ""
) -> None:
    """Trace production calibration for a single generated case."""
    case = {
        "generator": "blobs",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cluster_std": noise,
        "seed": seed,
        "name": label,
    }
    data_bin, labels, _, _ = generate_case_data(case)

    Z = linkage(
        pdist(data_bin.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_bin.index.tolist())
    tree.populate_node_divergences(leaf_data=data_bin)

    # ── Step 1: Gate 2 (edge significance) ──────────────────────────────
    min_proj_dim = resolve_minimum_projection_dimension_backend(
        config.PROJECTION_MINIMUM_DIMENSION, leaf_data=data_bin,
    )
    annotations_df = annotate_child_parent_divergence(
        tree, tree.annotations_df, significance_level_alpha=0.05,
        leaf_data=data_bin, minimum_projection_dimension=min_proj_dim,
    )

    # ── Step 2: Derive sibling spectral dims (geometric-mean-of-children) ─
    spectral_dims = derive_sibling_spectral_dims(tree, annotations_df)

    # ── Step 3: Derive parent + child PCA projections ───────────────────
    pca_projections, pca_eigenvalues = derive_sibling_pca_projections(
        annotations_df,
        spectral_dims,
    )
    child_pca_projections = derive_sibling_child_pca_projections(
        tree,
        annotations_df,
        spectral_dims,
    )

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # ── Step 4: Collect sibling pairs (production args) ─────────────────
    records, _non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=config.SIBLING_WHITENING,
    )

    n_null, n_focal, n_blocked = count_null_focal_pairs(records)

    print("=" * 90)
    print(f"PRODUCTION CALIBRATION TRACE: {label}")
    print(f"  n={n_samples}, p={n_features}, K={n_clusters}, noise={noise}")
    print(f"  Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")
    print(f"  Config: SIBLING_WHITENING={config.SIBLING_WHITENING}")
    print(f"  Config: FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}")
    print("=" * 90)
    print(
        f"Total pairs: {len(records)}  (null-like: {n_null}, focal: {n_focal}, gate2-blocked: {n_blocked})"
    )
    if spectral_dims:
        dims_vals = list(spectral_dims.values())
        print(
            f"Sibling spectral dims: {len(dims_vals)} parents, "
            f"min={min(dims_vals)}, max={max(dims_vals)}, median={sorted(dims_vals)[len(dims_vals)//2]}"
        )
    print(f"child_pca_projections: {'available' if child_pca_projections else 'None'}")
    print()

    # ── Per-pair table ──────────────────────────────────────────────────
    print(
        f"{'Parent':>8} {'T':>10} {'k':>6} {'struct_k':>8} "
        f"{'raw_p':>10} {'r=T/k':>8} {'null':>5} {'prior':>8} {'bl_sum':>8} {'n_par':>6}"
    )
    print("-" * 100)
    for r in sorted(records, key=lambda x: x.n_parent, reverse=True):
        ratio = r.stat / r.degrees_of_freedom if r.degrees_of_freedom > 0 else float("nan")
        print(
            f"{r.parent:>8} {r.stat:10.4f} {r.degrees_of_freedom:6.0f} {r.structural_dimension:8.2f} "
            f"{r.p_value:10.6f} {ratio:8.4f} {str(r.is_null_like):>5} "
            f"{r.sibling_null_prior_from_edge_pvalue:8.4f} "
            f"{r.branch_length_sum:8.4f} {r.n_parent:6d}"
        )

    # ── Step 5: Interpolation (if blocked pairs exist) ──────────────────
    if n_blocked > 0:
        print(f"\nInterpolating null priors for {n_blocked} gate2-blocked pairs...")
        records = interpolate_sibling_null_priors(records, tree, annotations_df)
        # Show updated priors
        blocked = [r for r in records if r.is_gate2_blocked]
        if blocked:
            priors_after = [r.sibling_null_prior_from_edge_pvalue for r in blocked]
            print(
                f"  Blocked pair priors after interpolation: min={min(priors_after):.4f}, "
                f"max={max(priors_after):.4f}, mean={sum(priors_after)/len(priors_after):.4f}"
            )

    # ── Step 6: Fit global inflation model ──────────────────────────────
    model = fit_inflation_model(records)
    print(f"\nModel method: {model.method}")
    print(f"Global c-hat: {model.global_inflation_factor:.4f}")
    print(f"Max observed ratio: {model.max_observed_ratio:.4f}")
    print(f"N calibration: {model.n_calibration}")
    if model.diagnostics:
        for k, v in sorted(model.diagnostics.items()):
            print(f"  diag.{k}: {v}")

    # ── Step 7: Compute pool stats (local kernel) ───────────────────────
    pool = compute_pool_stats(records, model)
    print("\nPool stats:")
    print(f"  c_global: {pool.c_global:.4f}")
    print(f"  geometric_mean_structural_dimension: {pool.geometric_mean_structural_dimension:.4f}")
    print(f"  bandwidth_log_structural_dimension: {pool.bandwidth_log_structural_dimension:.4f}")
    print(f"  bandwidth_status: {pool.bandwidth_status}")
    print(f"  max_ratio: {pool.max_ratio:.4f}")
    print(f"  n_records: {pool.n_records}")

    # ── Step 8: Per-node local deflation ────────────────────────────────
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
    root_children = list(tree.successors(root))

    print("\n--- PER-NODE DEFLATION (production path) ---")
    # Show focal pairs with local c
    for r in sorted(records, key=lambda x: x.n_parent, reverse=True):
        if r.is_null_like:
            continue
        c_local = predict_local_inflation_factor(model, pool, r.structural_dimension)
        c_global = predict_inflation_factor(model, r.branch_length_sum, r.n_parent)
        t_adj = r.stat / c_local
        p_adj = (
            float(chi2.sf(t_adj, df=r.degrees_of_freedom))
            if r.degrees_of_freedom > 0
            else float("nan")
        )
        print(
            f"  {r.parent:>8}  T={r.stat:8.2f}  k={r.degrees_of_freedom:5.1f}  struct_k={r.structural_dimension:6.2f}"
            f"  c_local={c_local:6.3f}  c_global={c_global:6.3f}  delta={c_local - c_global:+7.3f}"
            f"  T_adj={t_adj:8.2f}  p_adj={p_adj:.6f}  reject={p_adj < 0.05}"
        )

    # ── Root node detail ────────────────────────────────────────────────
    root_rec = [r for r in records if r.parent == root]
    if root_rec:
        rr = root_rec[0]
        c_local = predict_local_inflation_factor(model, pool, rr.structural_dimension)
        c_global = predict_inflation_factor(model, rr.branch_length_sum, rr.n_parent)
        t_adj_local = rr.stat / c_local
        t_adj_global = rr.stat / c_global
        p_local = (
            float(chi2.sf(t_adj_local, df=rr.degrees_of_freedom))
            if rr.degrees_of_freedom > 0
            else float("nan")
        )
        p_global = (
            float(chi2.sf(t_adj_global, df=rr.degrees_of_freedom))
            if rr.degrees_of_freedom > 0
            else float("nan")
        )

        print(f"\n--- ROOT NODE ({root}) ---")
        print(f"  Children: {root_children}")
        print(
            f"  T = {rr.stat:.4f},  k = {rr.degrees_of_freedom:.1f},  struct_k = {rr.structural_dimension:.2f}"
        )
        print(f"  raw p = {rr.p_value:.6f}")
        print(f"  is_null_like = {rr.is_null_like}")
        print(f"  sibling_null_prior = {rr.sibling_null_prior_from_edge_pvalue:.6f}")
        print()
        print(
            f"  LOCAL deflation (production):  c={c_local:.4f}  T_adj={t_adj_local:.4f}  p_adj={p_local:.6f}  reject={p_local < 0.05}"
        )
        print(
            f"  GLOBAL deflation (old trace):  c={c_global:.4f}  T_adj={t_adj_global:.4f}  p_adj={p_global:.6f}  reject={p_global < 0.05}"
        )

    # ── Step 9: Full pipeline comparison ────────────────────────────────
    print("\n--- FULL PIPELINE COMPARISON ---")
    tree2 = PosetTree.from_linkage(Z, leaf_names=data_bin.index.tolist())
    tree2.populate_node_divergences(leaf_data=data_bin)
    result = tree2.decompose(leaf_data=data_bin, alpha_local=0.05, sibling_alpha=0.05)
    found_k = result["num_clusters"]
    print(f"  decompose() found K = {found_k}")

    sdf = tree2.annotations_df
    if sdf is not None:
        root2 = [n for n in tree2.nodes if tree2.in_degree(n) == 0][0]
        root_children2 = list(tree2.successors(root2))

        inspect_cols = [
            "Child_Parent_Divergence_Significant",
            "Child_Parent_Divergence_P_Value_BH",
            "Sibling_Test_Statistic",
            "Sibling_Degrees_of_Freedom",
            "Sibling_Divergence_P_Value",
            "Sibling_Divergence_P_Value_Corrected",
            "Sibling_BH_Different",
            "Sibling_BH_Same",
            "Sibling_Divergence_Skipped",
            "Sibling_Test_Method",
        ]
        for node in [root2] + root_children2:
            if node in sdf.index:
                print(f"\n  {node}:")
                for col in inspect_cols:
                    if col in sdf.columns:
                        print(f"    {col}: {sdf.loc[node, col]}")

        audit = sdf.attrs.get("sibling_divergence_audit", {})
        if audit:
            print("\n  Pipeline calibration audit:")
            for k, v in audit.items():
                if k != "diagnostics":
                    print(f"    {k}: {v}")
            diag2 = audit.get("diagnostics", {})
            if diag2:
                for k, v in sorted(diag2.items()):
                    print(f"    diag.{k}: {v}")

        if "Child_Parent_Divergence_Significant" in sdf.columns:
            n_edge_sig = sdf["Child_Parent_Divergence_Significant"].sum()
            print(f"\n  Edge-significant nodes: {n_edge_sig}/{len(sdf)}")
        if "Sibling_BH_Different" in sdf.columns:
            n_diff = sdf["Sibling_BH_Different"].sum()
            n_same = sdf["Sibling_BH_Same"].sum()
            n_skip = sdf["Sibling_Divergence_Skipped"].sum()
            print(f"  Sibling outcomes: {n_diff} different, {n_same} same, {n_skip} skipped")

    # ── Verify trace matches pipeline ───────────────────────────────────
    if sdf is not None and root_rec:
        pipeline_t = sdf.loc[root, "Sibling_Test_Statistic"] if root in sdf.index else None
        trace_t = root_rec[0].stat
        if pipeline_t is not None:
            match = abs(pipeline_t - trace_t) < 1e-6
            print(
                f"\n  FIDELITY CHECK (root T): trace={trace_t:.6f}, pipeline={pipeline_t:.6f}, match={match}"
            )
            if not match:
                print(
                    f"  >>> WARNING: trace T differs from pipeline T by {abs(pipeline_t - trace_t):.6e}"
                )

    print("\n")


if __name__ == "__main__":
    cases = [
        ("gaussian_clear_1", 30, 30, 3, 0.6),
        ("gaussian_clear_2", 40, 40, 4, 0.5),
        ("gauss_clear_small", 30, 20, 3, 0.6),
        ("gauss_clear_medium", 60, 40, 4, 0.6),
    ]
    for label, n, p, k, noise in cases:
        trace_case(n, p, k, noise, seed=42, label=label)
