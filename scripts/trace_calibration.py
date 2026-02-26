"""Trace the weighted Gamma GLM calibration on a failing K=1 case.

Prints every sibling pair's raw Wald T, projection dimension k, weight,
ratio T/k, and the fitted model's predicted ĉ at the root node.
Shows exactly why the root is declared SAME (over-deflation).
"""

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _collect_weighted_pairs,
    _fit_weighted_inflation_model,
    predict_weighted_inflation_factor,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def trace_case(n_samples, n_features, n_clusters, noise, seed=42, label=""):
    """Run calibration trace for a single generated case."""
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

    # Gate 2: edge significance
    results_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05, leaf_data=data_bin
    )

    spectral_dims = results_df.attrs.get("_spectral_dims")
    pca_projections = results_df.attrs.get("_pca_projections")
    pca_eigenvalues = results_df.attrs.get("_pca_eigenvalues")
    mean_bl = compute_mean_branch_length(tree)

    # Collect all sibling pairs
    records = _collect_weighted_pairs(
        tree, results_df, mean_bl, spectral_dims, pca_projections, pca_eigenvalues
    )

    n_null = sum(1 for r in records if r.is_null_like)
    n_focal = sum(1 for r in records if not r.is_null_like)

    print("=" * 90)
    print(f"CALIBRATION TRACE: {label}")
    print(f"  n={n_samples}, p={n_features}, K={n_clusters}, noise={noise}")
    print(f"  Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")
    print(f"  Config: SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print(f"  Config: EIGENVALUE_WHITENING={config.EIGENVALUE_WHITENING}")
    print(f"  Config: FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}")
    print("=" * 90)
    print(f"Total pairs: {len(records)}  (null-like: {n_null}, focal: {n_focal})")
    print()

    # --- Per-pair table ---
    print(
        f"{'Parent':>8} {'T':>10} {'k':>4} {'raw_p':>10} "
        f"{'w':>8} {'r=T/k':>8} {'null':>5} {'bl_sum':>8} {'n_par':>6}"
    )
    print("-" * 90)
    for r in sorted(records, key=lambda x: x.weight, reverse=True):
        ratio = r.stat / r.df if r.df > 0 else float("nan")
        print(
            f"{r.parent:>8} {r.stat:10.4f} {r.df:4d} {r.pval:10.6f} "
            f"{r.weight:8.4f} {ratio:8.4f} {str(r.is_null_like):>5} "
            f"{r.bl_sum:8.4f} {r.n_parent:6d}"
        )

    # --- Fit model ---
    model = _fit_weighted_inflation_model(records)
    print(f"\nModel method: {model.method}")
    print(f"Global c-hat (weighted mean of r): {model.global_c_hat:.4f}")
    print(f"Max observed ratio (null-like): {model.max_observed_ratio:.4f}")
    if model.beta is not None:
        print(f"Beta: {model.beta.tolist()}")
    diag = model.diagnostics
    if diag:
        print(f"R-squared: {diag.get('r_squared', 'N/A')}")
        print(f"Effective n: {diag.get('effective_n', 'N/A')}")
        print(f"N null-like: {diag.get('n_null_like', 'N/A')}")

    # --- Root node analysis ---
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
    root_rec = [r for r in records if r.parent == root]
    if root_rec:
        rr = root_rec[0]
        c_hat = predict_weighted_inflation_factor(model, rr.bl_sum, rr.n_parent)
        ratio = rr.stat / rr.df if rr.df > 0 else float("nan")
        p_adj = chi2.sf(rr.stat / c_hat, rr.df) if c_hat > 0 and rr.df > 0 else float("nan")

        print(f"\n--- ROOT NODE ({root}) ---")
        print(f"  Children: {list(tree.successors(root))}")
        print(f"  T = {rr.stat:.4f},  k = {rr.df},  raw p = {rr.pval:.6f}")
        print(f"  r = T/k = {ratio:.4f}")
        print(f"  bl_sum = {rr.bl_sum:.4f},  n_parent = {rr.n_parent}")
        print(f"  weight = {rr.weight:.4f},  is_null_like = {rr.is_null_like}")
        print(f"  c-hat (predicted) = {c_hat:.4f}")
        print(f"  T_adj = T / c-hat = {rr.stat / c_hat:.4f}")
        print(f"  p_adj (pre-BH)    = {p_adj:.6f}")
        print(f"  Reject at alpha=0.05? {p_adj < 0.05}")
        print()
        print("  WITHOUT calibration (raw Wald):")
        print(f"    p = {rr.pval:.6f}, reject? {rr.pval < 0.05}")
        print(f"    DEFLATION FACTOR: T was divided by {c_hat:.2f}x")
        if c_hat > 1:
            print(f"    >>> Over-deflation: raw p={rr.pval:.4f} -> adj p={p_adj:.4f}")
    else:
        print(f"\nRoot {root} has no sibling pair record (non-binary?)")

    # --- Full pipeline comparison ---
    # Run the actual decompose() to see if results differ from manual trace
    print("--- FULL PIPELINE COMPARISON ---")
    tree2 = PosetTree.from_linkage(Z, leaf_names=data_bin.index.tolist())
    tree2.populate_node_divergences(leaf_data=data_bin)
    result = tree2.decompose(leaf_data=data_bin, alpha_local=0.05, sibling_alpha=0.05)
    cluster_assignments = result["cluster_assignments"]
    found_k = result["num_clusters"]
    print(f"  decompose() found K = {found_k}")

    # Inspect stats_df from decomposition
    sdf = tree2.stats_df
    if sdf is not None:
        root2 = [n for n in tree2.nodes if tree2.in_degree(n) == 0][0]
        root_children2 = list(tree2.successors(root2))
        print(f"  Root: {root2}, children: {root_children2}")

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
                        val = sdf.loc[node, col]
                        print(f"    {col}: {val}")

        # Compare calibration audit from full pipeline
        audit = sdf.attrs.get("sibling_divergence_audit", {})
        if audit:
            print("\n  Pipeline calibration audit:")
            for k, v in audit.items():
                if k != "diagnostics":
                    print(f"    {k}: {v}")
            diag2 = audit.get("diagnostics", {})
            if diag2:
                print(f"    diagnostics.r_squared: {diag2.get('r_squared')}")
                print(f"    diagnostics.max_observed_ratio: {diag2.get('max_observed_ratio')}")
                beta2 = diag2.get("beta")
                if beta2:
                    print(f"    diagnostics.beta: {beta2}")

        # Show edge significance counts
        if "Child_Parent_Divergence_Significant" in sdf.columns:
            n_edge_sig = sdf["Child_Parent_Divergence_Significant"].sum()
            print(f"\n  Edge-significant nodes: {n_edge_sig}/{len(sdf)}")

        # Show sibling test outcome distribution
        if "Sibling_BH_Different" in sdf.columns:
            n_diff = sdf["Sibling_BH_Different"].sum()
            n_same = sdf["Sibling_BH_Same"].sum() if "Sibling_BH_Same" in sdf.columns else 0
            n_skip = (
                sdf["Sibling_Divergence_Skipped"].sum()
                if "Sibling_Divergence_Skipped" in sdf.columns
                else 0
            )
            print(f"  Sibling outcomes: {n_diff} different, {n_same} same, {n_skip} skipped")

    print()
    return model, records


if __name__ == "__main__":
    cases = [
        {
            "n_samples": 30,
            "n_features": 30,
            "n_clusters": 3,
            "noise": 0.5,
            "label": "gaussian_clear_1 (true K=3)",
        },
        {
            "n_samples": 40,
            "n_features": 40,
            "n_clusters": 4,
            "noise": 0.8,
            "label": "gaussian_clear_2 (true K=4)",
        },
        {
            "n_samples": 30,
            "n_features": 20,
            "n_clusters": 3,
            "noise": 0.5,
            "label": "gauss_clear_small (true K=3)",
        },
        {
            "n_samples": 60,
            "n_features": 40,
            "n_clusters": 4,
            "noise": 0.6,
            "label": "gauss_clear_medium (true K=4)",
        },
    ]

    for case in cases:
        trace_case(**case)
        print("\n")
