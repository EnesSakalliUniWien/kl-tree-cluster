"""Compact diagnostic: show regression coefficients and root ĉ for K=1 cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald import (
    _collect_all_pairs,
    _fit_inflation_model,
    _predict_c,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

logging.basicConfig(level=logging.WARNING)

targets = [
    "gaussian_clear_1",
    "gaussian_clear_2",
    "gaussian_clear_3",
    "gaussian_mixed_4",
    "gauss_clear_small",
    "gauss_clear_medium",
    "gauss_moderate_3c",
    "binary_low_noise_2c",
]

cases = get_default_test_cases()
for case in cases:
    name = case.get("name", "")
    if name not in targets:
        continue

    data_df, true_labels, _, metadata = generate_case_data(case)
    dist_c = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist_c, method=config.TREE_LINKAGE_METHOD)

    # Run adj_wald
    config.SIBLING_TEST_METHOD = "cousin_adjusted_wald"
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)

    # Run wald for comparison
    config.SIBLING_TEST_METHOD = "wald"
    tree_w = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp_w = tree_w.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)

    sdf = tree.stats_df
    audit = sdf.attrs.get("sibling_divergence_audit", {})
    diag = audit.get("diagnostics", {})
    mean_bl = compute_mean_branch_length(tree)
    records = _collect_all_pairs(tree, sdf, mean_bl)
    model = _fit_inflation_model(records)

    n_null = sum(1 for r in records if r.is_null_like)
    n_focal = sum(1 for r in records if not r.is_null_like)

    print(f"\n{'='*70}")
    print(f"{name} (n={len(data_df)}, p={data_df.shape[1]}, true_k={case.get('n_clusters')})")
    print(f"  wald K={decomp_w['num_clusters']}, adj_wald K={decomp['num_clusters']}")
    print(f"  Pairs: {len(records)} total, {n_null} null-like, {n_focal} focal")
    print(
        f"  Calib: method={model.method}, n={model.n_calibration}, median_c={model.global_c_hat:.3f}"
    )
    if model.beta is not None:
        print(
            f"  Regression: β₀={model.beta[0]:.3f}, β₁={model.beta[1]:.3f}, β₂={model.beta[2]:.3f}, R²={diag.get('r_squared', 0):.3f}"
        )

    # Show top-3 nodes by n_parent (root and its children)
    focal_recs = sorted(
        [r for r in records if not r.is_null_like], key=lambda r: r.n_parent, reverse=True
    )
    if focal_recs:
        print("  --- Top focal nodes ---")
        for rec in focal_recs[:5]:
            c_hat = _predict_c(model, rec.bl_sum, rec.n_parent)
            t_adj = rec.stat / c_hat
            p_raw = float(chi2.sf(rec.stat, df=rec.df))
            p_adj = float(chi2.sf(t_adj, df=rec.df))
            # Simulated fixes
            c_cap = min(c_hat, model.global_c_hat * 3) if model.beta is not None else c_hat
            c_cap = max(c_cap, 1.0)
            c_noB2 = 1.0
            if model.beta is not None and rec.bl_sum > 0:
                c_noB2 = max(float(np.exp(model.beta[0] + model.beta[1] * np.log(rec.bl_sum))), 1.0)
            t_cap = rec.stat / c_cap
            t_noB2 = rec.stat / c_noB2
            p_cap = float(chi2.sf(t_cap, df=rec.df))
            p_noB2 = float(chi2.sf(t_noB2, df=rec.df))
            print(
                f"    {rec.parent}: n={rec.n_parent}, BL={rec.bl_sum:.4f}, T={rec.stat:.1f}, k={rec.df}"
            )
            print(
                f"      ĉ_reg={c_hat:.2f} → p={p_adj:.4f} | ĉ_cap={c_cap:.2f} → p={p_cap:.4f} | ĉ_noB2={c_noB2:.2f} → p={p_noB2:.4f} | p_raw={p_raw:.4f}"
            )
