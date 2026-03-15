#!/usr/bin/env python
"""Deep analysis of z-vectors and T statistics in the edge test.

Traces the EXACT computation for every edge on null + real data:
  z[j] = (θ_child[j] - θ_parent[j]) / √(θ_parent[j](1-θ_parent[j]) × (1/n_c - 1/n_p) × BL_factor)
  projected = R @ z      (R = PCA eigenvectors, k rows)
  T = Σ projected[i]² / λ[i]    (whitened Wald)
  T ~ χ²(k) under H₀

Questions to answer:
  1. What does ||z||² look like? Is it already inflated before projection?
  2. Does the PCA projection concentrate or amplify z?
  3. Is eigenvalue whitening (÷λ) causing inflation?
  4. What's the correlation between z and PCA eigenvectors? (Are they aligned?)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_basis import (
    build_projection_basis_with_padding,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_projected_wald import (
    compute_child_parent_standardized_z_scores,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_spectral_decomposition import (
    compute_child_parent_spectral_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.chi2_pvalue import (
    compute_projected_pvalue,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _generate_null(n=100, p=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


def _generate_block(n_per=50, k=4, p=100, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


def _build_tree(data):
    """Build tree, populate distributions, and compute spectral context."""
    orig = config.EDGE_CALIBRATION
    config.EDGE_CALIBRATION = False
    try:
        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
        # Compute spectral context independently so we have PCA projections/eigenvalues
        spectral_dims, pca_projections, pca_eigenvalues = compute_child_parent_spectral_context(
            tree,
            data,
            config.SPECTRAL_METHOD,
        )
        return tree, spectral_dims, pca_projections, pca_eigenvalues
    finally:
        config.EDGE_CALIBRATION = orig


def _analyse_edges(tree, label, spectral_dims, pca_projections, pca_eigenvalues):
    """Extract z, projected, T for every edge and analyse distributions."""

    # Get mean branch length
    from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
        compute_mean_branch_length,
    )

    mean_bl = compute_mean_branch_length(tree)

    # Collect per-edge intermediate values
    records = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        parent_dist = tree.nodes[parent].get("distribution")
        if parent_dist is None:
            continue
        n_parent = tree.nodes[parent].get("leaf_count", 0)
        if n_parent < 2:
            continue

        for child in children:
            child_dist = tree.nodes[child].get("distribution")
            if child_dist is None:
                continue
            n_child = tree.nodes[child].get("leaf_count", 0)
            if n_child < 1 or n_child >= n_parent:
                continue

            # Branch length
            bl = (
                tree.edges[parent, child].get("branch_length")
                if "branch_length" in tree.edges[parent, child]
                else None
            )

            # Step 1: Compute z-vector
            z = compute_child_parent_standardized_z_scores(
                child_dist,
                parent_dist,
                n_child,
                n_parent,
                branch_length=bl,
                mean_branch_length=mean_bl,
            )
            if not np.all(np.isfinite(z)):
                continue

            n_features = len(z)
            z_norm_sq = float(np.sum(z**2))
            z_mean = float(np.mean(z))
            z_std = float(np.std(z))
            z_max_abs = float(np.max(np.abs(z)))

            # Step 2: get the Marchenko-Pastur signal dimension for this parent node
            # and the corresponding PCA projection matrix + eigenvalues
            signal_dim_for_parent = spectral_dims.get(parent, 2) if spectral_dims else 2

            parent_pca_projection = pca_projections.get(parent) if pca_projections else None
            parent_pca_eigenvalues = pca_eigenvalues.get(parent) if pca_eigenvalues else None

            # projection_dim = degrees of freedom for the chi-square test
            projection_dim = min(signal_dim_for_parent, n_features) if signal_dim_for_parent else 2

            projection_matrix, whitening_eigenvalues = build_projection_basis_with_padding(
                n_features=n_features,
                k=projection_dim,
                pca_projection=parent_pca_projection,
                pca_eigenvalues=parent_pca_eigenvalues,
                random_state=config.PROJECTION_RANDOM_SEED,
            )

            # Step 3: compress z from d dimensions down to projection_dim
            projected_z = projection_matrix @ z

            # Step 4: compute chi-square statistic against chi2(projection_dim)
            test_statistic, eff_df, p_val = compute_projected_pvalue(
                projected_z, projection_dim, eigenvalues=whitening_eigenvalues
            )

            # Decompose the statistic: PCA (whitened) portion vs. random-padding portion
            if whitening_eigenvalues is not None and len(whitening_eigenvalues) > 0:
                n_pca_dims = len(whitening_eigenvalues)
                # PCA portion: each component divided by its eigenvalue (whitened)
                stat_pca_whitened = float(
                    np.sum(projected_z[:n_pca_dims] ** 2 / whitening_eigenvalues)
                )
                # Random-padding portion: plain squared sum (no eigenvalue scaling)
                stat_random_padding = (
                    float(np.sum(projected_z[n_pca_dims:] ** 2))
                    if n_pca_dims < len(projected_z)
                    else 0.0
                )
                # What the statistic would be WITHOUT eigenvalue whitening
                stat_unwhitened = float(np.sum(projected_z**2))
                mean_eig = float(np.mean(whitening_eigenvalues))
                min_eig = float(np.min(whitening_eigenvalues))
                max_eig = float(np.max(whitening_eigenvalues))
            else:
                stat_pca_whitened = test_statistic
                stat_random_padding = 0.0
                stat_unwhitened = test_statistic
                mean_eig = min_eig = max_eig = 1.0

            # Alignment: fraction of z's total energy captured in the PCA subspace
            if parent_pca_projection is not None and len(parent_pca_projection) > 0:
                z_projected_onto_pca = parent_pca_projection @ z  # shape (k_v,)
                z_energy_in_pca_subspace = float(np.sum(z_projected_onto_pca**2))
                z_energy_frac = z_energy_in_pca_subspace / z_norm_sq if z_norm_sq > 0 else 0.0
            else:
                z_energy_frac = float("nan")

            # Nested variance factor
            nested_factor = 1.0 / n_child - 1.0 / n_parent

            records.append(
                {
                    "parent": parent,
                    "child": child,
                    "n_parent": n_parent,
                    "n_child": n_child,
                    "nested_factor": nested_factor,
                    "bl": bl,
                    "bl_factor": (1 + bl / mean_bl) if bl and mean_bl else 1.0,
                    "n_features": n_features,
                    "projection_dim": projection_dim,
                    "signal_dim_spectral": signal_dim_for_parent,
                    "z_norm_sq": z_norm_sq,
                    "z_norm_sq_over_n_features": z_norm_sq / n_features,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "z_max_abs": z_max_abs,
                    "test_statistic": test_statistic,
                    "test_stat_over_df": (
                        test_statistic / projection_dim if projection_dim > 0 else float("nan")
                    ),
                    "stat_pca_whitened": stat_pca_whitened,
                    "stat_random_padding": stat_random_padding,
                    "stat_unwhitened": stat_unwhitened,
                    "stat_unwhitened_over_df": (
                        stat_unwhitened / projection_dim if projection_dim > 0 else float("nan")
                    ),
                    "mean_eig": mean_eig,
                    "min_eig": min_eig,
                    "max_eig": max_eig,
                    "z_energy_in_pca_frac": z_energy_frac,
                    "p_value": p_val,
                }
            )

    df = pd.DataFrame(records)
    return df


def _print_analysis(df, label, true_k):
    """Print key statistics about z and T distributions."""
    print(f"\n{'═' * 100}")
    print(f"  {label}  ({len(df)} edges, true K={true_k})")
    print(f"{'═' * 100}")

    if df.empty:
        print("  No valid edges.")
        return

    # Separate leaf edges (n_child == 1) and internal edges
    leaf_mask = df["n_child"] == 1
    internal_mask = ~leaf_mask

    for subset_name, mask in [
        ("ALL edges", slice(None)),
        ("Leaf edges (n_c=1)", leaf_mask),
        ("Internal edges (n_c>1)", internal_mask),
    ]:
        sub = df[mask] if not isinstance(mask, slice) else df
        if len(sub) == 0:
            continue
        print(f"\n  === {subset_name} ({len(sub)} edges) ===")

        # z-vector analysis
        print("  z-vector:")
        print(
            f"    ||z||²:     mean={sub['z_norm_sq'].mean():.2f},  median={sub['z_norm_sq'].median():.2f}"
        )
        print(
            f"    ||z||²/d:   mean={sub['z_norm_sq_over_n_features'].mean():.4f},  median={sub['z_norm_sq_over_n_features'].median():.4f}"
        )
        print(f"    z_std:      mean={sub['z_std'].mean():.4f}")
        print(f"    z_max|z|:   mean={sub['z_max_abs'].mean():.4f}")

        # Projection analysis
        print("  Projection (spectral signal dimension):")
        print(
            f"    projection_dim (df): mean={sub['projection_dim'].mean():.1f},  median={sub['projection_dim'].median():.1f}"
        )
        print(
            f"    eigenvalues: mean={sub['mean_eig'].mean():.4f},  min={sub['min_eig'].mean():.4f},  max={sub['max_eig'].mean():.4f}"
        )
        if not sub["z_energy_in_pca_frac"].isna().all():
            print(f"    z energy in PCA subspace: mean={sub['z_energy_in_pca_frac'].mean():.4f}")

        # T analysis
        print("  Test statistic (chi-square):")
        print(
            f"    T:          mean={sub['test_statistic'].mean():.2f},  median={sub['test_statistic'].median():.2f}"
        )
        print(
            f"    T/df:       mean={sub['test_stat_over_df'].mean():.3f},  median={sub['test_stat_over_df'].median():.3f}  (expect ~1.0 under null)"
        )
        print(
            f"    T_unwhitened/df: mean={sub['stat_unwhitened_over_df'].mean():.3f},  median={sub['stat_unwhitened_over_df'].median():.3f}"
        )
        print(f"    T_pca_whitened (PCA dims): mean={sub['stat_pca_whitened'].mean():.2f}")
        print(f"    T_random_padding (random dims): mean={sub['stat_random_padding'].mean():.2f}")

        # The KEY question: where does inflation come from?
        print("  INFLATION DECOMPOSITION:")
        # Under H₀, ||z||²/d ≈ 1 (each z[j] ~ N(0,1))
        z_inflation = sub["z_norm_sq_over_n_features"].median()
        print(
            f"    ||z||²/d (should be ~1.0):  {z_inflation:.3f}  → {'OK' if 0.7 < z_inflation < 1.5 else 'INFLATED' if z_inflation > 1.5 else 'DEFLATED'}"
        )

        # Under H₀ with random projection, T/k ≈ ||z||²/d ≈ 1
        # With PCA projection, T/k can differ due to alignment
        t_over_k_med = sub["test_stat_over_df"].median()
        t_no_wh_med = sub["stat_unwhitened_over_df"].median()
        print(
            f"    T_unwhitened/df (projection effect): {t_no_wh_med:.3f}  → {'OK' if 0.7 < t_no_wh_med < 1.5 else 'INFLATED'}"
        )
        print(
            f"    T_whitened/df (after eigenvalue scaling): {t_over_k_med:.3f}  → {'OK' if 0.7 < t_over_k_med < 1.5 else 'INFLATED'}"
        )

        if t_no_wh_med > 0:
            whitening_factor = t_over_k_med / t_no_wh_med
            print(f"    Whitening amplification:     {whitening_factor:.3f}x")
        if z_inflation > 0:
            projection_factor = t_no_wh_med / z_inflation
            print(f"    Projection amplification:    {projection_factor:.3f}x")

    # Per-edge detail for top 20 most inflated
    print("\n  TOP 20 MOST INFLATED EDGES (by test_stat/df):")
    top = df.nlargest(20, "test_stat_over_df")
    print(
        f"  {'parent':>8}→{'child':<8} {'n_p':>5} {'n_c':>5} {'df':>3} "
        f"{'||z²||/d':>9} {'T_uw/df':>7} {'T/df':>7} {'mean_λ':>7} {'z_pca%':>7} {'BL_f':>5}"
    )
    for _, r in top.iterrows():
        print(
            f"  {r['parent']:>8}→{r['child']:<8} {int(r['n_parent']):>5} {int(r['n_child']):>5} {int(r['projection_dim']):>3} "
            f"{r['z_norm_sq_over_n_features']:>9.3f} {r['stat_unwhitened_over_df']:>7.3f} {r['test_stat_over_df']:>7.3f} "
            f"{r['mean_eig']:>7.4f} {r['z_energy_in_pca_frac']:>7.3f} {r['bl_factor']:>5.2f}"
        )

    # Per-edge detail for top 20 LEAST inflated
    print("\n  BOTTOM 20 LEAST INFLATED EDGES (by test_stat/df):")
    bottom = df.nsmallest(20, "test_stat_over_df")
    print(
        f"  {'parent':>8}→{'child':<8} {'n_p':>5} {'n_c':>5} {'df':>3} "
        f"{'||z²||/d':>9} {'T_uw/df':>7} {'T/df':>7} {'mean_λ':>7} {'z_pca%':>7} {'BL_f':>5}"
    )
    for _, r in bottom.iterrows():
        print(
            f"  {r['parent']:>8}→{r['child']:<8} {int(r['n_parent']):>5} {int(r['n_child']):>5} {int(r['projection_dim']):>3} "
            f"{r['z_norm_sq_over_n_features']:>9.3f} {r['stat_unwhitened_over_df']:>7.3f} {r['test_stat_over_df']:>7.3f} "
            f"{r['mean_eig']:>7.4f} {r['z_energy_in_pca_frac']:>7.3f} {r['bl_factor']:>5.2f}"
        )


def _build_tree_with_method(data, spectral_method):
    """Build tree and decompose with a specific spectral method. Return tree + K."""
    orig_ec = config.EDGE_CALIBRATION
    orig_sm = config.SPECTRAL_METHOD
    config.EDGE_CALIBRATION = False
    config.SPECTRAL_METHOD = spectral_method
    try:
        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
        K = results["num_clusters"]
        return tree, K
    finally:
        config.EDGE_CALIBRATION = orig_ec
        config.SPECTRAL_METHOD = orig_sm


def _analyse_edges_random(tree, data):
    """Analyse edges using RANDOM projection (no PCA, no whitening)."""
    from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
        compute_projection_dimension_backend,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
        compute_mean_branch_length,
    )

    mean_bl = compute_mean_branch_length(tree)
    records = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        parent_dist = tree.nodes[parent].get("distribution")
        if parent_dist is None:
            continue
        n_parent = tree.nodes[parent].get("leaf_count", 0)
        if n_parent < 2:
            continue

        for child in children:
            child_dist = tree.nodes[child].get("distribution")
            if child_dist is None:
                continue
            n_child = tree.nodes[child].get("leaf_count", 0)
            if n_child < 1 or n_child >= n_parent:
                continue

            bl = (
                tree.edges[parent, child].get("branch_length")
                if "branch_length" in tree.edges[parent, child]
                else None
            )

            z = compute_child_parent_standardized_z_scores(
                child_dist,
                parent_dist,
                n_child,
                n_parent,
                branch_length=bl,
                mean_branch_length=mean_bl,
            )
            if not np.all(np.isfinite(z)):
                continue

            n_features = len(z)
            z_norm_sq = float(np.sum(z**2))

            # JL-based projection dimension
            projection_dim = compute_projection_dimension_backend(
                n_samples=n_child,
                n_features=n_features,
                eps=config.PROJECTION_EPS,
                minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
            )
            projection_dim = min(projection_dim, n_features)

            # Random orthonormal projection (no PCA, no eigenvalues)
            projection_matrix, whitening_eigenvalues = build_projection_basis_with_padding(
                n_features=n_features,
                k=projection_dim,
                pca_projection=None,
                pca_eigenvalues=None,
                random_state=config.PROJECTION_RANDOM_SEED,
            )

            projected_diff = projection_matrix @ z
            test_statistic, eff_df, p_val = compute_projected_pvalue(
                projected_diff, projection_dim, eigenvalues=whitening_eigenvalues
            )

            records.append(
                {
                    "parent": parent,
                    "child": child,
                    "n_parent": n_parent,
                    "n_child": n_child,
                    "n_features": n_features,
                    "projection_dim": projection_dim,
                    "z_norm_sq": z_norm_sq,
                    "z_norm_sq_over_n_features": z_norm_sq / n_features,
                    "test_statistic": test_statistic,
                    "test_stat_over_df": (
                        test_statistic / projection_dim if projection_dim > 0 else float("nan")
                    ),
                    "p_value": p_val,
                }
            )

    return pd.DataFrame(records)


def _print_comparison(label, true_k, df_pca, K_pca, df_rand, K_rand):
    """Side-by-side comparison of PCA vs random projection."""
    print(f"\n{'═' * 110}")
    print(f"  {label}  (true K={true_k})")
    print(f"{'═' * 110}")

    print(f"\n  {'':40} {'PCA (marchenko_pastur)':>25}   {'Random (JL)':>25}")
    print(f"  {'':40} {'─' * 25}   {'─' * 25}")
    print(f"  {'Clusters found (K)':40} {K_pca:>25}   {K_rand:>25}")

    for subset_name, pca_mask, rand_mask in [
        ("ALL edges", slice(None), slice(None)),
        ("Leaf edges (n_c=1)", df_pca["n_child"] == 1, df_rand["n_child"] == 1),
        ("Internal edges (n_c>1)", df_pca["n_child"] > 1, df_rand["n_child"] > 1),
    ]:
        sp = df_pca[pca_mask] if not isinstance(pca_mask, slice) else df_pca
        sr = df_rand[rand_mask] if not isinstance(rand_mask, slice) else df_rand
        if len(sp) == 0 and len(sr) == 0:
            continue

        print(f"\n  --- {subset_name} ({len(sp)} PCA / {len(sr)} rand edges) ---")

        def _fmt(s, col, fmt=".3f"):
            if len(s) == 0:
                return "N/A"
            return f"med={s[col].median():{fmt}}, mean={s[col].mean():{fmt}}"

        print(
            f"  {'||z||²/n_features':40} {_fmt(sp, 'z_norm_sq_over_n_features'):>25}   {_fmt(sr, 'z_norm_sq_over_n_features'):>25}"
        )
        print(
            f"  {'projection_dim':40} {_fmt(sp, 'projection_dim', '.1f'):>25}   {_fmt(sr, 'projection_dim', '.1f'):>25}"
        )
        print(
            f"  {'test_stat / projection_dim':40} {_fmt(sp, 'test_stat_over_df'):>25}   {_fmt(sr, 'test_stat_over_df'):>25}"
        )
        print(f"  {'p-value':40} {_fmt(sp, 'p_value'):>25}   {_fmt(sr, 'p_value'):>25}")

        # Significance rates at alpha=0.05
        pca_sig = (sp["p_value"] < 0.05).mean() if len(sp) > 0 else float("nan")
        rand_sig = (sr["p_value"] < 0.05).mean() if len(sr) > 0 else float("nan")
        print(f"  {'% significant (p<0.05, raw)':40} {pca_sig:>25.1%}   {rand_sig:>25.1%}")


def main():
    print("PCA vs Random Projection — z-vector and T analysis\n")

    cases = [
        ("NULL  n=100 p=50", _generate_null(100, 50, seed=42), 1),
        ("BLOCK 4c n=200 p=100", _generate_block(50, 4, 100, 0.05, seed=42), 4),
        ("NULL  n=200 p=100", _generate_null(200, 100, seed=99), 1),
    ]

    for label, data, true_k in cases:
        # PCA path (marchenko_pastur)
        tree_pca, K_pca = _build_tree_with_method(data, "marchenko_pastur")
        spectral_dims, pca_projections, pca_eigenvalues = compute_child_parent_spectral_context(
            tree_pca,
            data,
            "marchenko_pastur",
        )
        df_pca = _analyse_edges(tree_pca, label, spectral_dims, pca_projections, pca_eigenvalues)

        # Random path (JL)
        tree_rand, K_rand = _build_tree_with_method(data, None)
        df_rand = _analyse_edges_random(tree_rand, data)

        _print_comparison(label, true_k, df_pca, K_pca, df_rand, K_rand)


if __name__ == "__main__":
    main()
