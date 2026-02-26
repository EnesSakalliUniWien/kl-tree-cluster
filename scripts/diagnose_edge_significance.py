#!/usr/bin/env python3
"""Diagnose edge significance (Gate 2) behavior on feature_matrix.tsv.

Investigates WHY the projected Wald chi-square edge test fails at the root
and at most internal nodes for sparse binary data.

Key diagnostics:
- Raw z-score analysis at root (before and after projection)
- Projection dimension k vs feature dimension d
- Variance inflation from Felsenstein branch-length scaling
- Distribution of T/k ratios across all edges (should be ~1 under null)
- Comparison of raw p-values before and after BH correction
- Feature-level analysis: which features drive divergence

Usage
-----
    python scripts/diagnose_edge_significance.py
    python scripts/diagnose_edge_significance.py --alpha 0.001
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_projected_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
)
from kl_clustering_analysis.tree.io import tree_from_linkage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=Path("feature_matrix.tsv"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def main() -> None:
    args = parse_args()
    alpha = args.alpha
    out = args.output_dir or Path("benchmarks/results") / f"edge_diag_{_ts()}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data_df = pd.read_csv(args.input, sep="\t", index_col=0).astype(int)
    n, p = data_df.shape
    print(f"Data: {n} samples × {p} features")

    # 2. Build tree
    dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())

    # 3. Decompose (to populate distributions and annotations)
    decomp = tree.decompose(leaf_data=data_df, alpha_local=alpha, sibling_alpha=alpha)
    sdf = tree.stats_df
    K = decomp.get("num_clusters", -1)
    print(f"K = {K}  (alpha={alpha})")

    # 4. Get branch length info
    mean_bl = compute_mean_branch_length(tree)
    print(f"\nMean branch length: {mean_bl}")

    # Collect all branch lengths
    bls = [
        tree.edges[u, v]["branch_length"]
        for u, v in tree.edges()
        if "branch_length" in tree.edges[u, v]
    ]
    if bls:
        bls_arr = np.array(bls)
        print(
            f"Branch lengths: min={bls_arr.min():.6f}, max={bls_arr.max():.6f}, "
            f"mean={bls_arr.mean():.6f}, std={bls_arr.std():.6f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 5. ROOT NODE deep dive
    # ──────────────────────────────────────────────────────────────────────
    root = tree.root()
    children = list(tree.successors(root))
    print(f"\n{'═'*80}")
    print(f"ROOT: {root}  children: {children}")
    print(f"{'═'*80}")

    root_dist = np.asarray(tree.nodes[root]["distribution"], dtype=np.float64)
    n_root = len(tree.get_leaves(root))
    print(
        f"Root distribution: shape={root_dist.shape}, mean={root_dist.mean():.6f}, "
        f"min={root_dist.min():.6f}, max={root_dist.max():.6f}"
    )

    for ch in children:
        ch_dist = np.asarray(tree.nodes[ch]["distribution"], dtype=np.float64)
        n_ch = len(tree.get_leaves(ch))

        # Branch length
        bl = None
        if tree.has_edge(root, ch):
            bl = sanitize_positive_branch_length(tree.edges[root, ch].get("branch_length"))

        print(f"\n  Child {ch}: n_leaves={n_ch}, branch_length={bl}")
        print(
            f"    dist: mean={ch_dist.mean():.6f}, min={ch_dist.min():.6f}, max={ch_dist.max():.6f}"
        )

        # Compute raw difference
        diff = ch_dist - root_dist
        print(
            f"    diff (child - parent): mean={diff.mean():.6f}, std={diff.std():.6f}, "
            f"min={diff.min():.6f}, max={diff.max():.6f}"
        )

        # Compute nested variance factor
        nested_factor = 1.0 / n_ch - 1.0 / n_root
        print(f"    nested_factor (1/n_ch - 1/n_par): {nested_factor:.6f}")
        print(f"    n_ch={n_ch}, n_root={n_root}")

        # Compute raw variance (no Felsenstein)
        var_raw = root_dist * (1 - root_dist) * nested_factor
        var_raw = np.maximum(var_raw, 1e-10)

        # Felsenstein-adjusted variance
        var_fels = var_raw.copy()
        bl_norm = None
        if bl is not None and mean_bl is not None and mean_bl > 0:
            bl_norm = 1.0 + bl / mean_bl
            var_fels = var_raw * bl_norm
            print(f"    Felsenstein multiplier (1 + BL/mean_BL): {bl_norm:.4f}")
            print(f"    Variance inflation: {bl_norm:.4f}x")

        # z-scores WITHOUT Felsenstein
        z_raw = diff / np.sqrt(var_raw)
        print("\n    z-scores WITHOUT Felsenstein:")
        print(f"      mean={z_raw.mean():.4f}, std={z_raw.std():.4f}")
        print(f"      min={z_raw.min():.4f}, max={z_raw.max():.4f}")
        print(f"      |z| > 2: {(np.abs(z_raw) > 2).sum()} / {len(z_raw)}")
        print(f"      |z| > 3: {(np.abs(z_raw) > 3).sum()} / {len(z_raw)}")

        # z-scores WITH Felsenstein
        z_fels = diff / np.sqrt(var_fels)
        print("    z-scores WITH Felsenstein:")
        print(f"      mean={z_fels.mean():.4f}, std={z_fels.std():.4f}")
        print(f"      min={z_fels.min():.4f}, max={z_fels.max():.4f}")
        print(f"      |z| > 2: {(np.abs(z_fels) > 2).sum()} / {len(z_fels)}")
        print(f"      |z| > 3: {(np.abs(z_fels) > 3).sum()} / {len(z_fels)}")

        # Projection dimension
        d = len(z_fels)
        k = compute_projection_dimension(n_ch, d)
        print(f"\n    Projection: d={d} features → k={k} projected dims")
        print(f"    k/d ratio: {k/d:.4f}")

        # Compute projected test statistic
        seed = derive_projection_seed(
            config.PROJECTION_RANDOM_SEED,
            f"edge:{root}->{ch}",
        )
        stat, df_val, pval, invalid = _compute_projected_test(
            ch_dist, root_dist, n_ch, n_root, seed, bl, mean_bl
        )
        print(f"    T={stat:.4f}, df={df_val:.0f}, p={pval:.6g}, invalid={invalid}")
        print(f"    T/k ratio: {stat/k:.4f}  (expect ~1 under null)")

        # What WOULD the p-value be without Felsenstein?
        R = generate_projection_matrix(d, k, seed, use_cache=False)
        projected_raw = R.dot(z_raw)
        stat_raw = float(np.sum(projected_raw**2))
        pval_raw = float(chi2.sf(stat_raw, df=k))
        print(
            f"\n    WITHOUT Felsenstein: T={stat_raw:.4f}, p={pval_raw:.6g}, T/k={stat_raw/k:.4f}"
        )

        # What about the specific features driving the signal?
        # Identify top-10 features by |z|
        top_z_idx = np.argsort(np.abs(z_fels))[::-1][:10]
        print("\n    Top-10 features by |z| (Felsenstein):")
        feat_names = data_df.columns.tolist()
        for idx in top_z_idx:
            fname = feat_names[idx] if idx < len(feat_names) else f"F{idx}"
            print(
                f"      {fname}: z={z_fels[idx]:.4f}, "
                f"child_θ={ch_dist[idx]:.6f}, parent_θ={root_dist[idx]:.6f}, "
                f"diff={diff[idx]:.6f}, var={var_fels[idx]:.8f}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # 6. ALL EDGES — T/k distribution analysis
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("ALL EDGES — T/k distribution")
    print(f"{'═'*80}")

    edge_data = []
    for node in sdf.index:
        if node in {n for n in tree.nodes() if tree.out_degree(n) == 0}:
            continue  # skip leaves
        raw_p = sdf.loc[node].get("Child_Parent_Divergence_P_Value")
        bh_p = sdf.loc[node].get("Child_Parent_Divergence_P_Value_BH")
        sig = sdf.loc[node].get("Child_Parent_Divergence_Significant")
        df_val = sdf.loc[node].get("Child_Parent_Divergence_df")

        if raw_p is not None and df_val is not None:
            try:
                raw_p_f = float(raw_p)
                df_f = float(df_val)
                if np.isfinite(raw_p_f) and np.isfinite(df_f) and df_f > 0:
                    # Recover T from p-value: T = chi2.isf(p, df)
                    if raw_p_f < 1.0 and raw_p_f > 0:
                        T_recovered = float(chi2.isf(raw_p_f, df_f))
                        ratio = T_recovered / df_f
                    else:
                        T_recovered = np.nan
                        ratio = np.nan
                    edge_data.append(
                        {
                            "node": node,
                            "raw_p": raw_p_f,
                            "bh_p": float(bh_p) if bh_p is not None else np.nan,
                            "significant": bool(sig) if sig is not None else False,
                            "df": df_f,
                            "T_recovered": T_recovered,
                            "T_over_k": ratio,
                            "leaf_count": int(sdf.loc[node].get("leaf_count", 0)),
                        }
                    )
            except (TypeError, ValueError):
                pass

    edf = pd.DataFrame(edge_data)
    if not edf.empty:
        valid = edf["T_over_k"].dropna()
        print(f"  Total edges: {len(edf)}")
        print(f"  Significant (BH): {edf['significant'].sum()} / {len(edf)}")
        print(f"  T/k distribution (valid {len(valid)}):")
        print(f"    mean={valid.mean():.4f}, median={valid.median():.4f}")
        print(f"    std={valid.std():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")
        print(f"    T/k < 1: {(valid < 1).sum()} / {len(valid)}")
        print(f"    T/k > 1: {(valid > 1).sum()} / {len(valid)}")
        print(f"    T/k > 2: {(valid > 2).sum()} / {len(valid)}")

        # Raw p-value distribution
        raw_ps = edf["raw_p"].dropna()
        print("\n  Raw p-value distribution:")
        print(f"    < 0.001: {(raw_ps < 0.001).sum()}")
        print(f"    < 0.01:  {(raw_ps < 0.01).sum()}")
        print(f"    < 0.05:  {(raw_ps < 0.05).sum()}")
        print(f"    > 0.50:  {(raw_ps > 0.50).sum()}")
        print(f"    > 0.90:  {(raw_ps > 0.90).sum()}")
        print(f"    mean:    {raw_ps.mean():.4f}")

        # BH-corrected p-value distribution
        bh_ps = edf["bh_p"].dropna()
        print("\n  BH-corrected p-value distribution:")
        print(f"    < 0.001: {(bh_ps < 0.001).sum()}")
        print(f"    < 0.01:  {(bh_ps < 0.01).sum()}")
        print(f"    < 0.05:  {(bh_ps < 0.05).sum()}")
        print(f"    > 0.50:  {(bh_ps > 0.50).sum()}")
        print(f"    > 0.90:  {(bh_ps > 0.90).sum()}")

        # Save edge data
        edf.to_csv(out / "edge_data.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────
    # 7. VARIANCE ANALYSIS — why is variance so large?
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("VARIANCE ANALYSIS — sparse feature impact")
    print(f"{'═'*80}")

    col_means = data_df.mean(axis=0).values
    # θ(1-θ) is maximized at θ=0.5 and minimized near θ=0 or θ=1
    bernoulli_var = col_means * (1 - col_means)
    print("  θ(1-θ) distribution across features:")
    print(f"    mean={bernoulli_var.mean():.6f}, median={np.median(bernoulli_var):.6f}")
    print(f"    min={bernoulli_var.min():.6f}, max={bernoulli_var.max():.6f}")
    print(f"    #features with θ(1-θ) < 0.01: {(bernoulli_var < 0.01).sum()} / {p}")
    print(f"    #features with θ(1-θ) < 0.05: {(bernoulli_var < 0.05).sum()} / {p}")
    print(f"    #features with θ(1-θ) > 0.20: {(bernoulli_var > 0.20).sum()} / {p}")

    # For sparse data (θ≈0), θ(1-θ)≈θ which is very small
    # This means the DENOMINATOR of z is small → z should actually be LARGE
    # unless the NUMERATOR (child_θ - parent_θ) is also proportionally small
    print(f"\n  Sparse features (θ < 0.05): {(col_means < 0.05).sum()} / {p}")
    print("  These have θ(1-θ) < 0.0475, so variance denominator is SMALL")
    print("  BUT the numerator (child_θ - parent_θ) is also bounded by θ")
    print("  Net effect: z ≈ Δθ / √(θ × nested_factor)")
    print("  For purely random child subset: Δθ ≈ 0, so z ≈ 0 regardless of θ")

    # ──────────────────────────────────────────────────────────────────────
    # 8. THE REAL PROBLEM: projection dimension vs actual signal
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("PROJECTION DIMENSION ANALYSIS")
    print(f"{'═'*80}")

    # For different sample sizes, what is k?
    for n_test in [10, 50, 100, 200, 313, 626]:
        k_test = compute_projection_dimension(n_test, p)
        print(f"  n={n_test:4d}: k={k_test:3d}  (out of d={p}),  k/d={k_test/p:.4f}")

    # The issue: k ~ O(log n) but the signal is spread across d features
    # If only a few features carry the signal, projecting to k << d dimensions
    # may dilute it. But if z_i ~ N(0,1) for all i, then T = sum(projected_i²)
    # should follow χ²(k) regardless. The question is whether the z-scores
    # are actually standard normal under the alternative.

    # ──────────────────────────────────────────────────────────────────────
    # 9. Edge significance: per-node raw p-value by subtree size
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("EDGE P-VALUE vs SUBTREE SIZE")
    print(f"{'═'*80}")

    if not edf.empty:
        # Group by subtree size ranges
        for lo, hi in [(2, 10), (10, 50), (50, 100), (100, 200), (200, 626)]:
            mask = (edf["leaf_count"] >= lo) & (edf["leaf_count"] < hi)
            sub = edf.loc[mask]
            if len(sub) > 0:
                sig_count = sub["significant"].sum()
                mean_p = sub["raw_p"].mean()
                print(
                    f"  leaves [{lo:3d},{hi:3d}): {len(sub):4d} nodes, "
                    f"{sig_count:3d} significant, mean raw p={mean_p:.4f}"
                )

    # ──────────────────────────────────────────────────────────────────────
    # 10. Summary
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("SUMMARY")
    print(f"{'═'*80}")

    issues = []
    if mean_bl is not None and bls:
        root_bls = [
            tree.edges[root, ch].get("branch_length")
            for ch in children
            if tree.has_edge(root, ch) and "branch_length" in tree.edges[root, ch]
        ]
        if root_bls:
            for i, ch in enumerate(children):
                if i < len(root_bls) and root_bls[i] is not None:
                    bl_norm = 1.0 + root_bls[i] / mean_bl
                    if bl_norm > 2.0:
                        issues.append(
                            f"Root→{ch}: BL/mean_BL={root_bls[i]/mean_bl:.2f}, "
                            f"Felsenstein multiplier={bl_norm:.2f} (>{2:.0f}x variance inflation)"
                        )

    if not edf.empty:
        n_sig = edf["significant"].sum()
        total = len(edf)
        if n_sig / total < 0.10:
            issues.append(
                f"Only {n_sig}/{total} edges ({100*n_sig/total:.1f}%) pass Gate 2 — "
                f"very low signal detection rate"
            )

    col_means_arr = data_df.mean(axis=0).values
    sparse_frac = (col_means_arr < 0.05).sum() / len(col_means_arr)
    if sparse_frac > 0.5:
        issues.append(
            f"{100*sparse_frac:.0f}% of features have mean < 0.05 — "
            f"extremely sparse data, nested variance denominator is tiny"
        )

    for issue in issues:
        print(f"  ⚠  {issue}")

    if not issues:
        print("  No obvious issues detected.")

    # Save summary
    summary = {
        "input": str(args.input),
        "n_samples": n,
        "n_features": p,
        "alpha": alpha,
        "K": K,
        "mean_branch_length": float(mean_bl) if mean_bl else None,
        "n_edges_significant": int(edf["significant"].sum()) if not edf.empty else 0,
        "n_edges_total": len(edf) if not edf.empty else 0,
        "issues": issues,
    }
    (out / "edge_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs → {out}/")


if __name__ == "__main__":
    main()
