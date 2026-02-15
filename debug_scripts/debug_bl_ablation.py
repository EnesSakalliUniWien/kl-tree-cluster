"""Diagnostic: Isolate branch-length vs nested-variance vs projection effects on edge inflation.

Runs the edge test under null with three configurations:
A) Normal (with branch lengths + projection) — current production
B) No branch lengths (projection only) — tests nested variance alone
C) No projection (raw chi-square, no JL) — tests variance formula alone
D) No branch lengths AND no projection — pure z-score analysis

Also profiles:
- Distribution of n_child/n_parent ratios
- Distribution of BL/mean_BL ratios
- Rejection rate stratified by n_child/n_parent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import extract_leaf_counts
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_mean_branch_length,
    _compute_projected_test,
    _compute_standardized_z,
    _sanitize_positive_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    derive_projection_seed,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def generate_null_data(n_samples=200, n_features=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    return pd.DataFrame(
        X,
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )


def run_branch_length_ablation(seed=42, n=200, p=50):
    """Compare edge test with/without branch lengths and with/without projection."""
    data = generate_null_data(n_samples=n, n_features=p, seed=seed)

    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    results_df = tree.stats_df.copy()

    edge_list = list(tree.edges())
    parent_ids = [par for par, _ in edge_list]
    child_ids = [ch for _, ch in edge_list]
    n_edges = len(edge_list)

    child_leaf_counts = extract_leaf_counts(results_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(results_df, parent_ids)

    mean_bl = _compute_mean_branch_length(tree)

    # ---- Collect per-edge data ----
    records = []
    for i in range(n_edges):
        cid, pid = child_ids[i], parent_ids[i]
        n_child = int(child_leaf_counts[i])
        n_parent = int(parent_leaf_counts[i])
        ratio = n_child / n_parent if n_parent > 0 else 0

        child_dist = np.asarray(tree.nodes[cid].get("distribution"), dtype=np.float64)
        parent_dist = np.asarray(tree.nodes[pid].get("distribution"), dtype=np.float64)

        # Branch length
        bl = None
        if tree.has_edge(pid, cid):
            bl = _sanitize_positive_branch_length(tree.edges[pid, cid].get("branch_length"))

        bl_ratio = (
            bl / mean_bl if (bl is not None and mean_bl is not None and mean_bl > 0) else None
        )

        test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, f"edge:{pid}->{cid}")

        # A) With branch lengths (production)
        stat_a, df_a, pval_a, inv_a = _compute_projected_test(
            child_dist, parent_dist, n_child, n_parent, test_seed, bl, mean_bl
        )

        # B) Without branch lengths (projection still on)
        stat_b, df_b, pval_b, inv_b = _compute_projected_test(
            child_dist, parent_dist, n_child, n_parent, test_seed, None, None
        )

        # C) Raw z-scores with branch lengths (no projection)
        z_c = _compute_standardized_z(child_dist, parent_dist, n_child, n_parent, bl, mean_bl)
        if np.all(np.isfinite(z_c)):
            stat_c = float(np.sum(z_c**2))
            pval_c = float(chi2.sf(stat_c, df=len(z_c)))
        else:
            stat_c, pval_c = np.nan, np.nan

        # D) Raw z-scores without branch lengths (no projection)
        z_d = _compute_standardized_z(child_dist, parent_dist, n_child, n_parent, None, None)
        if np.all(np.isfinite(z_d)):
            stat_d = float(np.sum(z_d**2))
            pval_d = float(chi2.sf(stat_d, df=len(z_d)))
            z_mean = float(np.mean(z_d))
            z_std = float(np.std(z_d))
            z_max = float(np.max(np.abs(z_d)))
        else:
            stat_d, pval_d = np.nan, np.nan
            z_mean, z_std, z_max = np.nan, np.nan, np.nan

        records.append(
            {
                "child": cid,
                "parent": pid,
                "n_child": n_child,
                "n_parent": n_parent,
                "n_ratio": ratio,
                "branch_length": bl,
                "bl_over_mean": bl_ratio,
                "pval_with_bl_with_proj": pval_a,
                "pval_no_bl_with_proj": pval_b,
                "pval_with_bl_no_proj": pval_c,
                "pval_no_bl_no_proj": pval_d,
                "z_mean_no_bl": z_mean,
                "z_std_no_bl": z_std,
                "z_max_abs_no_bl": z_max,
                "proj_dim": df_a,
            }
        )

    df = pd.DataFrame(records)

    # ---- Print results ----
    print("=" * 70)
    print(f"BRANCH LENGTH ABLATION (n={n}, p={p}, seed={seed})")
    print("=" * 70)

    print(f"\nMean branch length: {mean_bl}")
    if mean_bl is not None:
        bl_vals = df["bl_over_mean"].dropna()
        print(
            f"BL/mean_BL: min={bl_vals.min():.3f}, median={bl_vals.median():.3f}, "
            f"max={bl_vals.max():.3f}, mean={bl_vals.mean():.3f}"
        )
        print(
            f"BL/mean_BL < 0.1: {(bl_vals < 0.1).sum()} edges "
            f"({(bl_vals < 0.1).mean():.1%}) — Felsenstein ≈ no effect"
        )
        print(
            f"BL/mean_BL > 2.0: {(bl_vals > 2.0).sum()} edges "
            f"({(bl_vals > 2.0).mean():.1%}) — Felsenstein triples variance"
        )

    n_ratio = df["n_ratio"]
    print(
        f"\nn_child/n_parent: min={n_ratio.min():.3f}, median={n_ratio.median():.3f}, "
        f"max={n_ratio.max():.3f}"
    )

    valid = df.dropna(subset=["pval_with_bl_with_proj"])
    n_valid = len(valid)

    print(f"\n{'Configuration':<35} {'Raw p<0.05':>12} {'Rate':>8}")
    print("-" * 60)
    for col, label in [
        ("pval_with_bl_with_proj", "A) BL + Projection (production)"),
        ("pval_no_bl_with_proj", "B) No BL + Projection"),
        ("pval_with_bl_no_proj", "C) BL + No Projection (raw χ²)"),
        ("pval_no_bl_no_proj", "D) No BL + No Projection (raw χ²)"),
    ]:
        vals = valid[col]
        n_sig = (vals < 0.05).sum()
        print(f"  {label:<33} {n_sig:>5}/{n_valid}    {n_sig/n_valid:.1%}")

    # ---- Stratify by n_ratio ----
    print(f"\n{'n_child/n_parent bin':<20} {'A (prod)':>10} {'B (no BL)':>10} {'D (raw)':>10}")
    print("-" * 55)
    bins = [
        (0, 0.1, "≤0.1"),
        (0.1, 0.2, "0.1-0.2"),
        (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
        (0.8, 1.0, ">0.8"),
    ]

    for lo, hi, label in bins:
        mask = (valid["n_ratio"] > lo) & (valid["n_ratio"] <= hi)
        sub = valid[mask]
        if len(sub) == 0:
            continue
        a_rate = (sub["pval_with_bl_with_proj"] < 0.05).mean()
        b_rate = (sub["pval_no_bl_with_proj"] < 0.05).mean()
        d_rate = (sub["pval_no_bl_no_proj"] < 0.05).mean()
        print(f"  {label:<18} {a_rate:>9.1%} {b_rate:>9.1%} {d_rate:>9.1%}  (n={len(sub)})")

    # ---- z-score diagnostics ----
    z_data = valid.dropna(subset=["z_std_no_bl"])
    print("\nz-score diagnostics (no BL, no projection):")
    print(f"  E[z_mean]:    {z_data['z_mean_no_bl'].mean():.4f}  (should be ~0)")
    print(f"  E[z_std]:     {z_data['z_std_no_bl'].mean():.4f}  (should be ~1.0)")
    print(f"  E[|z|_max]:   {z_data['z_max_abs_no_bl'].mean():.4f}")
    print(f"  Projection k: {z_data['proj_dim'].mean():.1f} (from d={p} features)")

    # ---- Stratify z_std by n_ratio ----
    print(f"\n{'n_ratio bin':<20} {'E[z_std]':>10} {'E[z²/d]':>10}  (z²/d should be ~1)")
    print("-" * 50)
    for lo, hi, label in bins:
        mask = (z_data["n_ratio"] > lo) & (z_data["n_ratio"] <= hi)
        sub = z_data[mask]
        if len(sub) == 0:
            continue
        # Compute mean z_std and mean(stat/dof) for raw chi-square
        sub_raw = valid[mask]
        raw_stat = sub_raw["pval_no_bl_no_proj"]
        # Re-derive stat/dof
        print(f"  {label:<18} {sub['z_std_no_bl'].mean():>9.4f}")

    return df


def run_multi_ablation(n_trials=10, n=200, p=50):
    """Run multiple trials, aggregate rejection rates per configuration."""
    results = {
        "A_with_bl_with_proj": [],
        "B_no_bl_with_proj": [],
        "C_with_bl_no_proj": [],
        "D_no_bl_no_proj": [],
    }

    for i in range(n_trials):
        data = generate_null_data(n_samples=n, n_features=p, seed=2000 + i)
        Z = linkage(
            pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        tree.populate_node_divergences(data)
        results_df = tree.stats_df.copy()

        edge_list = list(tree.edges())
        parent_ids = [par for par, _ in edge_list]
        child_ids = [ch for _, ch in edge_list]
        child_lc = extract_leaf_counts(results_df, child_ids)
        parent_lc = extract_leaf_counts(results_df, parent_ids)
        mean_bl = _compute_mean_branch_length(tree)

        pvals = {"A": [], "B": [], "C": [], "D": []}

        for j in range(len(child_ids)):
            cid, pid = child_ids[j], parent_ids[j]
            nc, np_ = int(child_lc[j]), int(parent_lc[j])
            cd = np.asarray(tree.nodes[cid]["distribution"], dtype=np.float64)
            pd_ = np.asarray(tree.nodes[pid]["distribution"], dtype=np.float64)
            bl = None
            if tree.has_edge(pid, cid):
                bl = _sanitize_positive_branch_length(tree.edges[pid, cid].get("branch_length"))
            seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, f"edge:{pid}->{cid}")

            _, _, pa, _ = _compute_projected_test(cd, pd_, nc, np_, seed, bl, mean_bl)
            _, _, pb, _ = _compute_projected_test(cd, pd_, nc, np_, seed, None, None)

            zc = _compute_standardized_z(cd, pd_, nc, np_, bl, mean_bl)
            zd = _compute_standardized_z(cd, pd_, nc, np_, None, None)

            if np.all(np.isfinite(zc)):
                pc = float(chi2.sf(float(np.sum(zc**2)), df=len(zc)))
            else:
                pc = 1.0
            if np.all(np.isfinite(zd)):
                pd_val = float(chi2.sf(float(np.sum(zd**2)), df=len(zd)))
            else:
                pd_val = 1.0

            pvals["A"].append(pa if np.isfinite(pa) else 1.0)
            pvals["B"].append(pb if np.isfinite(pb) else 1.0)
            pvals["C"].append(pc)
            pvals["D"].append(pd_val)

        m = len(child_ids)
        for key, full_key in [
            ("A", "A_with_bl_with_proj"),
            ("B", "B_no_bl_with_proj"),
            ("C", "C_with_bl_no_proj"),
            ("D", "D_no_bl_no_proj"),
        ]:
            arr = np.array(pvals[key])
            results[full_key].append((arr < 0.05).sum() / m)

    print("\n" + "=" * 70)
    print(f"MULTI-TRIAL ABLATION ({n_trials} trials, n={n}, p={p})")
    print("=" * 70)
    print(f"\n{'Configuration':<35} {'Mean rate':>10} {'Std':>10}")
    print("-" * 60)
    for key, label in [
        ("A_with_bl_with_proj", "A) BL + Projection (production)"),
        ("B_no_bl_with_proj", "B) No BL + Projection"),
        ("C_with_bl_no_proj", "C) BL + No Projection (raw χ²)"),
        ("D_no_bl_no_proj", "D) No BL + No Projection (raw χ²)"),
    ]:
        vals = np.array(results[key])
        print(f"  {label:<33} {vals.mean():>9.1%} {vals.std():>9.1%}")

    print("\n  Expected under null: 5.0%")
    print(
        f"  Difference A→B:     {np.mean(results['A_with_bl_with_proj']) - np.mean(results['B_no_bl_with_proj']):+.1%} (branch length effect)"
    )
    print(
        f"  Difference B→D:     {np.mean(results['B_no_bl_with_proj']) - np.mean(results['D_no_bl_no_proj']):+.1%} (projection effect)"
    )
    print(
        f"  Difference D→5%:    {np.mean(results['D_no_bl_no_proj']) - 0.05:+.1%} (nested variance effect)"
    )


if __name__ == "__main__":
    run_branch_length_ablation(seed=42, n=200, p=50)
    run_multi_ablation(n_trials=20, n=200, p=50)
