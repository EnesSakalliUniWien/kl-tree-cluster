"""Diagnostic: Test whether leaf-count-weighted branch lengths improve sibling calibration.

Compares:
A) Current: BL_norm = 1 + (bl_L + bl_R) / (2 * mean_BL)
B) Leaf-weighted: BL_norm = 1 + (bl_L + bl_R) / (2 * mean_BL) * harmonic_mean(n_L, n_R) / mean_n
C) Sqrt-leaf: BL_norm = 1 + (bl_L + bl_R) / (2 * mean_BL) * sqrt(min(n_L, n_R)) / sqrt(mean_n)
D) Direct n-adjust: Var *= (1 + BL_norm) * (1 + c/min(n_L, n_R))  -- penalize small groups
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2, hmean, kstest

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    _flatten_categorical,
    compute_pooled_proportion,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _get_binary_children,
    _get_sibling_data,
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


def sibling_test_with_custom_variance(
    left_dist,
    right_dist,
    n_left,
    n_right,
    bl_left,
    bl_right,
    mean_bl,
    mean_n,
    variant="current",
    test_id="test",
):
    """Run sibling test with different variance formulations.

    Returns (stat, df, pval).
    """
    left_dist = np.asarray(left_dist, dtype=np.float64)
    right_dist = np.asarray(right_dist, dtype=np.float64)

    # Pooled proportion
    pooled = compute_pooled_proportion(left_dist, right_dist, n_left, n_right)
    diff = left_dist - right_dist

    # Base variance: pooled * (1-pooled) * (1/n_L + 1/n_R)
    var = pooled * (1.0 - pooled) * (1.0 / n_left + 1.0 / n_right)

    # Branch length adjustment
    bl_l = sanitize_positive_branch_length(bl_left)
    bl_r = sanitize_positive_branch_length(bl_right)
    bl_sum = None
    if bl_l is not None and bl_r is not None:
        bl_sum = bl_l + bl_r

    if variant == "current":
        # Current: BL_norm = 1 + bl_sum / (2 * mean_bl)
        if bl_sum is not None and bl_sum > 0 and mean_bl is not None and mean_bl > 0:
            bl_norm = 1.0 + bl_sum / (2.0 * mean_bl)
            var = var * bl_norm

    elif variant == "leaf_harmonic":
        # Scale BL by harmonic mean of leaf counts
        if bl_sum is not None and bl_sum > 0 and mean_bl is not None and mean_bl > 0:
            h_n = hmean([n_left, n_right])
            bl_norm = 1.0 + (bl_sum / (2.0 * mean_bl)) * (h_n / mean_n)
            var = var * bl_norm

    elif variant == "leaf_sqrt":
        # Scale BL by sqrt of min leaf count
        if bl_sum is not None and bl_sum > 0 and mean_bl is not None and mean_bl > 0:
            n_min = min(n_left, n_right)
            bl_norm = 1.0 + (bl_sum / (2.0 * mean_bl)) * np.sqrt(n_min) / np.sqrt(mean_n)
            var = var * bl_norm

    elif variant == "n_penalty":
        # Add explicit penalty for small groups on top of BL
        if bl_sum is not None and bl_sum > 0 and mean_bl is not None and mean_bl > 0:
            bl_norm = 1.0 + bl_sum / (2.0 * mean_bl)
            var = var * bl_norm
        # Additional penalty: inflate variance for small min(n_L, n_R)
        n_min = min(n_left, n_right)
        penalty = 1.0 + 10.0 / n_min  # heuristic: 10/n_min extra inflation
        var = var * penalty

    elif variant == "no_bl":
        # No branch length adjustment at all
        pass

    elif variant == "double_bl":
        # Double the Felsenstein adjustment (more conservative)
        if bl_sum is not None and bl_sum > 0 and mean_bl is not None and mean_bl > 0:
            bl_norm = 1.0 + bl_sum / mean_bl  # remove the /2 factor
            var = var * bl_norm

    else:
        raise ValueError(f"Unknown variant: {variant}")

    var = np.maximum(var, 1e-10)
    z = _flatten_categorical(diff) / np.sqrt(_flatten_categorical(var))

    if not np.all(np.isfinite(z)):
        return np.nan, np.nan, np.nan

    d = len(z)
    n_eff = hmean([n_left, n_right])
    k = compute_projection_dimension(int(n_eff), d)
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)
    R = generate_projection_matrix(d, k, test_seed, use_cache=False)
    projected = R.dot(z) if hasattr(R, "dot") else R @ z
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))
    return stat, float(k), pval


def run_trial(seed=42, n=200, p=50):
    """Run one null trial with all variants."""
    data = generate_null_data(n_samples=n, n_features=p, seed=seed)
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    mean_bl = compute_mean_branch_length(tree)

    # Compute mean leaf count across all binary parents' children
    all_n = []
    for node in tree.nodes():
        children = _get_binary_children(tree, node)
        if children is None:
            continue
        left, right = children
        _, _, n_l, n_r, _, _ = _get_sibling_data(tree, node, left, right)
        all_n.extend([n_l, n_r])
    mean_n = np.mean(all_n) if all_n else 1.0

    variants = ["current", "no_bl", "double_bl", "leaf_harmonic", "leaf_sqrt", "n_penalty"]
    results = {v: [] for v in variants}

    for node in tree.nodes():
        children = _get_binary_children(tree, node)
        if children is None:
            continue
        left, right = children
        left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(tree, node, left, right)
        if n_l < 2 or n_r < 2:
            continue

        for variant in variants:
            stat, df, pval = sibling_test_with_custom_variance(
                left_dist,
                right_dist,
                n_l,
                n_r,
                bl_l,
                bl_r,
                mean_bl,
                mean_n,
                variant=variant,
                test_id=f"sibling:{node}",
            )
            results[variant].append(pval if np.isfinite(pval) else 1.0)

    return results


def run_multi(n_trials=20, n=200, p=50):
    """Run multiple trials and compare variants."""
    print("=" * 80)
    print(f"LEAF-WEIGHTED BRANCH LENGTH ABLATION: {n_trials} null trials, n={n}, p={p}")
    print("=" * 80)

    variant_pvals = {}
    variant_names = ["current", "no_bl", "double_bl", "leaf_harmonic", "leaf_sqrt", "n_penalty"]
    for v in variant_names:
        variant_pvals[v] = []

    for i in range(n_trials):
        results = run_trial(seed=4000 + i, n=n, p=p)
        for v in variant_names:
            variant_pvals[v].extend(results[v])
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_trials} trials")

    print()
    print(f"{'Variant':<25} {'Raw p<0.05':>12} {'Median p':>10} {'KS stat':>10} {'KS p-val':>12}")
    print("-" * 75)

    for v in variant_names:
        pvals = np.array(variant_pvals[v])
        rate = (pvals < 0.05).mean()
        med = np.median(pvals)
        ks_stat, ks_p = kstest(pvals[np.isfinite(pvals)], "uniform")
        label = {
            "current": "A) Current (BL/2mean)",
            "no_bl": "B) No BL at all",
            "double_bl": "C) Double BL (BL/mean)",
            "leaf_harmonic": "D) BL * hmean(n)/mean_n",
            "leaf_sqrt": "E) BL * sqrt(nmin)/sqrt(mn)",
            "n_penalty": "F) BL + 10/n_min penalty",
        }[v]
        print(f"  {label:<23} {rate:>11.1%} {med:>10.4f} {ks_stat:>10.4f} {ks_p:>12.2e}")

    print()
    print("  Expected under null:    5.0%       0.5000")

    # Stratify the best-looking variant by n_ratio
    # For that we need per-test metadata - run one more trial with tracking
    print()
    print("Detailed stratification for 'current' and 'n_penalty':")
    data = generate_null_data(n_samples=n, n_features=p, seed=9999)
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    mean_bl = compute_mean_branch_length(tree)
    all_n = []
    for node in tree.nodes():
        ch = _get_binary_children(tree, node)
        if ch is None:
            continue
        l, r = ch
        _, _, nl, nr, _, _ = _get_sibling_data(tree, node, l, r)
        all_n.extend([nl, nr])
    mean_n = np.mean(all_n) if all_n else 1.0

    for variant in ["current", "n_penalty"]:
        ratios = []
        pvals = []
        for node in tree.nodes():
            ch = _get_binary_children(tree, node)
            if ch is None:
                continue
            l, r = ch
            ld, rd, nl, nr, bll, blr = _get_sibling_data(tree, node, l, r)
            if nl < 2 or nr < 2:
                continue
            ratio = min(nl, nr) / (nl + nr)
            _, _, pval = sibling_test_with_custom_variance(
                ld,
                rd,
                nl,
                nr,
                bll,
                blr,
                mean_bl,
                mean_n,
                variant=variant,
                test_id=f"sibling:{node}",
            )
            ratios.append(ratio)
            pvals.append(pval if np.isfinite(pval) else 1.0)

        print(f"\n  {variant}:")
        ratios_arr = np.array(ratios)
        pvals_arr = np.array(pvals)
        bins = [
            (0, 0.1, "<0.1"),
            (0.1, 0.2, "0.1-0.2"),
            (0.2, 0.3, "0.2-0.3"),
            (0.3, 0.4, "0.3-0.4"),
            (0.4, 0.5, "0.4-0.5"),
        ]
        for lo, hi, label in bins:
            mask = (ratios_arr > lo) & (ratios_arr <= hi)
            if mask.sum() == 0:
                continue
            sub = pvals_arr[mask]
            print(f"    {label:>10}: {(sub < 0.05).mean():.1%} (n={mask.sum()})")


if __name__ == "__main__":
    run_multi(n_trials=20, n=200, p=50)
