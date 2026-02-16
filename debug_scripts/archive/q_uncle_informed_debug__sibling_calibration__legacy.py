"""
Purpose: Legacy uncle-informed sibling calibration experiment (kept for reference).
Inputs: Ad-hoc local setup inside script.
Outputs: Legacy debug console output.
Expected runtime: ~10-90 seconds.
How to run: python debug_scripts/archive/q_uncle_informed_debug__sibling_calibration__legacy.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2, hmean, kstest
from scipy.stats import f as f_dist

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
)
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


def compute_projected_stat(dist_a, dist_b, n_a, n_b, test_id, mean_bl=None,
                           bl_a=None, bl_b=None):
    """Compute projected Wald χ² statistic for two distributions.

    Returns (stat, k, pval, z_raw) where z_raw are the pre-projection z-scores.
    """
    dist_a = np.asarray(dist_a, dtype=np.float64)
    dist_b = np.asarray(dist_b, dtype=np.float64)

    pooled = compute_pooled_proportion(dist_a, dist_b, n_a, n_b)
    diff = dist_a - dist_b
    var = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)

    # Branch length adjustment
    bl_a_s = sanitize_positive_branch_length(bl_a)
    bl_b_s = sanitize_positive_branch_length(bl_b)
    if bl_a_s is not None and bl_b_s is not None and mean_bl is not None and mean_bl > 0:
        bl_sum = bl_a_s + bl_b_s
        if bl_sum > 0:
            bl_norm = 1.0 + bl_sum / (2.0 * mean_bl)
            var = var * bl_norm

    var = np.maximum(var, 1e-10)
    z = _flatten_categorical(diff) / np.sqrt(_flatten_categorical(var))

    if not np.all(np.isfinite(z)):
        return np.nan, np.nan, np.nan, None

    d = len(z)
    n_eff = hmean([n_a, n_b])
    k = compute_projection_dimension(int(n_eff), d)
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)
    R = generate_projection_matrix(d, k, test_seed, use_cache=False)
    projected = R.dot(z) if hasattr(R, "dot") else R @ z
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))
    return stat, k, pval, z


def get_uncle_info(tree, parent):
    """Get uncle node and grandparent for a given parent node.

    Returns (grandparent, uncle) or (None, None) if no uncle exists.
    """
    predecessors = list(tree.predecessors(parent))
    if not predecessors:
        return None, None

    grandparent = predecessors[0]
    gp_children = list(tree.successors(grandparent))
    if len(gp_children) != 2:
        return None, None

    uncle = gp_children[0] if gp_children[1] == parent else gp_children[1]
    return grandparent, uncle


# =============================================================================
# Approach 1: F-test (T_LR / T_PU ~ F(k1, k2))
# =============================================================================

def approach1_ftest(tree, parent, left, right, uncle, mean_bl):
    """Compare sibling divergence to parent-uncle divergence via F-test."""
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
        tree, parent, left, right
    )

    # Sibling statistic T_LR
    stat_lr, k_lr, _, z_lr = compute_projected_stat(
        left_dist, right_dist, n_l, n_r,
        test_id=f"ftest_sibling:{parent}",
        mean_bl=mean_bl, bl_a=bl_l, bl_b=bl_r,
    )

    # Parent-uncle statistic T_PU
    parent_dist = extract_node_distribution(tree, parent)
    uncle_dist = extract_node_distribution(tree, uncle)
    n_p = extract_node_sample_size(tree, parent)
    n_u = extract_node_sample_size(tree, uncle)

    grandparent = list(tree.predecessors(parent))[0]
    bl_p = tree.edges[grandparent, parent].get("branch_length") if tree.has_edge(grandparent, parent) else None
    bl_u = tree.edges[grandparent, uncle].get("branch_length") if tree.has_edge(grandparent, uncle) else None

    stat_pu, k_pu, _, z_pu = compute_projected_stat(
        parent_dist, uncle_dist, n_p, n_u,
        test_id=f"ftest_uncle:{parent}",
        mean_bl=mean_bl, bl_a=bl_p, bl_b=bl_u,
    )

    if np.isnan(stat_lr) or np.isnan(stat_pu) or stat_pu <= 0:
        return np.nan, np.nan, np.nan

    # Under null: T_LR/k_LR ~ χ²(k_lr)/k_lr and T_PU/k_PU ~ χ²(k_pu)/k_pu
    # Ratio ~ F(k_lr, k_pu)
    f_stat = (stat_lr / k_lr) / (stat_pu / k_pu)
    f_pval = float(f_dist.sf(f_stat, dfn=k_lr, dfd=k_pu))

    return f_stat, (k_lr, k_pu), f_pval


# =============================================================================
# Approach 2: Likelihood ratio — does L vs R improve over P vs U?
# =============================================================================

def approach2_likelihood_ratio(tree, parent, left, right, uncle, mean_bl):
    """Test if splitting P into L,R adds info beyond P-vs-U partition.

    Model 1 (reduced): {P} vs {U} — two groups, P is homogeneous
    Model 2 (full):    {L} vs {R} vs {U} — three groups

    LR statistic ≈ T_LR - contribution already explained by P-U split.
    Under null (L=R), improvement ~ χ²(k).
    """
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
        tree, parent, left, right
    )
    uncle_dist = extract_node_distribution(tree, uncle)
    n_u = extract_node_sample_size(tree, uncle)
    parent_dist = extract_node_distribution(tree, parent)
    n_p = extract_node_sample_size(tree, parent)

    # Model 2 has three groups: L, R, U
    # The within-group SS is fixed (each group has its own mean)
    # The between-group SS can be decomposed:
    #   SS_full = SS(L vs R vs U) = SS(P vs U) + SS(L vs R | P)
    #
    # So the "improvement" from splitting P into L,R is:
    #   ΔSS = SS_full - SS_reduced = SS(L vs R)  (within P)
    #
    # This is just the sibling test statistic! But the key difference
    # is how we compute the null distribution.
    #
    # Under the null (L=R), the sibling statistic should be compared
    # against a reference level set by the P-U divergence.

    # Sibling stat (improvement from splitting P)
    stat_lr, k_lr, pval_lr, z_lr = compute_projected_stat(
        left_dist, right_dist, n_l, n_r,
        test_id=f"lr_sibling:{parent}",
        mean_bl=mean_bl, bl_a=bl_l, bl_b=bl_r,
    )

    # P-U stat (reference: divergence at the level above)
    grandparent = list(tree.predecessors(parent))[0]
    bl_p = tree.edges[grandparent, parent].get("branch_length") if tree.has_edge(grandparent, parent) else None
    bl_u = tree.edges[grandparent, uncle].get("branch_length") if tree.has_edge(grandparent, uncle) else None

    stat_pu, k_pu, pval_pu, z_pu = compute_projected_stat(
        parent_dist, uncle_dist, n_p, n_u,
        test_id=f"lr_uncle:{parent}",
        mean_bl=mean_bl, bl_a=bl_p, bl_b=bl_u,
    )

    if np.isnan(stat_lr) or np.isnan(stat_pu):
        return np.nan, np.nan, np.nan

    # Under null (L=R=P), T_LR and T_PU are independent χ²(k) variables.
    # The improvement ratio follows F(k_lr, k_pu).
    # This is algebraically the same as Approach 1 for our projection setup,
    # BUT we frame it as a likelihood ratio improvement:
    #   -2 log(LR) ≈ T_LR  under Gaussian approximation
    # and we calibrate by testing T_LR against the noise floor set by T_PU.

    # Alternative: compute T_LR - E[T_LR | no signal] and test residual
    # E[T_LR | no signal] ≈ k_lr (chi-square mean under null)
    # But post-selection inflates E[T_LR] beyond k_lr.
    # Use T_PU/k_PU as empirical estimate of the inflation factor.
    inflation = stat_pu / k_pu  # empirical chi-square scale under post-selection
    if inflation <= 0:
        return np.nan, np.nan, np.nan

    # Adjust T_LR by the empirical inflation
    adjusted_stat = stat_lr / inflation
    adjusted_pval = float(chi2.sf(adjusted_stat, df=k_lr))

    return adjusted_stat, k_lr, adjusted_pval


# =============================================================================
# Approach 3: Uncle-calibrated variance
# =============================================================================

def approach3_uncle_variance(tree, parent, left, right, uncle, mean_bl):
    """Use uncle divergence to estimate post-selection variance inflation.

    The idea: under null, both T_LR and T_PU are inflated by the same
    post-selection factor. Use T_PU to estimate this factor, then deflate T_LR.
    """
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
        tree, parent, left, right
    )
    uncle_dist = extract_node_distribution(tree, uncle)
    n_u = extract_node_sample_size(tree, uncle)
    parent_dist = extract_node_distribution(tree, parent)
    n_p = extract_node_sample_size(tree, parent)

    grandparent = list(tree.predecessors(parent))[0]
    bl_p = tree.edges[grandparent, parent].get("branch_length") if tree.has_edge(grandparent, parent) else None
    bl_u = tree.edges[grandparent, uncle].get("branch_length") if tree.has_edge(grandparent, uncle) else None

    # Compute z-scores for both pairs (BEFORE projection)
    dist_l = np.asarray(left_dist, dtype=np.float64)
    dist_r = np.asarray(right_dist, dtype=np.float64)
    dist_p = np.asarray(parent_dist, dtype=np.float64)
    dist_u = np.asarray(uncle_dist, dtype=np.float64)

    # Sibling z-scores
    pooled_lr = compute_pooled_proportion(dist_l, dist_r, n_l, n_r)
    var_lr = pooled_lr * (1 - pooled_lr) * (1/n_l + 1/n_r)
    bl_ls = sanitize_positive_branch_length(bl_l)
    bl_rs = sanitize_positive_branch_length(bl_r)
    if bl_ls is not None and bl_rs is not None and mean_bl is not None and mean_bl > 0:
        bl_sum_lr = bl_ls + bl_rs
        if bl_sum_lr > 0:
            var_lr = var_lr * (1 + bl_sum_lr / (2 * mean_bl))
    var_lr = np.maximum(var_lr, 1e-10)
    z_lr = (dist_l - dist_r) / np.sqrt(var_lr)

    # Uncle z-scores
    pooled_pu = compute_pooled_proportion(dist_p, dist_u, n_p, n_u)
    var_pu = pooled_pu * (1 - pooled_pu) * (1/n_p + 1/n_u)
    bl_ps = sanitize_positive_branch_length(bl_p)
    bl_us = sanitize_positive_branch_length(bl_u)
    if bl_ps is not None and bl_us is not None and mean_bl is not None and mean_bl > 0:
        bl_sum_pu = bl_ps + bl_us
        if bl_sum_pu > 0:
            var_pu = var_pu * (1 + bl_sum_pu / (2 * mean_bl))
    var_pu = np.maximum(var_pu, 1e-10)
    z_pu = (dist_p - dist_u) / np.sqrt(var_pu)

    if not np.all(np.isfinite(z_lr)) or not np.all(np.isfinite(z_pu)):
        return np.nan, np.nan, np.nan

    z_lr = z_lr.ravel().astype(np.float64)
    z_pu = z_pu.ravel().astype(np.float64)

    # Estimate per-feature variance inflation from uncle
    # Under null, z_pu[j]² ~ χ²(1) with post-selection inflation.
    # Empirical inflation = mean(z_pu²). Divide z_lr by sqrt(inflation).
    empirical_var = np.mean(z_pu**2)

    if empirical_var <= 0 or not np.isfinite(empirical_var):
        return np.nan, np.nan, np.nan

    # Deflate sibling z-scores by uncle-estimated inflation
    z_adjusted = z_lr / np.sqrt(empirical_var)

    d = len(z_adjusted)
    n_eff = hmean([n_l, n_r])
    k = compute_projection_dimension(int(n_eff), d)
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, f"uncle_var:{parent}")
    R = generate_projection_matrix(d, k, test_seed, use_cache=False)
    projected = R.dot(z_adjusted)
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))
    return stat, k, pval


# =============================================================================
# Approach 4: Cousin-level F-test  (same-depth reference)
#
#           grandparent (G)
#          /              \
#       parent (P)      uncle (U)
#       /      \         /      \
#    left(L) right(R)  UL       UR
#
# Compare T_LR to T_UL_UR — both are sibling splits at the same tree depth,
# so post-selection inflation should be comparable.
# =============================================================================

def approach4_cousin_ftest(tree, parent, left, right, uncle, mean_bl):
    """F-test using uncle's own sibling split as reference (same depth)."""
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
        tree, parent, left, right
    )

    # Get uncle's children
    uncle_children = _get_binary_children(tree, uncle)
    if uncle_children is None:
        return np.nan, np.nan, np.nan

    ul, ur = uncle_children
    ul_dist, ur_dist, n_ul, n_ur, bl_ul, bl_ur = _get_sibling_data(
        tree, uncle, ul, ur
    )
    if n_ul < 2 or n_ur < 2:
        return np.nan, np.nan, np.nan

    # T_LR: sibling stat at parent
    stat_lr, k_lr, _, _ = compute_projected_stat(
        left_dist, right_dist, n_l, n_r,
        test_id=f"cousin_sibling:{parent}",
        mean_bl=mean_bl, bl_a=bl_l, bl_b=bl_r,
    )

    # T_UL_UR: sibling stat at uncle (same depth reference)
    stat_uu, k_uu, _, _ = compute_projected_stat(
        ul_dist, ur_dist, n_ul, n_ur,
        test_id=f"cousin_uncle_sib:{uncle}",
        mean_bl=mean_bl, bl_a=bl_ul, bl_b=bl_ur,
    )

    if np.isnan(stat_lr) or np.isnan(stat_uu) or stat_uu <= 0:
        return np.nan, np.nan, np.nan

    f_stat = (stat_lr / k_lr) / (stat_uu / k_uu)
    f_pval = float(f_dist.sf(f_stat, dfn=k_lr, dfd=k_uu))
    return f_stat, (k_lr, k_uu), f_pval


# =============================================================================
# Approach 5: Cousin-calibrated variance (same-depth z-score reference)
# =============================================================================

def approach5_cousin_variance(tree, parent, left, right, uncle, mean_bl):
    """Use uncle's sibling z-scores to estimate post-selection variance inflation."""
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
        tree, parent, left, right
    )

    uncle_children = _get_binary_children(tree, uncle)
    if uncle_children is None:
        return np.nan, np.nan, np.nan

    ul, ur = uncle_children
    ul_dist, ur_dist, n_ul, n_ur, bl_ul, bl_ur = _get_sibling_data(
        tree, uncle, ul, ur
    )
    if n_ul < 2 or n_ur < 2:
        return np.nan, np.nan, np.nan

    # Compute z-scores for both sibling pairs
    dist_l = np.asarray(left_dist, dtype=np.float64)
    dist_r = np.asarray(right_dist, dtype=np.float64)
    dist_ul = np.asarray(ul_dist, dtype=np.float64)
    dist_ur = np.asarray(ur_dist, dtype=np.float64)

    # Sibling z-scores at parent
    pooled_lr = compute_pooled_proportion(dist_l, dist_r, n_l, n_r)
    var_lr = pooled_lr * (1 - pooled_lr) * (1/n_l + 1/n_r)
    bl_ls = sanitize_positive_branch_length(bl_l)
    bl_rs = sanitize_positive_branch_length(bl_r)
    if bl_ls is not None and bl_rs is not None and mean_bl is not None and mean_bl > 0:
        var_lr = var_lr * (1 + (bl_ls + bl_rs) / (2 * mean_bl))
    var_lr = np.maximum(var_lr, 1e-10)
    z_lr = (dist_l - dist_r) / np.sqrt(var_lr)

    # Sibling z-scores at uncle (same depth reference)
    pooled_uu = compute_pooled_proportion(dist_ul, dist_ur, n_ul, n_ur)
    var_uu = pooled_uu * (1 - pooled_uu) * (1/n_ul + 1/n_ur)
    bl_uls = sanitize_positive_branch_length(bl_ul)
    bl_urs = sanitize_positive_branch_length(bl_ur)
    if bl_uls is not None and bl_urs is not None and mean_bl is not None and mean_bl > 0:
        var_uu = var_uu * (1 + (bl_uls + bl_urs) / (2 * mean_bl))
    var_uu = np.maximum(var_uu, 1e-10)
    z_uu = (dist_ul - dist_ur) / np.sqrt(var_uu)

    if not np.all(np.isfinite(z_lr)) or not np.all(np.isfinite(z_uu)):
        return np.nan, np.nan, np.nan

    z_lr = z_lr.ravel().astype(np.float64)
    z_uu = z_uu.ravel().astype(np.float64)

    # Empirical variance inflation from cousin-level reference
    cousin_var = np.mean(z_uu**2)
    if cousin_var <= 0 or not np.isfinite(cousin_var):
        return np.nan, np.nan, np.nan

    z_adjusted = z_lr / np.sqrt(cousin_var)

    d = len(z_adjusted)
    n_eff = hmean([n_l, n_r])
    k = compute_projection_dimension(int(n_eff), d)
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, f"cousin_var:{parent}")
    R = generate_projection_matrix(d, k, test_seed, use_cache=False)
    projected = R.dot(z_adjusted)
    stat = float(np.sum(projected**2))
    pval = float(chi2.sf(stat, df=k))
    return stat, k, pval


# =============================================================================
# Main diagnostic runner
# =============================================================================

def run_trial(seed=42, n=200, p=50):
    """Run one null trial, return p-values for all approaches at testable nodes."""
    data = generate_null_data(n_samples=n, n_features=p, seed=seed)
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    mean_bl = compute_mean_branch_length(tree)

    records = []
    for node in tree.nodes():
        children = _get_binary_children(tree, node)
        if children is None:
            continue
        left, right = children

        left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(
            tree, node, left, right
        )
        if n_l < 2 or n_r < 2:
            continue

        n_ratio = min(n_l, n_r) / (n_l + n_r)

        # Approach 0: Current baseline
        stat0, k0, pval0, _ = compute_projected_stat(
            left_dist, right_dist, n_l, n_r,
            test_id=f"sibling:{node}",
            mean_bl=mean_bl, bl_a=bl_l, bl_b=bl_r,
        )

        # Find uncle
        grandparent, uncle = get_uncle_info(tree, node)

        pval1 = pval2 = pval3 = pval4 = pval5 = np.nan
        has_uncle = grandparent is not None and uncle is not None

        if has_uncle:
            n_uncle = extract_node_sample_size(tree, uncle)
            if n_uncle >= 2:
                # Approach 1: F-test (parent-uncle level)
                _, _, pval1 = approach1_ftest(tree, node, left, right, uncle, mean_bl)

                # Approach 2: LR-nesting
                _, _, pval2 = approach2_likelihood_ratio(tree, node, left, right, uncle, mean_bl)

                # Approach 3: Uncle variance
                _, _, pval3 = approach3_uncle_variance(tree, node, left, right, uncle, mean_bl)

                # Approach 4: Cousin F-test (same-depth reference)
                _, _, pval4 = approach4_cousin_ftest(tree, node, left, right, uncle, mean_bl)

                # Approach 5: Cousin variance
                _, _, pval5 = approach5_cousin_variance(tree, node, left, right, uncle, mean_bl)

        records.append({
            "parent": node,
            "n_left": n_l,
            "n_right": n_r,
            "n_ratio": n_ratio,
            "has_uncle": has_uncle,
            "pval_baseline": pval0 if np.isfinite(pval0) else 1.0,
            "pval_ftest": pval1 if np.isfinite(pval1) else np.nan,
            "pval_lr_nest": pval2 if np.isfinite(pval2) else np.nan,
            "pval_uncle_var": pval3 if np.isfinite(pval3) else np.nan,
            "pval_cousin_f": pval4 if np.isfinite(pval4) else np.nan,
            "pval_cousin_var": pval5 if np.isfinite(pval5) else np.nan,
        })

    return pd.DataFrame(records)


def run_multi(n_trials=20, n=200, p=50):
    """Run multiple null trials and compare all approaches."""
    print("=" * 80)
    print("UNCLE-INFORMED SIBLING TEST DIAGNOSTIC")
    print(f"{n_trials} null trials, n={n}, p={p}")
    print("=" * 80)

    all_dfs = []
    for i in range(n_trials):
        df = run_trial(seed=5000 + i, n=n, p=p)
        all_dfs.append(df)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_trials} trials")

    combined = pd.concat(all_dfs, ignore_index=True)
    uncle_mask = combined["has_uncle"]

    print()
    print("=" * 80)
    print("OVERALL RESULTS (nodes with uncle available)")
    print("=" * 80)

    # Only use nodes with uncle for fair comparison
    uncle_df = combined[uncle_mask].copy()
    n_total = len(uncle_df)

    approaches = [
        ("pval_baseline",   "0) Current Wald χ²"),
        ("pval_ftest",      "1) F-test (T_LR/T_PU)"),
        ("pval_lr_nest",    "2) LR-nesting (T_LR/inflation)"),
        ("pval_uncle_var",  "3) Uncle-calibrated variance"),
        ("pval_cousin_f",   "4) Cousin F-test (same depth)"),
        ("pval_cousin_var", "5) Cousin-calibrated variance"),
    ]

    print(f"\n{'Approach':<35} {'Raw p<0.05':>12} {'Median p':>10} {'Mean p':>10} {'KS stat':>10} {'KS p-val':>12}")
    print("-" * 95)

    for col, label in approaches:
        pvals = uncle_df[col].dropna().values
        m = len(pvals)
        if m == 0:
            print(f"  {label:<33}  no data")
            continue
        rate = (pvals < 0.05).mean()
        med = np.median(pvals)
        mean = np.mean(pvals)
        ks_stat, ks_p = kstest(pvals, "uniform")
        print(f"  {label:<33} {rate:>7.1%} ({int(rate*m)}/{m}) {med:>10.4f} {mean:>10.4f} {ks_stat:>10.4f} {ks_p:>12.2e}")

    print("\n  Expected under null:                  5.0%        0.5000     0.5000")

    # Stratify by n_ratio
    print()
    print("=" * 80)
    print("STRATIFIED BY min(n_L, n_R) / n_P")
    print("=" * 80)

    bins = [(0, 0.1, "<0.1"), (0.1, 0.2, "0.1-0.2"), (0.2, 0.3, "0.2-0.3"),
            (0.3, 0.4, "0.3-0.4"), (0.4, 0.5, "0.4-0.5")]

    for lo, hi, label in bins:
        mask = (uncle_df["n_ratio"] > lo) & (uncle_df["n_ratio"] <= hi)
        sub = uncle_df[mask]
        if len(sub) < 5:
            continue
        parts = []
        for col, short in [("pval_baseline", "Base"), ("pval_ftest", "F-test"),
                           ("pval_lr_nest", "LR"), ("pval_uncle_var", "UncVar"),
                           ("pval_cousin_f", "CousF"), ("pval_cousin_var", "CousV")]:
            pvals = sub[col].dropna().values
            if len(pvals) > 0:
                rate = (pvals < 0.05).mean()
                parts.append(f"{short}={rate:.1%}")
            else:
                parts.append(f"{short}=N/A")
        print(f"  {label:>10} (n={len(sub):4d}):  {', '.join(parts)}")

    # Decile comparison for the best approach
    print()
    print("=" * 80)
    print("P-VALUE DECILES (should be 0.1, 0.2, ..., 0.9)")
    print("=" * 80)

    quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    expected = [q / 100 for q in quantiles]

    for col, label in approaches:
        pvals = uncle_df[col].dropna().values
        if len(pvals) == 0:
            continue
        deciles = np.percentile(pvals, quantiles)
        dec_str = ", ".join(f"{d:.3f}" for d in deciles)
        print(f"  {label}:")
        print(f"    Observed: {dec_str}")
    print(f"    Expected: {', '.join(f'{e:.3f}' for e in expected)}")

    # Also show nodes WITHOUT uncle (root's children)
    no_uncle = combined[~uncle_mask]
    if len(no_uncle) > 0:
        print()
        print(f"  Nodes without uncle (root children): {len(no_uncle)}")
        base_pvals = no_uncle["pval_baseline"].dropna().values
        if len(base_pvals) > 0:
            print(f"  Baseline p<0.05: {(base_pvals < 0.05).mean():.1%} (n={len(base_pvals)})")

    return combined


if __name__ == "__main__":
    run_multi(n_trials=20, n=200, p=50)
