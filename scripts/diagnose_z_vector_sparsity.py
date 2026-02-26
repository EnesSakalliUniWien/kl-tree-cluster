#!/usr/bin/env python3
"""Analyze the z-vector structure at internal edges to understand signal sparsity.

Key questions:
1. How many z_i are "large" (|z_i| > threshold) vs near-zero?
2. What is the distribution of |z_i| — dense or sparse alternative?
3. What is the effective dimension of the signal?
4. How does T = Σ z_i² compare to oracle tests on active features only?
5. What would Higher Criticism / Cauchy combination / power-enhanced tests give?

Usage
-----
    python scripts/diagnose_z_vector_sparsity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import cauchy, chi2, norm

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import extract_node_sample_size
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_standardized_z,
)
from kl_clustering_analysis.tree.io import tree_from_linkage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=Path("feature_matrix.tsv"))
    return p.parse_args()


# ---- Alternative test statistics ----


def higher_criticism(z: np.ndarray) -> tuple[float, float]:
    """Donoho & Jin (2004) Higher Criticism statistic.

    HC* = max_{i: p_i < 1/2} sqrt(d) * (i/d - p_(i)) / sqrt(p_(i) * (1 - p_(i)))

    Returns (HC_statistic, approximate_p_value).
    The p-value is from the asymptotic Gumbel distribution of HC under H0.
    """
    d = len(z)
    # Convert z-scores to two-sided p-values
    p_vals = 2.0 * norm.sf(np.abs(z))
    p_sorted = np.sort(p_vals)

    # HC* restricts to p < alpha_n = 1/2 (or sometimes 1/sqrt(d))
    alpha_n = 0.5
    valid = p_sorted < alpha_n
    if not np.any(valid):
        return 0.0, 1.0

    indices = np.arange(1, d + 1, dtype=float)
    expected = indices / d

    # Standardized HC
    numer = (expected - p_sorted) * np.sqrt(d)
    denom = np.sqrt(p_sorted * (1.0 - p_sorted))
    denom = np.maximum(denom, 1e-15)

    hc_values = numer[valid] / denom[valid]
    hc_stat = float(np.max(hc_values))

    # Approximate p-value using Gumbel (Donoho & Jin):
    # P(HC* > t) ≈ exp(-exp(-π²t²/8)) for large t
    # (very rough; for practical use, simulate under H0)
    if hc_stat > 0:
        log_p = -np.exp(-np.pi**2 * hc_stat**2 / 8.0)
        hc_p = 1.0 - np.exp(log_p)
    else:
        hc_p = 1.0

    return hc_stat, hc_p


def cauchy_combination(z: np.ndarray) -> tuple[float, float]:
    """Liu & Xie (2020) Cauchy Combination Test.

    T_CCT = (1/d) * Σ tan((0.5 - p_i) * π)
    Under H0: T_CCT ~ Cauchy(0, 1) regardless of correlation.

    Returns (CCT_statistic, p_value).
    """
    d = len(z)
    p_vals = 2.0 * norm.sf(np.abs(z))
    # Clamp to avoid tan(±π/2) = ±∞
    p_vals = np.clip(p_vals, 1e-15, 1.0 - 1e-15)

    cauchy_vals = np.tan((0.5 - p_vals) * np.pi)
    cct_stat = float(np.mean(cauchy_vals))
    cct_p = float(cauchy.sf(cct_stat))

    return cct_stat, cct_p


def power_enhancement(z: np.ndarray, screening_threshold: float = 3.0) -> tuple[float, float]:
    """Fan, Liao & Yao (2015) Power Enhancement test.

    J = J_1 + J_0 where:
    - J_1 = (Σ z_i² - d) / sqrt(2d) is the standardized sum-of-squares (N(0,1) under H0)
    - J_0 = Σ_{|z_i| > τ} (z_i² - τ²)  is the screening component (0 under H0 w.h.p.)

    Under H0: J →d N(0, 1) because P(J_0 = 0 | H0) → 1.
    Under sparse H1: J_0 catches the extreme coordinates that J_1 misses.

    Returns (J_statistic, p_value).
    """
    d = len(z)

    # Base statistic (standardized Wald)
    sum_z2 = np.sum(z**2)
    j1 = (sum_z2 - d) / np.sqrt(2.0 * d)

    # Screening component
    # Threshold: τ ∝ sqrt(2 log d) is the Bonferroni threshold for d coordinates
    tau = screening_threshold
    active = np.abs(z) > tau
    if np.any(active):
        j0 = float(np.sum(z[active] ** 2 - tau**2))
    else:
        j0 = 0.0

    j_stat = j1 + j0

    # Under H0: J ~ N(0, 1) (since J0 = 0 w.h.p.)
    j_p = float(norm.sf(j_stat))

    return j_stat, j_p


def oracle_test(z: np.ndarray, threshold: float = 2.0) -> tuple[float, int, float]:
    """Oracle test: Σ z_i² only for |z_i| > threshold.

    This is NOT a valid test (data-driven selection → circular).
    Used only for understanding the signal structure.

    Returns (statistic, n_active, p_value_if_it_were_valid).
    """
    active = np.abs(z) > threshold
    k_active = int(np.sum(active))
    if k_active == 0:
        return 0.0, 0, 1.0
    t_active = float(np.sum(z[active] ** 2))
    p_active = float(chi2.sf(t_active, df=k_active))
    return t_active, k_active, p_active


def main() -> None:
    args = parse_args()

    # 1. Load data and build tree
    data_df = pd.read_csv(args.input, sep="\t", index_col=0).astype(int)
    n, d = data_df.shape
    print(f"Data: {n} samples × {d} features")

    dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())

    # Populate distributions
    tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
    mean_bl = compute_mean_branch_length(tree)

    # 2. Find representative internal edges at different scales
    root = [n for n in tree.nodes() if tree.in_degree(n) == 0][0]
    children = list(tree.successors(root))

    print(f"\nRoot: {root}, children: {children}")
    print(f"Mean BL: {mean_bl:.6f}")
    print()

    # Collect a set of representative edges
    edges_to_analyze = []

    def collect_edges(node, depth=0, max_depth=8):
        """DFS to collect internal edges at different depths."""
        kids = list(tree.successors(node))
        if len(kids) == 0:
            return
        for kid in kids:
            n_leaves = extract_node_sample_size(tree, kid)
            if n_leaves > 1:  # internal node
                edges_to_analyze.append(
                    {
                        "parent": node,
                        "child": kid,
                        "n_leaves": n_leaves,
                        "depth": depth,
                    }
                )
            if depth < max_depth:
                collect_edges(kid, depth + 1, max_depth)

    collect_edges(root)

    # Sort by size for analysis
    edges_to_analyze.sort(key=lambda e: -e["n_leaves"])

    # Analyze top edges by size
    n_to_analyze = min(20, len(edges_to_analyze))
    print(f"Analyzing {n_to_analyze} largest internal edges...")
    print("=" * 120)

    header = (
        f"{'Parent':>8} → {'Child':>8}  "
        f"{'n_leaves':>8}  {'BL':>8}  {'Felsen':>7}  "
        f"|  {'T_full':>8} {'k=d':>5} {'p_wald':>10}  "
        f"|  {'HC':>8} {'p_HC':>10}  "
        f"|  {'CCT':>8} {'p_CCT':>10}  "
        f"|  {'J_PE':>8} {'p_PE':>10}  "
        f"|  {'s_eff':>5} {'T_orac':>8} {'p_orac':>10}"
    )
    print(header)
    print("-" * 120)

    all_results = []
    for edge_info in edges_to_analyze[:n_to_analyze]:
        parent_id = edge_info["parent"]
        child_id = edge_info["child"]
        n_leaves = edge_info["n_leaves"]

        child_dist = np.asarray(tree.nodes[child_id].get("distribution"), dtype=np.float64)
        parent_dist = np.asarray(tree.nodes[parent_id].get("distribution"), dtype=np.float64)

        if child_dist is None or parent_dist is None:
            continue

        n_child = n_leaves
        n_parent = extract_node_sample_size(tree, parent_id)

        # Branch length and Felsenstein
        bl = None
        if tree.has_edge(parent_id, child_id):
            bl_raw = tree.edges[parent_id, child_id].get("branch_length")
            if bl_raw is not None and np.isfinite(bl_raw) and bl_raw > 0:
                bl = bl_raw

        felsenstein = 1.0
        if bl is not None and mean_bl is not None and mean_bl > 0:
            felsenstein = 1.0 + bl / mean_bl

        # Compute z WITHOUT Felsenstein (to see raw signal)
        z_raw = _compute_standardized_z(
            child_dist, parent_dist, n_child, n_parent, branch_length=None, mean_branch_length=None
        )
        # Compute z WITH Felsenstein
        z_felsen = _compute_standardized_z(
            child_dist, parent_dist, n_child, n_parent, branch_length=bl, mean_branch_length=mean_bl
        )

        # Use z_raw for signal analysis (Felsenstein is a separate issue)
        z = z_raw
        d_z = len(z)

        # ---- Standard Wald (k = d) ----
        t_full = float(np.sum(z**2))
        p_wald = float(chi2.sf(t_full, df=d_z))

        # ---- Higher Criticism ----
        hc_stat, hc_p = higher_criticism(z)

        # ---- Cauchy Combination ----
        cct_stat, cct_p = cauchy_combination(z)

        # ---- Power Enhancement ----
        # Use τ = sqrt(2 log d) ≈ 3.5 for d=456 (Bonferroni-ish threshold)
        tau = np.sqrt(2.0 * np.log(d_z))
        pe_stat, pe_p = power_enhancement(z, screening_threshold=tau)

        # ---- Oracle (for understanding) ----
        orac_t, orac_s, orac_p = oracle_test(z, threshold=2.0)

        bl_str = f"{bl:.6f}" if bl is not None else "None"

        print(
            f"{parent_id:>8} → {child_id:>8}  "
            f"{n_leaves:>8}  {bl_str:>8}  {felsenstein:>7.3f}  "
            f"|  {t_full:>8.1f} {d_z:>5} {p_wald:>10.2e}  "
            f"|  {hc_stat:>8.2f} {hc_p:>10.2e}  "
            f"|  {cct_stat:>8.2f} {cct_p:>10.2e}  "
            f"|  {pe_stat:>8.2f} {pe_p:>10.2e}  "
            f"|  {orac_s:>5} {orac_t:>8.1f} {orac_p:>10.2e}"
        )

        all_results.append(
            {
                "parent": parent_id,
                "child": child_id,
                "n_leaves": n_leaves,
                "bl": bl,
                "felsenstein": felsenstein,
                "d": d_z,
                "T_full": t_full,
                "p_wald": p_wald,
                "HC": hc_stat,
                "p_HC": hc_p,
                "CCT": cct_stat,
                "p_CCT": cct_p,
                "PE": pe_stat,
                "p_PE": pe_p,
                "s_oracle": orac_s,
                "T_oracle": orac_t,
                "p_oracle": orac_p,
            }
        )

    print()
    print("=" * 120)

    # 3. Deep dive: z-vector sparsity at root
    print("\n" + "=" * 80)
    print("Z-VECTOR SPARSITY ANALYSIS AT ROOT")
    print("=" * 80)

    for child_id in children:
        child_dist = np.asarray(tree.nodes[child_id].get("distribution"), dtype=np.float64)
        parent_dist = np.asarray(tree.nodes[root].get("distribution"), dtype=np.float64)
        n_child = extract_node_sample_size(tree, child_id)
        n_parent = extract_node_sample_size(tree, root)

        z = _compute_standardized_z(
            child_dist, parent_dist, n_child, n_parent, branch_length=None, mean_branch_length=None
        )

        abs_z = np.abs(z)
        d_z = len(z)

        print(f"\n--- Edge: {root} → {child_id} (n={n_child}) ---")
        print(f"  d = {d_z}")
        print(f"  ||z||² = {np.sum(z**2):.1f}  (expected under H0: {d_z})")
        print(f"  max|z| = {np.max(abs_z):.2f}")
        print(f"  mean|z| = {np.mean(abs_z):.4f}")
        print(f"  median|z| = {np.median(abs_z):.4f}")

        # Count z's above various thresholds
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        print(
            "\n  Threshold  | #active | %active | Σ z_i² (active) | χ²(k) crit | p-value (oracle)"
        )
        print(f"  {'-'*85}")
        for t in thresholds:
            active = abs_z > t
            k_active = int(np.sum(active))
            if k_active == 0:
                print(
                    f"  |z| > {t:4.1f}  |     {k_active:>3} |  {0:>5.1f}% |             0.0 |        n/a |          1.0"
                )
                continue
            t_active = float(np.sum(z[active] ** 2))
            crit = float(chi2.ppf(0.95, df=k_active))
            p_active = float(chi2.sf(t_active, df=k_active))
            print(
                f"  |z| > {t:4.1f}  |     {k_active:>3} |  {100*k_active/d_z:>5.1f}% | "
                f"   {t_active:>10.1f} |  {crit:>9.1f} | {p_active:>10.2e}"
            )

        # Distribution of |z| in percentiles
        pcts = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
        print("\n  Percentile distribution of |z|:")
        for pct in pcts:
            val = np.percentile(abs_z, pct)
            print(f"    P{pct:>5.1f} = {val:.4f}")

        # Effective signal dimension: how many features needed to capture X% of T
        z_sq_sorted = np.sort(z**2)[::-1]  # descending
        cumsum = np.cumsum(z_sq_sorted)
        total = cumsum[-1]
        for frac in [0.5, 0.75, 0.9, 0.95, 0.99]:
            idx = np.searchsorted(cumsum, frac * total)
            print(
                f"    {frac*100:.0f}% of T captured by top {idx+1}/{d_z} features ({100*(idx+1)/d_z:.1f}%)"
            )

    # 4. Compare all test statistics at various edges
    print("\n" + "=" * 80)
    print("TEST STATISTIC COMPARISON SUMMARY")
    print("=" * 80)

    if all_results:
        df = pd.DataFrame(all_results)

        # Count rejections at alpha = 0.05 for each method
        alpha = 0.05
        for method, p_col in [
            ("Wald χ²(d)", "p_wald"),
            ("Higher Criticism", "p_HC"),
            ("Cauchy Combination", "p_CCT"),
            ("Power Enhancement", "p_PE"),
        ]:
            n_reject = int(np.sum(df[p_col] < alpha))
            print(f"  {method:25s}: {n_reject}/{len(df)} edges significant at α={alpha}")

        print("\n  Note: Oracle test is for understanding only (circular, not valid)")
        n_oracle = int(np.sum(df["p_oracle"] < alpha))
        print(f"  Oracle (|z|>2):          {n_oracle}/{len(df)} edges significant at α={alpha}")


if __name__ == "__main__":
    main()
