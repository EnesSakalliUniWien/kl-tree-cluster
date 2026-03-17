#!/usr/bin/env python
"""Experiment 12 — Deflation Diagnostics & Alternative Estimators.

The current cousin-adjusted-Wald calibration produces ĉ=1.0 (no deflation)
because the weighted mean of T/k ratios is ≤1.0 when k is small (1-3).
At χ²(2), T/k has mean=1 but std=1 — the estimator has no power.

This lab:
  A. Collects raw T and p-values for ALL pairs (null-like + focal).
  B. Diagnoses the null p-value distribution (should be Uniform if no inflation).
  C. Tests alternative inflation estimators that work at small k:
     1. Storey π₀ — fraction of true nulls from p-value distribution
     2. Anderson-Darling vs Uniform — departure from calibration
     3. Beta(a,b) MLE fit — shape parameter 'a' < 1 == inflation
     4. Median-ratio estimator — median(T/k) instead of weighted mean
     5. Upper-tail estimator — P(T/k > 2) vs expected P(χ²(k)/k > 2)
  D. Simulates what WOULD happen under each estimator:
     - Deflate focal T by estimated ĉ, recompute BH, count splits.
  E. Compares to ground truth (true vs false splits).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, kstest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab_helpers import FAILURE_CASES, REGRESSION_GUARD_CASES, build_tree_and_data
from statsmodels.stats.multitest import multipletests

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
)

ALL_CASES = FAILURE_CASES + REGRESSION_GUARD_CASES

# ─────────────────────────────────────────────────────────────────────
# Section A: Collect raw pair data with full annotations
# ─────────────────────────────────────────────────────────────────────


def collect_pair_data(case_name: str) -> pd.DataFrame:
    """Build tree, annotate Gates 1-2, collect ALL sibling pair records."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)

    # Use tree.stats_df (populated by build_tree_and_data → populate_node_divergences)
    # which contains leaf_count, distribution, is_leaf columns required by the pipeline
    assert tree.stats_df is not None, "stats_df must be populated"
    annotations_df = tree.stats_df.copy()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
        spectral_method=config.SPECTRAL_METHOD,
        sibling_method=config.SIBLING_TEST_METHOD,
    )
    annotated = bundle.annotated_df

    # Collect raw pair records (ALL pairs, not just focal)
    records, _ = collect_sibling_pair_records(
        tree,
        annotated,
        mean_branch_length=None,  # Felsenstein disabled
        spectral_dims=annotated.attrs.get("_sibling_spectral_dims"),
    )

    # Enrich with ground-truth labels
    if y_true is not None:
        leaf_to_label = {}
        for i, leaf_name in enumerate(data_df.index):
            leaf_to_label[leaf_name] = int(y_true[i])

        import networkx as nx

        def get_leaves(node):
            if tree.nodes[node].get("is_leaf", False):
                return {tree.nodes[node].get("label", node)}
            return {
                tree.nodes[n].get("label", n)
                for n in nx.descendants(tree, node)
                if tree.nodes[n].get("is_leaf", False)
            }

    rows = []
    for rec in records:
        row = {
            "parent": rec.parent,
            "left": rec.left,
            "right": rec.right,
            "stat": rec.stat,
            "k": rec.degrees_of_freedom,
            "p_value": rec.p_value,
            "n_parent": rec.n_parent,
            "is_null_like": rec.is_null_like,
            "edge_weight": rec.edge_weight,
            "bl_sum": rec.branch_length_sum,
        }
        if np.isfinite(rec.stat) and rec.degrees_of_freedom > 0:
            row["ratio"] = rec.stat / rec.degrees_of_freedom
        else:
            row["ratio"] = np.nan

        # Ground truth: true split if children have different label sets
        if y_true is not None:
            left_labels = {leaf_to_label.get(lf, -1) for lf in get_leaves(rec.left)}
            right_labels = {leaf_to_label.get(lf, -1) for lf in get_leaves(rec.right)}
            row["true_split"] = not bool(left_labels & right_labels)
        else:
            row["true_split"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Section B: P-value distribution diagnostics
# ─────────────────────────────────────────────────────────────────────


def diagnose_pvalue_distribution(pdf: pd.DataFrame, label: str = "") -> dict:
    """Analyze p-value distribution of null-like pairs for calibration quality."""
    null_pdf = pdf[pdf["is_null_like"] & np.isfinite(pdf["p_value"])].copy()
    focal_pdf = pdf[~pdf["is_null_like"] & np.isfinite(pdf["p_value"])].copy()

    result = {"label": label, "n_null": len(null_pdf), "n_focal": len(focal_pdf)}

    if len(null_pdf) < 3:
        result["diagnostic"] = "too_few_null_pairs"
        return result

    p_null = null_pdf["p_value"].values

    # KS test vs Uniform(0,1)
    ks_stat, ks_p = kstest(p_null, "uniform")
    result["ks_stat"] = ks_stat
    result["ks_p"] = ks_p

    # Anderson-Darling vs Uniform
    # (scipy doesn't have AD for uniform directly; use KS + histogram-based)
    # P-value histogram: count in 10 bins
    hist, _ = np.histogram(p_null, bins=10, range=(0, 1))
    expected = len(p_null) / 10
    chi2_stat = np.sum((hist - expected) ** 2 / expected)
    chi2_p = chi2.sf(chi2_stat, df=9)
    result["hist_chi2_stat"] = chi2_stat
    result["hist_chi2_p"] = chi2_p
    result["p_hist"] = hist.tolist()

    # Fraction of p < 0.05 (expected: 5%)
    frac_below_05 = np.mean(p_null < 0.05)
    result["frac_p_lt_05"] = frac_below_05
    result["expected_frac_05"] = 0.05

    # Fraction of p < 0.01 (expected: 1%)
    frac_below_01 = np.mean(p_null < 0.01)
    result["frac_p_lt_01"] = frac_below_01

    # Storey π₀ estimate: fraction of p > λ
    lam = 0.5
    pi0_est = np.mean(p_null > lam) / (1 - lam)
    pi0_est = min(pi0_est, 1.0)
    result["storey_pi0"] = pi0_est

    # Beta(a,b) MLE fit
    p_clamped = np.clip(p_null, 1e-12, 1 - 1e-12)
    try:
        a_hat, b_hat, _, _ = beta_dist.fit(p_clamped, floc=0, fscale=1)
        result["beta_a"] = a_hat
        result["beta_b"] = b_hat
        # Under Uniform: a=1, b=1. a<1 → pile-up near 0 (inflation)
    except Exception:
        result["beta_a"] = np.nan
        result["beta_b"] = np.nan

    # T/k ratio statistics for null-like pairs
    valid_null = null_pdf[np.isfinite(null_pdf["ratio"]) & (null_pdf["ratio"] > 0)]
    if len(valid_null) > 0:
        ratios = valid_null["ratio"].values
        result["null_ratio_mean"] = np.mean(ratios)
        result["null_ratio_median"] = np.median(ratios)
        result["null_ratio_std"] = np.std(ratios)
        result["null_ratio_max"] = np.max(ratios)

        # Per-k breakdown
        for k_val in sorted(valid_null["k"].unique()):
            k_ratios = valid_null[valid_null["k"] == k_val]["ratio"].values
            result[f"null_ratio_k{k_val}_n"] = len(k_ratios)
            result[f"null_ratio_k{k_val}_mean"] = np.mean(k_ratios)
            result[f"null_ratio_k{k_val}_median"] = np.median(k_ratios)
    else:
        result["null_ratio_mean"] = np.nan

    # Same for focal pairs
    valid_focal = focal_pdf[np.isfinite(focal_pdf["ratio"]) & (focal_pdf["ratio"] > 0)]
    if len(valid_focal) > 0:
        result["focal_ratio_mean"] = np.mean(valid_focal["ratio"].values)
        result["focal_ratio_median"] = np.median(valid_focal["ratio"].values)
        result["focal_ratio_max"] = np.max(valid_focal["ratio"].values)
    else:
        result["focal_ratio_mean"] = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────
# Section C: Alternative inflation estimators
# ─────────────────────────────────────────────────────────────────────


def estimate_inflation_methods(pdf: pd.DataFrame) -> dict:
    """Estimate ĉ using multiple methods. Returns dict of method → ĉ."""
    valid = pdf[np.isfinite(pdf["stat"]) & (pdf["k"] > 0)].copy()
    if len(valid) == 0:
        return {}

    ratios = (valid["stat"] / valid["k"]).values
    weights = valid["edge_weight"].values
    k_values = valid["k"].values
    p_values = valid["p_value"].values

    methods = {}

    # M0: Current method (weighted mean, clamped to [1, max])
    weighted_mean = np.average(ratios, weights=weights) if weights.sum() > 0 else 1.0
    methods["M0_weighted_mean"] = max(weighted_mean, 1.0)

    # M1: Unweighted median of T/k for null-like pairs only
    null_ratios = ratios[valid["is_null_like"].values]
    if len(null_ratios) > 0:
        methods["M1_null_median"] = max(float(np.median(null_ratios)), 1.0)
    else:
        methods["M1_null_median"] = 1.0

    # M2: Unweighted mean of T/k for null-like pairs only
    if len(null_ratios) > 0:
        methods["M2_null_mean"] = max(float(np.mean(null_ratios)), 1.0)
    else:
        methods["M2_null_mean"] = 1.0

    # M3: Per-k expected-value calibration
    # For each k, compare observed mean(T) to expected k under χ²(k)
    # ĉ_k = mean(T)/k; aggregate via weighted average across k-groups
    k_groups = valid.groupby("k")
    c_per_k = []
    w_per_k = []
    for k_val, grp in k_groups:
        if k_val <= 0:
            continue
        null_grp = grp[grp["is_null_like"]]
        if len(null_grp) < 2:
            continue
        c_k = float(null_grp["stat"].mean() / k_val)
        c_per_k.append(c_k)
        w_per_k.append(len(null_grp))
    if c_per_k:
        methods["M3_per_k_mean"] = max(float(np.average(c_per_k, weights=w_per_k)), 1.0)
    else:
        methods["M3_per_k_mean"] = 1.0

    # M4: Upper-tail calibration
    # Under χ²(k)/k, P(T/k > 2) has a known value. Compare observed to expected.
    # ĉ = observed_fraction / expected_fraction (excess upper-tail mass)
    null_mask = valid["is_null_like"].values
    for threshold in [1.5, 2.0, 3.0]:
        expected_frac_per_k = []
        observed_frac_per_k = []
        count_per_k = []
        for k_val, grp in k_groups:
            if k_val <= 0:
                continue
            null_grp = grp[grp["is_null_like"]]
            if len(null_grp) < 5:
                continue
            null_r = (null_grp["stat"] / k_val).values
            obs_frac = np.mean(null_r > threshold)
            exp_frac = float(chi2.sf(threshold * k_val, df=k_val))
            if exp_frac > 0.001:
                expected_frac_per_k.append(exp_frac)
                observed_frac_per_k.append(obs_frac)
                count_per_k.append(len(null_grp))
        if expected_frac_per_k:
            total_expected = np.average(expected_frac_per_k, weights=count_per_k)
            total_observed = np.average(observed_frac_per_k, weights=count_per_k)
            if total_expected > 0:
                c_tail = total_observed / total_expected
                methods[f"M4_upper_tail_{threshold}"] = max(c_tail, 1.0)

    # M5: P-value quantile calibration
    # If p-values are inflated, quantile(p, 0.5) < 0.5
    # ĉ estimated by matching: P(χ²(k) > T/ĉ) should have median 0.5
    # → T/ĉ should be the k-specific median of χ²(k)
    # → ĉ = median(T) / χ²_median(k)
    c_quantile_per_k = []
    w_quantile_per_k = []
    for k_val, grp in k_groups:
        if k_val <= 0:
            continue
        null_grp = grp[grp["is_null_like"]]
        if len(null_grp) < 5:
            continue
        median_T = float(null_grp["stat"].median())
        chi2_median = float(chi2.ppf(0.5, df=k_val))
        if chi2_median > 0:
            c_q = median_T / chi2_median
            c_quantile_per_k.append(c_q)
            w_quantile_per_k.append(len(null_grp))
    if c_quantile_per_k:
        methods["M5_quantile_match"] = max(
            float(np.average(c_quantile_per_k, weights=w_quantile_per_k)), 1.0
        )
    else:
        methods["M5_quantile_match"] = 1.0

    # M6: Storey-based inflation
    # π₀ from p-values → if π₀ < 1, some pairs are truly different
    # But we want inflation of null p-values, so use only null-like p-values
    null_p = p_values[null_mask]
    if len(null_p) >= 10:
        lam = 0.5
        pi0 = np.mean(null_p > lam) / (1 - lam)
        pi0 = min(pi0, 1.0)
        # If well-calibrated, pi0 ≈ 1.0 (all nulls are truly null)
        # If inflated, pi0 < 1.0 (some nulls wrongly rejected)
        # Estimate ĉ from the fraction that's "leaking" into small p:
        # frac(p < 0.05) should be 0.05; if it's F, then ĉ ≈ F/0.05
        frac_05 = np.mean(null_p < 0.05)
        if frac_05 > 0.05:
            methods["M6_type1_ratio"] = frac_05 / 0.05
        else:
            methods["M6_type1_ratio"] = 1.0
        methods["M6_storey_pi0"] = pi0
    else:
        methods["M6_type1_ratio"] = 1.0
        methods["M6_storey_pi0"] = 1.0

    return methods


# ─────────────────────────────────────────────────────────────────────
# Section D: Simulate deflation impact
# ─────────────────────────────────────────────────────────────────────


def simulate_deflation(pdf: pd.DataFrame, c_hat: float, alpha: float = 0.05) -> dict:
    """Simulate BH on focal pairs after deflating by c_hat."""
    focal = pdf[~pdf["is_null_like"] & np.isfinite(pdf["stat"]) & (pdf["k"] > 0)].copy()
    if len(focal) == 0:
        return {"n_focal": 0, "n_split": 0, "n_merge": 0, "c_hat": c_hat, "TP": 0, "FP": 0, "FN": 0, "TN": 0, "n_true_split": 0}

    t_adj = focal["stat"].values / c_hat
    k_vals = focal["k"].values
    p_adj = np.array([float(chi2.sf(t, df=int(k))) for t, k in zip(t_adj, k_vals)])

    # BH correction
    reject, _, _, _ = multipletests(p_adj, alpha=alpha, method="fdr_bh")

    focal = focal.copy()
    focal["t_deflated"] = t_adj
    focal["p_deflated"] = p_adj
    focal["reject_bh"] = reject

    n_true_split = focal["true_split"].sum() if "true_split" in focal.columns else 0
    n_tp = (
        int(focal[focal["reject_bh"] & (focal["true_split"] == True)].shape[0])
        if "true_split" in focal.columns
        else 0
    )
    n_fp = (
        int(focal[focal["reject_bh"] & (focal["true_split"] == False)].shape[0])
        if "true_split" in focal.columns
        else 0
    )
    n_fn = (
        int(focal[~focal["reject_bh"] & (focal["true_split"] == True)].shape[0])
        if "true_split" in focal.columns
        else 0
    )
    n_tn = (
        int(focal[~focal["reject_bh"] & (focal["true_split"] == False)].shape[0])
        if "true_split" in focal.columns
        else 0
    )

    return {
        "n_focal": len(focal),
        "c_hat": c_hat,
        "n_split": int(reject.sum()),
        "n_merge": int((~reject).sum()),
        "TP": n_tp,
        "FP": n_fp,
        "FN": n_fn,
        "TN": n_tn,
        "n_true_split": int(n_true_split),
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════


def main():
    header = "=" * 100
    print(header)
    print("  EXPERIMENT 12: Deflation Diagnostics & Alternative Estimators")
    print(header)

    all_pair_data = {}
    for case in ALL_CASES:
        print(f"  Collecting: {case} ...", end=" ", flush=True)
        try:
            pdf = collect_pair_data(case)
            all_pair_data[case] = pdf
            n_null = pdf["is_null_like"].sum()
            n_focal = (~pdf["is_null_like"]).sum()
            print(f"{len(pdf)} pairs ({n_null} null-like, {n_focal} focal)")
        except Exception as e:
            print(f"ERROR: {e}")

    # ═══════════════ Section B: P-value Distribution Diagnostics ═══════════════

    print(f"\n{header}")
    print("  Section B: Null-like P-value Distribution Diagnostics")
    print(header)
    print("  If well-calibrated, null-like p-values ~Uniform(0,1):")
    print("  - KS p > 0.05 → consistent with Uniform")
    print("  - frac(p<0.05) ≈ 5%")
    print("  - Beta(a,b) with a≈1, b≈1")
    print()

    diag_rows = []
    for case, pdf in all_pair_data.items():
        diag = diagnose_pvalue_distribution(pdf, label=case)
        diag_rows.append(diag)

    fmt = "{:>35s} {:>6s} {:>6s} {:>7s} {:>7s} {:>8s} {:>8s} {:>7s} {:>7s} {:>7s}"
    print(
        fmt.format("case", "n_nul", "n_foc", "KS_p", "χ²_p", "p<.05", "p<.01", "π₀", "β_a", "β_b")
    )
    print("-" * 110)
    for d in diag_rows:
        if d.get("diagnostic") == "too_few_null_pairs":
            print(
                f"  {d['label']:>33s} {d['n_null']:>6d} {d['n_focal']:>6d}   (too few null-like pairs)"
            )
            continue
        print(
            fmt.format(
                d["label"],
                str(d["n_null"]),
                str(d["n_focal"]),
                f"{d.get('ks_p', float('nan')):.4f}",
                f"{d.get('hist_chi2_p', float('nan')):.4f}",
                f"{d.get('frac_p_lt_05', float('nan')):.1%}",
                f"{d.get('frac_p_lt_01', float('nan')):.1%}",
                f"{d.get('storey_pi0', float('nan')):.3f}",
                f"{d.get('beta_a', float('nan')):.3f}",
                f"{d.get('beta_b', float('nan')):.3f}",
            )
        )

    # Per-k T/k ratio breakdown (all null-like pairs pooled)
    print(f"\n{'─' * 80}")
    print("  Per-k T/k ratio statistics (null-like pairs, pooled across cases)")
    print(f"{'─' * 80}")
    all_null = pd.concat(
        [pdf[pdf["is_null_like"] & np.isfinite(pdf["ratio"])] for pdf in all_pair_data.values()],
        ignore_index=True,
    )
    if len(all_null) > 0:
        for k_val in sorted(all_null["k"].unique()):
            sub = all_null[all_null["k"] == k_val]
            chi2_median_expected = float(chi2.ppf(0.5, df=k_val)) / k_val
            chi2_mean_expected = 1.0
            upper_tail_expected = float(chi2.sf(2 * k_val, df=k_val))
            upper_tail_observed = np.mean(sub["ratio"] > 2)
            print(
                f"  k={int(k_val):>3d}: n={len(sub):>5d}  "
                f"mean(T/k)={sub['ratio'].mean():.3f} [exp 1.000]  "
                f"median(T/k)={sub['ratio'].median():.3f} [exp {chi2_median_expected:.3f}]  "
                f"P(T/k>2)={upper_tail_observed:.3f} [exp {upper_tail_expected:.3f}]  "
                f"max={sub['ratio'].max():.2f}"
            )

    # P-value histogram (pooled null-like)
    print(f"\n{'─' * 80}")
    print("  P-value histogram (null-like pairs, pooled, 10 bins)")
    print(f"{'─' * 80}")
    all_null_p = all_null["p_value"].dropna().values
    if len(all_null_p) > 0:
        hist, edges = np.histogram(all_null_p, bins=10, range=(0, 1))
        expected_per_bin = len(all_null_p) / 10
        for i in range(10):
            bar = "█" * int(hist[i] / max(1, expected_per_bin) * 20)
            excess = hist[i] / expected_per_bin if expected_per_bin > 0 else 0
            print(
                f"  [{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]:>5d}  "
                f"(×{excess:.2f} expected)  {bar}"
            )

    # ═══════════════ Section C: Alternative Estimators ═══════════════

    print(f"\n{header}")
    print("  Section C: Alternative Inflation Estimators (ĉ)")
    print(header)

    estimator_rows = []
    for case, pdf in all_pair_data.items():
        methods = estimate_inflation_methods(pdf)
        methods["case"] = case
        estimator_rows.append(methods)

    est_df = pd.DataFrame(estimator_rows).set_index("case")
    est_cols = [c for c in est_df.columns if c.startswith("M")]
    print(f"\n  {'case':>35s}", end="")
    for col in est_cols:
        print(f"  {col:>16s}", end="")
    print()
    print("  " + "-" * (37 + 18 * len(est_cols)))
    for case in est_df.index:
        print(f"  {case:>35s}", end="")
        for col in est_cols:
            val = est_df.loc[case, col]
            if pd.isna(val):
                print(f"  {'—':>16s}", end="")
            else:
                print(f"  {val:>16.3f}", end="")
        print()

    # ═══════════════ Section D: Simulate Deflation Impact ═══════════════

    print(f"\n{header}")
    print("  Section D: Simulated Deflation Impact on Focal Pairs")
    print(header)
    print("  For each estimator: deflate focal T, BH-correct, count splits vs ground truth.\n")

    est_methods = [c for c in est_cols if "storey_pi0" not in c]

    for case in ALL_CASES:
        pdf = all_pair_data.get(case)
        if pdf is None:
            continue
        methods = estimate_inflation_methods(pdf)
        true_k = None
        for tc in FAILURE_CASES + REGRESSION_GUARD_CASES:
            pass  # true_k comes from the data
        n_true_splits = (
            int(pdf[~pdf["is_null_like"]]["true_split"].sum())
            if "true_split" in pdf.columns
            else "?"
        )

        print(
            f"\n  {case} (n_focal={int((~pdf['is_null_like']).sum())}, true_split_focal={n_true_splits})"
        )
        sim_fmt = (
            "    {:<22s}  ĉ={:>6.3f}  splits={:>3d}  TP={:>3d}  FP={:>3d}  FN={:>3d}  TN={:>3d}"
        )
        for method_name in est_methods:
            c_hat = methods.get(method_name, 1.0)
            if pd.isna(c_hat):
                continue
            sim = simulate_deflation(pdf, c_hat, alpha=config.SIBLING_ALPHA)
            print(
                sim_fmt.format(
                    method_name,
                    c_hat,
                    sim["n_split"],
                    sim["TP"],
                    sim["FP"],
                    sim["FN"],
                    sim["TN"],
                )
            )

    # ═══════════════ Section E: Summary ═══════════════

    print(f"\n{header}")
    print("  Section E: Cross-Case Summary — Which Estimator Helps Most?")
    print(header)

    summary_rows = []
    for case in ALL_CASES:
        pdf = all_pair_data.get(case)
        if pdf is None:
            continue
        methods = estimate_inflation_methods(pdf)
        for method_name in est_methods:
            c_hat = methods.get(method_name, 1.0)
            if pd.isna(c_hat):
                continue
            sim = simulate_deflation(pdf, c_hat, alpha=config.SIBLING_ALPHA)
            summary_rows.append(
                {
                    "case": case,
                    "method": method_name,
                    "c_hat": c_hat,
                    "TP": sim["TP"],
                    "FP": sim["FP"],
                    "FN": sim["FN"],
                    "TN": sim["TN"],
                    "accuracy": (sim["TP"] + sim["TN"]) / max(sim["n_focal"], 1),
                    "n_focal": sim["n_focal"],
                }
            )

    sdf = pd.DataFrame(summary_rows)
    if len(sdf) > 0:
        agg = (
            sdf.groupby("method")
            .agg(
                total_TP=("TP", "sum"),
                total_FP=("FP", "sum"),
                total_FN=("FN", "sum"),
                total_TN=("TN", "sum"),
                mean_c_hat=("c_hat", "mean"),
                mean_accuracy=("accuracy", "mean"),
            )
            .sort_values("mean_accuracy", ascending=False)
        )

        agg["precision"] = agg["total_TP"] / (agg["total_TP"] + agg["total_FP"]).replace(0, 1)
        agg["recall"] = agg["total_TP"] / (agg["total_TP"] + agg["total_FN"]).replace(0, 1)
        agg["f1"] = (
            2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"]).replace(0, 1)
        )

        print("\n  Aggregate across all cases:")
        print(agg.to_string())

    print(f"\n{header}")
    print("  Done.")
    print(header)


if __name__ == "__main__":
    main()
