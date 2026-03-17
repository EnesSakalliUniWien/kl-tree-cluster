#!/usr/bin/env python
"""Experiment 11 — Symmetric Pair Power Dissection.

Exp10 revealed: 91% of true splits are missed in symmetric pairs (leaf_ratio < 1.5).
This experiment dissects WHY by examining:

1. Signal strength: ||z||² distribution for true_split vs false_split
2. Per-feature signal: how many z_j² > threshold? (sparse vs diffuse signal)
3. Marginal BH test: BH correction on individual z_j² ~ χ²(1) tests
4. Higher Criticism: HC statistic for sparse signal detection
5. Max-|z| test: max_j |z_j| compared to Gumbel extreme-value distribution
6. Sample size: n_left, n_right — is this just a small-n problem?
7. Effect size: mean |z_j| for signal features vs noise features

The goal: understand whether the signal EXISTS but the projected Wald misses it,
or whether there's genuinely no detectable signal at these sample sizes.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from lab_helpers import FAILURE_CASES, REGRESSION_GUARD_CASES, build_tree_and_data

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.pooled_variance import (
    standardize_proportion_difference,
)


def _get_gt_label(tree, parent, left, right, y_true, leaf_data):
    if y_true is None:
        return None
    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}

    def _leaf_classes(node):
        if tree.out_degree(node) == 0:
            lbl = tree.nodes[node].get("label", node)
            idx = label_to_idx.get(lbl)
            return {y_true[idx]} if idx is not None else set()
        classes = set()
        for d in nx.descendants(tree, node):
            if tree.out_degree(d) == 0:
                lbl = tree.nodes[d].get("label", d)
                idx = label_to_idx.get(lbl)
                if idx is not None:
                    classes.add(y_true[idx])
        return classes

    lc = _leaf_classes(left)
    rc = _leaf_classes(right)
    if not lc or not rc:
        return None
    if lc == rc:
        return "false_split"
    elif lc & rc:
        return "mixed"
    else:
        return "true_split"


def _count_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return 1
    return sum(1 for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


# ── Signal analysis helpers ─────────────────────────────────────────────────


def _marginal_bh_test(z: np.ndarray, alpha: float = 0.05) -> Tuple[int, np.ndarray]:
    """BH correction on marginal z_j² ~ χ²(1) tests.

    Returns (n_significant, sorted_pvalues).
    """
    p_values = chi2.sf(z**2, df=1)
    sorted_p = np.sort(p_values)
    d = len(sorted_p)
    # BH threshold: p_(i) ≤ (i/d) × α
    bh_thresholds = np.arange(1, d + 1) / d * alpha
    rejections = sorted_p <= bh_thresholds
    if rejections.any():
        # Largest i where p_(i) ≤ threshold
        n_sig = int(np.max(np.where(rejections)[0]) + 1)
    else:
        n_sig = 0
    return n_sig, sorted_p


def _higher_criticism(z: np.ndarray) -> Tuple[float, float]:
    """Higher Criticism statistic (Donoho & Jin 2004).

    HC* = max_{1 ≤ i ≤ d/2} √d · (i/d - p_(i)) / √(p_(i)(1-p_(i)))

    Returns (HC_statistic, fraction_at_detection_boundary).
    """
    p_values = chi2.sf(z**2, df=1)
    sorted_p = np.sort(p_values)
    d = len(sorted_p)
    if d == 0:
        return 0.0, 0.0

    # Only scan up to d/2 (HC is meaningful in the left tail)
    n_scan = max(1, d // 2)
    hc_values = np.zeros(n_scan)
    for i in range(n_scan):
        p_i = sorted_p[i]
        expected = (i + 1) / d
        denom = math.sqrt(max(p_i * (1 - p_i), 1e-15))
        hc_values[i] = math.sqrt(d) * (expected - p_i) / denom

    hc_star = float(np.max(hc_values))
    i_star = int(np.argmax(hc_values))
    frac = (i_star + 1) / d
    return hc_star, frac


def _max_z_test(z: np.ndarray) -> Tuple[float, float]:
    """Max-|z| test against Gumbel extreme-value distribution.

    Under H₀, max|z_j| has approximately Gumbel distribution with
    location a_d = √(2 ln d) - (ln(4π ln d))/(2√(2 ln d))
    scale  b_d = 1/√(2 ln d)

    Returns (max_z, gumbel_pvalue).
    """
    d = len(z)
    if d <= 1:
        return float(np.max(np.abs(z))), 1.0

    max_z = float(np.max(np.abs(z)))
    # Gumbel parameters for max of d standard normals
    a_d = math.sqrt(2 * math.log(d)) - math.log(4 * math.pi * math.log(d)) / (
        2 * math.sqrt(2 * math.log(d))
    )
    b_d = 1.0 / math.sqrt(2 * math.log(d))
    # P(max|Z| > x) ≈ 1 - exp(-2·exp(-(x-a_d)/b_d))
    # Factor of 2 for two-sided (|Z| = max of Z and -Z)
    gumbel_p = 1.0 - math.exp(-2 * math.exp(-(max_z - a_d) / b_d))
    return max_z, max(0.0, min(1.0, gumbel_p))


def _signal_features_analysis(z: np.ndarray) -> dict:
    """Characterize the sparsity and strength of signal in z."""
    z_sq = z**2
    d = len(z)

    # Features with z² > various thresholds (under H₀, z²~χ²(1), mean=1)
    n_above_4 = int(np.sum(z_sq > 4))  # p < 0.046
    n_above_9 = int(np.sum(z_sq > 9))  # p < 0.003
    n_above_16 = int(np.sum(z_sq > 16))  # p < 6e-5

    # Signal fraction
    sig_frac = n_above_4 / max(d, 1)

    # Mean z² for top-10% features vs bottom-90%
    sorted_zsq = np.sort(z_sq)[::-1]
    top_10_pct = max(1, d // 10)
    mean_top10 = float(np.mean(sorted_zsq[:top_10_pct]))
    mean_bottom90 = float(np.mean(sorted_zsq[top_10_pct:])) if d > top_10_pct else 0.0

    # Effective signal dimension (entropy of normalized z²)
    z_sq_norm = z_sq / max(np.sum(z_sq), 1e-15)
    nonzero = z_sq_norm[z_sq_norm > 0]
    eff_dim = float(np.exp(-np.sum(nonzero * np.log(nonzero))))

    return {
        "n_features": d,
        "n_above_4": n_above_4,
        "n_above_9": n_above_9,
        "n_above_16": n_above_16,
        "signal_fraction": round(sig_frac, 4),
        "mean_zsq_top10pct": round(mean_top10, 2),
        "mean_zsq_bottom90pct": round(mean_bottom90, 2),
        "signal_noise_ratio": round(mean_top10 / max(mean_bottom90, 0.01), 2),
        "effective_signal_dim": round(eff_dim, 1),
    }


# ── Main diagnostic ─────────────────────────────────────────────────────────


def diagnose_case(case_name: str) -> pd.DataFrame:
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")
    n_features = data_df.shape[1]
    root = next(n for n, d in tree.in_degree() if d == 0)

    rows: List[dict] = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            left_dist = extract_node_distribution(tree, left)
            right_dist = extract_node_distribution(tree, right)
            n_left = extract_node_sample_size(tree, left)
            n_right = extract_node_sample_size(tree, right)
        except Exception:
            continue

        if n_left < 2 or n_right < 2:
            continue

        try:
            z, _var = standardize_proportion_difference(
                left_dist,
                right_dist,
                float(n_left),
                float(n_right),
            )
        except Exception:
            continue
        if not np.isfinite(z).all():
            continue

        n_leaves_left = _count_leaves(tree, left)
        n_leaves_right = _count_leaves(tree, right)
        leaf_ratio = max(n_leaves_left, n_leaves_right) / max(min(n_leaves_left, n_leaves_right), 1)

        gt = _get_gt_label(tree, parent, left, right, y_t, data_df)

        # Only focus on ground-truth-labeled pairs
        if gt not in ("true_split", "false_split"):
            continue

        z_norm_sq = float(np.sum(z**2))

        # ── Projected Wald at various k ──
        jl_k = compute_projection_dimension_backend(n_left + n_right, n_features)
        # Simulate random projection at k=5 (baseline)
        rng = np.random.RandomState(hash(parent) % (2**31))
        G = rng.randn(jl_k, n_features)
        Q, _ = np.linalg.qr(G.T)
        R = Q[:, :jl_k].T
        T_proj = float(np.sum((R @ z) ** 2))
        p_proj = float(chi2.sf(T_proj, jl_k))

        # ── Alternative tests ──
        n_bh_sig, _ = _marginal_bh_test(z, alpha=config.SIBLING_ALPHA)
        hc_stat, hc_frac = _higher_criticism(z)
        max_z, gumbel_p = _max_z_test(z)
        sig_info = _signal_features_analysis(z)

        # ── Bonferroni on max-z ──
        bonf_p = min(1.0, 2 * n_features * norm.sf(max_z))

        # ── Cauchy combination test ──
        p_marginals = chi2.sf(z**2, df=1)
        # Avoid exactly 0 or 1
        p_marginals = np.clip(p_marginals, 1e-15, 1 - 1e-15)
        cauchy_T = float(np.mean(np.tan(np.pi * (0.5 - p_marginals))))
        # Under H₀, Cauchy(0,1): P(T > t) = 0.5 - arctan(t)/π
        cauchy_p = 0.5 - math.atan(cauchy_T) / math.pi

        # ── depth ──
        try:
            depth = nx.shortest_path_length(tree, root, parent)
        except nx.NetworkXNoPath:
            depth = -1

        row = {
            "case": case_name,
            "true_k": true_k,
            "parent": parent,
            "gt": gt,
            "depth": depth,
            "n_left": n_left,
            "n_right": n_right,
            "n_total": n_left + n_right,
            "n_leaves_L": n_leaves_left,
            "n_leaves_R": n_leaves_right,
            "leaf_ratio": round(leaf_ratio, 2),
            "is_symmetric": leaf_ratio < 1.5,
            # Signal strength
            "z_norm_sq": round(z_norm_sq, 1),
            "d": n_features,
            # Projected Wald (baseline)
            "T_proj_k5": round(T_proj, 2),
            "p_proj_k5": p_proj,
            "split_proj_k5": p_proj < config.SIBLING_ALPHA,
            # Marginal BH
            "n_bh_significant": n_bh_sig,
            "split_bh": n_bh_sig > 0,
            # Higher Criticism
            "hc_stat": round(hc_stat, 2),
            "hc_detection_frac": round(hc_frac, 4),
            # Max-z (Gumbel)
            "max_z": round(max_z, 2),
            "gumbel_p": gumbel_p,
            "split_gumbel": gumbel_p < config.SIBLING_ALPHA,
            # Bonferroni
            "bonf_p": bonf_p,
            "split_bonf": bonf_p < config.SIBLING_ALPHA,
            # Cauchy combination
            "cauchy_T": round(cauchy_T, 2),
            "cauchy_p": cauchy_p,
            "split_cauchy": cauchy_p < config.SIBLING_ALPHA,
            # Signal structure
            **sig_info,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Reporting ────────────────────────────────────────────────────────────────


def summarize(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    sym = df[df["is_symmetric"]].copy()
    asym = df[~df["is_symmetric"]].copy()

    for subset_name, subset in [
        ("ALL PAIRS", df),
        ("SYMMETRIC (ratio<1.5)", sym),
        ("ASYMMETRIC (ratio≥1.5)", asym),
    ]:
        if len(subset) == 0:
            continue
        ts = subset[subset["gt"] == "true_split"]
        fs = subset[subset["gt"] == "false_split"]

        print(f"\n{'='*100}")
        print(f"  {subset_name}: {len(subset)} pairs ({len(ts)} true_split, {len(fs)} false_split)")
        print(f"{'='*100}")

        # ── Signal strength comparison ──
        print("\n  Signal strength (||z||²):")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['z_norm_sq'].mean():.0f}  median={ts['z_norm_sq'].median():.0f}"
                f"  min={ts['z_norm_sq'].min():.0f}  max={ts['z_norm_sq'].max():.0f}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['z_norm_sq'].mean():.0f}  median={fs['z_norm_sq'].median():.0f}"
                f"  min={fs['z_norm_sq'].min():.0f}  max={fs['z_norm_sq'].max():.0f}"
            )

        # ── Sample size ──
        print("\n  Sample sizes:")
        if len(ts) > 0:
            print(
                f"    true_split:  n_total mean={ts['n_total'].mean():.0f}  median={ts['n_total'].median():.0f}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: n_total mean={fs['n_total'].mean():.0f}  median={fs['n_total'].median():.0f}"
            )

        # ── Signal sparsity ──
        print("\n  Signal sparsity (features with z²>4, i.e. p<0.046):")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['n_above_4'].mean():.1f}  median={ts['n_above_4'].median():.0f}"
                f"  (of d={ts['d'].iloc[0]})"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['n_above_4'].mean():.1f}  median={fs['n_above_4'].median():.0f}"
            )

        print("\n  Signal-to-noise ratio (top-10% z² / bottom-90% z²):")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['signal_noise_ratio'].mean():.1f}  median={ts['signal_noise_ratio'].median():.1f}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['signal_noise_ratio'].mean():.1f}  median={fs['signal_noise_ratio'].median():.1f}"
            )

        print("\n  Effective signal dimension (entropy of z² spectrum):")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['effective_signal_dim'].mean():.1f}  median={ts['effective_signal_dim'].median():.1f}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['effective_signal_dim'].mean():.1f}  median={fs['effective_signal_dim'].median():.1f}"
            )

        # ── Test comparison ──
        print("\n  Test comparison (detection rates):")
        tests = [
            ("Projected Wald (k=5)", "split_proj_k5"),
            ("Marginal BH", "split_bh"),
            ("Max-z (Gumbel)", "split_gumbel"),
            ("Bonferroni max-z", "split_bonf"),
            ("Cauchy combination", "split_cauchy"),
        ]
        print(f"    {'Test':30s}  {'TP rate':>8s}  {'FP rate':>8s}  {'Accuracy':>8s}")
        for test_name, col in tests:
            if col not in subset.columns:
                continue
            tp_rate = ts[col].mean() if len(ts) > 0 else 0
            fp_rate = fs[col].mean() if len(fs) > 0 else 0
            # Weighted accuracy
            n_ts, n_fs = len(ts), len(fs)
            acc = (tp_rate * n_ts + (1 - fp_rate) * n_fs) / max(n_ts + n_fs, 1)
            print(f"    {test_name:30s}  {tp_rate:8.3f}  {fp_rate:8.3f}  {acc:8.3f}")

        # ── HC statistic distribution ──
        print("\n  Higher Criticism statistic:")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['hc_stat'].mean():.2f}  median={ts['hc_stat'].median():.2f}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['hc_stat'].mean():.2f}  median={fs['hc_stat'].median():.2f}"
            )

        # ── Cauchy p-value distribution ──
        print("\n  Cauchy p-value distribution:")
        if len(ts) > 0:
            print(
                f"    true_split:  mean={ts['cauchy_p'].mean():.4f}  median={ts['cauchy_p'].median():.4f}"
                f"  <0.01: {(ts['cauchy_p'] < 0.01).sum()}/{len(ts)}"
            )
        if len(fs) > 0:
            print(
                f"    false_split: mean={fs['cauchy_p'].mean():.4f}  median={fs['cauchy_p'].median():.4f}"
                f"  <0.01: {(fs['cauchy_p'] < 0.01).sum()}/{len(fs)}"
            )

    # ── Per-case breakdown for symmetric true splits ──
    sym_ts = sym[sym["gt"] == "true_split"]
    if len(sym_ts) > 0:
        print(f"\n{'='*100}")
        print(f"  SYMMETRIC TRUE SPLITS — detailed breakdown ({len(sym_ts)} pairs)")
        print(f"{'='*100}")

        case_summary = (
            sym_ts.groupby("case")
            .agg(
                true_k=("true_k", "first"),
                n_pairs=("parent", "count"),
                mean_n_total=("n_total", "mean"),
                mean_z_norm_sq=("z_norm_sq", "mean"),
                mean_n_above_4=("n_above_4", "mean"),
                mean_snr=("signal_noise_ratio", "mean"),
                wald_detect=("split_proj_k5", "mean"),
                bh_detect=("split_bh", "mean"),
                cauchy_detect=("split_cauchy", "mean"),
                gumbel_detect=("split_gumbel", "mean"),
            )
            .sort_values("case")
        )
        print(case_summary.round(3).to_string())

        # Show individual pairs for small cases
        small_cases = sym_ts.groupby("case").size()
        for case_name in small_cases[small_cases <= 20].index:
            pairs = sym_ts[sym_ts["case"] == case_name]
            print(f"\n  {case_name} — {len(pairs)} symmetric true_split pairs:")
            cols = [
                "parent",
                "n_total",
                "n_leaves_L",
                "n_leaves_R",
                "z_norm_sq",
                "n_above_4",
                "signal_noise_ratio",
                "p_proj_k5",
                "cauchy_p",
                "gumbel_p",
                "n_bh_significant",
            ]
            print(pairs[cols].to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    cases = FAILURE_CASES + REGRESSION_GUARD_CASES
    all_frames: List[pd.DataFrame] = []

    for i, name in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {name}", end=" ", flush=True)
        try:
            df = diagnose_case(name)
            all_frames.append(df)
            n_sym_ts = ((df["is_symmetric"]) & (df["gt"] == "true_split")).sum()
            print(f"→ {len(df)} labeled pairs ({n_sym_ts} sym true_split)")
        except Exception as e:
            print(f"→ ERROR: {e}")

    if not all_frames:
        print("No results.")
        return

    all_results = pd.concat(all_frames, ignore_index=True)
    summarize(all_results)


if __name__ == "__main__":
    print("=" * 100)
    print("  EXPERIMENT 11: Symmetric Pair Power Dissection")
    print("=" * 100)
    main()
