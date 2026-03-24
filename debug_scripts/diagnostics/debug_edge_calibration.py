#!/usr/bin/env python
"""Debug script: dissect edge calibration on a single null trial.

Generates ONE null dataset and traces every stage of the edge test pipeline:

  1. Spectral dimension assignment (Marchenko-Pastur k per node)
  2. Raw Wald T, df (k), and p — BEFORE calibration
  3. Null-like edge identification (which edges serve as calibration set?)
  4. Calibration model fit (regression β, ĉ, R²)
  5. Deflated T_adj and p_adj — AFTER calibration
  6. BH correction (TreeBH) on calibrated p-values
  7. Final SPLIT/MERGE decisions + resulting K

Prints a detailed per-edge table and a 6-panel diagnostic figure.

Usage
-----
    python debug_scripts/diagnostics/debug_edge_calibration.py [--seed 42] [--n 100] [--p 50]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

OUT_DIR = Path("debug_scripts/diagnostics/results")


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────


def _generate_null(n: int, p: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    return pd.DataFrame(
        X,
        index=[f"S{i}" for i in range(n)],
        columns=[f"F{j}" for j in range(p)],
    )


def _make_tree(data: pd.DataFrame) -> PosetTree:
    dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    return PosetTree.from_linkage(Z, leaf_names=data.index.tolist())


def _run_decompose(data: pd.DataFrame, alpha: float, edge_cal: bool) -> tuple[dict, pd.DataFrame]:
    """Run full pipeline with/without edge calibration.  Returns (decomp_result, annotations_df)."""
    orig = config.EDGE_CALIBRATION
    config.EDGE_CALIBRATION = edge_cal
    try:
        tree = _make_tree(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(leaf_data=data, alpha_local=alpha, sibling_alpha=alpha)
        return decomp, tree.annotations_df
    finally:
        config.EDGE_CALIBRATION = orig


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug edge calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--p", type=int, default=50, help="Number of features")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════════════════")
    print(f"  Edge Calibration Debug — seed={args.seed}, n={args.n}, p={args.p}, α={args.alpha}")
    print(f"  EDGE_CALIBRATION={config.EDGE_CALIBRATION}")
    print(f"  SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print(f"  SPECTRAL_MINIMUM_DIMENSION={config.SPECTRAL_MINIMUM_DIMENSION}")
    print("═══════════════════════════════════════════════════════════════\n")

    data = _generate_null(args.n, args.p, args.seed)

    # ── Run WITH calibration ──────────────────────────────────────────
    decomp_cal, stats_cal = _run_decompose(data, args.alpha, edge_cal=True)
    k_cal = decomp_cal.get("num_clusters", "?")

    # ── Run WITHOUT calibration ───────────────────────────────────────
    decomp_nocal, stats_nocal = _run_decompose(data, args.alpha, edge_cal=False)
    k_nocal = decomp_nocal.get("num_clusters", "?")

    # ── Extract raw test data (stored in attrs by annotate_child_parent_divergence) ──
    raw = stats_cal.attrs.get("_edge_raw_test_data", {})
    audit = stats_cal.attrs.get("edge_calibration_audit", {})
    spectral_dims = stats_cal.attrs.get("_spectral_dims", {})

    if not raw:
        print("ERROR: No raw edge test data found in annotations_df.attrs.")
        print("  Available attrs keys:", list(stats_cal.attrs.keys()))
        return

    child_ids = raw["child_ids"]
    parent_ids = raw["parent_ids"]
    T_raw = np.asarray(raw["test_stats"], dtype=float)
    df_raw = np.asarray(raw["degrees_of_freedom"], dtype=float)
    p_raw = np.asarray(raw["p_values"], dtype=float)
    plc = np.asarray(raw["parent_leaf_counts"], dtype=float)
    clc = np.asarray(raw["child_leaf_counts"], dtype=float)
    n_edges = len(child_ids)

    # invalid mask: non-finite T or degenerate df
    invalid = ~np.isfinite(T_raw) | (df_raw <= 0)
    valid = ~invalid

    # ── Identify null-like edges and fit calibration locally ──────────
    null_mask = _identify_null_like_edges(parent_ids, spectral_dims, invalid)
    model = fit_edge_inflation_model(T_raw, df_raw, plc, null_mask)

    c_hat_per_edge = np.array(
        [predict_edge_inflation(model, float(plc[i])) for i in range(n_edges)]
    )
    T_adj = T_raw.copy()
    p_adj = p_raw.copy()
    for i in range(n_edges):
        if invalid[i]:
            continue
        T_adj[i] = T_raw[i] / c_hat_per_edge[i]
        p_adj[i] = float(chi2.sf(T_adj[i], df=df_raw[i]))

    T_over_k_raw = T_raw[valid] / df_raw[valid]
    T_over_k_adj = T_adj[valid] / df_raw[valid]

    # ══════════════════════════════════════════════════════════════════
    # Print diagnostics
    # ══════════════════════════════════════════════════════════════════

    # Spectral dimension breakdown
    min_dim = config.SPECTRAL_MINIMUM_DIMENSION
    k_vals: list[int] = []
    print("Stage 1: Spectral dimension (Marchenko-Pastur)")
    if spectral_dims:
        kdist: dict[int, int] = {}
        for i, pid in enumerate(parent_ids):
            if invalid[i]:
                continue
            k = spectral_dims.get(pid, -1)
            kdist[k] = kdist.get(k, 0) + 1
            k_vals.append(k)
        print(f"  k distribution: {dict(sorted(kdist.items()))}")
        print(f"  Null-like threshold: k ≤ {min_dim}")
    print()

    # Raw edge test
    print("Stage 2: Raw Wald T (before calibration)")
    print(f"  Edges: {n_edges} total, {valid.sum()} valid")
    print(f"  T/k: mean={T_over_k_raw.mean():.3f}, median={np.median(T_over_k_raw):.3f}")
    print(
        f"  p < {args.alpha}: {(p_raw[valid] < args.alpha).sum()}/{valid.sum()} "
        f"({100 * (p_raw[valid] < args.alpha).mean():.1f}%)"
    )
    print(
        f"  p < 0.05: {(p_raw[valid] < 0.05).sum()}/{valid.sum()} "
        f"({100 * (p_raw[valid] < 0.05).mean():.1f}%)"
    )
    print()

    # Calibration model
    n_null_like = int(null_mask.sum())
    diag = model.diagnostics
    print("Stage 3-4: Calibration model")
    print(f"  Null-like edges: {n_null_like}/{n_edges}")
    print(f"  Model fit: n_cal={model.n_calibration}")
    if model.beta is not None:
        print(f"    β₀={model.beta[0]:.4f}, β₁={model.beta[1]:.4f}")
        print(f"    exp(β₀) = {np.exp(model.beta[0]):.4f}")
    print(f"    Global ĉ (median T/k): {model.global_inflation_factor:.4f}")
    print(f"    Max observed T/k: {model.max_observed_ratio:.4f}")
    if "r_squared" in diag:
        print(f"    R² = {diag['r_squared']:.4f}")
    print(f"    Fit status: {diag.get('fit_status', '?')}")
    print()

    # Deflated
    print("Stage 5: Deflated T (after calibration)")
    print(f"  T/k: mean={T_over_k_adj.mean():.3f}, median={np.median(T_over_k_adj):.3f}")
    print(f"  p < {args.alpha}: {(p_adj[valid] < args.alpha).sum()}/{valid.sum()}")
    print(f"  p < 0.05: {(p_adj[valid] < 0.05).sum()}/{valid.sum()}")
    print()

    # Pipeline comparison
    edge_cal_rows = stats_cal[stats_cal["Child_Parent_Divergence_P_Value"].notna()]
    edge_nocal_rows = stats_nocal[stats_nocal["Child_Parent_Divergence_P_Value"].notna()]
    n_sig_cal = int(edge_cal_rows["Child_Parent_Divergence_Significant"].sum())
    n_sig_nocal = int(edge_nocal_rows["Child_Parent_Divergence_Significant"].sum())

    print("Stage 6-7: End-to-end comparison")
    print(f"  {'':25s} {'Edge rejects':>15} {'K found':>10}")
    print(f"  {'Without calibration':25s} {n_sig_nocal:>15} {k_nocal!s:>10}")
    print(f"  {'With calibration':25s} {n_sig_cal:>15} {k_cal!s:>10}")
    print()

    # Pipeline audit
    if audit:
        print(
            f"  Pipeline audit: n_cal={audit.get('n_calibration')}, "
            f"ĉ={audit.get('global_inflation_factor', 0):.4f}, "
            f"max_ratio={audit.get('max_observed_ratio', 0):.4f}"
        )
        print()

    # ── Per-edge detail table ──────────────────────────────────────────
    bh_lookup: dict[str, dict] = {}
    for nid in edge_cal_rows.index:
        bh_lookup[nid] = {
            "sig": edge_cal_rows.loc[nid, "Child_Parent_Divergence_Significant"],
            "p_bh": edge_cal_rows.loc[nid, "Child_Parent_Divergence_P_Value_BH"],
        }

    print(f"{'─' * 130}")
    print(
        f"{'Edge':<15} {'n_p':>5} {'n_c':>5} {'k_sp':>4} "
        f"{'T_raw':>10} {'T/k_raw':>8} {'p_raw':>10} {'null?':>5} "
        f"{'ĉ_i':>6} {'T_adj':>10} {'T/k_adj':>8} {'p_adj':>10} "
        f"{'BH_sig':>6}"
    )
    print(f"{'─' * 130}")

    max_rows = min(n_edges, 50)
    for i in range(max_rows):
        if invalid[i]:
            continue
        pid, cid = parent_ids[i], child_ids[i]
        k_sp = spectral_dims.get(pid, -1) if spectral_dims else -1
        Tk_r = T_raw[i] / df_raw[i] if df_raw[i] > 0 else float("nan")
        Tk_a = T_adj[i] / df_raw[i] if df_raw[i] > 0 else float("nan")
        bh = bh_lookup.get(cid, {})
        sig = bh.get("sig", "?")
        sig_str = "SPLIT" if sig is True else ("merge" if sig is False else str(sig))

        print(
            f"{pid:>7}→{cid:<7} {int(plc[i]):>5} {int(clc[i]):>5} {k_sp:>4} "
            f"{T_raw[i]:>10.3f} {Tk_r:>8.3f} {p_raw[i]:>10.6f} "
            f"{'YES' if null_mask[i] else '':>5} "
            f"{c_hat_per_edge[i]:>6.3f} {T_adj[i]:>10.3f} {Tk_a:>8.3f} {p_adj[i]:>10.6f} "
            f"{sig_str:>6}"
        )

    print(f"{'─' * 130}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Plot  (6 panels)
    # ═══════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Edge Calibration Debug — seed={args.seed}, n={args.n}, p={args.p}, "
        f"α={args.alpha}  (null data, true K=1)",
        fontsize=13,
        fontweight="bold",
    )

    # ── A: Spectral k histogram ────────────────────────────────────────
    ax = axes[0, 0]
    if k_vals:
        bins = np.arange(min(k_vals) - 0.5, max(k_vals) + 1.5, 1)
        ax.hist(k_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(
            min_dim, color="red", ls="--", lw=1.5, label=f"Null-like threshold (k ≤ {min_dim})"
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Spectral dimension k (per parent)")
    ax.set_ylabel("Count")
    ax.set_title("A. Marchenko-Pastur spectral k")

    # ── B: Raw T/k histogram ──────────────────────────────────────────
    ax = axes[0, 1]
    null_valid = valid & null_mask
    signal_valid = valid & ~null_mask
    if null_valid.any():
        ax.hist(
            T_raw[null_valid] / df_raw[null_valid],
            bins=30,
            density=True,
            alpha=0.5,
            color="gray",
            label=f"Null-like (n={null_valid.sum()})",
        )
    if signal_valid.any():
        ax.hist(
            T_raw[signal_valid] / df_raw[signal_valid],
            bins=30,
            density=True,
            alpha=0.5,
            color="steelblue",
            label=f"Signal-like (n={signal_valid.sum()})",
        )
    k_mode = Counter(k_vals).most_common(1)[0][0] if k_vals else 2
    x_range = np.linspace(0.01, max(5, T_over_k_raw.max() * 1.1), 200)
    ax.plot(
        x_range,
        chi2.pdf(x_range * k_mode, df=k_mode) * k_mode,
        "k--",
        lw=1,
        label=f"χ²({k_mode})/{k_mode} reference",
    )
    ax.axvline(1.0, color="green", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("T/k")
    ax.set_ylabel("Density")
    ax.set_title("B. Raw T/k (before calibration)")
    ax.legend(fontsize=7)

    # ── C: Calibration scatter (T/k vs n_parent for null-like) ────────
    ax = axes[0, 2]
    if null_valid.any():
        null_ratio = T_raw[null_valid] / df_raw[null_valid]
        null_n = plc[null_valid]
        ax.scatter(null_n, null_ratio, s=20, alpha=0.6, color="gray", label="Null-like edges")
        ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.6)
        ax.axhline(
            model.global_inflation_factor,
            color="orange",
            ls="-",
            lw=1.5,
            label=f"Median ĉ = {model.global_inflation_factor:.3f}",
        )
        if model.beta is not None and abs(model.beta[1]) > 1e-8:
            n_range = np.linspace(max(null_n.min(), 2), null_n.max(), 100)
            pred = np.exp(model.beta[0] + model.beta[1] * np.log(n_range))
            ax.plot(
                n_range, pred, "r-", lw=1.5, label=f"Regression (R²={diag.get('r_squared', 0):.3f})"
            )
        ax.legend(fontsize=7)
    ax.set_xlabel("n_parent (leaf count)")
    ax.set_ylabel("T/k (null-like edges)")
    ax.set_title("C. Calibration: T/k vs n_parent")

    # ── D: p-value QQ (raw vs calibrated) ─────────────────────────────
    ax = axes[1, 0]
    p_r_v = p_raw[valid]
    p_a_v = p_adj[valid]
    n_pts = len(p_r_v)
    expected = np.linspace(0, 1, n_pts + 2)[1:-1]
    ax.plot(
        expected,
        np.sort(p_r_v),
        ".",
        color="red",
        alpha=0.4,
        ms=3,
        label=f"Raw (< α: {(p_r_v < args.alpha).sum()}/{n_pts})",
    )
    ax.plot(
        expected,
        np.sort(p_a_v),
        ".",
        color="steelblue",
        alpha=0.4,
        ms=3,
        label=f"Calibrated (< α: {(p_a_v < args.alpha).sum()}/{n_pts})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Expected (Uniform)")
    ax.set_ylabel("Observed p-value")
    ax.set_title("D. QQ plot: raw vs calibrated p-values")
    ax.legend(fontsize=7)

    # ── E: Per-edge ĉ_i vs n_parent ──────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(plc[valid], c_hat_per_edge[valid], s=15, alpha=0.5, c="steelblue")
    ax.axhline(1.0, color="green", ls=":", lw=1)
    ax.axhline(
        model.max_observed_ratio,
        color="red",
        ls="--",
        lw=1,
        label=f"Clamp (max = {model.max_observed_ratio:.3f})",
    )
    ax.set_xlabel("n_parent")
    ax.set_ylabel("Predicted ĉ_i")
    ax.set_title("E. Per-edge predicted inflation ĉ_i")
    ax.legend(fontsize=7)

    # ── F: Deflation scatter (T/k raw → T/k adj) ─────────────────────
    ax = axes[1, 2]
    ax.scatter(T_over_k_raw, T_over_k_adj, s=10, alpha=0.4, c="steelblue")
    lim = max(T_over_k_raw.max(), T_over_k_adj.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="No change")
    ax.axhline(1.0, color="green", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(1.0, color="green", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("T/k (raw)")
    ax.set_ylabel("T/k (calibrated)")
    ax.set_title("F. Deflation: T/k raw → T/k adj")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"edge_calibration_debug_seed{args.seed}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
