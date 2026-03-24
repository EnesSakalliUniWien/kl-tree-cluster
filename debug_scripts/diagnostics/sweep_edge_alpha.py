#!/usr/bin/env python
"""Sweep edge α: find the threshold where Gate 2 passes real splits but blocks noise.

For each test case (null + real), runs the full pipeline at various edge α values
(with EDGE_CALIBRATION=False) and records:
  - K found
  - Number of edges passing Gate 2
  - ARI (if true labels available)

This tells us: at what α does Gate 2 become permissive enough for real data
while still providing some filtering on null data?
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
from kl_clustering_analysis.tree.poset_tree import PosetTree

try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    adjusted_rand_score = None


# ─────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────


def _null_data(n=100, p=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    labels = np.zeros(n, dtype=int)
    return (
        pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]),
        labels,
    )


def _block_data(n_per=50, k=4, p=100, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    labels = np.repeat(np.arange(k), n_per)
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return (
        pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]),
        labels,
    )


def _moderate_data(n_per=30, k=3, p=50, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    labels = np.repeat(np.arange(k), n_per)
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return (
        pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]),
        labels,
    )


def _weak_data(n_per=25, k=2, p=40, noise=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    labels = np.repeat(np.arange(k), n_per)
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return (
        pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]),
        labels,
    )


# ─────────────────────────────────────────────────────────────────────
# Run pipeline at given edge α
# ─────────────────────────────────────────────────────────────────────


def _run_at_alpha(data, edge_alpha, sibling_alpha=0.05):
    """Run pipeline with specified edge alpha (no edge calibration)."""
    orig_cal = config.EDGE_CALIBRATION
    config.EDGE_CALIBRATION = False
    try:
        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(
                leaf_data=data,
                alpha_local=edge_alpha,
                sibling_alpha=sibling_alpha,
            )
        stats = tree.annotations_df

        # Count edges passing Gate 2
        n_sig = 0
        if "Child_Parent_Divergence_Significant" in stats.columns:
            n_sig = int(stats["Child_Parent_Divergence_Significant"].sum())

        # Raw edge p-values (pre-BH)
        raw = stats.attrs.get("_edge_raw_test_data", {})
        raw_pvals = np.asarray(raw.get("p_values", []), dtype=float) if raw else np.array([])

        return {
            "K": decomp.get("num_clusters", -1),
            "n_sig_edges": n_sig,
            "cluster_assignments": decomp.get("cluster_assignments", {}),
            "raw_pvals": raw_pvals,
        }
    finally:
        config.EDGE_CALIBRATION = orig_cal


def _compute_ari(labels_true, cluster_assignments, data):
    if adjusted_rand_score is None:
        return float("nan")
    if not cluster_assignments:
        return 0.0
    pred = np.full(len(labels_true), -1, dtype=int)
    leaf_names = data.index.tolist()
    for cid, info in cluster_assignments.items():
        for leaf in info.get("leaves", []):
            if leaf in leaf_names:
                idx = leaf_names.index(leaf)
                pred[idx] = cid
    if (pred == -1).all():
        return 0.0
    return adjusted_rand_score(labels_true, pred)


def main():
    sibling_alpha = 0.05

    # Edge alpha values to sweep
    alphas = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    cases = [
        ("null_100x50", *_null_data(100, 50, seed=42), 1),
        ("null_200x100", *_null_data(200, 100, seed=99), 1),
        ("block_4c_200x100", *_block_data(50, 4, 100, 0.05, seed=42), 4),
        ("block_4c_100x50", *_block_data(25, 4, 50, 0.05, seed=42), 4),
        ("moderate_3c_90x50", *_moderate_data(30, 3, 50, 0.2, seed=42), 3),
        ("moderate_3c_60x50", *_moderate_data(20, 3, 50, 0.2, seed=99), 3),
        ("weak_2c_50x40", *_weak_data(25, 2, 40, 0.3, seed=42), 2),
        ("weak_2c_30x40", *_weak_data(15, 2, 40, 0.3, seed=99), 2),
    ]

    # First: show raw edge p-value distribution at baseline
    print("═" * 100)
    print("  RAW EDGE P-VALUE DISTRIBUTIONS (before BH, no calibration)")
    print("═" * 100)

    for case_name, data, labels, true_k in cases:
        result = _run_at_alpha(data, edge_alpha=0.05, sibling_alpha=sibling_alpha)
        raw_p = result["raw_pvals"]
        if len(raw_p) > 0:
            valid_p = raw_p[np.isfinite(raw_p)]
            pctiles = np.percentile(valid_p, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            print(f"\n  {case_name} (n={data.shape[0]}, p={data.shape[1]}, K={true_k}):")
            print(f"    n_edges={len(valid_p)}, min={valid_p.min():.2e}, max={valid_p.max():.4f}")
            print(
                f"    Percentiles: P1={pctiles[0]:.2e} P5={pctiles[1]:.2e} P10={pctiles[2]:.2e} "
                f"P25={pctiles[3]:.2e} P50={pctiles[4]:.4f} P75={pctiles[5]:.4f} "
                f"P90={pctiles[6]:.4f} P95={pctiles[7]:.4f} P99={pctiles[8]:.4f}"
            )

    # Then: sweep edge α
    print("\n\n" + "═" * 100)
    print("  EDGE α SWEEP (sibling_α = {:.2f})".format(sibling_alpha))
    print("═" * 100)

    for case_name, data, labels, true_k in cases:
        print(f"\n{'─' * 95}")
        print(f"  {case_name}  (n={data.shape[0]}, p={data.shape[1]}, true K={true_k})")
        print(f"{'─' * 95}")
        print(f"  {'edge_α':>8}  {'K':>4}  {'ARI':>7}  {'Sig edges':>10}  {'Status':>12}")
        print(f"  {'─' * 50}")

        for alpha_e in alphas:
            result = _run_at_alpha(data, edge_alpha=alpha_e, sibling_alpha=sibling_alpha)
            ari = _compute_ari(labels, result["cluster_assignments"], data)
            k = result["K"]
            status = ""
            if true_k == 1:
                status = "✓" if k == 1 else f"FALSE K={k}"
            else:
                if k == true_k:
                    status = "✓ EXACT"
                elif k == 1:
                    status = "COLLAPSED"
                elif k < true_k:
                    status = "under-split"
                else:
                    status = "over-split"

            print(
                f"  {alpha_e:>8.3f}  {k:>4}  {ari:>7.3f}  {result['n_sig_edges']:>10}  {status:>12}"
            )

    # Summary: what α works for ALL cases?
    print("\n\n" + "═" * 100)
    print("  SUMMARY: Best edge α per case")
    print("═" * 100)

    for case_name, data, labels, true_k in cases:
        best_alpha = None
        best_ari = -1
        for alpha_e in alphas:
            result = _run_at_alpha(data, edge_alpha=alpha_e, sibling_alpha=sibling_alpha)
            ari = _compute_ari(labels, result["cluster_assignments"], data)
            k = result["K"]
            if true_k == 1:
                if k == 1 and alpha_e > (best_alpha or 0):
                    best_alpha = alpha_e
                    best_ari = ari
            else:
                if ari > best_ari:
                    best_ari = ari
                    best_alpha = alpha_e

        print(f"  {case_name:30s}  true_K={true_k}  best_α={best_alpha}  ARI={best_ari:.3f}")


if __name__ == "__main__":
    main()
