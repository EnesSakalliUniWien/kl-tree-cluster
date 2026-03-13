#!/usr/bin/env python
"""Investigate alternative null-identification strategies for edge calibration.

Runs several test cases (null + well-separated + moderate) WITHOUT edge
calibration, extracts raw edge T/k and spectral_k, then evaluates how
different oracle strategies partition edges into null-like vs signal.

Oracle strategies tested:
  A. spectral_k <= MIN_DIM  (current — broken)
  B. Weighted Wald analog: use raw p-value as continuous weight
  C. Leaf-pair edges only (n_child == 1)
  D. Two-component mixture on T/k
  E. Fraction-based: use bottom quartile of T/k as null
  F. Robust median-based: use median(T/k) for all edges (no oracle needed)

For each strategy, measure fitted ĉ and how many TRUE splits survive
deflation at α=0.01.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ─────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────


def _null_data(n=100, p=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


def _block_data(n_per=50, k=4, p=100, noise=0.05, seed=42):
    """Block-diagonal binary matrix with clean clusters."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


def _moderate_data(n_per=30, k=3, p=50, noise=0.2, seed=42):
    """Moderate difficulty clusters."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    X = rng.binomial(1, 0.5, size=(n, p))
    features_per_block = p // k
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * features_per_block, (c + 1) * features_per_block)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, features_per_block))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


# ─────────────────────────────────────────────────────────────────────
# Run pipeline and extract raw edge data
# ─────────────────────────────────────────────────────────────────────


def _extract_edge_data(data: pd.DataFrame, alpha: float = 0.01) -> dict | None:
    """Run decompose WITHOUT edge calibration and return raw edge data."""
    orig = config.EDGE_CALIBRATION
    config.EDGE_CALIBRATION = False
    try:
        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(leaf_data=data, alpha_local=alpha, sibling_alpha=alpha)

        stats = tree.stats_df
        raw = stats.attrs.get("_edge_raw_test_data", {})
        spectral_dims = stats.attrs.get("_spectral_dims", {})

        if not raw:
            return None

        T = np.asarray(raw["test_stats"], dtype=float)
        df = np.asarray(raw["degrees_of_freedom"], dtype=float)
        p = np.asarray(raw["p_values"], dtype=float)
        plc = np.asarray(raw["parent_leaf_counts"], dtype=float)
        clc = np.asarray(raw["child_leaf_counts"], dtype=float)
        child_ids = raw["child_ids"]
        parent_ids = raw["parent_ids"]

        valid = np.isfinite(T) & (df > 0)
        n_edges = len(T)

        # Spectral k per edge (parent's)
        spec_k = np.array([spectral_dims.get(pid, -1) for pid in parent_ids])

        # Edge significance from the full pipeline (post-BH)
        edge_sig = {}
        for nid in stats.index:
            if "Child_Parent_Divergence_Significant" in stats.columns:
                edge_sig[nid] = bool(stats.loc[nid, "Child_Parent_Divergence_Significant"])

        return {
            "T": T,
            "df": df,
            "p": p,
            "plc": plc,
            "clc": clc,
            "valid": valid,
            "spec_k": spec_k,
            "n_edges": n_edges,
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "edge_sig": edge_sig,
            "K_found": decomp.get("num_clusters", "?"),
        }
    finally:
        config.EDGE_CALIBRATION = orig


# ─────────────────────────────────────────────────────────────────────
# Oracle strategies
# ─────────────────────────────────────────────────────────────────────


def oracle_spectral(data: dict) -> np.ndarray:
    """Current: spectral_k <= MIN_DIM."""
    min_dim = getattr(config, "SPECTRAL_MINIMUM_DIMENSION", 2)
    mask = np.zeros(data["n_edges"], dtype=bool)
    for i in range(data["n_edges"]):
        if not data["valid"][i]:
            continue
        if data["spec_k"][i] <= min_dim:
            mask[i] = True
    return mask


def oracle_weighted_all(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Wald analog: every edge contributes, weight = raw p-value.
    Returns (mask=all-valid, weights)."""
    mask = data["valid"].copy()
    weights = data["p"].copy()
    weights[~mask] = 0.0
    return mask, weights


def oracle_leaf_edges(data: dict) -> np.ndarray:
    """Only edges where child is a leaf (n_child == 1)."""
    mask = np.zeros(data["n_edges"], dtype=bool)
    for i in range(data["n_edges"]):
        if data["valid"][i] and data["clc"][i] == 1:
            mask[i] = True
    return mask


def oracle_bottom_quartile(data: dict) -> np.ndarray:
    """Bottom quartile of T/k as null."""
    valid = data["valid"]
    ratios = np.full(data["n_edges"], np.nan)
    ratios[valid] = data["T"][valid] / data["df"][valid]
    q25 = np.nanpercentile(ratios, 25)
    return valid & (ratios <= q25)


def oracle_neither_child_sig(data: dict) -> np.ndarray:
    """Like sibling calibration: edges whose SIBLING pair has neither child sig.
    For edges, this means: for parent P with children L, R —
    if neither L nor R is edge-significant, both edges (P→L, P→R) are null-like."""
    mask = np.zeros(data["n_edges"], dtype=bool)
    edge_sig = data["edge_sig"]
    parent_ids = data["parent_ids"]
    child_ids = data["child_ids"]

    # Group edges by parent
    parent_children: dict[str, list[int]] = {}
    for i in range(data["n_edges"]):
        pid = parent_ids[i]
        parent_children.setdefault(pid, []).append(i)

    for pid, indices in parent_children.items():
        if len(indices) != 2:
            continue
        i0, i1 = indices
        c0, c1 = child_ids[i0], child_ids[i1]
        sig0 = edge_sig.get(c0, False)
        sig1 = edge_sig.get(c1, False)
        if not sig0 and not sig1:
            for i in indices:
                if data["valid"][i]:
                    mask[i] = True
    return mask


# ─────────────────────────────────────────────────────────────────────
# Evaluate an oracle: fit ĉ and measure deflation impact
# ─────────────────────────────────────────────────────────────────────


def evaluate_oracle(
    data: dict,
    null_mask: np.ndarray,
    name: str,
    alpha: float = 0.01,
    weights: np.ndarray | None = None,
) -> dict:
    """Fit ĉ from null_mask edges and measure deflated outcomes."""
    valid = data["valid"]
    T, df = data["T"], data["df"]

    # Fit ĉ
    if weights is not None:
        # Weighted mean: ĉ = Σ (w_i * T_i/k_i) / Σ w_i over valid edges
        w = weights[valid]
        ratios = T[valid] / df[valid]
        c_hat = float(np.average(ratios, weights=w))
    else:
        cal = null_mask & valid
        if cal.sum() == 0:
            c_hat = 1.0
        else:
            ratios_null = T[cal] / df[cal]
            c_hat = float(np.median(ratios_null))

    c_hat = max(c_hat, 1.0)

    # Deflate all valid edges
    T_adj = T.copy()
    p_adj = data["p"].copy()
    for i in range(data["n_edges"]):
        if not valid[i]:
            continue
        T_adj[i] = T[i] / c_hat
        p_adj[i] = float(chi2.sf(T_adj[i], df=df[i]))

    n_sig_raw = int((data["p"][valid] < alpha).sum())
    n_sig_adj = int((p_adj[valid] < alpha).sum())
    n_cal = int(null_mask.sum()) if weights is None else int(valid.sum())

    return {
        "oracle": name,
        "n_cal": n_cal,
        "c_hat": c_hat,
        "n_sig_raw": n_sig_raw,
        "n_sig_adj": n_sig_adj,
        "n_valid": int(valid.sum()),
        "pct_raw": 100 * n_sig_raw / valid.sum() if valid.sum() > 0 else 0,
        "pct_adj": 100 * n_sig_adj / valid.sum() if valid.sum() > 0 else 0,
    }


def main():
    alpha = 0.01

    cases = [
        ("null_100x50", _null_data(100, 50, seed=42), 1),
        ("null_200x100", _null_data(200, 100, seed=99), 1),
        ("block_4c_200x100", _block_data(50, 4, 100, noise=0.05, seed=42), 4),
        ("block_4c_100x50", _block_data(25, 4, 50, noise=0.05, seed=42), 4),
        ("moderate_3c_90x50", _moderate_data(30, 3, 50, noise=0.2, seed=42), 3),
        ("moderate_3c_60x50", _moderate_data(20, 3, 50, noise=0.2, seed=99), 3),
    ]

    for case_name, data, true_k in cases:
        print(f"\n{'═' * 90}")
        print(f"  Case: {case_name}  (n={data.shape[0]}, p={data.shape[1]}, true K={true_k})")
        print(f"{'═' * 90}")

        edge_data = _extract_edge_data(data, alpha)
        if edge_data is None:
            print("  ERROR: No edge data extracted.")
            continue

        valid = edge_data["valid"]
        T, df = edge_data["T"], edge_data["df"]
        ratios = T[valid] / df[valid]

        print(f"  Edges: {edge_data['n_edges']} total, {valid.sum()} valid")
        print(f"  Pipeline K (no edge cal): {edge_data['K_found']}")
        print(
            f"  T/k: mean={ratios.mean():.3f}, median={np.median(ratios):.3f}, "
            f"min={ratios.min():.3f}, max={ratios.max():.3f}"
        )
        print(
            f"  Spectral k dist: {dict(sorted(zip(*np.unique(edge_data['spec_k'][valid], return_counts=True))))}"
        )
        print()

        # Evaluate each oracle
        results = []

        # A: spectral (current)
        mask_a = oracle_spectral(edge_data)
        results.append(evaluate_oracle(edge_data, mask_a, "A:spectral_k<=2", alpha))

        # B: weighted (all edges, p-value weights)
        mask_b, weights_b = oracle_weighted_all(edge_data)
        results.append(
            evaluate_oracle(edge_data, mask_b, "B:weighted_pval", alpha, weights=weights_b)
        )

        # C: leaf edges only
        mask_c = oracle_leaf_edges(edge_data)
        results.append(evaluate_oracle(edge_data, mask_c, "C:leaf_edges", alpha))

        # D: bottom quartile
        mask_d = oracle_bottom_quartile(edge_data)
        results.append(evaluate_oracle(edge_data, mask_d, "D:bottom_Q1", alpha))

        # E: neither-child-significant (circular but informative)
        mask_e = oracle_neither_child_sig(edge_data)
        results.append(evaluate_oracle(edge_data, mask_e, "E:neither_sig", alpha))

        # F: no oracle — just use median of ALL edges
        mask_f = valid.copy()
        results.append(evaluate_oracle(edge_data, mask_f, "F:median_all", alpha))

        # Print comparison table
        print(
            f"  {'Oracle':<25s} {'n_cal':>6} {'ĉ':>8} {'Sig(raw)':>10} {'Sig(adj)':>10} {'%raw':>7} {'%adj':>7}"
        )
        print(f"  {'─' * 73}")
        for r in results:
            print(
                f"  {r['oracle']:<25s} {r['n_cal']:>6} {r['c_hat']:>8.3f} "
                f"{r['n_sig_raw']:>10} {r['n_sig_adj']:>10} "
                f"{r['pct_raw']:>6.1f}% {r['pct_adj']:>6.1f}%"
            )

        # Show what we WANT:
        # - null cases: sig_adj → 0  (no false splits)
        # - real cases: sig_adj close to sig_raw (preserve real splits)
        print()
        if true_k == 1:
            print("  GOAL: Sig(adj) should be ~0 (all false positives)")
        else:
            print("  GOAL: Sig(adj) should preserve real splits (not collapse to 0)")


if __name__ == "__main__":
    main()
