#!/usr/bin/env python3
"""
Debug the local chi-square gate and multiple-testing correction on a synthetic case.

It compares how many edges pass the local gate under different corrections and
under different degrees-of-freedom (df) choices:
 - df_total = number_of_features
 - df_eff   = number of informative features with |p_child - p_parent| > eps

It reports, for each (df_mode, correction) combination, how many child edges are
flagged significant, plus basic distribution stats.

Run:
  python misc/debug_local_gate.py

You can tweak the CASE dict below to match tests/test_cluster_validation.py cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

try:
    from statsmodels.stats.multitest import multipletests  # type: ignore
    HAVE_SM = True
except Exception:
    HAVE_SM = False

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence


CASE = dict(n_samples=40, n_features=40, n_clusters=4, cluster_std=7.5, seed=43)
ALPHA = 0.05
EPS = 0.02  # effective-df epsilon for |p_child - p_parent|


def build_case(case: dict):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=case["n_samples"],
        n_features=case["n_features"],
        centers=case["n_clusters"],
        cluster_std=case["cluster_std"],
        random_state=case["seed"],
    )
    Xb = (X > np.median(X, axis=0)).astype(int)
    df = pd.DataFrame(Xb, index=[f"S{i}" for i in range(case["n_samples"])])
    Z = linkage(pdist(df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
    stats = calculate_hierarchy_kl_divergence(tree, df)
    return tree, stats


def compute_local_pvals(tree, stats: pd.DataFrame, df_mode: str = "total", eps: float = EPS):
    # Collect children
    edges = list(tree.edges())
    children = [v for _, v in edges]
    # leaf_count, KL local
    leaf_cnt = (
        stats.get("leaf_count", pd.Series(index=stats.index, dtype=float))
        .reindex(children)
        .to_numpy()
    )
    kl_local = (
        stats.get("kl_divergence_local", pd.Series(index=stats.index, dtype=float))
        .reindex(children)
        .to_numpy()
    )
    # df values
    if df_mode == "total":
        dof = np.full(len(children), stats.iloc[0]["distribution"].size, dtype=int)
    else:
        # df_eff per child: count of features where |p_child - p_parent| > eps
        dist = stats["distribution"].to_dict()
        dof = np.zeros(len(children), dtype=int)
        for i, (parent, child) in enumerate(edges):
            pc = np.asarray(dist.get(child), float)
            pp = np.asarray(dist.get(parent), float)
            if pc is None or pp is None or pc.size != pp.size:
                dof[i] = 0
            else:
                dof[i] = int(np.sum(np.abs(pc - pp) > float(eps)))

    # chi-square stats and raw p-values
    chi2_stat = 2.0 * leaf_cnt * kl_local
    with np.errstate(divide="ignore", invalid="ignore"):
        pvals = np.ones_like(chi2_stat, dtype=float)
        mask = (dof > 0) & np.isfinite(chi2_stat)
        pvals[mask] = chi2.sf(chi2_stat[mask], df=dof[mask])
    return children, pvals


def apply_correction(pvals: np.ndarray, mode: str = "bh"):
    if mode == "none":
        return pvals <= ALPHA
    if HAVE_SM and mode in {"bh", "fdr_bh"}:
        rej, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        return rej.astype(bool)
    # Fallback: basic BH implementation
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    n = p_sorted.size
    ranks = np.arange(1, n + 1, dtype=float)
    thresh = (ranks / n) * ALPHA
    rej_sorted = p_sorted <= thresh
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    return rej_sorted[inv]


def main():
    tree, stats = build_case(CASE)
    print("Case:", CASE)
    print("Root:", tree.root())
    print("Edges:", tree.number_of_edges())

    modes_df = ["total", "eff"]
    corrections = ["none", "bh"]

    rows = []
    for df_mode in modes_df:
        children, p = compute_local_pvals(tree, stats, df_mode=df_mode)
        for corr in corrections:
            rej = apply_correction(p, mode=corr)
            rows.append({
                "df_mode": df_mode,
                "correction": corr,
                "tested": int(np.isfinite(p).sum()),
                "rejected": int(rej.sum()),
                "frac_rejected": float((rej.sum() / np.isfinite(p).sum()) if np.isfinite(p).sum() else 0.0),
                "p_median": float(np.median(p[np.isfinite(p)])) if np.isfinite(p).any() else 1.0,
            })

    out = pd.DataFrame(rows)
    print("\nLocal chi-square gate summary (child edges):")
    print(out.to_string(index=False))

    # Show a few smallest p-values per mode
    for df_mode in modes_df:
        _, p = compute_local_pvals(tree, stats, df_mode=df_mode)
        idx = np.argsort(p)[:5]
        print(f"\nTop-5 smallest p-values (df_mode={df_mode}):", p[idx])


if __name__ == "__main__":
    main()
