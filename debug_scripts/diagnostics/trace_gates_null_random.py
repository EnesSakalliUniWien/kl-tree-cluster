#!/usr/bin/env python
"""Trace gate decisions on null data with random projection (SPECTRAL_METHOD=None).

Shows exactly which gates fire at each node and why K != 1 on pure noise.
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
warnings.filterwarnings("ignore")

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _generate_null(n=100, p=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    return pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])


def _run(data, spectral_method, label):
    orig_sm = config.SPECTRAL_METHOD
    orig_ec = config.EDGE_CALIBRATION
    config.SPECTRAL_METHOD = spectral_method
    config.EDGE_CALIBRATION = False
    try:
        Z = linkage(
            pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        result = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
    finally:
        config.SPECTRAL_METHOD = orig_sm
        config.EDGE_CALIBRATION = orig_ec

    K = result["num_clusters"]
    stats = tree.annotations_df

    edge_sig = stats["Child_Parent_Divergence_Significant"]
    sib_diff = stats["Sibling_BH_Different"]
    sib_same = stats["Sibling_BH_Same"]
    sib_skip = stats.get("Sibling_Divergence_Skipped", pd.Series(dtype=bool))

    n_edge_sig = int(edge_sig.sum())
    n_diff = int(sib_diff.sum()) if not sib_diff.isna().all() else 0
    n_same = int(sib_same.sum()) if not sib_same.isna().all() else 0
    n_skip = int(sib_skip.sum()) if not sib_skip.isna().all() else 0

    sm_label = spectral_method or "None (random JL)"
    print(f"\n{'═' * 90}")
    print(f"  {label}   SPECTRAL_METHOD = {sm_label}")
    print(f"{'═' * 90}")
    print(f"  K found: {K}")
    print(f"  Edge significant: {n_edge_sig}/{len(stats)}")
    print(f"  Sibling: different={n_diff}, same={n_same}, skipped={n_skip}")

    # Gate trace for internal nodes that passed Gate 2
    split_nodes = []
    merge_nodes_g2 = []
    merge_nodes_g3 = []

    print(
        f"\n  {'node':>8}  {'n_p':>5}  {'G2_L':>5} {'G2_R':>5} {'G2':>6}  {'G3_diff':>7} {'G3_skip':>7} {'G3':>6}  {'SPLIT':>6}"
    )
    print(
        f"  {'─' * 8}  {'─' * 5}  {'─' * 5} {'─' * 5} {'─' * 6}  {'─' * 7} {'─' * 7} {'─' * 6}  {'─' * 6}"
    )

    for node in sorted(tree.nodes):
        children = list(tree.successors(node))
        if len(children) != 2:
            continue

        left, right = children
        l_sig = bool(edge_sig.get(left, False))
        r_sig = bool(edge_sig.get(right, False))
        g2_pass = l_sig or r_sig

        g3_diff = bool(sib_diff.get(node, False))
        g3_skip_val = (
            bool(sib_skip.get(node, False))
            if isinstance(sib_skip, pd.Series) and node in sib_skip.index
            else False
        )

        if g3_skip_val:
            g3_pass = False
        else:
            g3_pass = g3_diff

        final_split = g2_pass and g3_pass
        n_p = tree.nodes[node].get("leaf_count", 0)

        marker = ""
        if final_split:
            split_nodes.append(node)
            marker = " ← SPLIT"
        elif not g2_pass:
            merge_nodes_g2.append(node)
        elif not g3_pass:
            merge_nodes_g3.append(node)

        print(
            f"  {node:>8}  {n_p:>5}  {l_sig!s:>5} {r_sig!s:>5} {'PASS' if g2_pass else 'FAIL':>6}  "
            f"{g3_diff!s:>7} {g3_skip_val!s:>7} {'PASS' if g3_pass else 'FAIL':>6}  "
            f"{'YES' if final_split else 'no':>6}{marker}"
        )

    print("\n  SUMMARY:")
    print(
        f"    Total internal binary nodes: {len(split_nodes) + len(merge_nodes_g2) + len(merge_nodes_g3)}"
    )
    print(f"    SPLIT (G2+G3 pass): {len(split_nodes)}")
    print(f"    MERGE via G2 fail:  {len(merge_nodes_g2)}")
    print(f"    MERGE via G3 fail:  {len(merge_nodes_g3)}")

    # Edge p-value distribution
    if "Child_Parent_Divergence_P_Value_BH" in stats.columns:
        pvals_bh = stats["Child_Parent_Divergence_P_Value_BH"].dropna()
        pvals_raw = stats.get("Child_Parent_Divergence_P_Value", pd.Series(dtype=float)).dropna()
        print(
            f"\n  Edge p-values (BH-corrected): min={pvals_bh.min():.4e}, med={pvals_bh.median():.4e}, max={pvals_bh.max():.4e}"
        )
        if len(pvals_raw) > 0:
            print(
                f"  Edge p-values (raw):          min={pvals_raw.min():.4e}, med={pvals_raw.median():.4e}, max={pvals_raw.max():.4e}"
            )

    # Sibling p-value distribution
    if "Sibling_Divergence_P_Value_Corrected" in stats.columns:
        sib_pvals = stats["Sibling_Divergence_P_Value_Corrected"].dropna()
        sib_raw = stats.get("Sibling_Divergence_P_Value", pd.Series(dtype=float)).dropna()
        if len(sib_pvals) > 0:
            print(
                f"  Sibling p-values (corrected): min={sib_pvals.min():.4e}, med={sib_pvals.median():.4e}, max={sib_pvals.max():.4e}"
            )
        if len(sib_raw) > 0:
            print(
                f"  Sibling p-values (raw):       min={sib_raw.min():.4e}, med={sib_raw.median():.4e}, max={sib_raw.max():.4e}"
            )


def main():
    data = _generate_null(100, 50, seed=42)
    print(f"Null data: n={len(data)}, p={data.shape[1]}, true K=1\n")

    _run(data, "marchenko_pastur", "NULL n=100 p=50")
    _run(data, None, "NULL n=100 p=50")


if __name__ == "__main__":
    main()
