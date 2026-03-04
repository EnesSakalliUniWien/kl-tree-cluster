#!/usr/bin/env python3
"""
Diagnose why gauss_clear_small finds K=1.

The pass-through diagnostic showed:
  - G3yes = 0 (Gate 3 never passes anywhere in the tree)
  - block = 1, blk+s = 0 (one blocked node, no descendant signal)

This script investigates:
  1. What the binarized data looks like (cluster separability after median threshold)
  2. Gate 2 results (edge significance at root)
  3. Gate 3 results (sibling test at root and all internal nodes)
  4. Why Gate 3 fails everywhere
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def main():
    tc = next(c for c in get_default_test_cases() if c["name"] == "gauss_clear_small")
    print("Case config:")
    for k, v in tc.items():
        if k != "generator":
            print(f"  {k}: {v}")

    # ── Generate data ──
    data_t, y_t, x_original, meta = generate_case_data(tc)
    print(f"\nData shape: {data_t.shape}")
    print(f"True labels: {dict(zip(*np.unique(y_t, return_counts=True)))}")
    print(f"Sparsity: {1 - data_t.values.mean():.3f}")

    # ── Cluster separability after binarization ──
    print("\n--- Cluster means (per-feature average of binary values) ---")
    for label in sorted(np.unique(y_t)):
        mask = y_t == label
        cluster_mean = data_t.values[mask].mean(axis=0)
        print(
            f"  Cluster {int(label)}: n={mask.sum()}, "
            f"col_mean range=[{cluster_mean.min():.3f}, {cluster_mean.max():.3f}], "
            f"overall mean={cluster_mean.mean():.3f}"
        )

    # ── Build tree ──
    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())

    # ── Run decomposition (passthrough off to see raw v1 behavior) ──
    decomp = tree.decompose(
        leaf_data=data_t,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
        passthrough=False,
    )
    stats = tree.stats_df
    print(f"\nDecomposition: K={decomp['num_clusters']}")

    # ── Root analysis ──
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children = list(tree.successors(root))
    print(f"\nRoot: {root}, children: {children}")

    # Gate 2: child-parent divergence
    print("\n--- Gate 2 (child-parent divergence) ---")
    for c in children:
        sig = stats.loc[c, "Child_Parent_Divergence_Significant"]
        pval_bh = stats.loc[c, "Child_Parent_Divergence_P_Value_BH"]
        pval_raw = (
            stats.loc[c, "Child_Parent_Divergence_P_Value"]
            if "Child_Parent_Divergence_P_Value" in stats.columns
            else None
        )
        edge_df = (
            stats.loc[c, "Child_Parent_Divergence_df"]
            if "Child_Parent_Divergence_df" in stats.columns
            else None
        )
        pval_raw_str = f"{pval_raw:.6f}" if pval_raw is not None else "N/A"
        print(f"  {c}: sig={sig}, p_BH={pval_bh:.6f}, " f"p_raw={pval_raw_str}, df={edge_df}")

    # Gate 3: sibling divergence
    print("\n--- Gate 3 (sibling divergence at root) ---")
    sib_cols = [
        "Sibling_BH_Different",
        "Sibling_Divergence_P_Value",
        "Sibling_Divergence_P_Value_Corrected",
        "Sibling_Test_Statistic",
        "Sibling_Degrees_of_Freedom",
        "Sibling_Divergence_Skipped",
        "Sibling_Test_Method",
    ]
    for col in sib_cols:
        if col in stats.columns:
            val = stats.loc[root, col]
            print(f"  {col}: {val}")

    # ── All internal nodes: Gate 2 + Gate 3 overview ──
    internal = [n for n in tree.nodes() if tree.out_degree(n) > 0]
    print(f"\n--- All {len(internal)} internal nodes ---")

    gate2_pass_count = 0
    gate3_tested_count = 0
    gate3_pass_count = 0

    for n in sorted(internal):
        ch = list(tree.successors(n))
        if len(ch) != 2:
            continue

        l_sig = stats.loc[ch[0], "Child_Parent_Divergence_Significant"]
        r_sig = stats.loc[ch[1], "Child_Parent_Divergence_Significant"]
        gate2_pass = bool(l_sig or r_sig)
        if gate2_pass:
            gate2_pass_count += 1

        skipped = bool(stats.loc[n, "Sibling_Divergence_Skipped"])
        different = bool(stats.loc[n, "Sibling_BH_Different"])
        gate3_pass = different and not skipped
        if not skipped:
            gate3_tested_count += 1
        if gate3_pass:
            gate3_pass_count += 1

    print(f"  Gate 2 pass: {gate2_pass_count}/{len(internal)}")
    print(f"  Gate 3 tested (not skipped): {gate3_tested_count}/{len(internal)}")
    print(f"  Gate 3 pass: {gate3_pass_count}/{len(internal)}")

    # ── Show the top-5 nodes with lowest sibling p-value ──
    if "Sibling_Divergence_P_Value_Corrected" in stats.columns:
        sib_data = stats.loc[internal].copy()
        tested = sib_data[~sib_data["Sibling_Divergence_Skipped"]].copy()
        if len(tested) > 0:
            tested = tested.sort_values("Sibling_Divergence_P_Value_Corrected")
            print("\n--- Top 5 nodes by lowest corrected sibling p-value ---")
            for i, (node, row) in enumerate(tested.head(5).iterrows()):
                print(
                    f"  {node}: p_corr={row['Sibling_Divergence_P_Value_Corrected']:.6f}, "
                    f"stat={row['Sibling_Test_Statistic']:.2f}, "
                    f"df={row['Sibling_Degrees_of_Freedom']:.0f}, "
                    f"different={row['Sibling_BH_Different']}"
                )
        else:
            print("\n  No sibling tests were run (all skipped).")

    # ── Calibration audit ──
    audit = stats.attrs.get("sibling_divergence_audit", {})
    if audit:
        print("\n--- Sibling calibration audit ---")
        for k, v in audit.items():
            if k == "diagnostics" and isinstance(v, dict):
                print(f"  {k}:")
                for dk, dv in v.items():
                    print(f"    {dk}: {dv}")
            else:
                print(f"  {k}: {v}")
    else:
        print("\n  No sibling calibration audit found.")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    if gate2_pass_count == 0:
        print("Gate 2 fails everywhere — no edge signal detected.")
        print("This means the binarized data has no detectable parent-child divergence.")
    elif gate3_pass_count == 0:
        print("Gate 2 passes but Gate 3 never passes — sibling test kills all splits.")
        if gate3_tested_count == 0:
            print("All sibling tests were SKIPPED (not enough signal for testing).")
        else:
            print(f"{gate3_tested_count} sibling tests ran, all declared 'same'.")
    else:
        print(f"Gate 3 passes at {gate3_pass_count} nodes but DFS doesn't reach them.")


if __name__ == "__main__":
    main()
