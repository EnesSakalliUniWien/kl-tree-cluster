#!/usr/bin/env python3
"""
Generic gate-level diagnosis for any benchmark case.

Usage:
    python debug_scripts/diagnose_case.py <case_name> [case_name2 ...]

Example:
    python debug_scripts/diagnose_case.py gauss_overlap_3c_small overlap_heavy_4c_small_feat
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def diagnose_case(case_name: str) -> None:
    all_cases = get_default_test_cases()
    tc = next((c for c in all_cases if c["name"] == case_name), None)
    if tc is None:
        print(f"ERROR: Case '{case_name}' not found.")
        print("Available:", [c["name"] for c in all_cases][:20], "...")
        return

    print("=" * 72)
    print(f"  CASE: {case_name}")
    print("=" * 72)

    print("\nCase config:")
    for k, v in tc.items():
        if k != "generator":
            print(f"  {k}: {v}")

    # ── Generate data ──
    data_t, y_t, x_original, meta = generate_case_data(tc)
    n, p = data_t.shape
    true_k = tc.get("n_clusters", None)
    print(f"\nData: {n} samples × {p} features")
    if y_t is not None:
        label_counts = dict(zip(*np.unique(y_t, return_counts=True)))
        print(f"True labels (K={len(label_counts)}): {label_counts}")
    print(f"Sparsity: {1 - data_t.values.mean():.3f}")
    print(
        f"Column variance: min={data_t.values.var(0).min():.4f}, "
        f"max={data_t.values.var(0).max():.4f}, "
        f"mean={data_t.values.var(0).mean():.4f}"
    )
    n_const = (data_t.values.var(0) == 0).sum()
    print(f"Constant columns: {n_const}/{p}")

    # ── Cluster separability after binarization ──
    if y_t is not None:
        print("\n--- Cluster means (per-feature average of binary values) ---")
        cluster_means = []
        for label in sorted(np.unique(y_t)):
            mask = y_t == label
            cm = data_t.values[mask].mean(axis=0)
            cluster_means.append(cm)
            print(
                f"  Cluster {int(label)}: n={mask.sum()}, "
                f"col_mean range=[{cm.min():.3f}, {cm.max():.3f}], "
                f"overall θ̄={cm.mean():.3f}"
            )

        # Inter-cluster distance
        cluster_means_arr = np.array(cluster_means)
        if len(cluster_means_arr) > 1:
            from scipy.spatial.distance import pdist as cdist

            inter_dists = cdist(cluster_means_arr, metric="hamming")
            print(
                f"  Inter-cluster Hamming: min={inter_dists.min():.3f}, "
                f"max={inter_dists.max():.3f}, mean={inter_dists.mean():.3f}"
            )

    # ── Build tree ──
    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())

    # ── Spectral dimensions (before decompose) ──
    print(
        f"\nConfig: SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
        f"SPECTRAL_MIN_K={config.SPECTRAL_MIN_K}, "
        f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}"
    )

    # ── Run decomposition ──
    decomp = tree.decompose(
        leaf_data=data_t,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    stats = tree.stats_df
    K_found = decomp["num_clusters"]

    # Compute ARI
    ari_str = "N/A"
    if y_t is not None and true_k is not None:
        # Build predicted labels from cluster assignments
        y_pred = np.full(n, -1, dtype=int)
        for cid, cinfo in decomp["cluster_assignments"].items():
            for leaf in cinfo["leaves"]:
                idx = data_t.index.get_loc(leaf)
                y_pred[idx] = cid
        ari = adjusted_rand_score(y_t, y_pred)
        ari_str = f"{ari:.3f}"

    print(f"\n>>> K_found={K_found} (true={true_k}), ARI={ari_str}")

    # ── Root analysis ──
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children = list(tree.successors(root))
    print(f"\nRoot: {root}, children: {children}")

    # ── Gate 2 summary ──
    print("\n--- Gate 2 (child-parent divergence) at root ---")
    for c in children:
        sig = stats.loc[c, "Child_Parent_Divergence_Significant"]
        pval_bh = stats.loc[c, "Child_Parent_Divergence_P_Value_BH"]
        edge_df = (
            stats.loc[c, "Child_Parent_Divergence_df"]
            if "Child_Parent_Divergence_df" in stats.columns
            else "?"
        )
        print(f"  {c}: sig={sig}, p_BH={pval_bh:.2e}, df={edge_df}")

    # ── Gate 3 at root ──
    print("\n--- Gate 3 (sibling divergence) at root ---")
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
            print(f"  {col}: {stats.loc[root, col]}")

    # ── All internal nodes: Gate 2 + Gate 3 overview ──
    internal = [nd for nd in tree.nodes() if tree.out_degree(nd) > 0]
    print(f"\n--- All {len(internal)} internal nodes ---")

    gate2_pass = 0
    gate3_tested = 0
    gate3_pass = 0
    split_nodes = []

    for nd in sorted(internal):
        ch = list(tree.successors(nd))
        if len(ch) != 2:
            continue
        l_sig = bool(stats.loc[ch[0], "Child_Parent_Divergence_Significant"])
        r_sig = bool(stats.loc[ch[1], "Child_Parent_Divergence_Significant"])
        g2 = l_sig or r_sig
        if g2:
            gate2_pass += 1
        skipped = bool(stats.loc[nd, "Sibling_Divergence_Skipped"])
        different = bool(stats.loc[nd, "Sibling_BH_Different"])
        g3 = different and not skipped
        if not skipped:
            gate3_tested += 1
        if g3:
            gate3_pass += 1
        if g2 and g3:
            split_nodes.append(nd)

    print(f"  Gate 2 pass: {gate2_pass}/{len(internal)}")
    print(f"  Gate 3 tested: {gate3_tested}/{len(internal)}")
    print(f"  Gate 3 pass: {gate3_pass}/{len(internal)}")
    print(f"  Both gates pass (split candidates): {len(split_nodes)}/{len(internal)}")

    # ── Spectral dimensions distribution ──
    spectral_dims = stats.attrs.get("_spectral_dims", {})
    if spectral_dims:
        dims_internal = [v for k, v in spectral_dims.items() if tree.out_degree(k) > 0]
        if dims_internal:
            print("\n--- Spectral dimensions (Gate 2) ---")
            print(
                f"  min={min(dims_internal)}, max={max(dims_internal)}, "
                f"median={np.median(dims_internal):.0f}, mean={np.mean(dims_internal):.1f}"
            )

    # ── Top split nodes (where the DFS actually splits) ──
    if split_nodes:
        print("\n--- Top 10 split-candidate nodes (both gates pass) ---")
        # Sort by tree depth (shallowest first)
        from networkx import shortest_path_length

        depths = {nd: shortest_path_length(tree, root, nd) for nd in split_nodes}
        sorted_splits = sorted(split_nodes, key=lambda x: depths[x])
        for nd in sorted_splits[:10]:
            ch = list(tree.successors(nd))
            n_desc_l = len(list(tree.successors(ch[0]))) if tree.out_degree(ch[0]) > 0 else 0
            n_desc_r = len(list(tree.successors(ch[1]))) if tree.out_degree(ch[1]) > 0 else 0
            leaf_count = int(stats.loc[nd, "leaf_count"]) if "leaf_count" in stats.columns else "?"
            sib_p = (
                stats.loc[nd, "Sibling_Divergence_P_Value_Corrected"]
                if "Sibling_Divergence_P_Value_Corrected" in stats.columns
                else "?"
            )
            sib_stat = (
                stats.loc[nd, "Sibling_Test_Statistic"]
                if "Sibling_Test_Statistic" in stats.columns
                else "?"
            )
            sib_df = (
                stats.loc[nd, "Sibling_Degrees_of_Freedom"]
                if "Sibling_Degrees_of_Freedom" in stats.columns
                else "?"
            )
            print(
                (
                    f"  {nd} (depth={depths[nd]}, leaves={leaf_count}): " f"sib_stat={sib_stat:.1f}"
                    if isinstance(sib_stat, float)
                    else f"  {nd} (depth={depths[nd]}, leaves={leaf_count}): sib_stat={sib_stat}"
                ),
                end="",
            )
            if isinstance(sib_df, (int, float)):
                print(f", df={sib_df:.0f}", end="")
            if isinstance(sib_p, float):
                print(f", p_corr={sib_p:.2e}", end="")
            print()

    # ── Top 5 nodes by lowest corrected sibling p-value ──
    if "Sibling_Divergence_P_Value_Corrected" in stats.columns:
        sib_data = stats.loc[internal].copy()
        tested_df = sib_data[~sib_data["Sibling_Divergence_Skipped"]].copy()
        if len(tested_df) > 0:
            tested_df = tested_df.sort_values("Sibling_Divergence_P_Value_Corrected")
            print("\n--- Top 5 nodes by lowest corrected sibling p-value ---")
            for i, (node, row) in enumerate(tested_df.head(5).iterrows()):
                ch = list(tree.successors(node))
                l_sig = bool(stats.loc[ch[0], "Child_Parent_Divergence_Significant"])
                r_sig = bool(stats.loc[ch[1], "Child_Parent_Divergence_Significant"])
                g2_str = "G2✓" if (l_sig or r_sig) else "G2✗"
                print(
                    f"  {node}: p_corr={row['Sibling_Divergence_P_Value_Corrected']:.2e}, "
                    f"stat={row['Sibling_Test_Statistic']:.1f}, "
                    f"df={row['Sibling_Degrees_of_Freedom']:.0f}, "
                    f"diff={row['Sibling_BH_Different']}, {g2_str}"
                )

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

    # ── Post-hoc merge audit ──
    merge_audit = decomp.get("posthoc_merge_audit", [])
    if merge_audit:
        merged = [m for m in merge_audit if m.get("action") == "merged"]
        blocked = [m for m in merge_audit if m.get("action") != "merged"]
        print(f"\n--- Post-hoc merge audit: {len(merged)} merged, {len(blocked)} blocked ---")
        for m in merge_audit[:8]:
            print(f"  {m.get('action', '?')}: {m.get('pair', '?')}, p={m.get('p_value', '?')}")
    else:
        print("\n  No post-hoc merges attempted.")

    # ── DFS tree trace (top 3 levels) ──
    print("\n--- DFS decision tree (first 3 levels from root) ---")
    _print_dfs_tree(tree, stats, root, depth=0, max_depth=3)

    print()


def _print_dfs_tree(tree, stats, node, depth, max_depth):
    """Print the DFS decision at each node."""
    if depth > max_depth:
        return
    indent = "  " * depth
    ch = list(tree.successors(node))
    if len(ch) == 0:
        leaf_count = int(stats.loc[node, "leaf_count"]) if "leaf_count" in stats.columns else 1
        print(f"{indent}🍃 {node} (leaf, n={leaf_count})")
        return
    if len(ch) != 2:
        print(f"{indent}⚠ {node} (non-binary, {len(ch)} children)")
        return

    leaf_count = int(stats.loc[node, "leaf_count"]) if "leaf_count" in stats.columns else "?"
    l_sig = bool(stats.loc[ch[0], "Child_Parent_Divergence_Significant"])
    r_sig = bool(stats.loc[ch[1], "Child_Parent_Divergence_Significant"])
    g2 = l_sig or r_sig
    skipped = bool(stats.loc[node, "Sibling_Divergence_Skipped"])
    different = bool(stats.loc[node, "Sibling_BH_Different"])
    g3 = different and not skipped

    if g2 and g3:
        decision = "SPLIT"
        marker = "✂"
    elif not g2:
        decision = "MERGE (Gate 2 fail)"
        marker = "🔴"
    else:
        decision = "MERGE (Gate 3 fail)"
        marker = "🟡"

    print(
        f"{indent}{marker} {node} (n={leaf_count}): {decision}  "
        f"[edge L={l_sig}, R={r_sig}; sib={'DIFF' if different else 'SAME'}{' SKIP' if skipped else ''}]"
    )

    for c in ch:
        _print_dfs_tree(tree, stats, c, depth + 1, max_depth)


if __name__ == "__main__":
    cases = sys.argv[1:] if len(sys.argv) > 1 else ["gauss_clear_small"]
    for case_name in cases:
        diagnose_case(case_name)
