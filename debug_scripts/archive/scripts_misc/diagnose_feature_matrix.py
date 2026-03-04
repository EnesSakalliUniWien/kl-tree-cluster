#!/usr/bin/env python3
"""Diagnose why KL decomposition produces K=1 on feature_matrix.tsv.

Runs the full pipeline, then inspects the stats_df (per-node gate decisions)
to show exactly which statistical gate blocked splitting at every internal
node.  Also produces a UMAP plot colored by the (trivial) cluster labels
and saves diagnostic tables.

Usage
-----
    python scripts/diagnose_feature_matrix.py
    python scripts/diagnose_feature_matrix.py --alpha 0.001
    python scripts/diagnose_feature_matrix.py --alpha 0.05 --input feature_matrix.tsv
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.io import tree_from_linkage
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import build_sample_cluster_assignments


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("feature_matrix.tsv"))
    p.add_argument("--alpha", type=float, default=0.001, help="Significance level for both gates (default 0.001)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-seed", type=int, default=42)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _scalar_notna(val) -> bool:
    """Return True if *val* is a non-null scalar (safe for arrays)."""
    if val is None:
        return False
    try:
        return bool(pd.notna(val))
    except (ValueError, TypeError):
        return False


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _load_binary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    num = df.apply(pd.to_numeric, errors="raise").astype(int)
    vals = set(np.unique(num.values))
    if not vals <= {0, 1}:
        raise ValueError(f"Non-binary values found: {vals - {0,1}}")
    return num


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    alpha = args.alpha

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    out = args.output_dir or Path("benchmarks/results") / f"diagnose_fm_{_ts()}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data_df = _load_binary(args.input)
    n, p = data_df.shape
    print(f"Loaded {args.input}  →  {n} samples × {p} features")

    # 2. Build tree + decompose
    dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=alpha,
        sibling_alpha=alpha,
    )
    num_k = decomp.get("num_clusters", -1)
    print(f"Decomposition K = {num_k}  (alpha_local={alpha}, sibling_alpha={alpha})")

    # 3. stats_df — gate decisions for EVERY internal node
    sdf = tree.stats_df
    if sdf is None:
        print("ERROR: stats_df is None after decomposition — cannot diagnose.")
        return

    # Identify internal (non-leaf) nodes
    all_nodes = list(sdf.index)
    leaf_set = {n for n in tree.nodes() if tree.out_degree(n) == 0}
    internal = [n for n in all_nodes if n not in leaf_set]

    # Key columns we want (actual stats_df column names)
    gate_cols = [
        "leaf_count",
        # Gate 2 — edge (child-parent)
        "Child_Parent_Divergence_P_Value",
        "Child_Parent_Divergence_P_Value_BH",
        "Child_Parent_Divergence_Significant",
        "Child_Parent_Divergence_df",
        "Child_Parent_Divergence_Invalid",
        # Gate 3 — sibling
        "Sibling_Test_Statistic",
        "Sibling_Degrees_of_Freedom",
        "Sibling_Divergence_P_Value",
        "Sibling_Divergence_P_Value_Corrected",
        "Sibling_Divergence_Invalid",
        "Sibling_BH_Different",
        "Sibling_BH_Same",
        "Sibling_Divergence_Skipped",
        "Sibling_Test_Method",
    ]
    present = [c for c in gate_cols if c in sdf.columns]
    diag = sdf.loc[internal, present].copy()

    # Sort by leaf_count descending so root is first
    if "leaf_count" in diag.columns:
        diag = diag.sort_values("leaf_count", ascending=False)

    # Save full diagnostic table
    diag_csv = out / "gate_diagnostics.csv"
    diag.to_csv(diag_csv)
    print(f"\nFull gate diagnostics ({len(diag)} internal nodes) → {diag_csv}")

    # 4. Console summary — top 20 internal nodes (largest subtrees)
    show = diag.head(20)
    print(f"\n{'─'*100}")
    print("Top-20 internal nodes (largest subtrees first):")
    print(f"{'─'*100}")
    for node in show.index:
        row = show.loc[node]
        lc = int(row.get("leaf_count", -1)) if _scalar_notna(row.get("leaf_count")) else -1

        # Gate 2
        edge_sig = row.get("Child_Parent_Divergence_Significant")
        edge_pval_bh = row.get("Child_Parent_Divergence_P_Value_BH")
        edge_pval_raw = row.get("Child_Parent_Divergence_P_Value")
        edge_pval = edge_pval_bh if _scalar_notna(edge_pval_bh) else edge_pval_raw

        # Gate 3
        sib_diff = row.get("Sibling_BH_Different")
        sib_same = row.get("Sibling_BH_Same")
        sib_skip = row.get("Sibling_Divergence_Skipped")
        sib_pval_corr = row.get("Sibling_Divergence_P_Value_Corrected")
        sib_pval_raw = row.get("Sibling_Divergence_P_Value")
        sib_pval = sib_pval_corr if _scalar_notna(sib_pval_corr) else sib_pval_raw
        sib_method = row.get("Sibling_Test_Method", "?")

        # Safe bool conversions
        gate2_pass = bool(edge_sig) if _scalar_notna(edge_sig) else None
        gate3_diff = bool(sib_diff) if _scalar_notna(sib_diff) else None
        gate3_same = bool(sib_same) if _scalar_notna(sib_same) else None
        gate3_skip = bool(sib_skip) if _scalar_notna(sib_skip) else None

        # Verdict
        if gate2_pass is False:
            verdict = "MERGE (Gate 2: edge not significant)"
        elif gate3_skip:
            verdict = "MERGE (Gate 3: sibling test skipped)"
        elif gate3_same:
            verdict = "MERGE (Gate 3: siblings same)"
        elif gate3_diff:
            verdict = "SPLIT"
        else:
            verdict = "MERGE (unknown / missing annotation)"

        edge_p_str = f"{edge_pval:.4g}" if _scalar_notna(edge_pval) else "NA"
        sib_p_str = f"{sib_pval:.4g}" if _scalar_notna(sib_pval) else "NA"

        print(
            f"  {node:>8s}  leaves={lc:4d}  "
            f"| G2: edge_sig={gate2_pass!s:>5s} p={edge_p_str:>10s}  "
            f"| G3: diff={gate3_diff!s:>5s} same={gate3_same!s:>5s} skip={gate3_skip!s:>5s} p={sib_p_str:>10s} method={sib_method}  "
            f"| {verdict}"
        )

    # 5. Summary stats
    if "Child_Parent_Divergence_Significant" in diag.columns:
        n_edge_sig = diag["Child_Parent_Divergence_Significant"].sum()
        print(f"\nGate 2 (edge significant):  {n_edge_sig} / {len(diag)} internal nodes")
    if "Sibling_BH_Different" in diag.columns:
        n_sib_diff = diag["Sibling_BH_Different"].sum()
        print(f"Gate 3 (sibling different): {n_sib_diff} / {len(diag)} internal nodes")
    if "Sibling_BH_Same" in diag.columns:
        n_sib_same = diag["Sibling_BH_Same"].sum()
        print(f"Gate 3 (sibling same):      {n_sib_same} / {len(diag)} internal nodes")
    if "Sibling_Divergence_Skipped" in diag.columns:
        n_sib_skip = diag["Sibling_Divergence_Skipped"].sum()
        print(f"Gate 3 (sibling skipped):   {n_sib_skip} / {len(diag)} internal nodes")

    # 6. Root-node deep dive
    root = tree.root()
    print(f"\n{'─'*100}")
    print(f"ROOT NODE: {root}")
    print(f"{'─'*100}")
    children = list(tree.successors(root))
    print(f"  children: {children}")
    for ch in children:
        leaves_under = tree.get_leaves(ch)
        print(f"  {ch}: {len(leaves_under)} leaves underneath")

    if root in sdf.index:
        root_row = sdf.loc[root]
        # Only print scalar columns (skip array-valued ones like 'distribution')
        for col in present:
            val = root_row.get(col)
            if _scalar_notna(val):
                print(f"    {col}: {val}")

    # 7. Data-level diagnostics
    print(f"\n{'─'*100}")
    print("Data-level diagnostics")
    print(f"{'─'*100}")
    col_means = data_df.mean(axis=0)
    print(f"  Feature sparsity (mean of column means): {col_means.mean():.4f}")
    print(f"  Min column mean: {col_means.min():.4f},  Max column mean: {col_means.max():.4f}")
    print(f"  #features with mean in (0.4, 0.6): {((col_means > 0.4) & (col_means < 0.6)).sum()}")
    print(f"  #features with mean < 0.05 or > 0.95: {((col_means < 0.05) | (col_means > 0.95)).sum()}")

    row_sums = data_df.sum(axis=1)
    print(f"  Row sums — min: {row_sums.min()}, max: {row_sums.max()}, mean: {row_sums.mean():.1f}")

    hamm_dists = pdist(data_df.values, metric="hamming")
    print(f"  Pairwise Hamming distances — min: {hamm_dists.min():.4f}, "
          f"max: {hamm_dists.max():.4f}, mean: {hamm_dists.mean():.4f}, std: {hamm_dists.std():.4f}")

    # 8. Calibration / inflation audit (cousin-adjusted wald)
    if hasattr(sdf, "attrs"):
        audit = sdf.attrs.get("sibling_divergence_audit")
        if audit:
            print(f"\n{'─'*100}")
            print("Sibling divergence calibration audit")
            print(f"{'─'*100}")
            for k, v in audit.items():
                print(f"  {k}: {v}")

    # 9. UMAP plot
    print(f"\n{'─'*100}")
    print("Generating UMAP plot …")
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.umap_seed,
        )
        emb = reducer.fit_transform(data_df.values)

        assignments = build_sample_cluster_assignments(decomp)
        labels = assignments.loc[data_df.index, "cluster_id"].to_numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=30, alpha=0.85)
        ax.set_title(f"UMAP — KL clusters (K={num_k}, α={alpha})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(scatter, ax=ax, label="cluster_id")
        fig.tight_layout()
        umap_path = out / "umap_clusters.png"
        fig.savefig(umap_path, dpi=180)
        plt.close(fig)
        print(f"  saved → {umap_path}")
    except ImportError:
        print("  umap-learn not installed — skipping UMAP plot")

    # 10. Final summary JSON
    summary = {
        "input": str(args.input),
        "n_samples": n,
        "n_features": p,
        "alpha": alpha,
        "num_clusters": num_k,
        "sibling_test_method": config.SIBLING_TEST_METHOD,
        "distance_metric": config.TREE_DISTANCE_METRIC,
        "linkage_method": config.TREE_LINKAGE_METHOD,
        "n_internal_nodes": len(internal),
        "gate2_edge_significant_count": int(diag["Child_Parent_Divergence_Significant"].sum()) if "Child_Parent_Divergence_Significant" in diag.columns else 0,
        "gate3_sibling_different_count": int(diag["Sibling_BH_Different"].sum()) if "Sibling_BH_Different" in diag.columns else 0,
        "gate3_sibling_same_count": int(diag["Sibling_BH_Same"].sum()) if "Sibling_BH_Same" in diag.columns else 0,
        "gate3_sibling_skipped_count": int(diag["Sibling_Divergence_Skipped"].sum()) if "Sibling_Divergence_Skipped" in diag.columns else 0,
        "hamming_dist_mean": float(hamm_dists.mean()),
        "hamming_dist_std": float(hamm_dists.std()),
        "feature_mean_of_means": float(col_means.mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll outputs → {out}/")
    print("Done.")


def _is_descendant(tree, root: str, node: str) -> bool:
    """Check if *node* is a descendant of *root* in a directed tree."""
    import networkx as nx

    try:
        nx.shortest_path(tree, source=root, target=node)
        return True
    except nx.NetworkXNoPath:
        return False


if __name__ == "__main__":
    main()
