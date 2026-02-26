#!/usr/bin/env python3
"""Per-node dimension diagnostic with pooled-within spectral analysis.

For every internal node in the tree, computes:
  1. d_active   — features with variance > 0 among descendants
  2. erank      — effective rank (Shannon entropy of eigenvalue spectrum)
  3. erank_pw   — pooled within-cluster effective rank (binary nodes)
  4. k_MP       — Marchenko-Pastur signal eigenvalue count
  5. k_JL       — JL-based projection dimension
  6. edge_k     — pipeline spectral_k (edge test)
  7. sib_k      — pipeline sibling spectral_k (pooled within)

Analyses:
  - Summary statistics and per-node table
  - Dimension vs n_descendants (binned)
  - Sample nodes (small / medium / large)
  - Ratio analysis (per-node k vs JL)
  - d_active vs n_desc scaling (log-log regression)
  - Sibling test power comparison (different k strategies)
  - End-to-end pipeline K/ARI comparison (synthetic cases only)

Data sources:
  --input FILE     Run on a TSV feature matrix (default: feature_matrix.tsv)
  --synthetic      Run on synthetic benchmark cases (blobs, binary)
  --all            Run on both (default)

Usage:
    python scripts/diagnose_per_node_dimension.py                  # both
    python scripts/diagnose_per_node_dimension.py --input data.tsv # file only
    python scripts/diagnose_per_node_dimension.py --synthetic      # synthetic only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.random_projection import johnson_lindenstrauss_min_dim

sys.path.insert(0, ".")
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral_dimension import (
    compute_sibling_spectral_dimensions,
    compute_spectral_decomposition,
    effective_rank,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# =====================================================================
# Dimension estimators
# =====================================================================


def count_active_features(data_sub: np.ndarray) -> int:
    """Count features with non-zero variance (not all 0 or all 1)."""
    if data_sub.shape[0] <= 1:
        return 0
    col_var = np.var(data_sub, axis=0)
    return int(np.sum(col_var > 0))


def mp_signal_count(eigenvalues: np.ndarray, n: int, d: int) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound."""
    gamma = d / n
    sigma2 = float(np.median(eigenvalues))
    if sigma2 <= 0:
        return 0
    upper = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return int(np.sum(eigenvalues > upper))


def jl_k(n_child: int, d: int, eps: float = 0.3) -> int:
    """Current JL-based dimension for comparison."""
    k = johnson_lindenstrauss_min_dim(n_samples=max(n_child, 1), eps=eps)
    return int(min(max(k, config.PROJECTION_MIN_K), d))


def _compute_erank_from_data(data_sub: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute effective rank via correlation matrix (matching library).

    Returns (erank, eigenvalues).
    """
    n, d = data_sub.shape
    col_var = np.var(data_sub, axis=0)
    d_active = int(np.sum(col_var > 0))
    if d_active < 2:
        return 1.0, np.array([1.0])

    active = data_sub[:, col_var > 0]
    if n < d_active:
        # Dual-form: n×n Gram instead of d×d correlation
        X_std = active - active.mean(axis=0)
        stds = active.std(axis=0, ddof=0)
        stds[stds == 0] = 1.0
        X_std /= stds
        gram = X_std @ X_std.T / d_active
        evals = np.sort(np.linalg.eigvalsh(gram))[::-1]
    else:
        corr = np.corrcoef(active.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        evals = np.sort(np.linalg.eigvalsh(corr))[::-1]

    evals = np.maximum(evals, 0.0)
    return effective_rank(evals), evals


def _compute_pooled_within_erank(
    X: np.ndarray, left_idx: List[int], right_idx: List[int]
) -> Optional[float]:
    """Pooled within-cluster effective rank for binary split."""
    n_l, n_r = len(left_idx), len(right_idx)
    if n_l < 2 or n_r < 2:
        return None
    left_rows = X[left_idx, :]
    right_rows = X[right_idx, :]
    resid_l = left_rows - left_rows.mean(axis=0)
    resid_r = right_rows - right_rows.mean(axis=0)
    pooled = np.vstack([resid_l, resid_r])
    er, _ = _compute_erank_from_data(pooled)
    return er


# =====================================================================
# Data sources
# =====================================================================

SYNTHETIC_CASES = [
    {
        "name": "gauss_clear_small",
        "generator": "blobs",
        "n_samples": 30,
        "n_features": 100,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 42,
    },
    {
        "name": "gauss_clear_med",
        "generator": "blobs",
        "n_samples": 100,
        "n_features": 100,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 42,
    },
    {
        "name": "block_4c",
        "generator": "binary",
        "n_rows": 80,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.3,
        "seed": 42,
    },
    {
        "name": "trivial_2c",
        "generator": "blobs",
        "n_samples": 40,
        "n_features": 50,
        "n_clusters": 2,
        "cluster_std": 1.0,
        "seed": 42,
    },
]


def load_file_dataset(path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray], str]:
    """Load a TSV feature matrix. Returns (data_df, true_labels_or_None, name)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df, None, path.stem


def load_synthetic_dataset(
    case: dict,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], str]:
    """Generate a synthetic benchmark case."""
    from benchmarks.shared.generators import generate_case_data

    data_df, true_labels, _X_raw, _meta = generate_case_data(case)
    return data_df, true_labels, case["name"]


def build_tree(data_df: pd.DataFrame) -> PosetTree:
    """Build a PosetTree from a data DataFrame."""
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)
    return tree


# =====================================================================
# Per-node analysis
# =====================================================================


def compute_per_node_dimensions(
    tree: PosetTree,
    data_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Compute dimension estimates for every internal node.

    Returns (records_df, pipeline_edge_dims, pipeline_sibling_dims).
    """
    import networkx as nx

    X = data_df.values.astype(np.float64)
    n, d = X.shape
    label_to_idx = {label: i for i, label in enumerate(data_df.index)}

    # Pipeline spectral dims
    edge_dims, _, _ = compute_spectral_decomposition(
        tree,
        data_df,
        method="effective_rank",
        min_k=1,
        compute_projections=True,
    )
    sib_dims = compute_sibling_spectral_dimensions(
        tree,
        data_df,
        method="effective_rank",
        min_k=1,
    )

    root = tree.graph.get("root", list(nx.topological_sort(tree))[0])

    # Precompute descendant leaf indices bottom-up
    desc_leaf_idx: Dict[str, List[int]] = {}
    for node_id in reversed(list(nx.topological_sort(tree))):
        if tree._is_leaf(node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_leaf_idx[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
        else:
            indices = []
            for child in tree.successors(node_id):
                indices.extend(desc_leaf_idx.get(child, []))
            desc_leaf_idx[node_id] = indices

    records = []
    internal_nodes = [nid for nid in tree.nodes if not tree._is_leaf(nid)]

    for node_id in internal_nodes:
        row_indices = desc_leaf_idx[node_id]
        n_desc = len(row_indices)
        if n_desc < 2:
            continue

        data_sub = X[row_indices, :]
        children = list(tree.successors(node_id))
        n_children = len(children)

        # 1. Active features
        d_act = count_active_features(data_sub)

        # 2. Effective rank (leaves only, correlation matrix)
        erank_val, eigenvalues = _compute_erank_from_data(data_sub)

        # 3. Marchenko-Pastur signal count
        k_mp = mp_signal_count(eigenvalues, n_desc, d)

        # 4. JL baseline
        k_jl_val = jl_k(n_desc, d)

        # 5. Variance captured thresholds
        total_var = np.sum(eigenvalues)
        cum_var = (
            np.cumsum(eigenvalues) / total_var if total_var > 0 else np.zeros(len(eigenvalues))
        )
        k_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        k_95 = int(np.searchsorted(cum_var, 0.95)) + 1

        # 6. Children eranks
        children_eranks: Dict[str, float] = {}
        for child in children:
            if tree._is_leaf(child):
                children_eranks[child] = 0.0
                continue
            cidx = desc_leaf_idx.get(child, [])
            if len(cidx) < 2:
                children_eranks[child] = 1.0
                continue
            er_c, _ = _compute_erank_from_data(X[cidx, :])
            children_eranks[child] = er_c

        # 7. Pooled within-cluster erank (binary nodes only)
        erank_pw = None
        if n_children == 2:
            l_idx = desc_leaf_idx.get(children[0], [])
            r_idx = desc_leaf_idx.get(children[1], [])
            erank_pw = _compute_pooled_within_erank(X, l_idx, r_idx)

        try:
            depth = nx.shortest_path_length(tree, root, node_id)
        except nx.NetworkXNoPath:
            depth = -1

        records.append(
            {
                "node": node_id,
                "n_desc": n_desc,
                "depth": depth,
                "n_children": n_children,
                "d_active": d_act,
                "erank": round(erank_val, 1),
                "erank_pw": (round(erank_pw, 1) if erank_pw is not None else None),
                "k_MP": k_mp,
                "k_90pct": min(k_90, d),
                "k_95pct": min(k_95, d),
                "k_JL": k_jl_val,
                "edge_k": edge_dims.get(node_id),
                "sib_k": sib_dims.get(node_id),
                "max_child_erank": (
                    round(max(children_eranks.values()), 1) if children_eranks else None
                ),
                "children_eranks": {k: round(v, 1) for k, v in children_eranks.items()},
                "d": d,
                "top_eig_pct": (round(100 * eigenvalues[0] / total_var, 1) if total_var > 0 else 0),
            }
        )

    df = pd.DataFrame(records).sort_values("n_desc")
    return df, edge_dims, sib_dims


# =====================================================================
# Output sections
# =====================================================================


def print_section(title: str, level: int = 1) -> None:
    sep = "=" * 90 if level == 1 else "-" * 90
    print(f"\n{sep}")
    print(f"{'SECTION' if level == 1 else '  '} {title}")
    print(sep)


def section_summary(results: pd.DataFrame) -> None:
    """Section 1: Summary statistics."""
    print_section("1: Per-Node Dimension Estimates — Summary")
    print(f"\nTotal internal nodes analyzed: {len(results)}")
    print(f"\nn_desc range:  {results['n_desc'].min()} – {results['n_desc'].max()}")
    print(f"d_active range: {results['d_active'].min()} – {results['d_active'].max()}")
    print(f"erank range:    {results['erank'].min()} – {results['erank'].max()}")
    pw = results["erank_pw"].dropna()
    if len(pw) > 0:
        print(f"erank_pw range: {pw.min()} – {pw.max()}")
    print(f"k_MP range:     {results['k_MP'].min()} – {results['k_MP'].max()}")

    print("\n--- Descriptive statistics ---")
    for col in [
        "d_active",
        "erank",
        "erank_pw",
        "k_MP",
        "k_JL",
        "edge_k",
        "sib_k",
        "k_90pct",
    ]:
        if col not in results.columns:
            continue
        vals = results[col].dropna()
        if len(vals) == 0:
            continue
        print(
            f"  {col:12s}:  mean={vals.mean():7.1f}  median={vals.median():7.1f}  "
            f"std={vals.std():7.1f}  min={vals.min():5.0f}  max={vals.max():5.0f}"
        )


def section_binned(results: pd.DataFrame) -> None:
    """Section 2: Dimension vs n_descendants (binned)."""
    print_section("2: Dimension vs n_descendants (binned)")

    max_n = results["n_desc"].max()
    bins = [b for b in [2, 5, 10, 20, 50, 100, 200, 400, 700, 1500] if b <= max_n + 1]
    bins.append(max_n + 1)
    results = results.copy()
    results["n_bin"] = pd.cut(results["n_desc"], bins=bins, right=False)

    agg_cols = {
        "node": "count",
        "d_active": ["mean", "median"],
        "erank": ["mean", "median"],
        "k_MP": ["mean", "median"],
        "k_JL": ["mean", "median"],
    }
    if results["sib_k"].notna().any():
        agg_cols["sib_k"] = ["mean", "median"]

    grouped = results.groupby("n_bin", observed=True).agg(agg_cols)

    header = (
        f"{'n_desc bin':>15s}  {'cnt':>4s}  {'d_act(med)':>10s}  {'erank(med)':>10s}  "
        f"{'k_MP(med)':>10s}  {'k_JL(med)':>10s}"
    )
    has_sib = "sib_k" in agg_cols
    if has_sib:
        header += f"  {'sib_k(med)':>10s}"
    print(f"\n{header}")
    print("-" * len(header))

    for interval, row in grouped.iterrows():
        cnt = int(row[("node", "count")])
        line = (
            f"{str(interval):>15s}  {cnt:4d}  "
            f"{row[('d_active', 'median')]:10.0f}  "
            f"{row[('erank', 'median')]:10.1f}  "
            f"{row[('k_MP', 'median')]:10.0f}  "
            f"{row[('k_JL', 'median')]:10.0f}"
        )
        if has_sib:
            line += f"  {row[('sib_k', 'median')]:10.0f}"
        print(line)


def section_sample_nodes(results: pd.DataFrame) -> None:
    """Section 3: Sample nodes (small / medium / large)."""
    print_section("3: Sample Nodes (small / medium / large)")

    quantiles = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    sample_indices = sorted(set(int(q * (len(results) - 1)) for q in quantiles))
    sample_indices = [i for i in sample_indices if 0 <= i < len(results)]
    sample = results.iloc[sample_indices]

    header = (
        f"{'node':>8s}  {'n_desc':>6s}  {'d_active':>8s}  {'erank':>6s}  "
        f"{'PW':>6s}  {'k_MP':>5s}  {'k_90%':>5s}  {'k_JL':>5s}  "
        f"{'edge_k':>6s}  {'sib_k':>6s}  {'top%':>5s}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for _, row in sample.iterrows():
        pw_str = f"{row['erank_pw']:6.1f}" if pd.notna(row.get("erank_pw")) else "   N/A"
        ek_str = f"{int(row['edge_k']):6d}" if pd.notna(row.get("edge_k")) else "   N/A"
        sk_str = f"{int(row['sib_k']):6d}" if pd.notna(row.get("sib_k")) else "   N/A"
        print(
            f"{row['node']:>8s}  {row['n_desc']:6d}  {row['d_active']:8d}  "
            f"{row['erank']:6.1f}  {pw_str}  {row['k_MP']:5d}  {row['k_90pct']:5d}  "
            f"{row['k_JL']:5d}  {ek_str}  {sk_str}  {row['top_eig_pct']:5.1f}"
        )


def section_ratio_analysis(results: pd.DataFrame) -> None:
    """Section 4: Ratio analysis (per-node k vs JL)."""
    print_section("4: Ratio Analysis (per-node k vs JL)")

    for col, label in [
        ("d_active", "d_active / k_JL"),
        ("erank", "erank / k_JL"),
        ("k_MP", "k_MP / k_JL"),
        ("sib_k", "sib_k / k_JL"),
    ]:
        valid = results[results[col].notna() & (results["k_JL"] > 0)]
        if len(valid) == 0:
            continue
        r = valid[col] / valid["k_JL"]
        print(
            f"\n--- {label} ---\n"
            f"  mean={r.mean():.3f}  median={r.median():.3f}  "
            f"min={r.min():.3f}  max={r.max():.3f}"
        )


def section_scaling_regression(results: pd.DataFrame) -> None:
    """Section 5: d_active / erank vs n_desc scaling."""
    print_section("5: d_active / erank vs n_desc Scaling")

    valid = results[results["d_active"] > 0].copy()
    if len(valid) <= 5:
        print("\nInsufficient data for regression")
        return

    log_n = np.log(valid["n_desc"].values)

    for col, label in [("d_active", "d_active"), ("erank", "erank")]:
        log_y = np.log(valid[col].values.astype(float))
        A = np.column_stack([np.ones_like(log_n), log_n])
        beta, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        predicted = A @ beta
        ss_res = np.sum((log_y - predicted) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n  log({label}) = {beta[0]:.3f} + {beta[1]:.3f} * log(n_desc)")
        print(f"  => {label} ≈ {np.exp(beta[0]):.1f} * n_desc^{beta[1]:.3f}")
        print(f"  R² = {r2:.4f}")


def section_root_analysis(results: pd.DataFrame, tree: PosetTree) -> None:
    """Section 6: Root node detailed analysis."""
    import networkx as nx

    print_section("6: Root Node Analysis")
    root = tree.graph.get("root", list(nx.topological_sort(tree))[0])
    root_row = results[results["node"] == root]
    if root_row.empty:
        print("\n  Root node not in results")
        return

    r = root_row.iloc[0]
    print(f"\n  Root: {root}")
    print(f"    n_descendants     = {r['n_desc']}")
    print(f"    d_active          = {r['d_active']}")
    print(f"    erank (leaves)    = {r['erank']}")
    pw = r.get("erank_pw")
    print(f"    erank (pooled PW) = {pw if pd.notna(pw) else 'N/A'}")
    print(f"    max_child_erank   = {r.get('max_child_erank', 'N/A')}")
    print(f"    k_MP              = {r['k_MP']}")
    print(f"    k_JL              = {r['k_JL']}")
    print(f"    edge_k (pipeline) = {r.get('edge_k', 'N/A')}")
    print(f"    sib_k  (pipeline) = {r.get('sib_k', 'N/A')}")
    print(f"    top eigenvalue %  = {r['top_eig_pct']}%")

    children_eranks = r.get("children_eranks", {})
    if children_eranks:
        print(f"    children eranks   = {children_eranks}")


def section_sibling_test_comparison(results: pd.DataFrame, tree: PosetTree) -> None:
    """Section 7: Sibling test power comparison with different k strategies."""
    print_section("7: Sibling Test Power Comparison")
    print("  Tests KL sibling divergence at binary nodes with different k values.\n")

    binary = results[results["n_children"] == 2].sort_values(
        ["depth", "n_desc"], ascending=[True, False]
    )
    if binary.empty:
        print("  No binary internal nodes found.")
        return

    # Show top 15 binary nodes (shallowest first)
    for _, row in binary.head(15).iterrows():
        node_id = row["node"]
        children = list(tree.successors(node_id))
        if len(children) != 2:
            continue
        left, right = children[0], children[1]

        left_dist = tree.nodes[left].get("distribution")
        right_dist = tree.nodes[right].get("distribution")
        if left_dist is None or right_dist is None:
            continue

        left_dist = np.asarray(left_dist, dtype=np.float64)
        right_dist = np.asarray(right_dist, dtype=np.float64)
        n_left = float(tree.nodes[left].get("leaf_count", 1))
        n_right = float(tree.nodes[right].get("leaf_count", 1))

        k_variants = {
            "erank (parent)": int(round(row["erank"])),
            "erank (PW)": (int(round(row["erank_pw"])) if pd.notna(row.get("erank_pw")) else None),
            "max_child": (
                int(round(row["max_child_erank"])) if pd.notna(row.get("max_child_erank")) else None
            ),
            "sib_k (pipeline)": (int(row["sib_k"]) if pd.notna(row.get("sib_k")) else None),
            "JL": row["k_JL"],
        }

        print(
            f"--- {node_id} (depth={int(row['depth'])}, n={int(row['n_desc'])}, "
            f"left={left} n={int(n_left)}, right={right} n={int(n_right)}) ---"
        )

        for label, k in k_variants.items():
            if k is None or k < 1:
                print(f"  {label:22s}: k=N/A")
                continue
            k = max(int(k), 1)
            T, df, p = sibling_divergence_test(
                left_dist,
                right_dist,
                n_left,
                n_right,
                spectral_k=k,
                pca_projection=None,
                pca_eigenvalues=None,
            )
            verdict = "REJECT" if p < 0.05 else "SAME"
            marker = " << PIPELINE" if label == "sib_k (pipeline)" else ""
            print(
                f"  {label:22s}: k={k:3d}  T={T:8.2f}  df={df:5.0f}  "
                f"p={p:.6f}  -> {verdict}{marker}"
            )
        print()


def section_gate_trace(
    datasets: List[Tuple[str, pd.DataFrame, Optional[np.ndarray]]],
) -> None:
    """Section 8: Per-node gate trace for every dataset.

    Runs the full pipeline and shows Gate 1/2/3 decisions at each
    internal node on the decomposition path (root → leaves), plus
    calibration audit info.
    """
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score

    print_section("8: Pipeline Gate Trace & K/ARI Comparison")

    for ds_name, data_df, true_labels in datasets:
        tree = build_tree(data_df)
        result = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
        )
        rdf = tree.stats_df
        found_k = result["num_clusters"]

        true_k_str = ""
        ari_str = ""
        if true_labels is not None:
            true_k = len(set(true_labels))
            sample_to_cluster: Dict[str, int] = {}
            for cid, cinfo in result["cluster_assignments"].items():
                for leaf in cinfo["leaves"]:
                    sample_to_cluster[leaf] = cid
            pred = [sample_to_cluster.get(s, -1) for s in data_df.index]
            ari = adjusted_rand_score(list(true_labels), pred)
            true_k_str = f", true_K={true_k}"
            status = "OK" if found_k == true_k else "MISS"
            ari_str = f", ARI={ari:.3f} ({status})"

        n, d = data_df.shape
        print(f"\n--- {ds_name} (n={n}, d={d}{true_k_str}) " f"→ found_K={found_k}{ari_str} ---")

        # Calibration audit
        audit = rdf.attrs.get("sibling_divergence_audit", {})
        if audit:
            method = audit.get("calibration_method", "?")
            n_cal = audit.get("calibration_n", "?")
            c_hat = audit.get("global_c_hat", "?")
            max_r = audit.get("max_observed_ratio", "?")
            diag = audit.get("diagnostics", {})
            r2 = diag.get("R2", "?")
            beta = diag.get("beta", "?")
            print(
                f"  Calibration: method={method}, n_pairs={n_cal}, "
                f"global_ĉ={c_hat}, max_ratio={max_r}, R²={r2}"
            )
            if beta and beta != "?":
                print(f"  Regression β = {beta}")

        # Gate columns
        edge_sig_col = "Child_Parent_Divergence_Significant"
        edge_p_col = "Child_Parent_Divergence_P_Value_BH"
        edge_df_col = "Child_Parent_Divergence_df"
        sib_diff_col = "Sibling_BH_Different"
        sib_p_col = "Sibling_Divergence_P_Value_Corrected"
        sib_raw_p_col = "Sibling_Divergence_P_Value"
        sib_stat_col = "Sibling_Test_Statistic"
        sib_df_col = "Sibling_Degrees_of_Freedom"
        sib_skip_col = "Sibling_Divergence_Skipped"
        sib_method_col = "Sibling_Test_Method"

        root = tree.graph.get("root", list(nx.topological_sort(tree))[0])

        # BFS from root, showing all split-path nodes
        queue = [root]
        visited = set()
        print(
            f"\n  {'Node':>8s}  {'n':>4s}  {'G1':>3s}  "
            f"{'EdgeL_p':>8s}  {'EdgeR_p':>8s}  {'G2':>5s}  "
            f"{'Sib_T':>8s}  {'Sib_df':>6s}  {'Sib_p':>8s}  {'Sib_padj':>8s}  "
            f"{'G3':>5s}  {'method':>12s}  {'Verdict':>7s}"
        )
        print("  " + "-" * 120)

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            if tree._is_leaf(node_id):
                continue

            children = list(tree.successors(node_id))
            n_children = len(children)
            n_desc = tree.nodes[node_id].get("leaf_count", "?")

            # Gate 1: binary?
            g1 = "OK" if n_children == 2 else "FAIL"

            # Gate 2: edge significance for children
            edge_l_p = "N/A"
            edge_r_p = "N/A"
            g2 = "N/A"
            if n_children == 2:
                left, right = children[0], children[1]
                l_sig = _gate_val(rdf, left, edge_sig_col, False)
                r_sig = _gate_val(rdf, right, edge_sig_col, False)
                l_p = _gate_val(rdf, left, edge_p_col, np.nan)
                r_p = _gate_val(rdf, right, edge_p_col, np.nan)
                edge_l_p = f"{l_p:.4f}" if np.isfinite(l_p) else "N/A"
                edge_r_p = f"{r_p:.4f}" if np.isfinite(r_p) else "N/A"
                if l_sig or r_sig:
                    g2 = "SIG"
                else:
                    g2 = "NONE"

            # Gate 3: sibling divergence
            sib_t = _gate_val(rdf, node_id, sib_stat_col, np.nan)
            sib_df = _gate_val(rdf, node_id, sib_df_col, np.nan)
            sib_p = _gate_val(rdf, node_id, sib_raw_p_col, np.nan)
            sib_padj = _gate_val(rdf, node_id, sib_p_col, np.nan)
            sib_diff = _gate_val(rdf, node_id, sib_diff_col, False)
            sib_skip = _gate_val(rdf, node_id, sib_skip_col, False)
            sib_method = _gate_val(rdf, node_id, sib_method_col, "")

            if sib_skip:
                g3 = "SKIP"
            elif sib_diff:
                g3 = "DIFF"
            else:
                g3 = "SAME"

            # Final verdict
            if g1 != "OK":
                verdict = "MERGE"
            elif g2 == "NONE":
                verdict = "MERGE"
            elif g3 == "DIFF":
                verdict = "SPLIT"
            else:
                verdict = "MERGE"

            sib_t_str = f"{sib_t:8.2f}" if np.isfinite(sib_t) else "     N/A"
            sib_df_str = f"{sib_df:6.0f}" if np.isfinite(sib_df) else "   N/A"
            sib_p_str = f"{sib_p:.4f}" if np.isfinite(sib_p) else "     N/A"
            sib_padj_str = f"{sib_padj:.4f}" if np.isfinite(sib_padj) else "     N/A"

            print(
                f"  {node_id:>8s}  {str(n_desc):>4s}  {g1:>3s}  "
                f"{edge_l_p:>8s}  {edge_r_p:>8s}  {g2:>5s}  "
                f"{sib_t_str}  {sib_df_str}  {sib_p_str}  {sib_padj_str}  "
                f"{g3:>5s}  {str(sib_method):>12s}  {verdict:>7s}"
            )

            # Only descend into children if this node SPLIT
            if verdict == "SPLIT":
                for child in children:
                    if not tree._is_leaf(child):
                        queue.append(child)

        # ---------- Deep failure analysis for MERGE nodes on the decomposition path ----------
        _print_failure_analysis(tree, rdf, ds_name, data_df)


def _print_failure_analysis(
    tree: "PosetTree",
    rdf: pd.DataFrame,
    ds_name: str,
    data_df: pd.DataFrame,
) -> None:
    """Print deep failure analysis for each MERGE on the decomposition path.

    For Gate 2 failures: shows raw vs BH-corrected p-values, spectral k,
    how many BH tests, how close the raw p was to alpha.

    For Gate 3 failures: shows raw Wald T, deflated T_adj, ĉ used,
    what raw Wald (no calibration) would have concluded.
    """
    import networkx as nx
    from scipy.stats import chi2

    root = tree.graph.get("root", list(nx.topological_sort(tree))[0])

    edge_sig_col = "Child_Parent_Divergence_Significant"
    edge_p_raw_col = "Child_Parent_Divergence_P_Value"
    edge_p_bh_col = "Child_Parent_Divergence_P_Value_BH"
    edge_df_col = "Child_Parent_Divergence_df"
    sib_stat_col = "Sibling_Test_Statistic"
    sib_df_col = "Sibling_Degrees_of_Freedom"
    sib_raw_p_col = "Sibling_Divergence_P_Value"
    sib_p_col = "Sibling_Divergence_P_Value_Corrected"
    sib_skip_col = "Sibling_Divergence_Skipped"
    sib_diff_col = "Sibling_BH_Different"

    # Walk decomposition path collecting MERGE nodes
    queue = [root]
    visited = set()
    merge_nodes = []

    while queue:
        node_id = queue.pop(0)
        if node_id in visited or tree._is_leaf(node_id):
            continue
        visited.add(node_id)

        children = list(tree.successors(node_id))
        if len(children) != 2:
            merge_nodes.append((node_id, "gate1"))
            continue

        left, right = children[0], children[1]
        l_sig = _gate_val(rdf, left, edge_sig_col, False)
        r_sig = _gate_val(rdf, right, edge_sig_col, False)

        if not (l_sig or r_sig):
            merge_nodes.append((node_id, "gate2"))
            continue

        sib_skip = _gate_val(rdf, node_id, sib_skip_col, False)
        sib_diff = _gate_val(rdf, node_id, sib_diff_col, False)

        if sib_skip:
            merge_nodes.append((node_id, "gate3_skip"))
            continue
        if not sib_diff:
            merge_nodes.append((node_id, "gate3_same"))
            continue

        # SPLIT — descend
        for child in children:
            if not tree._is_leaf(child):
                queue.append(child)

    if not merge_nodes:
        return

    print(f"\n  --- FAILURE ANALYSIS ({ds_name}) ---")

    # Edge test summary
    edge_ps = rdf[edge_p_raw_col].dropna()
    n_edge_tests = len(edge_ps)
    n_edge_sig = int(rdf[edge_sig_col].sum()) if edge_sig_col in rdf.columns else 0
    print(
        f"  Edge tests: {n_edge_sig}/{n_edge_tests} significant "
        f"(BH α=0.05, {n_edge_tests} tests)"
    )

    for node_id, failure_gate in merge_nodes:
        children = list(tree.successors(node_id))
        n_desc = tree.nodes[node_id].get("leaf_count", "?")

        if failure_gate == "gate1":
            print(
                f"\n  {node_id} (n={n_desc}): Gate 1 FAIL — " f"{len(children)} children (need 2)"
            )

        elif failure_gate == "gate2":
            left, right = children[0], children[1]
            l_p_raw = _gate_val(rdf, left, edge_p_raw_col, np.nan)
            r_p_raw = _gate_val(rdf, right, edge_p_raw_col, np.nan)
            l_p_bh = _gate_val(rdf, left, edge_p_bh_col, np.nan)
            r_p_bh = _gate_val(rdf, right, edge_p_bh_col, np.nan)
            l_df = _gate_val(rdf, left, edge_df_col, np.nan)
            r_df = _gate_val(rdf, right, edge_df_col, np.nan)

            print(f"\n  {node_id} (n={n_desc}): Gate 2 FAIL — " f"neither child edge-significant")
            print(f"    Left  {left}: p_raw={l_p_raw:.6f}  p_BH={l_p_bh:.6f}  " f"df={l_df}")
            print(f"    Right {right}: p_raw={r_p_raw:.6f}  p_BH={r_p_bh:.6f}  " f"df={r_df}")
            min_raw = min(
                l_p_raw if np.isfinite(l_p_raw) else 1.0,
                r_p_raw if np.isfinite(r_p_raw) else 1.0,
            )
            if min_raw < 0.10:
                print(
                    f"    ⚠ Near-miss: min raw p={min_raw:.4f} "
                    f"(rejected by BH with {n_edge_tests} tests)"
                )
            else:
                print(f"    Edge test sees no signal (min raw p={min_raw:.4f})")

        elif failure_gate in ("gate3_same", "gate3_skip"):
            left, right = children[0], children[1]
            sib_t = _gate_val(rdf, node_id, sib_stat_col, np.nan)
            sib_df_val = _gate_val(rdf, node_id, sib_df_col, np.nan)
            sib_p = _gate_val(rdf, node_id, sib_raw_p_col, np.nan)
            sib_padj = _gate_val(rdf, node_id, sib_p_col, np.nan)

            label = "Gate 3 SKIP" if failure_gate == "gate3_skip" else "Gate 3 SAME"
            print(f"\n  {node_id} (n={n_desc}): {label}")

            if np.isfinite(sib_t) and np.isfinite(sib_df_val) and sib_df_val > 0:
                # What would raw Wald (no deflation) say?
                p_raw_wald = float(chi2.sf(sib_t, df=sib_df_val))
                print(
                    f"    T_adj={sib_t:.2f}  df={sib_df_val:.0f}  "
                    f"p_raw={sib_p:.6f}  p_BH={sib_padj:.6f}"
                )

                # Compute what the un-deflated raw Wald T would be
                # T_adj = T_raw / ĉ, so T_raw = T_adj * ĉ
                # But we don't store T_raw in stats_df. We can reconstruct:
                # The sibling_divergence_test gives raw T.
                left_dist = tree.nodes[left].get("distribution")
                right_dist = tree.nodes[right].get("distribution")
                n_left = float(tree.nodes[left].get("leaf_count", 1))
                n_right = float(tree.nodes[right].get("leaf_count", 1))

                if left_dist is not None and right_dist is not None:
                    left_dist = np.asarray(left_dist, dtype=np.float64)
                    right_dist = np.asarray(right_dist, dtype=np.float64)
                    T_raw, df_raw, p_raw = sibling_divergence_test(
                        left_dist,
                        right_dist,
                        n_left,
                        n_right,
                        spectral_k=int(sib_df_val),
                        pca_projection=None,
                        pca_eigenvalues=None,
                    )
                    c_hat_est = T_raw / sib_t if sib_t > 0 else float("nan")
                    verdict_raw = "REJECT" if p_raw < 0.05 else "SAME"
                    print(f"    Raw Wald: T_raw={T_raw:.2f}  p={p_raw:.6f}  " f"→ {verdict_raw}")
                    print(f"    Implied ĉ = T_raw/T_adj = {c_hat_est:.3f}")
                    if p_raw < 0.05 and sib_padj >= 0.05:
                        print(
                            "    ⚠ Raw Wald rejects but calibrated test does not — "
                            "calibration deflation is the cause"
                        )
            else:
                print("    Sibling test was skipped (insufficient data)")


def _gate_val(rdf: pd.DataFrame, node_id: str, col: str, default):
    """Safely extract a gate column value from results_df."""
    if col not in rdf.columns:
        return default
    if node_id not in rdf.index:
        return default
    val = rdf.loc[node_id, col]
    if pd.isna(val):
        return default
    return val


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="TSV feature matrix (default: feature_matrix.tsv if it exists)",
    )
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic cases")
    p.add_argument(
        "--all",
        action="store_true",
        help="Run on both file + synthetic (default)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which data sources to use
    run_file = False
    run_synthetic = False

    if args.synthetic:
        run_synthetic = True
    elif args.input is not None:
        run_file = True
    elif args.all:
        run_file = True
        run_synthetic = True
    else:
        # Default: both if feature_matrix.tsv exists, else synthetic only
        default_path = Path("feature_matrix.tsv")
        if default_path.exists():
            args.input = default_path
            run_file = True
        run_synthetic = True

    # Collect datasets: (name, data_df, true_labels_or_None)
    datasets: List[Tuple[str, pd.DataFrame, Optional[np.ndarray]]] = []

    if run_file and args.input is not None:
        data_df, labels, name = load_file_dataset(args.input)
        datasets.append((name, data_df, labels))

    if run_synthetic:
        for case in SYNTHETIC_CASES:
            data_df, labels, name = load_synthetic_dataset(case)
            datasets.append((name, data_df, labels))

    if not datasets:
        print("No datasets to analyze. Use --input FILE or --synthetic.")
        return

    # Process each dataset
    for ds_name, data_df, true_labels in datasets:
        n, d = data_df.shape
        true_k_str = f", true_K={len(set(true_labels))}" if true_labels is not None else ""
        print("\n" + "#" * 90)
        print(f"# DATASET: {ds_name} (n={n}, d={d}{true_k_str})")
        print("#" * 90)

        tree = build_tree(data_df)
        results, edge_dims, sib_dims = compute_per_node_dimensions(tree, data_df)

        if results.empty:
            print("  No internal nodes with ≥2 descendants.")
            continue

        section_summary(results)
        section_binned(results)
        section_sample_nodes(results)
        section_ratio_analysis(results)
        section_scaling_regression(results)
        section_root_analysis(results, tree)
        section_sibling_test_comparison(results, tree)

    # Pipeline gate trace across all datasets
    section_gate_trace(datasets)


if __name__ == "__main__":
    main()
