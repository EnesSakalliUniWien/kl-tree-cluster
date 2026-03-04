#!/usr/bin/env python3
"""Diagnose spectral decomposition across benchmark cases.

For each case:
  1. Build tree, populate distributions
  2. Run compute_spectral_decomposition with effective_rank and marchenko_pastur
  3. Run compute_sibling_spectral_dimensions
  4. Show per-node dimensions, eigenvalue spectra, and comparison with JL
  5. Compare INCLUDE_INTERNAL_IN_SPECTRAL True vs False
  6. Run full decomposition and show resulting K, ARI

Usage:
  python debug_scripts/projection_power/diagnose_spectral_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.estimators import (
    effective_rank,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    compute_projection_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.sibling_spectral_dimension import (
    compute_sibling_spectral_dimensions,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral_dimension import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Cases to diagnose (mix of easy, moderate, and hard)
TARGET_CASES = [
    "gauss_clear_small",
    "gauss_clear_large",
    "gauss_moderate_3c",
    "binary_perfect_2c",
    "binary_perfect_4c",
    "binary_perfect_8c",
    "binary_low_noise_4c",
    "sparse_features_72x72",
    "gauss_null_small",
    "binary_null_small",
]


def find_case(name: str) -> dict | None:
    for c in get_default_test_cases():
        if c.get("name") == name:
            return c
    return None


def build_tree(case_config: dict):
    """Generate data and build a PosetTree."""
    data, labels, _x_orig, _meta = generate_case_data(case_config)

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        n, p = data.shape
        df = pd.DataFrame(
            data, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]
        )

    Z = linkage(
        pdist(df.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
    tree.populate_node_divergences(df)
    return tree, df, labels


def format_eigenvalues(eigenvalues: np.ndarray, max_show: int = 15) -> str:
    """Format eigenvalue array for display."""
    if eigenvalues is None or len(eigenvalues) == 0:
        return "(none)"
    sorted_eigs = np.sort(eigenvalues)[::-1]
    top = sorted_eigs[:max_show]
    parts = [f"{v:.3f}" for v in top]
    if len(sorted_eigs) > max_show:
        parts.append(f"... ({len(sorted_eigs)} total)")
    return "[" + ", ".join(parts) + "]"


def analyze_case(name: str, case_config: dict):
    """Run spectral decomposition diagnosis for one case."""
    print(f"\n{'='*80}")
    print(f"CASE: {name}")
    true_k = case_config.get("n_clusters", case_config.get("true_k", "?"))
    print(f"True K: {true_k}")
    print(f"{'='*80}")

    tree, df, labels = build_tree(case_config)
    n_samples, n_features = df.shape
    root = tree.root()
    n_internal = sum(1 for n in tree.nodes if not tree.nodes[n].get("is_leaf", False))

    print(f"  Data: n={n_samples}, d={n_features}")
    print(f"  Tree: {len(tree.nodes)} nodes, {n_internal} internal")

    # JL-based dimension for reference
    jl_k = compute_projection_dimension(n_samples, n_features)
    print(f"  JL dimension (global): k={jl_k} (eps={config.PROJECTION_EPS})")

    # ---- Effective Rank (includes internal = True) ----
    print("\n  --- Spectral: effective_rank, include_internal=True ---")
    orig_incl = config.INCLUDE_INTERNAL_IN_SPECTRAL
    config.INCLUDE_INTERNAL_IN_SPECTRAL = True

    dims_er_incl, projs_er_incl, eigs_er_incl = compute_spectral_decomposition(
        tree, df, method="effective_rank", min_k=1, compute_projections=True
    )
    internal_ks = {
        nid: k for nid, k in dims_er_incl.items() if not tree.nodes[nid].get("is_leaf", False)
    }
    ks_list = list(internal_ks.values())
    print(
        f"    k stats: min={min(ks_list)}, max={max(ks_list)}, median={np.median(ks_list):.0f}, mean={np.mean(ks_list):.1f}"
    )
    print(f"    Root ({root}): k={dims_er_incl.get(root)}")

    # Show root eigenvalue spectrum
    root_eigs = eigs_er_incl.get(root)
    if root_eigs is not None:
        er = effective_rank(root_eigs)
        print(
            f"    Root eigenvalues ({len(root_eigs)} PCA components): {format_eigenvalues(root_eigs)}"
        )
        print(f"    Root effective_rank(top-k eigs): {er:.2f}")

    # Show first few binary split nodes
    children = list(tree.successors(root))
    if len(children) == 2:
        for i, child in enumerate(children):
            child_k = dims_er_incl.get(child)
            child_eigs = eigs_er_incl.get(child)
            n_leaves = len(tree.get_leaves(child))
            print(f"    Child[{i}] ({child}): k={child_k}, n_leaves={n_leaves}")
            if child_eigs is not None:
                print(f"      eigenvalues: {format_eigenvalues(child_eigs, 10)}")

    # ---- Effective Rank (includes internal = False) ----
    print("\n  --- Spectral: effective_rank, include_internal=False ---")
    config.INCLUDE_INTERNAL_IN_SPECTRAL = False

    dims_er_excl, _, eigs_er_excl = compute_spectral_decomposition(
        tree, df, method="effective_rank", min_k=1, compute_projections=True
    )
    internal_ks_excl = {
        nid: k for nid, k in dims_er_excl.items() if not tree.nodes[nid].get("is_leaf", False)
    }
    ks_list_excl = list(internal_ks_excl.values())
    print(
        f"    k stats: min={min(ks_list_excl)}, max={max(ks_list_excl)}, median={np.median(ks_list_excl):.0f}, mean={np.mean(ks_list_excl):.1f}"
    )
    print(f"    Root ({root}): k={dims_er_excl.get(root)}")

    root_eigs_excl = eigs_er_excl.get(root)
    if root_eigs_excl is not None:
        er_excl = effective_rank(root_eigs_excl)
        print(f"    Root effective_rank(top-k eigs): {er_excl:.2f}")

    # Compare include vs exclude
    diffs = []
    for nid in internal_ks:
        k_incl = internal_ks[nid]
        k_excl = internal_ks_excl.get(nid, k_incl)
        diffs.append(k_incl - k_excl)
    print(
        f"    Δ(include - exclude): mean={np.mean(diffs):.2f}, median={np.median(diffs):.0f}, "
        f"min={min(diffs)}, max={max(diffs)}"
    )

    config.INCLUDE_INTERNAL_IN_SPECTRAL = orig_incl

    # ---- Marchenko-Pastur ----
    print("\n  --- Spectral: marchenko_pastur ---")
    dims_mp, _, _ = compute_spectral_decomposition(
        tree, df, method="marchenko_pastur", min_k=1, compute_projections=False
    )
    internal_ks_mp = {
        nid: k for nid, k in dims_mp.items() if not tree.nodes[nid].get("is_leaf", False)
    }
    ks_mp = list(internal_ks_mp.values())
    print(
        f"    k stats: min={min(ks_mp)}, max={max(ks_mp)}, median={np.median(ks_mp):.0f}, mean={np.mean(ks_mp):.1f}"
    )
    print(f"    Root: k={dims_mp.get(root)}")

    # ---- Sibling-specific spectral dimensions (pooled within-cluster) ----
    print("\n  --- Sibling spectral dims (pooled within-cluster, effective_rank) ---")
    sib_dims = compute_sibling_spectral_dimensions(tree, df, method="effective_rank", min_k=1)
    sib_internal = {
        nid: k for nid, k in sib_dims.items() if not tree.nodes[nid].get("is_leaf", False)
    }
    sib_ks = list(sib_internal.values())
    print(
        f"    k stats: min={min(sib_ks)}, max={max(sib_ks)}, median={np.median(sib_ks):.0f}, mean={np.mean(sib_ks):.1f}"
    )
    print(f"    Root: k_sibling={sib_dims.get(root)} vs k_overall={dims_er_incl.get(root)}")

    # Compare sibling vs overall dimensions
    ratios = []
    for nid in sib_internal:
        overall = internal_ks.get(nid, 1)
        sibling = sib_internal[nid]
        if overall > 0:
            ratios.append(sibling / overall)
    if ratios:
        print(
            f"    Ratio(sibling/overall): mean={np.mean(ratios):.2f}, median={np.median(ratios):.2f}"
        )

    # ---- Active Features (simple count) ----
    print("\n  --- Spectral: active_features ---")
    dims_af, _, _ = compute_spectral_decomposition(
        tree, df, method="active_features", min_k=1, compute_projections=False
    )
    af_root = dims_af.get(root)
    print(f"    Root active features: {af_root} / {n_features}")

    # ---- Run pipeline to show clustering result ----
    print("\n  --- Pipeline result ---")
    results = tree.decompose(leaf_data=df, alpha_local=0.05, sibling_alpha=0.05)
    found_k = results["num_clusters"]
    cluster_assignments = results["cluster_assignments"]
    label_map = {}
    for cid, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cid
    leaf_names = df.index.tolist()
    pred = np.array([label_map.get(leaf_names[i], -1) for i in range(n_samples)])
    ari = adjusted_rand_score(labels, pred) if labels is not None else "N/A"
    print(f"    Found K={found_k} (true K={true_k}), ARI={ari:.3f}")

    # Show spectral dims from stats_df
    stats_df = tree.stats_df
    spectral_dims_cached = stats_df.attrs.get("_spectral_dims", {})
    if spectral_dims_cached:
        cached_ks = [k for nid, k in spectral_dims_cached.items() if k is not None and k > 0]
        if cached_ks:
            print(
                f"    Cached spectral dims: min={min(cached_ks)}, max={max(cached_ks)}, median={np.median(cached_ks):.0f}"
            )

    return {
        "name": name,
        "true_k": true_k,
        "found_k": found_k,
        "ari": ari,
        "n": n_samples,
        "d": n_features,
        "jl_k": jl_k,
        "er_root_k": dims_er_incl.get(root),
        "er_root_k_excl": dims_er_excl.get(root),
        "mp_root_k": dims_mp.get(root),
        "sib_root_k": sib_dims.get(root),
        "af_root": af_root,
        "er_median": np.median(ks_list),
        "er_mean": np.mean(ks_list),
    }


def main():
    print("=" * 80)
    print("SPECTRAL DECOMPOSITION DIAGNOSTIC")
    print(
        f"Config: SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
        f"INCLUDE_INTERNAL_IN_SPECTRAL={config.INCLUDE_INTERNAL_IN_SPECTRAL}"
    )
    print(
        f"        EIGENVALUE_WHITENING={config.EIGENVALUE_WHITENING}, "
        f"PROJECTION_EPS={config.PROJECTION_EPS}"
    )
    print(
        f"        PROJECTION_MIN_K={config.PROJECTION_MIN_K}, "
        f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}"
    )
    print("=" * 80)

    results = []
    for name in TARGET_CASES:
        case = find_case(name)
        if case is None:
            print(f"\n  SKIP: {name} not found")
            continue
        try:
            r = analyze_case(name, case)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(
        f"{'Case':<30} {'n':>5} {'d':>5} {'JL':>4} {'ER':>4} {'ER-':>4} {'MP':>4} "
        f"{'Sib':>4} {'AF':>5} {'TrK':>4} {'K':>4} {'ARI':>6}"
    )
    print("-" * 100)
    for r in results:
        ari_str = f"{r['ari']:.3f}" if isinstance(r["ari"], float) else r["ari"]
        print(
            f"{r['name']:<30} {r['n']:>5} {r['d']:>5} {r['jl_k']:>4} "
            f"{r.get('er_root_k', '?'):>4} {r.get('er_root_k_excl', '?'):>4} "
            f"{r.get('mp_root_k', '?'):>4} {r.get('sib_root_k', '?'):>4} "
            f"{r.get('af_root', '?'):>5} {r['true_k']:>4} {r['found_k']:>4} {ari_str:>6}"
        )

    print("\nColumns: JL=Johnson-Lindenstrauss k, ER=effective_rank (include_internal=True),")
    print("         ER-=effective_rank (include_internal=False), MP=marchenko_pastur,")
    print("         Sib=sibling spectral k (pooled within-cluster), AF=active features at root")


if __name__ == "__main__":
    main()
