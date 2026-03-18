"""Lab exp26: Parent-PCA projection basis for Gate 3 sibling test.

Hypothesis: Gate 3's projection should use the PARENT's PCA directions
(which capture between-group variance) rather than random directions or
per-child PCA.  The parent's eigendecomposition is already computed in
Gate 2 — we just need to pass it through.

This experiment separates two independent concerns:
  A. Projection DIRECTIONS: random vs parent PCA eigenvectors
  B. Projection DIMENSION k: min-child vs parent-k vs lam12

Previous experiments tested only (B) — various k strategies with random
directions.  This experiment tests (A) in combination with (B).

Configurations:
  1. baseline          — min-child k, NO parent PCA (pre-change behavior)
  2. parent_pca_minchild — min-child k, parent PCA directions + eigenvalues
  3. parent_pca_parentk  — parent k, parent PCA directions + eigenvalues
  4. parent_pca_lam12    — lam12_frac × JL k, parent PCA directions + eigenvalues
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data, compute_ari, temporary_experiment_overrides

from benchmarks.shared.cases import get_default_test_cases
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)

_leaf_data_cache: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════


def _descendant_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        if tree.out_degree(n) == 0:
            leaves.append(n)
        else:
            stack.extend(tree.successors(n))
    return leaves


def _node_data(tree, node, leaf_data):
    leaves = _descendant_leaves(tree, node)
    labels = [tree.nodes[lf].get("label", lf) for lf in leaves]
    return leaf_data.loc[labels].values


def _eigendecompose(data_matrix):
    if data_matrix.shape[0] < 2:
        return None
    result = eigendecompose_correlation_backend(data_matrix, need_eigh=False)
    if result is None:
        return None
    return result.eigenvalues, data_matrix.shape[0], result.d_active


def _clamp_k(x, minimum=2):
    return max(minimum, int(round(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Dimension strategies (k derivation)
# ═════════════════════════════════════════════════════════════════════════════


def _derive_dims_min_child(tree, annotated_df):
    """Min-child spectral k (original production behavior)."""
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None
    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        child_ks = [
            k for k in (edge_spectral_dims.get(left, 0), edge_spectral_dims.get(right, 0)) if k > 0
        ]
        if not child_ks:
            continue
        sibling_dims[parent] = min(child_ks)
    return sibling_dims if sibling_dims else None


def _derive_dims_parent_k(tree, annotated_df):
    """Parent spectral k (parent's MP signal count on L∪R descendants)."""
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None
    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        parent_k = edge_spectral_dims.get(parent, 0)
        if parent_k > 0:
            sibling_dims[parent] = parent_k
    return sibling_dims if sibling_dims else None


def _derive_dims_lam12(tree, annotated_df):
    """Lam12_frac × JL dimension (from exp25)."""
    leaf_data = _leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None
    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        try:
            data_P = np.vstack(
                [_node_data(tree, left, leaf_data), _node_data(tree, right, leaf_data)]
            )
        except (KeyError, IndexError):
            continue
        eig_P = _eigendecompose(data_P)
        if eig_P is None:
            continue
        ev_P, ns_P, da_P = eig_P
        n_features = leaf_data.shape[1]
        n_P = len(_descendant_leaves(tree, left)) + len(_descendant_leaves(tree, right))
        jl_k = compute_jl_dim(n_P, n_features)
        trace_P = float(np.sum(ev_P))
        lam1 = float(ev_P[0]) if len(ev_P) > 0 else 0.0
        lam2 = float(ev_P[1]) if len(ev_P) > 1 else 0.0
        frac = (lam1 + lam2) / max(trace_P, 1e-12)
        sibling_dims[parent] = _clamp_k(frac * jl_k)
    return sibling_dims if sibling_dims else None


# ═════════════════════════════════════════════════════════════════════════════
# PCA projection strategies
# ═════════════════════════════════════════════════════════════════════════════


def _derive_pca_none(annotated_df, sibling_dims):
    """No PCA — forces random projection (baseline behavior)."""
    return None, None


def _derive_pca_parent(annotated_df, sibling_dims):
    """Parent PCA projections + eigenvalues from Gate 2 attrs."""
    if sibling_dims is None:
        return None, None
    pca_projections = annotated_df.attrs.get("_pca_projections")
    pca_eigenvalues = annotated_df.attrs.get("_pca_eigenvalues")
    if not pca_projections:
        return None, None
    sibling_projections = {}
    sibling_eigenvalues = {}
    for parent in sibling_dims:
        proj = pca_projections.get(parent)
        if proj is not None:
            sibling_projections[parent] = proj
        eig = pca_eigenvalues.get(parent) if pca_eigenvalues else None
        if eig is not None:
            sibling_eigenvalues[parent] = eig
    return (
        sibling_projections if sibling_projections else None,
        sibling_eigenvalues if sibling_eigenvalues else None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Configurations: (k_strategy, pca_strategy)
# ═════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "baseline": (_derive_dims_min_child, _derive_pca_none),
    "parent_pca_minchild": (_derive_dims_min_child, _derive_pca_parent),
    "parent_pca_parentk": (_derive_dims_parent_k, _derive_pca_parent),
    "parent_pca_lam12": (_derive_dims_lam12, _derive_pca_parent),
}

CONFIG_NAMES = list(CONFIGS.keys())
N_CONFIGS = len(CONFIG_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


def run_case(case_name: str) -> dict[str, dict]:
    results = {}
    for cfg_name in CONFIG_NAMES:
        tree, data_df, y_true, tc = build_tree_and_data(case_name)
        true_k = tc.get("n_clusters", "?")

        k_fn, pca_fn = CONFIGS[cfg_name]

        try:
            with temporary_experiment_overrides(
                leaf_data_cache=_leaf_data_cache,
                leaf_data=data_df,
                sibling_dims=k_fn,
                sibling_pca=pca_fn,
            ):
                decomp = tree.decompose(
                    leaf_data=data_df,
                    alpha_local=config.SIBLING_ALPHA,
                    sibling_alpha=config.SIBLING_ALPHA,
                )
            ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
            results[cfg_name] = {
                "true_k": true_k,
                "found_k": decomp["num_clusters"],
                "ari": round(ari, 4),
            }
        except Exception as exc:
            results[cfg_name] = {
                "true_k": true_k,
                "found_k": "ERR",
                "ari": 0.0,
                "error": str(exc)[:80],
            }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    n_cases = len(case_names)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}, EPS={config.PROJECTION_EPS}")
    print(f"\n{'='*80}")
    print(f"EXP26: Parent-PCA Projection Basis — {n_cases} cases × {N_CONFIGS} configs")
    print(f"{'='*80}")
    print("\n  Strategies:")
    for i, cn in enumerate(CONFIG_NAMES, 1):
        print(f"    {i}. {cn}")
    print()

    all_results: list[tuple[str, dict]] = []
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        print(f"  [{i:3d}/{n_cases}] {case_name:<45}", end="", flush=True)
        t1 = time.time()
        try:
            res = run_case(case_name)
        except Exception as exc:
            print(f" FATAL: {str(exc)[:60]}")
            res = {cn: {"true_k": "?", "found_k": "ERR", "ari": 0.0} for cn in CONFIG_NAMES}
        dt = time.time() - t1

        base = res["baseline"]["ari"]
        ppca = res["parent_pca_minchild"]["ari"]
        delta = ppca - base
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f" {dt:5.1f}s  base={base:.3f}  pca_min={ppca:.3f}  [{marker}]")
        all_results.append((case_name, res))

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time / 60:.1f}min)\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Aggregate
    # ═════════════════════════════════════════════════════════════════════════

    print(f"{'='*80}")
    print(f"AGGREGATE ({n_cases} cases)")
    print(f"{'='*80}\n")
    print(
        f"  {'Config':<22} │ {'Mean ARI':>9} │ {'Med ARI':>9} │ "
        f"{'Exact K':>9} │ {'Perfect':>8} │ {'K=1':>5} │ {'Wins':>5}"
    )
    print("  " + "─" * 86)

    strat_aris = {cn: [] for cn in CONFIG_NAMES}
    strat_exact = {cn: 0 for cn in CONFIG_NAMES}
    strat_k1 = {cn: 0 for cn in CONFIG_NAMES}
    strat_perfect = {cn: 0 for cn in CONFIG_NAMES}

    for _, res in all_results:
        for cn in CONFIG_NAMES:
            r = res[cn]
            strat_aris[cn].append(r["ari"])
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]:
                strat_exact[cn] += 1
            if isinstance(r["found_k"], int) and r["found_k"] == 1:
                strat_k1[cn] += 1
            if isinstance(r["ari"], float) and r["ari"] >= 0.999:
                strat_perfect[cn] += 1

    countable = sum(1 for _, res in all_results if res[CONFIG_NAMES[0]]["true_k"] != "?")

    strat_wins = {cn: 0 for cn in CONFIG_NAMES}
    for _, res in all_results:
        best_ari = max(res[cn]["ari"] for cn in CONFIG_NAMES)
        for cn in CONFIG_NAMES:
            if res[cn]["ari"] >= best_ari - 1e-6:
                strat_wins[cn] += 1

    for cn in CONFIG_NAMES:
        aris = strat_aris[cn]
        mean_a = float(np.mean(aris))
        med_a = float(np.median(aris))
        print(
            f"  {cn:<22} │ {mean_a:9.3f} │ {med_a:9.3f} │ "
            f"{strat_exact[cn]:5d}/{countable:<3d} │ {strat_perfect[cn]:8d} │ "
            f"{strat_k1[cn]:5d} │ {strat_wins[cn]:5d}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Head-to-head: baseline vs parent_pca_minchild
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("HEAD-TO-HEAD: baseline vs parent_pca_minchild")
    print(f"{'='*80}\n")

    wins_pca = 0
    wins_base = 0
    ties = 0
    deltas = []
    improved_cases = []
    regressed_cases = []

    for case_name, res in all_results:
        a_base = res["baseline"]["ari"]
        a_pca = res["parent_pca_minchild"]["ari"]
        d = a_pca - a_base
        deltas.append(d)
        if d > 0.01:
            wins_pca += 1
            improved_cases.append(
                (
                    case_name,
                    d,
                    res["baseline"]["found_k"],
                    res["parent_pca_minchild"]["found_k"],
                    res["baseline"]["true_k"],
                )
            )
        elif d < -0.01:
            wins_base += 1
            regressed_cases.append(
                (
                    case_name,
                    d,
                    res["baseline"]["found_k"],
                    res["parent_pca_minchild"]["found_k"],
                    res["baseline"]["true_k"],
                )
            )
        else:
            ties += 1

    print(f"  PCA wins: {wins_pca}  |  Baseline wins: {wins_base}  |  Ties: {ties}")
    print(f"  Mean delta ARI: {np.mean(deltas):+.4f}")
    print(f"  Median delta ARI: {np.median(deltas):+.4f}")

    if improved_cases:
        improved_cases.sort(key=lambda x: -x[1])
        print("\n  Top improvements (Δ > 0.01):")
        for name, d, k_base, k_pca, true_k in improved_cases[:15]:
            print(f"    {name:<40} Δ={d:+.3f}  K: {k_base}→{k_pca} (true={true_k})")

    if regressed_cases:
        regressed_cases.sort(key=lambda x: x[1])
        print("\n  Top regressions (Δ < -0.01):")
        for name, d, k_base, k_pca, true_k in regressed_cases[:15]:
            print(f"    {name:<40} Δ={d:+.3f}  K: {k_base}→{k_pca} (true={true_k})")

    # ═════════════════════════════════════════════════════════════════════════
    # Category breakdown
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*80}\n")

    categories: dict[str, list[tuple[str, dict]]] = {}
    for case_name, res in all_results:
        tc = next((c for c in all_cases if c["name"] == case_name), {})
        cat = tc.get("category", "unknown")
        if "gaussian" in cat or "gauss" in cat:
            macro = "Gaussian"
        elif "binary" in cat or "sparse" in cat:
            macro = "Binary"
        elif "sbm" in cat:
            macro = "SBM"
        elif "cat" in cat:
            macro = "Categorical"
        elif "phylo" in cat:
            macro = "Phylogenetic"
        elif "overlap" in cat:
            macro = "Overlap"
        else:
            macro = "Other"
        categories.setdefault(macro, []).append((case_name, res))

    for macro in sorted(categories):
        cat_cases = categories[macro]
        print(f"  {macro} ({len(cat_cases)} cases):")
        print(f"    {'Config':<22} │ {'Mean ARI':>9} │ {'Exact K':>9} │ {'K=1':>5}")
        print("    " + "─" * 56)
        for cn in CONFIG_NAMES:
            aris = [res[cn]["ari"] for _, res in cat_cases]
            exact = sum(
                1
                for _, res in cat_cases
                if isinstance(res[cn]["found_k"], int)
                and res[cn]["true_k"] != "?"
                and res[cn]["found_k"] == res[cn]["true_k"]
            )
            k1 = sum(
                1
                for _, res in cat_cases
                if isinstance(res[cn]["found_k"], int) and res[cn]["found_k"] == 1
            )
            n_countable = sum(1 for _, res in cat_cases if res[cn]["true_k"] != "?")
            print(f"    {cn:<22} │ {np.mean(aris):9.3f} │ {exact:5d}/{n_countable:<3d} │ {k1:5d}")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # Per-case detail table
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("PER-CASE DETAIL")
    print(f"{'='*80}\n")
    print(f"  {'Case':<40} │ {'TrueK':>5} │ " + " │ ".join(f"{cn[:10]:>10}" for cn in CONFIG_NAMES))
    print("  " + "─" * (48 + 13 * N_CONFIGS))

    for case_name, res in all_results:
        true_k = res[CONFIG_NAMES[0]]["true_k"]
        parts = []
        for cn in CONFIG_NAMES:
            r = res[cn]
            k = r["found_k"]
            a = r["ari"]
            parts.append(f" K={k:<2} {a:.2f}")
        print(f"  {case_name:<40} │ {str(true_k):>5} │ " + " │ ".join(parts))
