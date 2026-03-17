#!/usr/bin/env python
"""Experiment 9 — Sibling Projection Strategy Comparison.

Tests four strategies for choosing the sibling test projection:

  A. **baseline**       — Current production: JL k capped at 5, random projection
  B. **parent_spectral** — Parent's MP spectral k, parent's PCA eigenvectors,
                           eigenvalue whitening.  χ²(k_parent) exact under H₀.
  C. **min_child**      — min(k_left, k_right) as projection dim, random projection
  D. **max_child**      — max(k_left, k_right) as projection dim, random projection

Strategy B is the main hypothesis: since both children come from the parent
distribution under H₀, the parent's eigenvectors are the correct null basis,
and k_parent is the correct degrees of freedom.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from lab_helpers import FAILURE_CASES, REGRESSION_GUARD_CASES, build_tree_and_data, compute_ari

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ── Strategy helpers ─────────────────────────────────────────────────────────


def _build_spectral_dicts(
    tree: PosetTree,
    leaf_data: pd.DataFrame,
) -> Dict[str, Any]:
    spectral_dims, pca_projections, pca_eigenvalues = compute_spectral_decomposition(
        tree,
        leaf_data,
        method="marchenko_pastur",
        minimum_projection_dimension=config.SPECTRAL_MINIMUM_DIMENSION,
        compute_projections=True,
    )
    return {
        "spectral_dims": spectral_dims,
        "pca_projections": pca_projections,
        "pca_eigenvalues": pca_eigenvalues,
    }


def _build_child_k_dicts(
    tree: PosetTree,
    spectral_dims: Dict[str, int],
    mode: str,  # "min" or "max"
) -> Dict[str, int]:
    """Build a parent→k dict using min or max of children's spectral k.

    For parents without two binary children, returns the parent's own k.
    """
    fn = min if mode == "min" else max
    parent_k: Dict[str, int] = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) == 2:
            k_l = spectral_dims.get(children[0], 2)
            k_r = spectral_dims.get(children[1], 2)
            parent_k[parent] = max(fn(k_l, k_r), 1)  # floor at 1
        else:
            parent_k[parent] = spectral_dims.get(parent, 2)
    return parent_k


# ── Decomposition with injected sibling strategy ────────────────────────────


def decompose_with_strategy(
    tree: PosetTree,
    data_df: pd.DataFrame,
    strategy: str,
    spectral_info: Dict[str, Any],
) -> dict:
    """Run full decomposition, monkey-patching sibling spectral injection.

    strategy: "baseline" | "parent_spectral" | "min_child" | "max_child"
    """
    spectral_dims = spectral_info["spectral_dims"]
    pca_projections = spectral_info["pca_projections"]
    pca_eigenvalues = spectral_info["pca_eigenvalues"]

    # Build the sibling_spectral_dims / pca dicts for the strategy
    if strategy == "baseline":
        sib_dims = None
        sib_pca = None
        sib_eig = None

    elif strategy == "parent_spectral":
        # Parent's spectral k, PCA projections, and eigenvalues
        sib_dims = spectral_dims  # keyed by node → includes parents
        sib_pca = pca_projections
        sib_eig = pca_eigenvalues

    elif strategy == "min_child":
        # min(k_left, k_right) as projection dim, random projection (no PCA/whitening)
        sib_dims = _build_child_k_dicts(tree, spectral_dims, "min")
        sib_pca = None
        sib_eig = None

    elif strategy == "max_child":
        # max(k_left, k_right) as projection dim, random projection (no PCA/whitening)
        sib_dims = _build_child_k_dicts(tree, spectral_dims, "max")
        sib_pca = None
        sib_eig = None

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # Monkey-patch _prepare_annotations to inject our spectral dicts
    original_prepare = TreeDecomposition._prepare_annotations

    def patched_prepare(self_td, annotations_df):
        return run_gate_annotation_pipeline(
            self_td.tree,
            annotations_df,
            alpha_local=self_td.alpha_local,
            sibling_alpha=self_td.sibling_alpha,
            leaf_data=self_td._leaf_data,
            spectral_method=self_td._spectral_method,
            minimum_projection_dimension=self_td._resolved_minimum_projection_dimension,
            sibling_method=config.SIBLING_TEST_METHOD,
            fdr_method="tree_bh",
            sibling_spectral_dims=sib_dims,
            sibling_pca_projections=sib_pca,
            sibling_pca_eigenvalues=sib_eig,
        ).annotated_df

    TreeDecomposition._prepare_annotations = patched_prepare
    try:
        result = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )
    finally:
        TreeDecomposition._prepare_annotations = original_prepare

    return result


# ── Per-case runner ──────────────────────────────────────────────────────────


def run_case(case_name: str) -> list[dict]:
    """Run all four strategies for one case, return list of result rows."""
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")

    # Pre-compute spectral decomposition (shared across strategies)
    spectral_info = _build_spectral_dicts(tree, data_df)

    rows = []
    strategies = ["baseline", "parent_spectral", "min_child", "max_child"]

    for strat in strategies:
        # Need a fresh tree for each strategy (decompose mutates state)
        tree_copy, data_df_copy, _, _ = build_tree_and_data(case_name)
        # Re-use spectral info from the original tree (same topology)
        try:
            decomp = decompose_with_strategy(tree_copy, data_df_copy, strat, spectral_info)
            found_k = decomp["num_clusters"]
            ari = compute_ari(decomp, data_df_copy, y_t) if y_t is not None else float("nan")
        except Exception as e:
            found_k = None
            ari = None
            print(f"    ERROR [{strat}]: {e}")

        # Collect aggregate k stats for this strategy
        k_vals = list(spectral_info["spectral_dims"].values())
        rows.append(
            {
                "case": case_name,
                "true_k": true_k,
                "strategy": strat,
                "found_k": found_k,
                "ari": round(ari, 3) if ari is not None else None,
                "delta_k": (
                    (found_k - true_k) if found_k is not None and isinstance(true_k, int) else None
                ),
                "mean_spectral_k": round(np.mean(k_vals), 1) if k_vals else None,
            }
        )

    return rows


# ── Reporting ────────────────────────────────────────────────────────────────


def summarize(results_df: pd.DataFrame) -> None:
    """Print comprehensive comparison."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 100)
    print("  Per-case, per-strategy results")
    print("=" * 100)

    pivot_k = results_df.pivot(index="case", columns="strategy", values="found_k")
    pivot_ari = results_df.pivot(index="case", columns="strategy", values="ari")
    pivot_dk = results_df.pivot(index="case", columns="strategy", values="delta_k")

    # Add true_k column
    true_k_map = results_df.drop_duplicates("case").set_index("case")["true_k"]

    print("\n  Found K:")
    display_k = pivot_k.copy()
    display_k.insert(0, "true_k", true_k_map)
    print(display_k.to_string())

    print("\n  ARI:")
    display_ari = pivot_ari.copy()
    display_ari.insert(0, "true_k", true_k_map)
    print(display_ari.to_string())

    print("\n  Delta K (found - true):")
    display_dk = pivot_dk.copy()
    display_dk.insert(0, "true_k", true_k_map)
    print(display_dk.to_string())

    # Aggregate metrics per strategy
    print("\n" + "=" * 100)
    print("  Aggregate metrics by strategy")
    print("=" * 100)
    agg = results_df.groupby("strategy").agg(
        mean_ari=("ari", "mean"),
        median_ari=("ari", "median"),
        exact_k=("delta_k", lambda x: (x == 0).sum()),
        mean_abs_delta_k=("delta_k", lambda x: x.abs().mean()),
        over_split=("delta_k", lambda x: (x > 0).sum()),
        under_split=("delta_k", lambda x: (x < 0).sum()),
        k_eq_1=("found_k", lambda x: (x == 1).sum()),
    )
    agg["n_cases"] = results_df.groupby("strategy")["case"].nunique()
    print(agg.to_string())

    # Improvement over baseline
    print("\n" + "=" * 100)
    print("  Per-case ARI difference vs baseline")
    print("=" * 100)
    if "baseline" in pivot_ari.columns:
        for strat in ["parent_spectral", "min_child", "max_child"]:
            if strat in pivot_ari.columns:
                diff = pivot_ari[strat] - pivot_ari["baseline"]
                improved = (diff > 0.01).sum()
                degraded = (diff < -0.01).sum()
                unchanged = len(diff) - improved - degraded
                print(f"\n  {strat} vs baseline:")
                print(f"    Improved: {improved}  Degraded: {degraded}  Unchanged: {unchanged}")
                print(f"    Mean ARI delta: {diff.mean():+.3f}")
                # Show individual deltas
                for case_name in diff.index:
                    d = diff[case_name]
                    if abs(d) > 0.01:
                        marker = "+" if d > 0 else "-"
                        print(f"      {marker} {case_name}: {d:+.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    cases = FAILURE_CASES + REGRESSION_GUARD_CASES
    all_rows = []

    for i, name in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {name}", flush=True)
        rows = run_case(name)
        for r in rows:
            status = f"K={r['found_k']}" if r["found_k"] is not None else "ERR"
            if r["strategy"] != "baseline":
                print(f"    {r['strategy']:20s} → {status}  ARI={r['ari']}")
            else:
                print(f"    {r['strategy']:20s} → {status}  ARI={r['ari']}  (baseline)")
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    summarize(results_df)


if __name__ == "__main__":
    print("=" * 100)
    print("  EXPERIMENT 9: Sibling Projection Strategy Comparison")
    print("  Strategies: baseline / parent_spectral / min_child / max_child")
    print("=" * 100)
    main()
