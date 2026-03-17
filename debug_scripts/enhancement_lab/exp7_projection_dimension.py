#!/usr/bin/env python3
"""Experiment 7 — Gate 3 projection dimension.

Gate 2 (edge test) uses Marchenko-Pastur spectral k (data-adaptive,
typically 1–10).  Gate 3 (sibling test) uses JL-based k ≈ 8·ln(n)/ε²
(typically 20–50).  This power imbalance means Gate 3 projects into a
much higher dimension, giving more df → higher expected T under null →
more spurious splits when the inflation calibration ĉ doesn't fully
compensate.

This experiment tests three interventions on Gate 3 projection:

  A.  JL epsilon sweep — vary config.PROJECTION_EPS to control JL k.
      Higher ε → fewer dimensions → less over-splitting risk.

  B.  Hard k cap — monkey-patch compute_projection_dimension_backend
      to clamp k ≤ cap.  Tests whether limiting df alone helps.

  C.  Spectral dims for Gate 3 — reuse the per-node spectral dimensions
      computed by Gate 2 (Marchenko-Pastur) as the sibling projection
      dimension.  Eliminates the power imbalance entirely.

Pass-through is always active (config.PASSTHROUGH = True).

Usage:
    python debug_scripts/enhancement_lab/exp7_projection_dimension.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    build_tree_and_data,
    compute_ari,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends import (
    random_projection_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    child_parent_projected_wald as _edge_wald_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (
    wald_statistic as _wald_statistic_module,
)

# ---------------------------------------------------------------------------
# Section A: JL epsilon sweep
# ---------------------------------------------------------------------------


def _decompose_with_eps(case_name: str, eps: float) -> dict:
    """Decompose with a custom PROJECTION_EPS (controls JL k for Gate 3)."""
    original_eps = config.PROJECTION_EPS
    # Clear projection cache so stale matrices from prior eps don't interfere
    random_projection_backend._PROJECTION_CACHE.clear()
    try:
        config.PROJECTION_EPS = eps
        tree, data_df, y_t, tc = build_tree_and_data(case_name)
        decomp = tree.decompose(leaf_data=data_df)
        true_k = tc.get("n_clusters", None)
        found_k = decomp["num_clusters"]
        ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

        # Report what k values were actually used
        stats = tree.stats_df
        sib_df_col = "Sibling_Degrees_of_Freedom"
        sib_dfs = (
            stats[sib_df_col].dropna() if sib_df_col in stats.columns else pd.Series(dtype=float)
        )
        mean_k = sib_dfs.mean() if len(sib_dfs) else float("nan")
        max_k = sib_dfs.max() if len(sib_dfs) else float("nan")

        return {
            "case": case_name,
            "eps": eps,
            "true_k": true_k,
            "found_k": found_k,
            "ari": round(ari, 3),
            "delta_k": found_k - (true_k or 0),
            "mean_sib_df": round(mean_k, 1),
            "max_sib_df": round(max_k, 1),
        }
    finally:
        config.PROJECTION_EPS = original_eps


def sweep_eps(cases: list[str], eps_values: list[float]) -> pd.DataFrame:
    """Sweep PROJECTION_EPS across cases."""
    rows: list[dict] = []
    total = len(cases) * len(eps_values)
    done = 0
    for case_name in cases:
        for eps in eps_values:
            done += 1
            try:
                rows.append(_decompose_with_eps(case_name, eps))
            except Exception as e:
                rows.append({"case": case_name, "eps": eps, "error": str(e)})
            if done % 5 == 0:
                print(f"  [{done}/{total}]", end="", flush=True)
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section B: Hard k cap
# ---------------------------------------------------------------------------


def _decompose_with_k_cap(case_name: str, k_cap: int) -> dict:
    """Decompose with a hard ceiling on the JL projection dimension."""
    _original_fn = random_projection_backend.compute_projection_dimension_backend

    def _capped_fn(n_samples: int, n_features: int, **kwargs):
        k = _original_fn(n_samples, n_features, **kwargs)
        return min(k, k_cap)

    # Patch at ALL import sites: the backend module AND the local aliases
    # in wald_statistic (Gate 3) and child_parent_projected_wald (Gate 2).
    with (
        patch.object(
            random_projection_backend,
            "compute_projection_dimension_backend",
            side_effect=_capped_fn,
        ),
        patch.object(
            _wald_statistic_module,
            "compute_projection_dimension",
            side_effect=_capped_fn,
        ),
        patch.object(
            _edge_wald_module,
            "compute_projection_dimension",
            side_effect=_capped_fn,
        ),
    ):
        tree, data_df, y_t, tc = build_tree_and_data(case_name)
        decomp = tree.decompose(leaf_data=data_df)
        true_k = tc.get("n_clusters", None)
        found_k = decomp["num_clusters"]
        ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

        stats = tree.stats_df
        sib_df_col = "Sibling_Degrees_of_Freedom"
        sib_dfs = (
            stats[sib_df_col].dropna() if sib_df_col in stats.columns else pd.Series(dtype=float)
        )
        mean_k = sib_dfs.mean() if len(sib_dfs) else float("nan")

        return {
            "case": case_name,
            "k_cap": k_cap,
            "true_k": true_k,
            "found_k": found_k,
            "ari": round(ari, 3),
            "delta_k": found_k - (true_k or 0),
            "mean_sib_df": round(mean_k, 1),
        }


def sweep_k_cap(cases: list[str], k_caps: list[int]) -> pd.DataFrame:
    """Sweep k cap across cases."""
    rows: list[dict] = []
    total = len(cases) * len(k_caps)
    done = 0
    for case_name in cases:
        for k_cap in k_caps:
            done += 1
            try:
                rows.append(_decompose_with_k_cap(case_name, k_cap))
            except Exception as e:
                rows.append({"case": case_name, "k_cap": k_cap, "error": str(e)})
            if done % 5 == 0:
                print(f"  [{done}/{total}]", end="", flush=True)
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section C: Spectral dims for Gate 3 (reuse Gate 2 Marchenko-Pastur)
# ---------------------------------------------------------------------------


def _decompose_with_spectral_sibling(case_name: str) -> dict:
    """Decompose using Gate 2's spectral dimensions for Gate 3.

    This feeds the Marchenko-Pastur per-node k from the edge gate
    directly into the sibling gate, eliminating the JL/spectral
    power imbalance.
    """
    from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

    tree, data_df, y_t, tc = build_tree_and_data(case_name)

    # Step 1: Run edge gate to get spectral dims
    from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
        resolve_minimum_projection_dimension_backend,
    )
    from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.edge_gate import (
        annotate_edge_gate,
    )

    min_proj = resolve_minimum_projection_dimension_backend(
        config.PROJECTION_MINIMUM_DIMENSION,
        leaf_data=data_df,
    )

    edge_bundle = annotate_edge_gate(
        tree,
        tree.stats_df.copy(),
        significance_level_alpha=config.EDGE_ALPHA,
        leaf_data=data_df,
        spectral_method=config.SPECTRAL_METHOD,
        minimum_projection_dimension=min_proj,
    )

    # Extract spectral dims computed by Gate 2
    spectral_dims = edge_bundle.annotated_df.attrs.get("_spectral_dims", None)

    # Step 2: Run sibling gate with spectral dims injected
    from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.sibling_gate import (
        annotate_sibling_gate,
    )

    sibling_bundle = annotate_sibling_gate(
        tree,
        edge_bundle.annotated_df,
        significance_level_alpha=config.SIBLING_ALPHA,
        sibling_method=config.SIBLING_TEST_METHOD,
        minimum_projection_dimension=min_proj,
        spectral_dims=spectral_dims,
    )

    annotated_df = sibling_bundle.annotated_df

    # Step 3: Build decomposer from the fully-annotated df (skip re-annotation)
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=annotated_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
    )
    tree.stats_df = decomposer.annotations_df
    result = decomposer.decompose_tree()

    true_k = tc.get("n_clusters", None)
    found_k = result["num_clusters"]
    ari = compute_ari(result, data_df, y_t) if y_t is not None else float("nan")

    # Stats on spectral dims used
    sib_df_col = "Sibling_Degrees_of_Freedom"
    stats = tree.stats_df
    sib_dfs = stats[sib_df_col].dropna() if sib_df_col in stats.columns else pd.Series(dtype=float)
    mean_k = sib_dfs.mean() if len(sib_dfs) else float("nan")
    max_k = sib_dfs.max() if len(sib_dfs) else float("nan")

    # Spectral dim stats (from Gate 2)
    if spectral_dims:
        spec_vals = list(spectral_dims.values())
        mean_spec = sum(spec_vals) / len(spec_vals) if spec_vals else float("nan")
        max_spec = max(spec_vals) if spec_vals else float("nan")
    else:
        mean_spec = max_spec = float("nan")

    return {
        "case": case_name,
        "method": "spectral_sibling",
        "true_k": true_k,
        "found_k": found_k,
        "ari": round(ari, 3),
        "delta_k": found_k - (true_k or 0),
        "mean_sib_df": round(mean_k, 1),
        "max_sib_df": round(max_k, 1),
        "mean_spectral_k": round(mean_spec, 1),
        "max_spectral_k": round(max_spec, 1),
    }


def _decompose_baseline(case_name: str) -> dict:
    """Standard baseline decomposition for comparison."""
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    decomp = tree.decompose(leaf_data=data_df)
    true_k = tc.get("n_clusters", None)
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

    stats = tree.stats_df
    sib_df_col = "Sibling_Degrees_of_Freedom"
    sib_dfs = stats[sib_df_col].dropna() if sib_df_col in stats.columns else pd.Series(dtype=float)
    mean_k = sib_dfs.mean() if len(sib_dfs) else float("nan")
    max_k = sib_dfs.max() if len(sib_dfs) else float("nan")

    return {
        "case": case_name,
        "method": "baseline_JL",
        "true_k": true_k,
        "found_k": found_k,
        "ari": round(ari, 3),
        "delta_k": found_k - (true_k or 0),
        "mean_sib_df": round(mean_k, 1),
        "max_sib_df": round(max_k, 1),
    }


def run_spectral_comparison(cases: list[str]) -> pd.DataFrame:
    """Compare baseline JL vs spectral-dim sibling on all cases."""
    rows: list[dict] = []
    for i, case_name in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case_name}", end="", flush=True)
        try:
            rows.append(_decompose_baseline(case_name))
            print(" [baseline]", end="", flush=True)
        except Exception as e:
            rows.append({"case": case_name, "method": "baseline_JL", "error": str(e)})

        try:
            rows.append(_decompose_with_spectral_sibling(case_name))
            print(" [spectral]", end="", flush=True)
        except Exception as e:
            rows.append({"case": case_name, "method": "spectral_sibling", "error": str(e)})
        print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 7: Gate 3 Projection Dimension")
    print("=" * 72)
    print(f"  Pass-through: ALWAYS ON (config.PASSTHROUGH={config.PASSTHROUGH})")

    cases = FAILURE_CASES + REGRESSION_GUARD_CASES

    # ── Section A: JL epsilon sweep ──
    print(f"\n{'=' * 72}")
    print("  Section A: JL Epsilon Sweep")
    print(f"{'=' * 72}")
    print("  Higher ε → fewer projection dims → fewer df → less over-splitting")

    eps_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # 0.3 = current default; must be in (0, 1)
    df_eps = sweep_eps(cases, eps_values)

    print("\n  Found K by eps:")
    pivot = df_eps.pivot_table(index="case", columns="eps", values="found_k", aggfunc="first")
    print(pivot.to_string())

    print("\n  ARI by eps:")
    pivot = df_eps.pivot_table(index="case", columns="eps", values="ari", aggfunc="first")
    print(pivot.to_string())

    print("\n  Mean Sibling df by eps:")
    pivot = df_eps.pivot_table(index="case", columns="eps", values="mean_sib_df", aggfunc="first")
    print(pivot.to_string())

    # ── Section B: Hard k cap ──
    print(f"\n{'=' * 72}")
    print("  Section B: Hard k Cap")
    print(f"{'=' * 72}")
    print("  Clamp JL dimension to at most k_cap (regardless of n_samples)")

    k_caps = [3, 5, 8, 10, 15, 20, 50]
    df_cap = sweep_k_cap(cases, k_caps)

    print("\n  Found K by k_cap:")
    pivot = df_cap.pivot_table(index="case", columns="k_cap", values="found_k", aggfunc="first")
    print(pivot.to_string())

    print("\n  ARI by k_cap:")
    pivot = df_cap.pivot_table(index="case", columns="k_cap", values="ari", aggfunc="first")
    print(pivot.to_string())

    # ── Section C: Spectral dims for Gate 3 ──
    print(f"\n{'=' * 72}")
    print("  Section C: Spectral Dims (MP) for Gate 3 Sibling Test")
    print(f"{'=' * 72}")
    print("  Reuses Gate 2's Marchenko-Pastur per-node k for Gate 3")
    print("  Eliminates the JL/spectral power imbalance\n")

    df_spec = run_spectral_comparison(cases)

    # Side-by-side comparison
    baseline = df_spec[df_spec["method"] == "baseline_JL"].set_index("case")
    spectral = df_spec[df_spec["method"] == "spectral_sibling"].set_index("case")

    comparison_rows = []
    for case_name in cases:
        row = {"case": case_name}
        if case_name in baseline.index:
            b = baseline.loc[case_name]
            row["true_k"] = b.get("true_k")
            row["baseline_k"] = b.get("found_k")
            row["baseline_ari"] = b.get("ari")
            row["baseline_mean_df"] = b.get("mean_sib_df")
        if case_name in spectral.index:
            s = spectral.loc[case_name]
            row["spectral_k"] = s.get("found_k")
            row["spectral_ari"] = s.get("ari")
            row["spectral_mean_df"] = s.get("mean_sib_df")
            row["spectral_mean_spec_k"] = s.get("mean_spectral_k")
        comparison_rows.append(row)

    df_compare = pd.DataFrame(comparison_rows)
    print("\n  Side-by-side comparison:")
    print(df_compare.to_string(index=False))

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")

    # Best eps
    if "error" not in df_eps.columns or not df_eps["error"].notna().all():
        valid_eps = df_eps.dropna(subset=["ari"])
        eps_summary = valid_eps.groupby("eps").agg(
            mean_ari=("ari", "mean"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_abs_delta=("delta_k", lambda x: x.abs().mean()),
        )
        print("\n  Epsilon sweep summary:")
        print(eps_summary.to_string())

    # Best k_cap
    if "error" not in df_cap.columns or not df_cap["error"].notna().all():
        valid_cap = df_cap.dropna(subset=["ari"])
        cap_summary = valid_cap.groupby("k_cap").agg(
            mean_ari=("ari", "mean"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_abs_delta=("delta_k", lambda x: x.abs().mean()),
        )
        print("\n  K cap sweep summary:")
        print(cap_summary.to_string())

    # Spectral comparison
    if "baseline_ari" in df_compare.columns and "spectral_ari" in df_compare.columns:
        bl_ari = df_compare["baseline_ari"].dropna()
        sp_ari = df_compare["spectral_ari"].dropna()
        print(
            f"\n  Baseline JL:       mean ARI = {bl_ari.mean():.3f}  "
            f"median = {bl_ari.median():.3f}"
        )
        print(
            f"  Spectral sibling:  mean ARI = {sp_ari.mean():.3f}  "
            f"median = {sp_ari.median():.3f}"
        )

        # Per-case winners
        both = df_compare.dropna(subset=["baseline_ari", "spectral_ari"])
        if len(both):
            spec_wins = (both["spectral_ari"] > both["baseline_ari"]).sum()
            bl_wins = (both["baseline_ari"] > both["spectral_ari"]).sum()
            ties = (both["baseline_ari"] == both["spectral_ari"]).sum()
            print(f"  Spectral wins: {spec_wins}  Baseline wins: {bl_wins}  Ties: {ties}")


if __name__ == "__main__":
    main()
