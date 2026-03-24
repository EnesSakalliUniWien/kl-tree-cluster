"""Lab: diagnose WHY auto-derived sibling spectral dims cause regression.

Compares Gate 3 projection dimensions and sibling-test outcomes under
two modes:

  A) auto-derive (HEAD) — min-child spectral k from Gate 2 → Gate 3
  B) JL fallback       — sibling test uses JL-based projection dim

For each case, traces every binary parent node:
  - spectral_k from Gate 2 (per child)
  - min-child k passed to Gate 3
  - JL-based k (what the fallback would use)
  - sibling test p-value under both modes
  - whether Gate 3 decision flips (SPLIT → MERGE or MERGE → SPLIT)

This reveals the mechanism: spectral k is too small at nodes that span
cluster boundaries, causing low-power sibling tests → false MERGE.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
import pandas as pd
from lab_helpers import build_tree_and_data, compute_ari, temporary_experiment_overrides

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    _derive_sibling_spectral_dims,
    run_gate_annotation_pipeline,
)

# Cases with known regression from auto-derive
CASES = [
    "binary_balanced_low_noise__2",  # ARI: 0.000 → 1.000
    "gauss_clear_small",  # ARI: 0.554 → 1.000
    "binary_low_noise_12c",  # ARI: 0.614 → 1.000
    "binary_perfect_8c",  # ARI: 0.757 → 1.000
    "binary_hard_4c",  # ARI: 0.708 → 0.950
    "gauss_noisy_3c",  # ARI: 1.000 → 0.927 (reverse)
    "gauss_overlap_4c_med",  # ARI: 1.000 → 0.850 (reverse)
]


def get_jl_dim(n_samples: int, n_features: int) -> int:
    """What the JL fallback would compute."""
    return compute_projection_dimension(n_samples, n_features)


def run_annotation_pipeline(tree, data_df, use_auto_derive: bool):
    """Run Gate 2 + Gate 3 and return annotated DataFrame + spectral info."""
    annotations_df = tree.annotations_df.copy() if tree.annotations_df is not None else pd.DataFrame()

    if use_auto_derive:
        bundle = run_gate_annotation_pipeline(
            tree,
            annotations_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            leaf_data=data_df,
            spectral_method=config.SPECTRAL_METHOD,
            sibling_method=config.SIBLING_TEST_METHOD,
        )
    else:
        # Explicitly pass None to disable auto-derive
        bundle = run_gate_annotation_pipeline(
            tree,
            annotations_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            leaf_data=data_df,
            spectral_method=config.SPECTRAL_METHOD,
            sibling_method=config.SIBLING_TEST_METHOD,
            sibling_spectral_dims={},  # Empty dict → no spectral override per node
        )

    return bundle


def analyze_case(case_name: str) -> dict:
    """Deep trace of one case, comparing auto-derive vs JL mode."""
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)
    n_samples, n_features = data_df.shape
    true_k = tc.get("n_clusters", "?")

    # --- Mode A: Auto-derive (HEAD behavior) ---
    bundle_auto = run_annotation_pipeline(tree, data_df, use_auto_derive=True)
    df_auto = bundle_auto.annotated_df

    # Extract the spectral dims that Gate 2 produced
    edge_spectral_dims = df_auto.attrs.get("_spectral_dims", {})

    # Derive sibling dims (same logic as orchestrator)
    sibling_dims = _derive_sibling_spectral_dims(tree, df_auto) or {}

    # --- Mode B: JL fallback ---
    tree2, data_df2, _, _ = build_tree_and_data(case_name)
    bundle_jl = run_annotation_pipeline(tree2, data_df2, use_auto_derive=False)
    df_jl = bundle_jl.annotated_df

    # --- Compute ARIs ---
    decomp_auto = tree.decompose(
        leaf_data=data_df, alpha_local=config.SIBLING_ALPHA, sibling_alpha=config.SIBLING_ALPHA
    )
    tree3, data_df3, _, _ = build_tree_and_data(case_name)
    with temporary_experiment_overrides(sibling_dims=lambda t, d: None):
        decomp_jl = tree3.decompose(
            leaf_data=data_df3, alpha_local=config.SIBLING_ALPHA, sibling_alpha=config.SIBLING_ALPHA
        )

    ari_auto = compute_ari(decomp_auto, data_df, true_labels)
    ari_jl = compute_ari(decomp_jl, data_df3, true_labels)

    # --- Per-node trace ---
    node_traces = []
    jl_dim = get_jl_dim(n_samples, n_features)

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children

        # Gate 2 spectral dims per child
        k_left = edge_spectral_dims.get(left, 0)
        k_right = edge_spectral_dims.get(right, 0)
        sibling_k = sibling_dims.get(parent, None)

        # Gate 2 edge significance
        edge_sig_l = (
            df_auto.loc[left, "Child_Parent_Divergence_Significant"]
            if left in df_auto.index
            else None
        )
        edge_sig_r = (
            df_auto.loc[right, "Child_Parent_Divergence_Significant"]
            if right in df_auto.index
            else None
        )

        # Gate 3 sibling result under both modes
        sib_pval_auto = (
            df_auto.loc[parent, "Sibling_Divergence_P_Value_Corrected"]
            if parent in df_auto.index
            else None
        )
        sib_diff_auto = (
            df_auto.loc[parent, "Sibling_BH_Different"] if parent in df_auto.index else None
        )
        sib_skip_auto = (
            df_auto.loc[parent, "Sibling_Divergence_Skipped"] if parent in df_auto.index else None
        )
        sib_df_auto = (
            df_auto.loc[parent, "Sibling_Degrees_of_Freedom"] if parent in df_auto.index else None
        )

        sib_pval_jl = (
            df_jl.loc[parent, "Sibling_Divergence_P_Value_Corrected"]
            if parent in df_jl.index
            else None
        )
        sib_diff_jl = df_jl.loc[parent, "Sibling_BH_Different"] if parent in df_jl.index else None
        sib_df_jl = (
            df_jl.loc[parent, "Sibling_Degrees_of_Freedom"] if parent in df_jl.index else None
        )

        # Leaf counts
        n_desc_l = tree.nodes[left].get("leaf_count", 0) if left in tree.nodes else 0
        n_desc_r = tree.nodes[right].get("leaf_count", 0) if right in tree.nodes else 0

        # Decision flip?
        flip = ""
        if sib_diff_auto is not None and sib_diff_jl is not None:
            if bool(sib_diff_auto) != bool(sib_diff_jl):
                if bool(sib_diff_auto) and not bool(sib_diff_jl):
                    flip = "JL:MERGE→auto:SPLIT"
                else:
                    flip = "JL:SPLIT→auto:MERGE"

        node_traces.append(
            {
                "parent": parent,
                "n_desc_L": n_desc_l,
                "n_desc_R": n_desc_r,
                "edge_sig_L": edge_sig_l,
                "edge_sig_R": edge_sig_r,
                "spectral_k_L": k_left,
                "spectral_k_R": k_right,
                "sibling_k_auto": sibling_k,
                "jl_k": jl_dim,
                "sib_df_auto": sib_df_auto,
                "sib_df_jl": sib_df_jl,
                "sib_pval_auto": sib_pval_auto,
                "sib_pval_jl": sib_pval_jl,
                "sib_diff_auto": sib_diff_auto,
                "sib_diff_jl": sib_diff_jl,
                "sib_skip": sib_skip_auto,
                "flip": flip,
            }
        )

    return {
        "case": case_name,
        "true_k": true_k,
        "n_samples": n_samples,
        "n_features": n_features,
        "jl_dim": jl_dim,
        "k_auto": decomp_auto["num_clusters"],
        "k_jl": decomp_jl["num_clusters"],
        "ari_auto": round(ari_auto, 3),
        "ari_jl": round(ari_jl, 3),
        "node_traces": node_traces,
    }


def print_case_report(result: dict):
    """Print detailed per-case analysis."""
    print(f"\n{'='*90}")
    print(f"Case: {result['case']}")
    print(
        f"  True K={result['true_k']}, n={result['n_samples']}, p={result['n_features']}, JL dim={result['jl_dim']}"
    )
    print(f"  Auto-derive: K={result['k_auto']}, ARI={result['ari_auto']}")
    print(f"  JL fallback:  K={result['k_jl']}, ARI={result['ari_jl']}")

    traces = result["node_traces"]
    # Show only nodes with flips, or that have spectral dims set
    interesting = [t for t in traces if t["flip"] or t["sibling_k_auto"] is not None]
    if not interesting:
        interesting = traces[:5]

    # Sort: flips first, then by parent name
    interesting.sort(key=lambda t: (0 if t["flip"] else 1, t["parent"]))

    print(
        f"\n  {'Parent':<8} {'nL':>4} {'nR':>4} {'edgL':>4} {'edgR':>4} "
        f"{'kL':>3} {'kR':>3} {'k_sib':>5} {'JLk':>4} "
        f"{'df_a':>5} {'df_j':>5} {'p_auto':>8} {'p_jl':>8} "
        f"{'diff_a':>6} {'diff_j':>6} {'FLIP'}"
    )
    print(f"  {'-'*110}")

    for t in interesting:

        def fmt_p(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "  ---  "
            return f"{v:8.4f}"

        def fmt_b(v):
            if v is None:
                return " --- "
            return "  T  " if bool(v) else "  F  "

        def fmt_df(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return " --- "
            return f"{v:5.0f}"

        k_sib = str(t["sibling_k_auto"]) if t["sibling_k_auto"] is not None else " JL "
        print(
            f"  {t['parent']:<8} "
            f"{t['n_desc_L']:>4} {t['n_desc_R']:>4} "
            f"{fmt_b(t['edge_sig_L'])} {fmt_b(t['edge_sig_R'])} "
            f"{t['spectral_k_L']:>3} {t['spectral_k_R']:>3} {k_sib:>5} {t['jl_k']:>4} "
            f"{fmt_df(t['sib_df_auto'])} {fmt_df(t['sib_df_jl'])} "
            f"{fmt_p(t['sib_pval_auto'])} {fmt_p(t['sib_pval_jl'])} "
            f"{fmt_b(t['sib_diff_auto'])} {fmt_b(t['sib_diff_jl'])} "
            f"{t['flip']}"
        )

    # Count flips
    n_flips = sum(1 for t in traces if t["flip"])
    n_split_to_merge = sum(1 for t in traces if "SPLIT→auto:MERGE" in t["flip"])
    n_merge_to_split = sum(1 for t in traces if "MERGE→auto:SPLIT" in t["flip"])
    print(
        f"\n  Decision flips: {n_flips} total  "
        f"(JL:SPLIT→auto:MERGE: {n_split_to_merge}, JL:MERGE→auto:SPLIT: {n_merge_to_split})"
    )

    # Spectral k distribution
    all_ks = [t["sibling_k_auto"] for t in traces if t["sibling_k_auto"] is not None]
    if all_ks:
        print(
            f"  Spectral k distribution: min={min(all_ks)}, max={max(all_ks)}, "
            f"median={np.median(all_ks):.0f}, mean={np.mean(all_ks):.1f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print(f"        PROJECTION_MINIMUM_DIMENSION={config.PROJECTION_MINIMUM_DIMENSION}")

    all_results = []
    for name in CASES:
        try:
            result = analyze_case(name)
            all_results.append(result)
            print_case_report(result)
        except Exception as e:
            print(f"\n{'='*90}")
            print(f"Case: {name}  — ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(
        f"{'Case':<35} {'TK':>3} {'K_auto':>6} {'K_JL':>5} "
        f"{'ARI_a':>7} {'ARI_j':>7} {'delta':>7} {'flips':>5}"
    )
    print("-" * 85)
    for r in all_results:
        n_flips = sum(1 for t in r["node_traces"] if t["flip"])
        delta = r["ari_jl"] - r["ari_auto"]
        print(
            f"{r['case']:<35} {r['true_k']:>3} {r['k_auto']:>6} {r['k_jl']:>5} "
            f"{r['ari_auto']:>7.3f} {r['ari_jl']:>7.3f} {delta:>+7.3f} {n_flips:>5}"
        )
