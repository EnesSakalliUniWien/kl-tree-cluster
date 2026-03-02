#!/usr/bin/env python3
"""
Diagnostic analysis of over-splitting overlap benchmark cases.

Analyzes gate decisions, p-value distributions, calibration info,
and spectral dimensions for cases that severely over-split.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault("KL_TE_N_JOBS", "1")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

TARGET_CASES = {
    "gauss_overlap_3c_small",
    "overlap_heavy_4c_small_feat",
    "overlap_mod_4c_small",
}


def analyze_case(tc: dict) -> None:
    name = tc["name"]
    true_k = tc.get("n_clusters", "?")

    data, labels, _original, _meta = generate_case_data(tc)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Binarize continuous data
    if data.dtypes.apply(lambda d: d.kind == "f").any():
        data = (data > np.median(data.values, axis=0)).astype(int)

    n, d = data.shape
    print(f"\n{'=' * 70}")
    print(f"CASE: {name}  (n={n}, d={d}, true_K={true_k})")
    print(f"{'=' * 70}")

    # --- Build tree and decompose ---
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    results = tree.decompose(
        leaf_data=data,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    found_k = results["num_clusters"]
    print(f"Found K={found_k}")

    # --- Gate statistics ---
    df = tree.stats_df
    internal = df[~df.index.str.startswith("L")]

    # Gate 2: Edge significance
    g2_sig = internal["Child_Parent_Divergence_Significant"].sum()
    g2_total = internal["Child_Parent_Divergence_Significant"].count()
    print(
        f"\nGate 2 (Edge): {g2_sig}/{g2_total} significant ({100 * g2_sig / max(g2_total, 1):.1f}%)"
    )

    # Gate 3: Sibling divergence
    tested = internal[internal["Sibling_Divergence_Skipped"] == False]  # noqa: E712
    g3_diff = (
        tested["Sibling_BH_Different"].sum() if "Sibling_BH_Different" in tested.columns else 0
    )
    g3_same = tested["Sibling_BH_Same"].sum() if "Sibling_BH_Same" in tested.columns else 0
    g3_total = len(tested)
    print(f"Gate 3 (Sibling): {g3_diff} different, {g3_same} same out of {g3_total} tested")

    # --- Edge p-value distribution ---
    pvals = internal["Child_Parent_Divergence_P_Value_BH"].dropna()
    print(
        f"\nEdge BH p-values: min={pvals.min():.2e}, median={pvals.median():.2e}, max={pvals.max():.2e}"
    )
    print(
        f"  p < 0.05: {(pvals < 0.05).sum()}, p < 0.01: {(pvals < 0.01).sum()}, p < 0.001: {(pvals < 0.001).sum()}"
    )

    # --- Sibling p-value distribution ---
    sib_pvals = (
        tested["Sibling_Divergence_P_Value_Corrected"].dropna()
        if "Sibling_Divergence_P_Value_Corrected" in tested.columns
        else pd.Series(dtype=float)
    )
    if len(sib_pvals) > 0:
        print(
            f"\nSibling BH p-values: min={sib_pvals.min():.2e}, median={sib_pvals.median():.2e}, max={sib_pvals.max():.2e}"
        )
        print(f"  p < 0.05: {(sib_pvals < 0.05).sum()}, p >= 0.05: {(sib_pvals >= 0.05).sum()}")

    # --- Calibration audit ---
    if hasattr(df, "attrs") and "sibling_divergence_audit" in df.attrs:
        audit = df.attrs["sibling_divergence_audit"]
        c_hat = audit.get("global_c_hat", "?")
        c_hat_str = f"{c_hat:.3f}" if isinstance(c_hat, (int, float)) else str(c_hat)
        print(
            f"\nCalibration: method={audit.get('calibration_method', '?')}, "
            f"c_hat={c_hat_str}, n_cal={audit.get('calibration_n', '?')}"
        )
        diag = audit.get("diagnostics", {})
        if diag:
            eff_n = diag.get("effective_n", "?")
            r2 = diag.get("R2", "?")
            max_obs = audit.get("max_observed_ratio", "?")
            print(f"  effective_n={eff_n}, R2={r2}, max_obs_ratio={max_obs}")

    # --- Spectral dimensions ---
    if "Child_Parent_Divergence_df" in internal.columns:
        dfs = internal["Child_Parent_Divergence_df"].dropna()
        print(
            f"\nSpectral dims (edge df): min={dfs.min():.0f}, median={dfs.median():.0f}, max={dfs.max():.0f}, mean={dfs.mean():.1f}"
        )

    if "Sibling_Degrees_of_Freedom" in tested.columns:
        sib_df = tested["Sibling_Degrees_of_Freedom"].dropna()
        if len(sib_df) > 0:
            print(
                f"Sibling df: min={sib_df.min():.0f}, median={sib_df.median():.0f}, max={sib_df.max():.0f}, mean={sib_df.mean():.1f}"
            )

    # --- Cluster size distribution ---
    assignments = results["cluster_assignments"]
    sizes = sorted([v["size"] for v in assignments.values()], reverse=True)
    print(f"\nCluster sizes (top 10): {sizes[:10]}")
    print(f"  singletons: {sizes.count(1)}, pairs: {sizes.count(2)}")

    # --- Data characteristics ---
    col_means = data.mean(axis=0)
    print(
        f"\nData: col_mean range=[{col_means.min():.3f}, {col_means.max():.3f}], "
        f"overall_mean={col_means.mean():.3f}"
    )

    # Per-true-cluster signal check (if labels available)
    if labels is not None:
        unique_labels = np.unique(labels)
        print("\nPer-cluster column means (θ):")
        for lbl in unique_labels[:6]:
            mask = labels == lbl
            cluster_means = data.values[mask].mean(axis=0)
            print(
                f"  Cluster {lbl} (n={mask.sum()}): θ range=[{cluster_means.min():.3f}, {cluster_means.max():.3f}], "
                f"mean={cluster_means.mean():.3f}, std={cluster_means.std():.3f}"
            )


def main():
    all_cases = get_default_test_cases()
    cases = [c for c in all_cases if c["name"] in TARGET_CASES]

    if not cases:
        print(f"No cases found matching {TARGET_CASES}")
        sys.exit(1)

    print(f"Analyzing {len(cases)} overlap cases...")
    print(
        f"Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}, "
        f"FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}, "
        f"SPECTRAL_METHOD={config.SPECTRAL_METHOD}"
    )

    for tc in cases:
        analyze_case(tc)


if __name__ == "__main__":
    main()
