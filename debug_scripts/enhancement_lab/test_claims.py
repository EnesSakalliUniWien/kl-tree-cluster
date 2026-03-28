"""Test claims from copilot-instructions.md against empirical data.

Claims to test:
  C1: "Under H₀, c is constant and independent of n and BL"
  C2: "Covariates log(BL_sum) and log(n_parent) confound signal strength
       with post-selection inflation"
  C3: "Intercept-only model is better because c is constant"
  C4: "Per-case power-law R² is high" (positive claim — should hold)
  C5: "Cross-case generalization fails" (positive claim — should hold)

Uses the pipeline directly, no permutations needed. Tests C1 by checking
whether T/k ratios from null-like pairs correlate with n_parent within
individual cases — especially gauss_null_small (pure null, K=1).

Usage:
    python debug_scripts/enhancement_lab/test_claims.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402

from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

# Cases: mix of null (K=1) and signal cases
TEST_CASES = [
    "gauss_null_small",  # Pure null — K=1, no real clusters
    "gauss_moderate_3c",  # Signal — K=3
    "gauss_noisy_3c",  # Signal — K=3
    "gauss_clear_small",  # Signal — K=3
    "binary_hard_4c",  # Signal — K=4
    "gauss_overlap_4c_med",  # Signal — K=4
    "binary_balanced_low_noise",  # Signal — K=4 (but found K=1)
]


def collect_pairs_for_case(case_name: str) -> list[dict]:
    """Run the pipeline and extract all sibling pair records."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    spectral_dims = derive_sibling_spectral_dims(tree, annotations_df)
    pca_proj, pca_eig = derive_sibling_pca_projections(annotations_df, spectral_dims)
    child_pca = derive_sibling_child_pca_projections(tree, annotations_df, spectral_dims)

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_proj,
        pca_eigenvalues=pca_eig,
        whitening=config.SIBLING_WHITENING,
    )

    rows = []
    for r in records:
        if r.degrees_of_freedom <= 0 or not np.isfinite(r.stat):
            continue
        ratio = r.stat / r.degrees_of_freedom

        # Get branch length for this pair
        bl_sum = r.branch_length_sum if r.branch_length_sum is not None else 0.0

        rows.append(
            {
                "case": case_name,
                "parent": r.parent,
                "n_parent": r.n_parent,
                "k": r.degrees_of_freedom,
                "T": r.stat,
                "T_over_k": ratio,
                "bl_sum": bl_sum,
                "edge_weight": r.edge_weight,
                "is_null_like": r.is_null_like,
                "is_gate2_blocked": r.is_gate2_blocked,
                "true_k": tc.get("n_clusters"),
                "found_k": decomp["num_clusters"],
            }
        )
    return rows


def test_claim_c1(all_rows: list[dict]) -> None:
    """C1: Under H₀, c is constant and independent of n and BL.

    Test: Within each case, check if T/k correlates with n_parent for
    null-like pairs. Focus especially on gauss_null_small (pure null).
    """
    print("\n" + "=" * 72)
    print("CLAIM C1: Under H₀, c is constant and independent of n and BL")
    print("=" * 72)

    for case in TEST_CASES:
        null_rows = [r for r in all_rows if r["case"] == case and r["is_null_like"]]
        if len(null_rows) < 3:
            print(f"  {case:<30s} null-like={len(null_rows):>3d} — too few")
            continue

        n_vals = np.array([r["n_parent"] for r in null_rows])
        ratios = np.array([r["T_over_k"] for r in null_rows])
        bl_vals = np.array([r["bl_sum"] for r in null_rows])

        rho_n, p_n = spearmanr(n_vals, ratios)
        rho_bl, p_bl = (
            spearmanr(bl_vals, ratios) if np.std(bl_vals) > 0 else (float("nan"), float("nan"))
        )

        marker = "***" if p_n < 0.05 else ""
        bl_marker = "***" if p_bl < 0.05 else ""

        median_ratio = float(np.median(ratios))
        cv = float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else 0
        ratio_range = float(np.max(ratios) - np.min(ratios))

        print(
            f"  {case:<30s} null={len(null_rows):>3d}  "
            f"T/k: med={median_ratio:.2f} CV={cv:.2f} range={ratio_range:.2f}  "
            f"ρ(n,T/k)={rho_n:+.3f} p={p_n:.4f}{marker}  "
            f"ρ(BL,T/k)={rho_bl:+.3f} p={p_bl:.4f}{bl_marker}"
        )

    # Special analysis: gauss_null_small is pure null — all nodes are null
    null_case_rows = [r for r in all_rows if r["case"] == "gauss_null_small"]
    if null_case_rows:
        print("\n  ── Detailed: gauss_null_small (pure null, K=1) ──")
        n_vals = np.array([r["n_parent"] for r in null_case_rows])
        ratios = np.array([r["T_over_k"] for r in null_case_rows])
        print(f"  All {len(null_case_rows)} pairs (null & focal, since true K=1):")
        rho, p = spearmanr(n_vals, ratios)
        print(f"    ρ(n_parent, T/k) = {rho:+.3f}, p = {p:.6f}")
        print(f"    T/k range: [{np.min(ratios):.2f}, {np.max(ratios):.2f}]")
        print("    T/k by n_parent strata:")
        for lo, hi in [(2, 5), (5, 10), (10, 30), (30, 100)]:
            mask = (n_vals >= lo) & (n_vals < hi)
            if mask.sum() > 0:
                print(
                    f"      n∈[{lo},{hi}): n={mask.sum()}, mean T/k={np.mean(ratios[mask]):.2f}, "
                    f"range=[{np.min(ratios[mask]):.2f}, {np.max(ratios[mask]):.2f}]"
                )

    print("\n  VERDICT: ", end="")
    # Check gauss_null_small specifically
    null_only = [r for r in all_rows if r["case"] == "gauss_null_small"]
    if len(null_only) >= 3:
        n_vals = np.array([r["n_parent"] for r in null_only])
        ratios = np.array([r["T_over_k"] for r in null_only])
        rho, p = spearmanr(n_vals, ratios)
        cv = float(np.std(ratios) / np.mean(ratios))
        if abs(rho) > 0.3 or cv > 0.3:
            print(f"CLAIM C1 IS FALSE. Even under pure null: ρ={rho:+.3f}, CV={cv:.2f}")
        else:
            print(f"CLAIM C1 IS PLAUSIBLE. Under pure null: ρ={rho:+.3f}, CV={cv:.2f}")


def test_claim_c2(all_rows: list[dict]) -> None:
    """C2: Covariates confound signal strength with post-selection inflation.

    Test: If this were true, among null-like pairs (where there's no signal),
    log(n_parent) should NOT predict T/k. If it DOES predict T/k even among
    null-like pairs, then the variation is real inflation heterogeneity,
    not confounding.
    """
    print("\n" + "=" * 72)
    print("CLAIM C2: Covariates confound signal with post-selection inflation")
    print("=" * 72)

    # Pool all null-like pairs across all cases
    null_rows = [r for r in all_rows if r["is_null_like"]]
    focal_rows = [r for r in all_rows if not r["is_null_like"]]

    if len(null_rows) >= 3:
        n_vals = np.array([r["n_parent"] for r in null_rows])
        ratios = np.array([r["T_over_k"] for r in null_rows])
        log_n = np.log(np.maximum(n_vals, 1))
        log_ratio = np.log(np.maximum(ratios, 1e-9))

        rho, p = spearmanr(log_n, log_ratio)
        print(
            f"  Null-like pairs (n={len(null_rows)}): ρ(log(n), log(T/k)) = {rho:+.3f}, p = {p:.6f}"
        )

        # Fit OLS: log(T/k) ~ a + b * log(n)
        A = np.column_stack([np.ones(len(log_n)), log_n])
        params, _, _, _ = np.linalg.lstsq(A, log_ratio, rcond=None)
        a, b = params
        ss_res = np.sum((log_ratio - A @ params) ** 2)
        ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  OLS fit: log(T/k) = {a:.3f} + {b:.3f} * log(n),  R² = {r2:.3f}")
        print(f"  Interpretation: slope b={b:.3f} means T/k ~ n^{b:.3f}")

    if len(focal_rows) >= 3:
        n_vals = np.array([r["n_parent"] for r in focal_rows])
        ratios = np.array([r["T_over_k"] for r in focal_rows])
        log_n = np.log(np.maximum(n_vals, 1))
        log_ratio = np.log(np.maximum(ratios, 1e-9))

        rho, p = spearmanr(log_n, log_ratio)
        print(
            f"\n  Focal pairs (n={len(focal_rows)}):    ρ(log(n), log(T/k)) = {rho:+.3f}, p = {p:.6f}"
        )

        A = np.column_stack([np.ones(len(log_n)), log_n])
        params, _, _, _ = np.linalg.lstsq(A, log_ratio, rcond=None)
        a, b = params
        ss_res = np.sum((log_ratio - A @ params) ** 2)
        ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  OLS fit: log(T/k) = {a:.3f} + {b:.3f} * log(n),  R² = {r2:.3f}")

    print("\n  CONFOUNDING TEST: If claim C2 is right, the regression slope b")
    print("  for null-like pairs should be ~0 (no n-dependence under H₀).")
    print("  If slope is significantly nonzero, then n-dependence is real")
    print("  inflation heterogeneity, not signal confounding.")

    if len(null_rows) >= 3:
        n_vals = np.array([r["n_parent"] for r in null_rows])
        ratios = np.array([r["T_over_k"] for r in null_rows])
        rho, p = spearmanr(n_vals, ratios)
        if p < 0.05 and abs(rho) > 0.2:
            print("\n  VERDICT: CLAIM C2 IS FALSE (or misleading). Null-like pairs show")
            print(f"  ρ(n, T/k) = {rho:+.3f} (p={p:.6f}). The n-dependence exists even")
            print("  WITHOUT signal. It's real inflation heterogeneity, not confounding.")
        else:
            print(f"\n  VERDICT: CLAIM C2 IS PLAUSIBLE. Null-like ρ = {rho:+.3f} (p={p:.4f})")


def test_claim_c3(all_rows: list[dict]) -> None:
    """C3: Intercept-only is better than regression.

    Test: Compute within-case R² of log(T/k) ~ a + b*log(n) for null-like
    pairs. If R² is consistently high, the claim that intercept-only is
    better is empirically questionable.
    """
    print("\n" + "=" * 72)
    print("CLAIM C3: Intercept-only model is better (c is constant)")
    print("=" * 72)

    for case in TEST_CASES:
        null_rows = [r for r in all_rows if r["case"] == case and r["is_null_like"]]
        if len(null_rows) < 5:
            print(f"  {case:<30s} null={len(null_rows):>3d} — too few for regression")
            continue

        n_vals = np.array([r["n_parent"] for r in null_rows])
        ratios = np.array([r["T_over_k"] for r in null_rows])
        log_n = np.log(np.maximum(n_vals, 1))
        log_ratio = np.log(np.maximum(ratios, 1e-9))

        # Fit: log(T/k) = a + b * log(n)
        A = np.column_stack([np.ones(len(log_n)), log_n])
        params, _, _, _ = np.linalg.lstsq(A, log_ratio, rcond=None)
        a, b = params
        pred = A @ params
        ss_res = np.sum((log_ratio - pred) ** 2)
        ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Intercept-only residual
        mean_pred = np.mean(log_ratio)
        ss_intercept = ss_tot  # by definition

        improvement = 1 - ss_res / ss_intercept if ss_intercept > 0 else 0

        print(
            f"  {case:<30s} null={len(null_rows):>3d}  "
            f"slope={b:+.3f}  R²={r2:.3f}  "
            f"residual reduction={improvement:.1%}"
        )

    print("\n  VERDICT: If R² > 0.3 in most cases, intercept-only is NOT optimal")
    print("  and a within-tree regression would capture real c(n) heterogeneity.")


def test_claim_c4_c5(all_rows: list[dict]) -> None:
    """C4: Per-case power-law R² is high.
    C5: Cross-case generalization fails.

    Test: Fit log(T/k) ~ a + b*log(n) within each case (C4),
    then do LOO: train on 6, predict on held-out (C5).
    """
    print("\n" + "=" * 72)
    print("CLAIMS C4 & C5: Per-case R² vs cross-case generalization")
    print("=" * 72)

    # Per-case fits
    case_models = {}
    for case in TEST_CASES:
        null_rows = [r for r in all_rows if r["case"] == case and r["is_null_like"]]
        if len(null_rows) < 5:
            continue

        n_vals = np.array([r["n_parent"] for r in null_rows])
        ratios = np.array([r["T_over_k"] for r in null_rows])
        log_n = np.log(np.maximum(n_vals, 1))
        log_ratio = np.log(np.maximum(ratios, 1e-9))

        A = np.column_stack([np.ones(len(log_n)), log_n])
        params, _, _, _ = np.linalg.lstsq(A, log_ratio, rcond=None)
        a, b = params
        pred = A @ params
        ss_res = np.sum((log_ratio - pred) ** 2)
        ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        alpha_c = math.exp(a)
        case_models[case] = {
            "a": a,
            "b": b,
            "r2": r2,
            "alpha": alpha_c,
            "n": len(null_rows),
            "log_n": log_n,
            "log_ratio": log_ratio,
        }

        print(
            f"  C4 {case:<30s}  c(n) = {alpha_c:.2f} × n^({b:+.3f})  R²={r2:.3f}  n={len(null_rows)}"
        )

    # LOO cross-case
    print("\n  C5: Leave-one-case-out cross-validation:")
    cases_with_models = list(case_models.keys())
    for held_out in cases_with_models:
        train_log_n = np.concatenate(
            [case_models[c]["log_n"] for c in cases_with_models if c != held_out]
        )
        train_log_ratio = np.concatenate(
            [case_models[c]["log_ratio"] for c in cases_with_models if c != held_out]
        )

        A_train = np.column_stack([np.ones(len(train_log_n)), train_log_n])
        params, _, _, _ = np.linalg.lstsq(A_train, train_log_ratio, rcond=None)

        test_log_n = case_models[held_out]["log_n"]
        test_log_ratio = case_models[held_out]["log_ratio"]
        A_test = np.column_stack([np.ones(len(test_log_n)), test_log_n])
        pred_test = A_test @ params

        ss_res = np.sum((test_log_ratio - pred_test) ** 2)
        ss_tot = np.sum((test_log_ratio - np.mean(test_log_ratio)) ** 2)
        r2_loo = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        within_r2 = case_models[held_out]["r2"]
        print(
            f"    {held_out:<30s}  within R²={within_r2:.3f}  LOO R²={r2_loo:+.3f}  "
            f"{'DEGRADED' if r2_loo < within_r2 - 0.1 else 'OK'}"
        )


def main():
    print("Collecting sibling pair data from pipeline...")
    all_rows = []
    for case in TEST_CASES:
        print(f"  {case}...", end=" ", flush=True)
        rows = collect_pairs_for_case(case)
        all_rows.extend(rows)
        null_count = sum(1 for r in rows if r["is_null_like"])
        print(f"{len(rows)} pairs ({null_count} null-like)")

    print(f"\nTotal: {len(all_rows)} pairs across {len(TEST_CASES)} cases")
    print(f"  Null-like: {sum(1 for r in all_rows if r['is_null_like'])}")
    print(f"  Focal:     {sum(1 for r in all_rows if not r['is_null_like'])}")

    test_claim_c1(all_rows)
    test_claim_c2(all_rows)
    test_claim_c3(all_rows)
    test_claim_c4_c5(all_rows)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        """
  C1 "c is constant under H₀": Check gauss_null_small verdict above.
     If T/k varies with n even for pure-null data, claim is FALSE.

  C2 "Covariates confound signal with inflation": Check null-like ρ above.
     If null-like pairs show significant ρ(n, T/k), the n-dependence
     is real heterogeneity, not signal confounding.

  C3 "Intercept-only is better": Check per-case R² above.
     If within-case R² >> 0, intercept-only leaves variance on the table.

  C4 "Per-case R² is high": Direct measurement above.
  C5 "Cross-case generalization fails": LOO R² vs within-case R² above.
"""
    )


if __name__ == "__main__":
    main()
