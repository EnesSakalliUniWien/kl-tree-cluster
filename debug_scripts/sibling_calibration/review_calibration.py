"""Multi-case calibration review: diagnoses the cousin-adjusted Wald pipeline.

Runs the full calibration chain (collect → interpolate → fit → pool → deflate)
across a representative set of benchmark cases and reports:

  1. Global inflation ĉ, effective_n, max_ratio, null-like/focal/blocked counts
  2. Pool geometry: center, bandwidth, bandwidth_status
  3. Local kernel c(k) profile: inflation factor at each unique structural_k
  4. Per-focal-pair: T, df, structural_k, c_local, T_adj, p_adj
  5. Pipeline K vs true K, ARI
  6. Cross-case summary table

Usage:
    python debug_scripts/sibling_calibration/review_calibration.py
    python debug_scripts/sibling_calibration/review_calibration.py --cases gauss_clear_small gauss_clear_medium
    python debug_scripts/sibling_calibration/review_calibration.py --all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    resolve_minimum_projection_dimension_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.conditional_deflation import (
    compute_pool_stats,
    predict_local_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    fit_inflation_model,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_null_prior_interpolation import (
    interpolate_sibling_null_priors,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
    count_null_focal_pairs,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ── Default representative cases ──────────────────────────────────────────────
# Covers: small/medium/large, binary/gaussian, different K, null, noisy
DEFAULT_CASE_NAMES = [
    "gauss_clear_small",      # n=30,  p=20, K=3  — small Gaussian, clear signal
    "gauss_clear_medium",     # n=60,  p=40, K=4  — medium Gaussian
    "gauss_clear_large",      # n=200, p=80, K=4  — large Gaussian
    "gauss_null_small",       # n=30,  p=20, K=1  — null case (no clusters)
    "gauss_moderate_3c",      # n=50,  p=30, K=3  — moderate noise
    "binary_perfect_4c",      # n=100, p=50, K=4  — binary, no noise
    "binary_low_noise_4c",    # n=100, p=50, K=4  — binary, low noise
    "binary_moderate_4c",     # n=100, p=50, K=4  — binary, moderate noise
    "sparse_features_72x72",  # n=72,  p=72, K=4  — sparse blocks
    "gauss_noisy_3c",         # n=50,  p=30, K=3  — heavy noise
]

ALPHA = 0.05

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_case_map() -> dict[str, dict]:
    """Build name → case dict from the benchmark registry."""
    cases = {}
    for case in get_default_test_cases():
        cases[case["name"]] = case
    return cases


def _build_tree_and_annotate(
    data_bin: pd.DataFrame,
) -> tuple[PosetTree, pd.DataFrame, list[SiblingPairRecord], list[str]]:
    """Build PosetTree, run Gate 2, collect sibling pair records."""
    Z = linkage(
        pdist(data_bin.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_bin.index.tolist())
    tree.populate_node_divergences(leaf_data=data_bin)

    min_proj_dim = resolve_minimum_projection_dimension_backend(
        config.PROJECTION_MINIMUM_DIMENSION,
        leaf_data=data_bin,
    )
    annotations_df = annotate_child_parent_divergence(
        tree,
        tree.annotations_df,
        significance_level_alpha=ALPHA,
        leaf_data=data_bin,
        minimum_projection_dimension=min_proj_dim,
    )
    spectral_dims = derive_sibling_spectral_dims(tree, annotations_df)
    pca_projections, pca_eigenvalues = derive_sibling_pca_projections(
        annotations_df, spectral_dims,
    )
    child_pca_projections = derive_sibling_child_pca_projections(
        tree, annotations_df, spectral_dims,
    )
    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=config.SIBLING_WHITENING,
    )
    return tree, annotations_df, records, non_binary


def _compute_ari(labels_true: np.ndarray, assignments: dict) -> float:
    """Compute ARI from decompose() cluster_assignments dict."""
    from sklearn.metrics import adjusted_rand_score

    n = len(labels_true)
    labels_pred = np.full(n, -1, dtype=int)
    for cid, info in assignments.items():
        for leaf in info["leaves"]:
            # leaf names are "S0", "S1", ... — extract index
            idx = int(leaf[1:]) if leaf.startswith("S") else -1
            if 0 <= idx < n:
                labels_pred[idx] = cid
    return float(adjusted_rand_score(labels_true, labels_pred))


# ── Per-case analysis ─────────────────────────────────────────────────────────

def analyze_case(
    case: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full calibration chain on a single case and return diagnostics."""

    name = case["name"]
    data_bin, labels, _, metadata = generate_case_data(case)
    n_samples, n_features = data_bin.shape
    true_k = case.get("n_clusters", int(len(np.unique(labels))))

    tree, ann_df, records, non_binary = _build_tree_and_annotate(data_bin)
    n_null, n_focal, n_blocked = count_null_focal_pairs(records)
    total_pairs = len(records)

    # Interpolation
    if n_blocked > 0:
        records = interpolate_sibling_null_priors(records, tree, ann_df)

    # Fit inflation model
    model = fit_inflation_model(records)
    c_global = model.global_inflation_factor
    max_ratio = model.max_observed_ratio
    diag = model.diagnostics or {}
    effective_n = diag.get("effective_n", 0.0)

    # Pool stats
    pool = compute_pool_stats(records, model)

    # Collect T/df ratios and structural dims
    all_ratios = []
    all_struct_k = []
    all_priors = []
    for r in records:
        if np.isfinite(r.stat) and r.degrees_of_freedom > 0:
            ratio = r.stat / r.degrees_of_freedom
            if ratio > 0:
                all_ratios.append(ratio)
                sk = r.structural_dimension if np.isfinite(r.structural_dimension) and r.structural_dimension > 0 else r.degrees_of_freedom
                all_struct_k.append(sk)
                all_priors.append(r.sibling_null_prior_from_edge_pvalue)

    # Local kernel profile: c(k) at unique structural dimensions
    unique_k = sorted(set(all_struct_k))
    c_local_profile = {}
    for k in unique_k:
        c_local_profile[k] = predict_local_inflation_factor(pool, k)

    # Focal pair deflation
    focal_details = []
    for r in records:
        if r.is_null_like:
            continue
        sk = r.structural_dimension if np.isfinite(r.structural_dimension) and r.structural_dimension > 0 else r.degrees_of_freedom
        c_local = predict_local_inflation_factor(pool, sk)
        t_adj = r.stat / c_local if c_local > 0 else r.stat
        p_adj = float(chi2.sf(t_adj, df=r.degrees_of_freedom)) if r.degrees_of_freedom > 0 else float("nan")
        focal_details.append({
            "parent": r.parent,
            "T": r.stat,
            "df": r.degrees_of_freedom,
            "struct_k": sk,
            "c_local": c_local,
            "T_adj": t_adj,
            "p_adj": p_adj,
            "prior": r.sibling_null_prior_from_edge_pvalue,
            "n_parent": r.n_parent,
        })

    # Run full pipeline to get K and ARI
    result = tree.decompose(leaf_data=data_bin, alpha_local=ALPHA, sibling_alpha=ALPHA)
    pipeline_k = result["num_clusters"]
    ari = _compute_ari(labels, result["cluster_assignments"]) if labels is not None and len(np.unique(labels)) > 1 else float("nan")

    # Also extract pipeline audit
    sdf = tree.annotations_df
    audit = sdf.attrs.get("sibling_divergence_audit", {})

    summary = {
        "name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "true_k": true_k,
        "pipeline_k": pipeline_k,
        "ari": ari,
        "total_pairs": total_pairs,
        "n_null": n_null,
        "n_focal": n_focal,
        "n_blocked": n_blocked,
        "c_global": c_global,
        "effective_n": effective_n,
        "max_ratio": max_ratio,
        "pool_center_k": pool.geometric_mean_structural_dimension,
        "pool_bandwidth": pool.bandwidth_log_structural_dimension,
        "pool_bw_status": pool.bandwidth_status,
        "pool_n_records": pool.n_records,
        "c_local_profile": c_local_profile,
        "focal_details": focal_details,
        "ratio_stats": {
            "min": float(np.min(all_ratios)) if all_ratios else float("nan"),
            "median": float(np.median(all_ratios)) if all_ratios else float("nan"),
            "mean": float(np.mean(all_ratios)) if all_ratios else float("nan"),
            "max": float(np.max(all_ratios)) if all_ratios else float("nan"),
            "std": float(np.std(all_ratios)) if all_ratios else float("nan"),
        },
        "struct_k_stats": {
            "min": float(np.min(all_struct_k)) if all_struct_k else float("nan"),
            "median": float(np.median(all_struct_k)) if all_struct_k else float("nan"),
            "max": float(np.max(all_struct_k)) if all_struct_k else float("nan"),
            "unique_count": len(unique_k),
        },
        "prior_stats": {
            "min": float(np.min(all_priors)) if all_priors else float("nan"),
            "median": float(np.median(all_priors)) if all_priors else float("nan"),
            "mean": float(np.mean(all_priors)) if all_priors else float("nan"),
            "max": float(np.max(all_priors)) if all_priors else float("nan"),
        },
    }

    if verbose:
        _print_case_report(summary)

    return summary


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_case_report(s: dict) -> None:
    """Print a detailed single-case report."""
    print()
    print("=" * 100)
    print(f"CASE: {s['name']}  (n={s['n_samples']}, p={s['n_features']}, true_K={s['true_k']})")
    print("=" * 100)

    # ── Pipeline result ──
    k_match = "✓" if s["pipeline_k"] == s["true_k"] else "✗"
    print(f"  Pipeline K = {s['pipeline_k']}  (true K = {s['true_k']})  {k_match}    ARI = {s['ari']:.4f}")

    # ── Pair counts ──
    print(f"\n  Pairs: {s['total_pairs']} total  |  null-like: {s['n_null']}  |  focal: {s['n_focal']}  |  blocked: {s['n_blocked']}")
    null_frac = s["n_null"] / s["total_pairs"] * 100 if s["total_pairs"] > 0 else 0
    focal_frac = s["n_focal"] / s["total_pairs"] * 100 if s["total_pairs"] > 0 else 0
    print(f"         null-like fraction: {null_frac:.1f}%    focal fraction: {focal_frac:.1f}%")

    # ── Global inflation ──
    print("\n  Global inflation:")
    print(f"    ĉ_global      = {s['c_global']:.6f}")
    print(f"    effective_n    = {s['effective_n']:.1f}")
    print(f"    max_ratio      = {s['max_ratio']:.4f}")

    # ── T/df ratio distribution ──
    rs = s["ratio_stats"]
    print("\n  T/df ratio distribution (all valid pairs):")
    print(f"    min={rs['min']:.4f}  median={rs['median']:.4f}  mean={rs['mean']:.4f}  max={rs['max']:.4f}  std={rs['std']:.4f}")

    # ── Structural dimension ──
    ks = s["struct_k_stats"]
    print("\n  Structural dimension (k):")
    print(f"    min={ks['min']:.0f}  median={ks['median']:.0f}  max={ks['max']:.0f}  unique values: {ks['unique_count']}")

    # ── Prior distribution ──
    ps = s["prior_stats"]
    print("\n  Sibling null prior distribution:")
    print(f"    min={ps['min']:.6f}  median={ps['median']:.6f}  mean={ps['mean']:.6f}  max={ps['max']:.6f}")

    # ── Pool stats ──
    print("\n  Pool stats:")
    print(f"    center (geom_mean_k) = {s['pool_center_k']:.4f}")
    print(f"    bandwidth (log_k)    = {s['pool_bandwidth']:.6f}")
    print(f"    bandwidth_status     = {s['pool_bw_status']}")
    print(f"    n_records            = {s['pool_n_records']}")

    # ── Local kernel c(k) profile ──
    profile = s["c_local_profile"]
    if profile:
        print("\n  Local kernel c(k) profile:")
        print(f"    {'k':>8} {'c_local':>10} {'c/c_global':>12}")
        print(f"    {'─'*8} {'─'*10} {'─'*12}")
        for k in sorted(profile.keys()):
            c = profile[k]
            ratio = c / s["c_global"] if s["c_global"] > 0 else float("nan")
            bar = "█" * min(int(c / max(profile.values()) * 30), 30) if max(profile.values()) > 0 else ""
            print(f"    {k:8.1f} {c:10.4f} {ratio:12.4f}  {bar}")
        c_vals = list(profile.values())
        c_range = max(c_vals) / min(c_vals) if min(c_vals) > 0 else float("inf")
        print(f"    c_local range: {min(c_vals):.4f} → {max(c_vals):.4f}  (ratio: {c_range:.1f}×)")

    # ── Focal pair details ──
    focal = s["focal_details"]
    if focal:
        print(f"\n  Focal pair deflation ({len(focal)} pairs):")
        print(f"    {'Parent':>8} {'T':>10} {'df':>6} {'sk':>6} {'c_local':>8} {'T_adj':>10} {'p_adj':>10} {'prior':>8} {'n_par':>6}")
        print(f"    {'─'*8} {'─'*10} {'─'*6} {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6}")
        for d in sorted(focal, key=lambda x: x["T"], reverse=True):
            sig = "*" if d["p_adj"] < ALPHA else " "
            print(
                f"    {d['parent']:>8} {d['T']:10.4f} {d['df']:6.1f} {d['struct_k']:6.1f} "
                f"{d['c_local']:8.4f} {d['T_adj']:10.4f} {d['p_adj']:10.6f}{sig} "
                f"{d['prior']:8.4f} {d['n_parent']:6}"
            )

        # Deflation impact: how many pairs are significant before vs after
        n_sig_raw = sum(1 for d in focal if chi2.sf(d["T"], df=d["df"]) < ALPHA)
        n_sig_adj = sum(1 for d in focal if d["p_adj"] < ALPHA)
        print(f"\n    Sig before deflation: {n_sig_raw}/{len(focal)}")
        print(f"    Sig after deflation:  {n_sig_adj}/{len(focal)}")
        print(f"    Deflation killed:     {n_sig_raw - n_sig_adj} significances")


def _print_cross_case_summary(summaries: list[dict]) -> None:
    """Print a cross-case comparison table."""
    print()
    print("=" * 140)
    print("CROSS-CASE SUMMARY")
    print("=" * 140)
    print(
        f"  {'Case':<30} {'n':>5} {'p':>5} {'K*':>3} {'K':>3} {'ARI':>6} "
        f"{'pairs':>5} {'null%':>5} {'focal':>5} {'blk':>4} "
        f"{'ĉ_glob':>7} {'eff_n':>6} {'max_R':>8} "
        f"{'bw':>6} {'c_min':>6} {'c_max':>6} {'c_×':>5}"
    )
    print("  " + "─" * 136)

    k_exact = 0
    total = 0
    for s in summaries:
        total += 1
        if s["pipeline_k"] == s["true_k"]:
            k_exact += 1
        null_pct = s["n_null"] / s["total_pairs"] * 100 if s["total_pairs"] > 0 else 0
        profile = s["c_local_profile"]
        c_vals = list(profile.values()) if profile else [0]
        c_min = min(c_vals) if c_vals else 0
        c_max = max(c_vals) if c_vals else 0
        c_range = c_max / c_min if c_min > 0 else float("inf")
        k_match = "✓" if s["pipeline_k"] == s["true_k"] else " "
        print(
            f"  {s['name']:<30} {s['n_samples']:5} {s['n_features']:5} {s['true_k']:3} "
            f"{s['pipeline_k']:3}{k_match} {s['ari']:6.3f} "
            f"{s['total_pairs']:5} {null_pct:5.1f} {s['n_focal']:5} {s['n_blocked']:4} "
            f"{s['c_global']:7.3f} {s['effective_n']:6.1f} {s['max_ratio']:8.2f} "
            f"{s['pool_bandwidth']:6.3f} {c_min:6.2f} {c_max:6.2f} {c_range:5.1f}"
        )

    print("  " + "─" * 136)
    print(f"  Exact K: {k_exact}/{total}")
    aris = [s["ari"] for s in summaries if np.isfinite(s["ari"])]
    if aris:
        print(f"  Mean ARI: {np.mean(aris):.4f}   Median ARI: {np.median(aris):.4f}")

    # Calibration quality indicators
    print("\n  Calibration diagnostics:")
    c_globals = [s["c_global"] for s in summaries]
    print(f"    ĉ_global range:   {min(c_globals):.3f} – {max(c_globals):.3f}")
    eff_ns = [s["effective_n"] for s in summaries]
    print(f"    effective_n range: {min(eff_ns):.1f} – {max(eff_ns):.1f}")
    bws = [s["pool_bandwidth"] for s in summaries if s["pool_bandwidth"] > 0]
    if bws:
        print(f"    bandwidth range:  {min(bws):.4f} – {max(bws):.4f}")
    max_rs = [s["max_ratio"] for s in summaries]
    print(f"    max_ratio range:  {min(max_rs):.2f} – {max(max_rs):.2f}")

    # Over/under-splitting diagnosis
    over_split = [s for s in summaries if s["pipeline_k"] > s["true_k"] * 1.5]
    under_split = [s for s in summaries if s["pipeline_k"] < s["true_k"] and s["true_k"] > 1]
    if over_split:
        print(f"\n  Over-splitting cases ({len(over_split)}):")
        for s in over_split:
            print(f"    {s['name']}: K={s['pipeline_k']} vs K*={s['true_k']},  ĉ={s['c_global']:.3f}, c_max={max(s['c_local_profile'].values()) if s['c_local_profile'] else 0:.3f}")
    if under_split:
        print(f"\n  Under-splitting cases ({len(under_split)}):")
        for s in under_split:
            print(f"    {s['name']}: K={s['pipeline_k']} vs K*={s['true_k']},  ĉ={s['c_global']:.3f}, eff_n={s['effective_n']:.1f}")

    # Null case analysis
    null_cases = [s for s in summaries if s["true_k"] == 1]
    if null_cases:
        print("\n  Null cases (true_K=1):")
        for s in null_cases:
            status = "✓ K=1" if s["pipeline_k"] == 1 else f"✗ K={s['pipeline_k']} (false positive)"
            print(f"    {s['name']}: {status},  ĉ={s['c_global']:.3f}, null%={s['n_null']/s['total_pairs']*100:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-case calibration review")
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Case names to analyze (default: curated set of 10)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available benchmark cases",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-case detail, show only cross-case summary",
    )
    args = parser.parse_args()

    case_map = _build_case_map()

    if args.all:
        case_names = list(case_map.keys())
    elif args.cases:
        case_names = args.cases
    else:
        case_names = [n for n in DEFAULT_CASE_NAMES if n in case_map]

    # Validate
    missing = [n for n in case_names if n not in case_map]
    if missing:
        print(f"Unknown cases: {missing}", file=sys.stderr)
        print(f"Available: {sorted(case_map.keys())}", file=sys.stderr)
        sys.exit(1)

    print("╔" + "═" * 98 + "╗")
    print(f"║  CALIBRATION REVIEW: {len(case_names)} cases" + " " * (98 - 24 - len(str(len(case_names)))) + "║")
    print("╚" + "═" * 98 + "╝")
    print("\n  Config:")
    print(f"    SIBLING_TEST_METHOD         = {config.SIBLING_TEST_METHOD}")
    print(f"    SIBLING_WHITENING           = {config.SIBLING_WHITENING}")
    print(f"    FELSENSTEIN_SCALING          = {config.FELSENSTEIN_SCALING}")
    print(f"    SPECTRAL_MINIMUM_DIMENSION  = {config.SPECTRAL_MINIMUM_DIMENSION}")
    print(f"    PROJECTION_MINIMUM_DIMENSION= {config.PROJECTION_MINIMUM_DIMENSION}")
    print(f"    EDGE_ALPHA                  = {config.EDGE_ALPHA}")
    print(f"    SIBLING_ALPHA               = {config.SIBLING_ALPHA}")
    print(f"    Alpha (this script)         = {ALPHA}")

    summaries = []
    for i, name in enumerate(case_names, 1):
        if not args.quiet:
            print(f"\n{'▸'*3} [{i}/{len(case_names)}] {name}")
        try:
            s = analyze_case(case_map[name], verbose=not args.quiet)
            summaries.append(s)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    if len(summaries) > 1:
        _print_cross_case_summary(summaries)

    print(f"\nDone. Analyzed {len(summaries)}/{len(case_names)} cases.")


if __name__ == "__main__":
    main()
