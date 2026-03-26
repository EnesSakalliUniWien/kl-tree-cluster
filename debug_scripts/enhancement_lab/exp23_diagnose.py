"""Exp23 diagnostic: Trace projection dimensions for conflict cases.

For each conflict case, traces Gate 2 and Gate 3 projection dimensions under
all 4 configurations to identify WHY the combined approach hurts.

Hypothesis: lam12_frac scales k by (λ₁+λ₂)/trace. When data has diffuse
eigenvalue spectrum (no dominant components), this fraction is low → k shrinks
→ under-powered tests → false merges → K=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data, temporary_experiment_overrides

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    marchenko_pastur as mp_module,
)

_orig_estimate_k = mp_module.estimate_k_marchenko_pastur
# ═════════════════════════════════════════════════════════════════════════════
# Conflict cases from exp23 interaction analysis
# ═════════════════════════════════════════════════════════════════════════════

CONFLICT_CASES = [
    # (case_name, combined_ARI, best_single_ARI, delta)
    ("cat_highcard_20cat_4c", 0.000, 0.511, -0.511),
    ("gauss_extreme_noise_highd", 0.000, 0.480, -0.480),
    ("gauss_overlap_3c_small_q5", 0.787, 0.980, -0.193),
    ("cat_overlap_3cat_4c", 0.697, 0.882, -0.185),
    ("cat_clear_5cat_6c", 0.819, 1.000, -0.181),
    ("cat_highd_4cat_1000feat", 0.978, 1.000, -0.022),
    ("overlap_heavy_8c_large_feat", 0.000, 0.024, -0.024),
    ("overlap_heavy_4c_med_feat", 0.281, 0.293, -0.012),
    ("sbm_moderate", 0.000, 0.011, -0.011),
    ("overlap_heavy_4c_small_feat", 0.035, 0.043, -0.009),
]


# ═════════════════════════════════════════════════════════════════════════════
# Instrumented Gate 2 estimator that logs every call
# ═════════════════════════════════════════════════════════════════════════════

_gate2_log: list[dict] = []


def _gate2_mp_instrumented(eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1):
    """Production MP estimator with logging."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k = _orig_estimate_k(
        ev,
        n_samples=n_samples,
        n_features=n_features,
        minimum_projection_dimension=minimum_projection_dimension,
    )
    trace = float(np.sum(ev))
    lam1 = float(ev[0]) if len(ev) > 0 else 0.0
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    frac = (lam1 + lam2) / max(trace, 1e-12)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    k_lam12 = max(2, int(round(frac * k_mp)))
    _gate2_log.append(
        {
            "n": n_samples,
            "d": n_features,
            "k_mp": k_mp,
            "k_returned": k,
            "lam1": round(lam1, 4),
            "lam2": round(lam2, 4),
            "trace": round(trace, 4),
            "frac": round(frac, 4),
            "k_lam12_would_be": k_lam12,
            "top5_ev": [round(float(ev[i]), 4) for i in range(min(5, len(ev)))],
        }
    )
    return k


def _gate2_lam12_instrumented(
    eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1
):
    """Lam12 estimator with logging."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    trace = float(np.sum(ev))
    lam1 = float(ev[0]) if len(ev) > 0 else 0.0
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    frac = (lam1 + lam2) / max(trace, 1e-12)
    k = max(2, int(round(frac * k_mp)))
    k = max(k, int(minimum_projection_dimension))
    k = min(k, int(n_features))
    _gate2_log.append(
        {
            "n": n_samples,
            "d": n_features,
            "k_mp": k_mp,
            "k_returned": k,
            "lam1": round(lam1, 4),
            "lam2": round(lam2, 4),
            "trace": round(trace, 4),
            "frac": round(frac, 4),
            "k_lam12_would_be": k,
            "top5_ev": [round(float(ev[i]), 4) for i in range(min(5, len(ev)))],
        }
    )
    return k


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
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


def _eigendecompose(data_matrix):
    if data_matrix.shape[0] < 2:
        return None
    result = eigendecompose_correlation_backend(data_matrix, compute_eigenvectors=False)
    if result is None:
        return None
    return result.eigenvalues, data_matrix.shape[0], result.active_feature_count


def analyze_case(case_name: str):
    """Deep analysis of eigenvalue structure and dimension decisions for a case."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")
    n_samples, n_features = data_df.shape

    print(f"\n{'═' * 80}")
    print(f"  CASE: {case_name}")
    print(f"  True K={true_k}, n={n_samples}, d={n_features}")
    print(f"{'═' * 80}")

    # ── Global eigenvalue analysis ──────────────────────────────────────────
    X = data_df.values
    eig = _eigendecompose(X)
    if eig is not None:
        ev, ns, da = eig
        k_mp = marchenko_pastur_signal_count(ev, ns, da)
        jl_k = compute_jl_dim(n_samples, n_features)
        trace = float(np.sum(ev))
        lam1 = float(ev[0]) if len(ev) > 0 else 0.0
        lam2 = float(ev[1]) if len(ev) > 1 else 0.0
        frac12 = (lam1 + lam2) / max(trace, 1e-12)

        print("\n  GLOBAL EIGENVALUE SPECTRUM (root node equivalent):")
        print(f"    k_MP={k_mp}, JL={jl_k}, JL/4={max(1, jl_k//4)}")
        print(f"    λ₁={lam1:.4f}, λ₂={lam2:.4f}, trace={trace:.4f}")
        print(f"    (λ₁+λ₂)/trace = {frac12:.4f}")
        print(
            f"    lam12 × k_MP = {frac12:.4f} × {k_mp} = {frac12 * k_mp:.1f} → k={max(2, int(round(frac12 * k_mp)))}"
        )
        print(
            f"    lam12 × JL   = {frac12:.4f} × {jl_k} = {frac12 * jl_k:.1f} → k={max(2, int(round(frac12 * jl_k)))}"
        )
        print(
            f"    Top 10 eigenvalues: {[round(float(ev[i]), 4) for i in range(min(10, len(ev)))]}"
        )

        # Cumulative variance
        cum_var = np.cumsum(ev) / trace
        for pct in [0.5, 0.8, 0.9, 0.95]:
            idx = np.searchsorted(cum_var, pct) + 1
            print(f"    Components for {pct*100:.0f}% variance: {idx}")
    else:
        print("  EIGENDECOMPOSITION FAILED")

    # ── Run all 4 configs with Gate 2 instrumentation ───────────────────────
    from exp23_combined_gate_validation import (
        CONFIG_NAMES,
        _derive_min_child,
        _leaf_data_cache,
        run_case_config,
    )

    print("\n  RESULTS ACROSS CONFIGS:")
    print(f"  {'Config':<18} {'K':>4} {'ARI':>7}")
    print(f"  {'─' * 32}")
    for cn in CONFIG_NAMES:
        r = run_case_config(case_name, cn)
        marker = (
            "✓"
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]
            else " "
        )
        print(f"  {cn:<18} K={r['found_k']:>3} {r['ari']:7.3f} {marker}")

    # ── Trace Gate 2 k values under MP vs lam12 ────────────────────────────
    print("\n  GATE 2 TRACE (per-node k values):")

    # Run with MP instrumented
    _gate2_log.clear()
    try:
        tree2, data_df2, _, _ = build_tree_and_data(case_name)
        with temporary_experiment_overrides(
            leaf_data_cache=_leaf_data_cache,
            leaf_data=data_df2,
            gate2_estimator=_gate2_mp_instrumented,
            sibling_dims=_derive_min_child,
        ):
            tree2.decompose(
                leaf_data=data_df2,
                alpha_local=config.SIBLING_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
    except Exception:
        pass
    mp_log = list(_gate2_log)

    # Run with lam12 instrumented
    _gate2_log.clear()
    try:
        tree3, data_df3, _, _ = build_tree_and_data(case_name)
        with temporary_experiment_overrides(
            leaf_data_cache=_leaf_data_cache,
            leaf_data=data_df3,
            gate2_estimator=_gate2_lam12_instrumented,
            sibling_dims=_derive_min_child,
        ):
            tree3.decompose(
                leaf_data=data_df3,
                alpha_local=config.SIBLING_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
    except Exception:
        pass
    lam12_log = list(_gate2_log)

    # Compare k distributions
    mp_ks = [e["k_returned"] for e in mp_log]
    lam12_ks = [e["k_returned"] for e in lam12_log]
    fracs = [e["frac"] for e in mp_log]

    if mp_ks:
        print(
            f"    MP   nodes: {len(mp_ks)}, k range: [{min(mp_ks)}, {max(mp_ks)}], "
            f"mean k: {np.mean(mp_ks):.1f}, median k: {np.median(mp_ks):.0f}"
        )
    if lam12_ks:
        print(
            f"    lam12 nodes: {len(lam12_ks)}, k range: [{min(lam12_ks)}, {max(lam12_ks)}], "
            f"mean k: {np.mean(lam12_ks):.1f}, median k: {np.median(lam12_ks):.0f}"
        )
    if fracs:
        print(
            f"    (λ₁+λ₂)/trace: min={min(fracs):.4f}, max={max(fracs):.4f}, "
            f"mean={np.mean(fracs):.4f}, median={np.median(fracs):.4f}"
        )

    # Show nodes where lam12 significantly reduces k
    print("\n    Nodes with largest k reduction (MP→lam12):")
    paired = list(zip(mp_log, lam12_log))
    if len(mp_log) != len(lam12_log):
        print(
            f"    WARNING: different number of gate2 calls (MP={len(mp_log)}, lam12={len(lam12_log)})"
        )
        paired = list(
            zip(
                mp_log[: min(len(mp_log), len(lam12_log))],
                lam12_log[: min(len(mp_log), len(lam12_log))],
            )
        )

    reductions = []
    for m, l in paired:
        delta = m["k_returned"] - l["k_returned"]
        if delta > 0:
            reductions.append((delta, m, l))
    reductions.sort(key=lambda x: -x[0])

    for delta, m, l in reductions[:8]:
        print(
            f"      n={m['n']:>4} d={m['d']:>5}  "
            f"MP k={m['k_returned']:>3} → lam12 k={l['k_returned']:>3}  "
            f"(Δ={-delta:+d})  frac={m['frac']:.4f}  "
            f"k_mp={m['k_mp']}"
        )

    # Nodes where lam12 INCREASES k (rare but interesting)
    increases = [
        (l["k_returned"] - m["k_returned"], m, l)
        for m, l in paired
        if l["k_returned"] > m["k_returned"]
    ]
    if increases:
        increases.sort(key=lambda x: -x[0])
        print("\n    Nodes where lam12 INCREASES k:")
        for delta, m, l in increases[:5]:
            print(
                f"      n={m['n']:>4} d={m['d']:>5}  "
                f"MP k={m['k_returned']:>3} → lam12 k={l['k_returned']:>3}  "
                f"(Δ={+delta:+d})  frac={m['frac']:.4f}"
            )

    # Fraction histogram
    if fracs:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(fracs, bins=bins)
        print("\n    (λ₁+λ₂)/trace distribution:")
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            bar = "█" * hist[i]
            print(f"      [{lo:.1f}, {hi:.1f}): {hist[i]:3d} {bar}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print("        SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)")
    print(f"\n═══ EXP23 DIAGNOSTIC: {len(CONFLICT_CASES)} conflict cases ═══")

    for case_name, comb_ari, best_ari, delta in CONFLICT_CASES:
        analyze_case(case_name)
