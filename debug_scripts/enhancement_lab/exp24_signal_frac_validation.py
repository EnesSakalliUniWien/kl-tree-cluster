"""Lab exp24: Signal-fraction adaptive projection dimension.

Previous experiments found lam12_frac = (λ₁+λ₂)/trace collapses on diffuse-
spectrum data (categorical, SBM, heavy overlap) where top-2 eigenvalues hold
only ~3% of variance → k≈2 → χ²(2) has no power → K=1.

This experiment tests signal_frac = sum(λᵢ > MP_upper) / trace — which captures
ALL signal eigenvalues, not just top-2.  For diffuse spectra with many modest
signal components, signal_frac >> lam12_frac.

Configurations:
  1. production       — Gate2=MP,             Gate3=min_child  (baseline)
  2. lam12_combined   — Gate2=lam12_frac_mp,  Gate3=lam12_frac_jl  (exp23 champ)
  3. sigfrac_combined — Gate2=sigfrac_mp,     Gate3=sigfrac_jl     (new)
  4. sigfrac_g2_only  — Gate2=sigfrac_mp,     Gate3=jl_floor_qrt
  5. sigfrac_g3_only  — Gate2=MP,             Gate3=sigfrac_jl
  6. hybrid_combined  — Gate2=max(lam12,sigfrac)×k, Gate3=max(lam12,sigfrac)×k
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
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

_leaf_data_cache: dict = {}


def _clamp_k(x, minimum=2):
    return max(minimum, int(round(x)))


def _mp_upper_bound(eigenvalues, n_samples, n_features):
    """Compute Marchenko-Pastur upper bound for noise eigenvalues."""
    pos = eigenvalues[eigenvalues > 0]
    sigma2 = float(np.median(pos)) if len(pos) > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    gamma = n_features / max(n_samples, 1)
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _lam12_frac(eigenvalues):
    """(λ₁+λ₂)/trace — top-2 variance fraction."""
    trace = float(np.sum(eigenvalues))
    lam1 = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
    lam2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    return (lam1 + lam2) / max(trace, 1e-12)


def _signal_frac(eigenvalues, n_samples, n_features):
    """sum(λᵢ for λᵢ > MP_upper) / trace — signal variance fraction."""
    trace = float(np.sum(eigenvalues))
    if trace <= 0:
        return 0.0
    mp_ub = _mp_upper_bound(eigenvalues, n_samples, n_features)
    if mp_ub <= 0:
        return 0.0
    signal_sum = float(np.sum(eigenvalues[eigenvalues > mp_ub]))
    return signal_sum / max(trace, 1e-12)


def _hybrid_frac(eigenvalues, n_samples, n_features):
    """max(lam12_frac, signal_frac) — picks whichever is larger."""
    return max(
        _lam12_frac(eigenvalues),
        _signal_frac(eigenvalues, n_samples, n_features),
    )


# ── Tree utilities ──────────────────────────────────────────────────────────


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
    result = eigendecompose_correlation_backend(data_matrix, compute_eigenvectors=False)
    if result is None:
        return None
    return result.eigenvalues, data_matrix.shape[0], result.active_feature_count


# ═════════════════════════════════════════════════════════════════════════════
# Gate 2 estimators
# ═════════════════════════════════════════════════════════════════════════════


def _gate2_lam12_frac_mp(eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1):
    """Gate 2: (λ₁+λ₂)/trace × k_MP."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    frac = _lam12_frac(ev)
    k = _clamp_k(frac * k_mp)
    k = max(k, int(minimum_projection_dimension))
    return min(k, int(n_features))


def _gate2_sigfrac_mp(eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1):
    """Gate 2: signal_frac × k_MP."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    frac = _signal_frac(ev, n_samples, n_features)
    k = _clamp_k(frac * k_mp)
    k = max(k, int(minimum_projection_dimension))
    return min(k, int(n_features))


def _gate2_hybrid_mp(eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1):
    """Gate 2: max(lam12, signal_frac) × k_MP."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    frac = _hybrid_frac(ev, n_samples, n_features)
    k = _clamp_k(frac * k_mp)
    k = max(k, int(minimum_projection_dimension))
    return min(k, int(n_features))


# ═════════════════════════════════════════════════════════════════════════════
# Gate 3 strategies
# ═════════════════════════════════════════════════════════════════════════════


def _derive_min_child(tree, annotated_df):
    """Production default: returns None → orchestrator falls back to min-child."""
    return None


def _derive_jl_floor_qrt(tree, annotated_df):
    """max(spectral_min, JL/4) — validated baseline."""
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None
    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        k_left = edge_spectral_dims.get(left, 0)
        k_right = edge_spectral_dims.get(right, 0)
        positive = [k for k in (k_left, k_right) if k > 0]
        if not positive:
            continue
        spectral_k = min(positive)
        n_left = tree.nodes[left].get("leaf_count", 1)
        n_right = tree.nodes[right].get("leaf_count", 1)
        n_parent = n_left + n_right
        dist = tree.nodes[left].get("distribution")
        n_features = len(dist) if dist is not None else 100
        jl_k = compute_jl_dim(n_parent, n_features)
        floor_k = max(1, jl_k // 4)
        sibling_dims[parent] = max(spectral_k, floor_k)
    return sibling_dims if sibling_dims else None


def _make_gate3_frac_strategy(frac_fn):
    """Factory: create a Gate 3 strategy from any fraction function.

    frac_fn(eigenvalues, n_samples, d_active) → float
    """

    def _derive(tree, annotated_df):
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
                data_L = _node_data(tree, left, leaf_data)
                data_R = _node_data(tree, right, leaf_data)
            except (KeyError, IndexError):
                continue

            data_P = np.vstack([data_L, data_R])
            eig_P = _eigendecompose(data_P)
            if eig_P is None:
                continue

            ev_P, ns_P, da_P = eig_P
            n_features = leaf_data.shape[1]
            n_L = len(_descendant_leaves(tree, left))
            n_R = len(_descendant_leaves(tree, right))
            n_P = n_L + n_R

            jl_k_val = compute_jl_dim(n_P, n_features)

            frac = frac_fn(ev_P, ns_P, da_P)
            k = _clamp_k(frac * jl_k_val)
            sibling_dims[parent] = k

        return sibling_dims if sibling_dims else None

    return _derive


# Concrete Gate 3 strategies
_derive_gate3_lam12 = _make_gate3_frac_strategy(lambda ev, ns, da: _lam12_frac(ev))
_derive_gate3_sigfrac = _make_gate3_frac_strategy(lambda ev, ns, da: _signal_frac(ev, ns, da))
_derive_gate3_hybrid = _make_gate3_frac_strategy(lambda ev, ns, da: _hybrid_frac(ev, ns, da))


# ═════════════════════════════════════════════════════════════════════════════
# 6 configurations
# ═════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "production": {
        "gate2": None,
        "gate3": _derive_min_child,
    },
    "lam12_combined": {
        "gate2": _gate2_lam12_frac_mp,
        "gate3": _derive_gate3_lam12,
    },
    "sigfrac_combined": {
        "gate2": _gate2_sigfrac_mp,
        "gate3": _derive_gate3_sigfrac,
    },
    "sigfrac_g2_only": {
        "gate2": _gate2_sigfrac_mp,
        "gate3": _derive_jl_floor_qrt,
    },
    "sigfrac_g3_only": {
        "gate2": None,
        "gate3": _derive_gate3_sigfrac,
    },
    "hybrid_combined": {
        "gate2": _gate2_hybrid_mp,
        "gate3": _derive_gate3_hybrid,
    },
}

CONFIG_NAMES = list(CONFIGS.keys())
N_CONFIGS = len(CONFIG_NAMES)

# Conflict cases from exp23 — we track these specially
CONFLICT_CASES = {
    "cat_highcard_20cat_4c",
    "gauss_extreme_noise_highd",
    "gauss_overlap_3c_small_q5",
    "cat_overlap_3cat_4c",
    "cat_clear_5cat_6c",
    "cat_highd_4cat_1000feat",
    "overlap_heavy_8c_large_feat",
    "overlap_heavy_4c_med_feat",
    "sbm_moderate",
    "overlap_heavy_4c_small_feat",
}


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


def run_case_config(case_name: str, cfg_name: str) -> dict:
    """Run one case with a specific Gate 2 + Gate 3 configuration."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)

    cfg = CONFIGS[cfg_name]
    override_kwargs = {
        "leaf_data_cache": _leaf_data_cache,
        "leaf_data": data_df,
        "sibling_dims": cfg["gate3"],
    }
    if cfg["gate2"] is not None:
        override_kwargs["gate2_estimator"] = cfg["gate2"]

    try:
        with temporary_experiment_overrides(**override_kwargs):
            decomp = tree.decompose(
                leaf_data=data_df,
                alpha_local=config.SIBLING_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
        ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
        return {
            "true_k": tc.get("n_clusters", "?"),
            "found_k": decomp["num_clusters"],
            "ari": round(ari, 4),
        }
    except Exception as exc:
        return {
            "true_k": tc.get("n_clusters", "?"),
            "found_k": "ERR",
            "ari": 0.0,
            "error": str(exc)[:80],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Fraction diagnostic — run once per case to compare frac values
# ═════════════════════════════════════════════════════════════════════════════


def diagnose_fractions(case_name: str) -> dict | None:
    """Compute lam12_frac vs signal_frac at the root split of a case."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
    children = list(tree.successors(root))
    if len(children) != 2:
        return None

    left, right = children

    try:
        data_L = _node_data(tree, left, data_df)
        data_R = _node_data(tree, right, data_df)
    except (KeyError, IndexError):
        return None

    data_P = np.vstack([data_L, data_R])
    eig_P = _eigendecompose(data_P)
    if eig_P is None:
        return None

    ev_P, ns_P, da_P = eig_P
    n_features = data_df.shape[1]
    n_P = data_P.shape[0]

    k_mp = marchenko_pastur_signal_count(ev_P, ns_P, da_P)
    jl_k = compute_jl_dim(n_P, n_features)
    mp_ub = _mp_upper_bound(ev_P, ns_P, da_P)

    frac_lam12 = _lam12_frac(ev_P)
    frac_signal = _signal_frac(ev_P, ns_P, da_P)
    frac_hybrid = _hybrid_frac(ev_P, ns_P, da_P)

    return {
        "n": n_P,
        "d": n_features,
        "k_mp": k_mp,
        "jl_k": jl_k,
        "mp_upper": round(mp_ub, 4),
        "frac_lam12": round(frac_lam12, 4),
        "frac_signal": round(frac_signal, 4),
        "frac_hybrid": round(frac_hybrid, 4),
        "k_lam12_g2": _clamp_k(frac_lam12 * k_mp),
        "k_sigfrac_g2": _clamp_k(frac_signal * k_mp),
        "k_lam12_g3": _clamp_k(frac_lam12 * jl_k),
        "k_sigfrac_g3": _clamp_k(frac_signal * jl_k),
        "k_hybrid_g3": _clamp_k(frac_hybrid * jl_k),
        "ratio": round(frac_signal / max(frac_lam12, 1e-12), 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    n_cases = len(case_names)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print("        SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)")
    print(f"\n═══ EXP24: Signal-Fraction Validation — {n_cases} cases × {N_CONFIGS} configs ═══")
    print("    1. production       = Gate2:MP           + Gate3:min_child")
    print("    2. lam12_combined   = Gate2:lam12×k_MP   + Gate3:lam12×JL")
    print("    3. sigfrac_combined = Gate2:sigfrac×k_MP  + Gate3:sigfrac×JL")
    print("    4. sigfrac_g2_only  = Gate2:sigfrac×k_MP  + Gate3:jl_floor_qrt")
    print("    5. sigfrac_g3_only  = Gate2:MP            + Gate3:sigfrac×JL")
    print("    6. hybrid_combined  = Gate2:hybrid×k_MP   + Gate3:hybrid×JL")
    print()

    # ── Phase 1: Fraction diagnostics (conflict cases) ──────────────────────

    print("═══ FRACTION DIAGNOSTICS (conflict cases) ═══\n")
    print(
        f"  {'Case':<40} {'n':>5} {'d':>5} {'k_MP':>5} "
        f"{'lam12':>7} {'sigfrac':>8} {'hybrid':>7} {'ratio':>6} "
        f"{'k_l12_g2':>8} {'k_sf_g2':>7} {'k_l12_g3':>8} {'k_sf_g3':>7}"
    )
    print("  " + "─" * 120)

    for case_name in sorted(CONFLICT_CASES):
        diag = diagnose_fractions(case_name)
        if diag is None:
            print(f"  {case_name:<40} — skipped")
            continue
        print(
            f"  {case_name:<40} {diag['n']:5d} {diag['d']:5d} {diag['k_mp']:5d} "
            f"{diag['frac_lam12']:7.4f} {diag['frac_signal']:8.4f} "
            f"{diag['frac_hybrid']:7.4f} {diag['ratio']:6.1f}x "
            f"{diag['k_lam12_g2']:8d} {diag['k_sigfrac_g2']:7d} "
            f"{diag['k_lam12_g3']:8d} {diag['k_sigfrac_g3']:7d}"
        )

    # ── Phase 2: Full benchmark ─────────────────────────────────────────────

    print(f"\n═══ FULL BENCHMARK — {n_cases} cases × {N_CONFIGS} configs ═══\n")

    results: dict[str, list[dict]] = {c: [] for c in CONFIG_NAMES}
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        is_conflict = "★" if case_name in CONFLICT_CASES else " "
        print(f" {is_conflict}[{i:3d}/{n_cases}] {case_name:<42}", end="", flush=True)
        t1 = time.time()
        for cn in CONFIG_NAMES:
            r = run_case_config(case_name, cn)
            results[cn].append(r)
        dt = time.time() - t1
        prod = results["production"][-1]["ari"]
        lam12 = results["lam12_combined"][-1]["ari"]
        sigfr = results["sigfrac_combined"][-1]["ari"]
        print(f" {dt:5.1f}s  prod={prod:.3f}  lam12={lam12:.3f}  sigfrac={sigfr:.3f}")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ── Aggregate ───────────────────────────────────────────────────────────

    print(f"═══ AGGREGATE ({n_cases} cases) ═══\n")
    print(
        f"  {'Config':<20} {'Mean ARI':>9} {'Med ARI':>9} "
        f"{'Exact K':>9} {'Perfect':>8} {'K=1':>5}"
    )
    print("  " + "─" * 75)

    for cn in CONFIG_NAMES:
        aris = [r["ari"] for r in results[cn] if isinstance(r["ari"], float)]
        exact_k = sum(
            1
            for r in results[cn]
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]
        )
        countable = sum(1 for r in results[cn] if r["true_k"] != "?")
        perfect = sum(1 for r in results[cn] if isinstance(r["ari"], float) and r["ari"] >= 0.999)
        k1 = sum(1 for r in results[cn] if isinstance(r["found_k"], int) and r["found_k"] == 1)
        mean_ari = float(np.mean(aris)) if aris else 0.0
        med_ari = float(np.median(aris)) if aris else 0.0
        print(
            f"  {cn:<20} {mean_ari:9.3f} {med_ari:9.3f} "
            f"{exact_k:5d}/{countable:<3d} {perfect:8d} {k1:5d}"
        )

    # ── Per-case detail ─────────────────────────────────────────────────────

    print("\n═══ PER-CASE DETAIL ═══\n")
    header = f"  {'Case':<42} {'TK':>4}"
    for cn in CONFIG_NAMES:
        short = cn[:8]
        header += f" │ {short:>12}"
    print(header)
    print("  " + "─" * (48 + 15 * N_CONFIGS))

    for ci, case_name in enumerate(case_names):
        is_conflict = "★" if case_name in CONFLICT_CASES else " "
        tk = results[CONFIG_NAMES[0]][ci]["true_k"]
        line = f" {is_conflict}{case_name:<41} {str(tk):>4}"
        for cn in CONFIG_NAMES:
            r = results[cn][ci]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>3} {ari:.2f}{check}"
        print(line)

    # ── Head-to-head: sigfrac_combined vs lam12_combined ────────────────────

    print("\n═══ HEAD-TO-HEAD: sigfrac_combined vs lam12_combined ═══\n")
    wins = losses = ties = 0
    delta_lines: list[tuple[str, float]] = []
    for ci, case_name in enumerate(case_names):
        r_sf = results["sigfrac_combined"][ci]
        r_l12 = results["lam12_combined"][ci]
        delta = r_sf["ari"] - r_l12["ari"]
        tag = "★" if case_name in CONFLICT_CASES else " "
        if abs(delta) <= 0.005:
            ties += 1
        elif delta > 0:
            wins += 1
            delta_lines.append(
                (
                    f" {tag}WIN  {case_name:<38} TK={r_sf['true_k']:>3}  "
                    f"sf: K={r_sf['found_k']:>3} {r_sf['ari']:.3f}  "
                    f"l12: K={r_l12['found_k']:>3} {r_l12['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )
        else:
            losses += 1
            delta_lines.append(
                (
                    f" {tag}LOSS {case_name:<38} TK={r_sf['true_k']:>3}  "
                    f"sf: K={r_sf['found_k']:>3} {r_sf['ari']:.3f}  "
                    f"l12: K={r_l12['found_k']:>3} {r_l12['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )

    for ln, _ in sorted(delta_lines, key=lambda x: -x[1]):
        print(ln)
    print(f"\n  sigfrac wins: {wins}, lam12 wins: {losses}, ties: {ties}")

    # ── Head-to-head: sigfrac_combined vs production ────────────────────────

    print("\n═══ HEAD-TO-HEAD: sigfrac_combined vs production ═══\n")
    wins2 = losses2 = ties2 = 0
    delta_lines2: list[tuple[str, float]] = []
    for ci, case_name in enumerate(case_names):
        r_sf = results["sigfrac_combined"][ci]
        r_p = results["production"][ci]
        delta = r_sf["ari"] - r_p["ari"]
        tag = "★" if case_name in CONFLICT_CASES else " "
        if abs(delta) <= 0.005:
            ties2 += 1
        elif delta > 0:
            wins2 += 1
            delta_lines2.append(
                (
                    f" {tag}WIN  {case_name:<38} TK={r_sf['true_k']:>3}  "
                    f"sf: K={r_sf['found_k']:>3} {r_sf['ari']:.3f}  "
                    f"prod: K={r_p['found_k']:>3} {r_p['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )
        else:
            losses2 += 1
            delta_lines2.append(
                (
                    f" {tag}LOSS {case_name:<38} TK={r_sf['true_k']:>3}  "
                    f"sf: K={r_sf['found_k']:>3} {r_sf['ari']:.3f}  "
                    f"prod: K={r_p['found_k']:>3} {r_p['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )

    for ln, _ in sorted(delta_lines2, key=lambda x: -x[1]):
        print(ln)
    print(f"\n  sigfrac wins: {wins2}, production wins: {losses2}, ties: {ties2}")

    # ── Head-to-head: hybrid_combined vs lam12_combined ─────────────────────

    print("\n═══ HEAD-TO-HEAD: hybrid_combined vs lam12_combined ═══\n")
    wins3 = losses3 = ties3 = 0
    delta_lines3: list[tuple[str, float]] = []
    for ci, case_name in enumerate(case_names):
        r_h = results["hybrid_combined"][ci]
        r_l12 = results["lam12_combined"][ci]
        delta = r_h["ari"] - r_l12["ari"]
        tag = "★" if case_name in CONFLICT_CASES else " "
        if abs(delta) <= 0.005:
            ties3 += 1
        elif delta > 0:
            wins3 += 1
            delta_lines3.append(
                (
                    f" {tag}WIN  {case_name:<38} TK={r_h['true_k']:>3}  "
                    f"hyb: K={r_h['found_k']:>3} {r_h['ari']:.3f}  "
                    f"l12: K={r_l12['found_k']:>3} {r_l12['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )
        else:
            losses3 += 1
            delta_lines3.append(
                (
                    f" {tag}LOSS {case_name:<38} TK={r_h['true_k']:>3}  "
                    f"hyb: K={r_h['found_k']:>3} {r_h['ari']:.3f}  "
                    f"l12: K={r_l12['found_k']:>3} {r_l12['ari']:.3f}  "
                    f"Δ={delta:+.3f}",
                    delta,
                )
            )

    for ln, _ in sorted(delta_lines3, key=lambda x: -x[1]):
        print(ln)
    print(f"\n  hybrid wins: {wins3}, lam12 wins: {losses3}, ties: {ties3}")

    # ── Conflict case spotlight ─────────────────────────────────────────────

    print("\n═══ CONFLICT CASE SPOTLIGHT ═══\n")
    print(
        f"  {'Case':<40} {'TK':>4} │ {'prod':>8} │ {'lam12':>8} │ "
        f"{'sigfrac':>8} │ {'hybrid':>8} │ {'sf_g2':>8} │ {'sf_g3':>8}"
    )
    print("  " + "─" * 108)

    for ci, case_name in enumerate(case_names):
        if case_name not in CONFLICT_CASES:
            continue
        tk = results["production"][ci]["true_k"]
        line = f"  {case_name:<40} {str(tk):>4}"
        for cn in [
            "production",
            "lam12_combined",
            "sigfrac_combined",
            "hybrid_combined",
            "sigfrac_g2_only",
            "sigfrac_g3_only",
        ]:
            r = results[cn][ci]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>2} {ari:.2f}{check}"
        print(line)

    # ── Best config per case ────────────────────────────────────────────────

    print("\n═══ BEST CONFIG PER CASE ═══\n")
    best_counts = {cn: 0 for cn in CONFIG_NAMES}
    for ci, case_name in enumerate(case_names):
        best_ari = -1.0
        best_cn = ""
        for cn in CONFIG_NAMES:
            a = results[cn][ci]["ari"]
            if a > best_ari:
                best_ari = a
                best_cn = cn
        best_counts[best_cn] += 1

    for cn in CONFIG_NAMES:
        bar = "█" * best_counts[cn]
        print(f"  {cn:<20} {best_counts[cn]:3d}  {bar}")
