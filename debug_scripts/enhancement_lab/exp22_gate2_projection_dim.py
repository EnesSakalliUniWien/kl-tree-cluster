"""Lab exp22: Gate 2 (edge test) projection dimension strategies.

Mirrors the Gate 3 investigation (exp15-21) but for the EDGE test.
Gate 2 currently uses Marchenko-Pastur spectral k per node via
`estimate_k_marchenko_pastur()`.  We monkey-patch this function in the
marchenko_pastur module to test alternative projection-dimension equations.

The same 6 equation families from exp20 are adapted for Gate 2:
  Family 1 — Arithmetic combinations of spectral features
  Family 2 — Trigonometric transformations
  Family 3 — Gap-based eigenvalue analysis
  Family 4 — Eigenvalue relationships
  Family 5 — Bayesian/conditional approaches
  Family 6 — Hybrid blends with JL floor

Monkey-patch target:
  marchenko_pastur.estimate_k_marchenko_pastur  (module-level reference)
  Called by _process_node() for each tree node during Gate 2 spectral context.

NOTE: Gate 3 uses the CURRENT BEST sibling strategy (jl_floor_qrt) throughout
all runs, so we isolate the effect of changing Gate 2 only.

Phase A — 15-sentinel screening of ~30 equations
Phase B — Full 93-case validation of top performers (if warranted)
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

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    marchenko_pastur as mp_module,
)

# Apply jl_floor_qrt as the FIXED Gate 3 strategy for all runs
# (isolate Gate 2 effect from Gate 3)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as _compute_jl_dim_g3,
)


def _derive_jl_floor_qrt(tree, annotated_df):
    """Gate 3: jl_floor_qrt (fixed across all Gate 2 experiments)."""
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
        jl_k = _compute_jl_dim_g3(n_parent, n_features)
        floor_k = max(1, jl_k // 4)
        sibling_dims[parent] = max(spectral_k, floor_k)
    return sibling_dims if sibling_dims else None


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers for equations
# ═════════════════════════════════════════════════════════════════════════════


def _mp_upper_bound(eigenvalues, n_samples, n_features):
    """Marchenko-Pastur upper bound."""
    pos = eigenvalues[eigenvalues > 0]
    sigma2 = float(np.median(pos)) if len(pos) > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    gamma = n_features / max(n_samples, 1)
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _max_consecutive_gap_index(eigenvalues, max_k=20):
    """Index of largest consecutive eigenvalue ratio gap."""
    best_gap = 0.0
    best_idx = 1
    for i in range(min(max_k, len(eigenvalues) - 1)):
        denom = max(eigenvalues[i + 1], 1e-12)
        gap = eigenvalues[i] / denom
        if gap > best_gap:
            best_gap = gap
            best_idx = i + 1
    return best_idx


def _safe_div(a, b, default=0.0):
    return a / b if abs(b) > 1e-12 else default


def _clamp_k(x, minimum=2):
    return max(minimum, int(round(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Equation factory: wraps an equation function into estimate_k_marchenko_pastur
# ═════════════════════════════════════════════════════════════════════════════


def _make_gate2_estimator(eq_fn):
    """Create a replacement for estimate_k_marchenko_pastur from an equation.

    The replacement receives the same args as the original:
      (eigenvalues, *, n_samples, n_features, minimum_projection_dimension)
    and returns an int projection dimension.
    """

    def _estimator(
        eigenvalues,
        *,
        n_samples,
        n_features,
        minimum_projection_dimension=1,
    ):
        ev = np.asarray(eigenvalues, dtype=np.float64)
        k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
        jl_k = compute_jl_dim(n_samples, n_features)
        jl_qrt = max(1, jl_k // 4)
        trace = float(np.sum(ev))
        mp_bound = _mp_upper_bound(ev, n_samples, n_features)
        max_gap_idx = _max_consecutive_gap_index(ev)

        row = {
            "n": n_samples,
            "d": n_features,
            "k_mp": k_mp,
            "jl_k": jl_k,
            "jl_qrt": jl_qrt,
            "lam1": float(ev[0]) if len(ev) > 0 else 0.0,
            "lam2": float(ev[1]) if len(ev) > 1 else 0.0,
            "lam3": float(ev[2]) if len(ev) > 2 else 0.0,
            "trace": trace,
            "mp_bound": mp_bound,
            "max_gap_idx": max_gap_idx,
            "min_k": minimum_projection_dimension,
        }

        k = eq_fn(row)
        k = max(k, int(minimum_projection_dimension))
        k = min(k, int(n_features))
        return k

    return _estimator


# ═════════════════════════════════════════════════════════════════════════════
# Equation definitions — organized by family
# ═════════════════════════════════════════════════════════════════════════════

# ── Baseline strategies ─────────────────────────────────────────────────────


def eq_mp_only(row):
    """Current production: pure Marchenko-Pastur count."""
    return row["k_mp"]


def eq_jl_only(row):
    """Pure JL dimension (no spectral info)."""
    return row["jl_k"]


def eq_jl_quarter(row):
    """JL / 4 floor."""
    return row["jl_qrt"]


# ── Family 1: Arithmetic ────────────────────────────────────────────────────


def eq_mp_plus_jl_qrt(row):
    """k_MP + JL/4: boost MP by a floor."""
    return _clamp_k(row["k_mp"] + row["jl_qrt"])


def eq_max_mp_jl_qrt(row):
    """max(k_MP, JL/4): simple floor."""
    return max(row["k_mp"], row["jl_qrt"])


def eq_mp_times_2(row):
    """2 × k_MP: double the spectral count."""
    return _clamp_k(2 * row["k_mp"])


def eq_avg_mp_jl(row):
    """(k_MP + JL) / 2: arithmetic mean."""
    return _clamp_k((row["k_mp"] + row["jl_k"]) / 2)


def eq_min_mp_jl(row):
    """min(k_MP, JL): conservative cap."""
    return max(2, min(row["k_mp"], row["jl_k"]))


def eq_mp_minus_1(row):
    """max(k_MP − 1, 2): slightly conservative."""
    return max(2, row["k_mp"] - 1)


def eq_mp_plus_1(row):
    """k_MP + 1: slightly aggressive."""
    return row["k_mp"] + 1


# ── Family 2: Trigonometric ─────────────────────────────────────────────────


def eq_sin_spectral_frac(row):
    """sin(π/2 × k_MP/JL) × JL: smooth interpolation."""
    ratio = min(row["k_mp"] / max(row["jl_k"], 1), 1.0)
    return _clamp_k(np.sin(np.pi / 2 * ratio) * row["jl_k"])


def eq_cos_noise_frac(row):
    """cos(π/2 × noise_ratio) × JL: penalize noise."""
    noise_frac = 1.0 - min(row["k_mp"] / max(row["jl_k"], 1), 1.0)
    return _clamp_k(np.cos(np.pi / 2 * noise_frac) * row["jl_k"])


def eq_sin_lam_ratio_g2(row):
    """sin(π/2 × λ₁/max_λ) × k_MP: scale MP by leading eigenvalue dominance."""
    max_lam = max(row["lam1"], 1.0)
    ratio = min(row["lam1"] / max_lam, 1.0)
    return _clamp_k(np.sin(np.pi / 2 * ratio) * row["k_mp"])


# ── Family 3: Gap-based ────────────────────────────────────────────────────


def eq_max_gap_idx(row):
    """Pure max consecutive gap index."""
    return max(2, row["max_gap_idx"])


def eq_max_gap_idx_jl_qrt(row):
    """max(gap_idx, JL/4): gap with JL floor."""
    return max(2, row["max_gap_idx"], row["jl_qrt"])


def eq_max_gap_idx_mp(row):
    """max(gap_idx, k_MP): gap with MP floor."""
    return max(2, row["max_gap_idx"], row["k_mp"])


def eq_avg_gap_mp(row):
    """(gap_idx + k_MP) / 2: average gap and MP."""
    return _clamp_k((row["max_gap_idx"] + row["k_mp"]) / 2)


# ── Family 4: Eigenvalue relationships ──────────────────────────────────────


def eq_lam12_frac_jl(row):
    """(λ₁+λ₂)/trace × JL: top-2 dominance × JL."""
    frac = (row["lam1"] + row["lam2"]) / max(row["trace"], 1e-12)
    return _clamp_k(frac * row["jl_k"])


def eq_lam12_frac_mp(row):
    """(λ₁+λ₂)/trace × k_MP: top-2 dominance × MP count."""
    frac = (row["lam1"] + row["lam2"]) / max(row["trace"], 1e-12)
    return _clamp_k(frac * row["k_mp"])


def eq_lam1_over_mp(row):
    """λ₁/mp_bound × k_MP: signal strength scaling."""
    ratio = _safe_div(row["lam1"], row["mp_bound"], 1.0)
    return _clamp_k(ratio * row["k_mp"])


def eq_leading_frac_jl(row):
    """λ₁/trace × JL: single leading eigenvalue dominance."""
    frac = _safe_div(row["lam1"], row["trace"], 0.5)
    return _clamp_k(frac * row["jl_k"])


def eq_spectral_entropy_jl(row):
    """exp(H(normalized eigenvalues)) as effective rank, floored by JL/4."""
    ev = np.array([row["lam1"], row["lam2"], row["lam3"]])
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return row["jl_qrt"]
    p = ev / np.sum(ev)
    entropy = -float(np.sum(p * np.log(p)))
    eff_rank = float(np.exp(entropy))
    return max(_clamp_k(eff_rank), row["jl_qrt"])


# ── Family 5: Bayesian / conditional ───────────────────────────────────────


def eq_posterior_mp(row):
    """Posterior k: JL/4 + sigmoid(evidence) × k_MP."""
    evidence = _safe_div(row["lam1"], row["mp_bound"], 0.0) - 1.0
    sigmoid = 1.0 / (1.0 + np.exp(-2 * evidence))
    return _clamp_k(row["jl_qrt"] + sigmoid * row["k_mp"])


def eq_soft_threshold(row):
    """Soft threshold: 1 per eigenvalue above mp_bound * 0.8."""
    mp08 = row["mp_bound"] * 0.8
    count = sum(1 for lam in [row["lam1"], row["lam2"], row["lam3"]] if lam > mp08)
    return max(2, count, row["jl_qrt"])


def eq_evidence_weighted_avg(row):
    """Weight MP and JL by signal strength evidence."""
    evidence = min(_safe_div(row["lam1"], row["mp_bound"], 0.5), 2.0)  # clamp at 2
    w = evidence / (1.0 + evidence)  # increasing weight to MP as evidence grows
    return _clamp_k(w * row["k_mp"] + (1 - w) * row["jl_qrt"])


# ── Family 6: Hybrid blends ────────────────────────────────────────────────


def eq_harmonic_mp_jl(row):
    """Harmonic mean of k_MP and JL/4."""
    a, b = max(row["k_mp"], 1), max(row["jl_qrt"], 1)
    return _clamp_k(2 * a * b / (a + b))


def eq_geometric_mp_jl(row):
    """Geometric mean of k_MP and JL/4."""
    return _clamp_k(np.sqrt(max(row["k_mp"], 1) * max(row["jl_qrt"], 1)))


def eq_mp_capped_jl(row):
    """min(k_MP, JL): MP capped at JL (never exceed JL)."""
    return max(2, min(row["k_mp"], row["jl_k"]))


def eq_jl_half(row):
    """JL / 2: intermediate between JL and JL/4."""
    return max(2, row["jl_k"] // 2)


# ═════════════════════════════════════════════════════════════════════════════
# Assemble all equations
# ═════════════════════════════════════════════════════════════════════════════

ALL_EQUATIONS = {
    # Baselines
    "mp_only": eq_mp_only,
    "jl_only": eq_jl_only,
    "jl_quarter": eq_jl_quarter,
    # Family 1: Arithmetic
    "mp_plus_jl_qrt": eq_mp_plus_jl_qrt,
    "max_mp_jl_qrt": eq_max_mp_jl_qrt,
    "mp_x2": eq_mp_times_2,
    "avg_mp_jl": eq_avg_mp_jl,
    "min_mp_jl": eq_min_mp_jl,
    "mp_minus_1": eq_mp_minus_1,
    "mp_plus_1": eq_mp_plus_1,
    # Family 2: Trigonometric
    "sin_spectral_frac": eq_sin_spectral_frac,
    "cos_noise_frac": eq_cos_noise_frac,
    "sin_lam_ratio_g2": eq_sin_lam_ratio_g2,
    # Family 3: Gap-based
    "max_gap_idx": eq_max_gap_idx,
    "max_gap_jl_qrt": eq_max_gap_idx_jl_qrt,
    "max_gap_mp": eq_max_gap_idx_mp,
    "avg_gap_mp": eq_avg_gap_mp,
    # Family 4: Eigenvalue relationships
    "lam12_frac_jl": eq_lam12_frac_jl,
    "lam12_frac_mp": eq_lam12_frac_mp,
    "lam1_over_mp": eq_lam1_over_mp,
    "leading_frac_jl": eq_leading_frac_jl,
    "spectral_entropy_jl": eq_spectral_entropy_jl,
    # Family 5: Bayesian/conditional
    "posterior_mp": eq_posterior_mp,
    "soft_threshold": eq_soft_threshold,
    "evidence_weighted": eq_evidence_weighted_avg,
    # Family 6: Hybrid
    "harmonic_mp_jl": eq_harmonic_mp_jl,
    "geometric_mp_jl": eq_geometric_mp_jl,
    "mp_capped_jl": eq_mp_capped_jl,
    "jl_half": eq_jl_half,
}


# ═════════════════════════════════════════════════════════════════════════════
# 15 sentinel cases (same as exp20)
# ═════════════════════════════════════════════════════════════════════════════

CASES = [
    "binary_balanced_low_noise__2",
    "gauss_clear_small",
    "binary_low_noise_12c",
    "binary_perfect_8c",
    "binary_hard_4c",
    "gauss_noisy_3c",
    "gauss_overlap_4c_med",
    "gauss_moderate_3c",
    "gauss_moderate_5c",
    "binary_low_noise_4c",
    "binary_perfect_4c",
    "gauss_clear_large",
    "binary_multiscale_4c",
    "binary_many_features",
    "gauss_noisy_many",
]


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


def run_case_strategy(case_name, eq_name, eq_fn):
    """Run one case with a custom Gate 2 projection-dimension equation.

    Gate 2 k: controlled by monkey-patching mp_module.estimate_k_marchenko_pastur
    Gate 3 k: fixed to jl_floor_qrt for all runs
    """
    tree, data_df, y_true, tc = build_tree_and_data(case_name)

    try:
        with temporary_experiment_overrides(
            gate2_estimator=_make_gate2_estimator(eq_fn),
            sibling_dims=_derive_jl_floor_qrt,
        ):
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
# Main — Phase A: 15-sentinel screening
# ═════════════════════════════════════════════════════════════════════════════


def run_phase_b():
    """Phase B: full 93-case validation of top Gate 2 strategies."""
    from benchmarks.shared.cases import get_default_test_cases

    PHASE_B_STRATEGIES = {
        "mp_only": eq_mp_only,
        "lam12_frac_mp": eq_lam12_frac_mp,
        "max_gap_idx": eq_max_gap_idx,
        "avg_gap_mp": eq_avg_gap_mp,
        "max_gap_mp": eq_max_gap_idx_mp,
        "geometric_mp_jl": eq_geometric_mp_jl,
    }

    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    strat_names = list(PHASE_B_STRATEGIES.keys())
    n_cases = len(case_names)
    n_strat = len(strat_names)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print("        Gate 3 fixed to: jl_floor_qrt")
    print(f"\n═══ PHASE B: {n_cases} cases × {n_strat} Gate 2 strategies ═══\n")

    results: dict[str, list[dict]] = {name: [] for name in strat_names}
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        print(f"  [{i:3d}/{n_cases}] {case_name:<45}", end="", flush=True)
        t1 = time.time()
        for sn in strat_names:
            r = run_case_strategy(case_name, sn, PHASE_B_STRATEGIES[sn])
            results[sn].append(r)
        dt = time.time() - t1
        mp_ari = results["mp_only"][-1]["ari"]
        lam12_ari = results["lam12_frac_mp"][-1]["ari"]
        print(f" {dt:5.1f}s  MP={mp_ari:.3f}  lam12_mp={lam12_ari:.3f}")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ── Aggregate ───────────────────────────────────────────────────────────

    print(f"═══ AGGREGATE ({n_cases} cases) ═══\n")
    print(f"  {'Strategy':<25} {'Mean ARI':>9} {'Med ARI':>9} {'Exact K':>9} {'Perfect':>8}")
    print("  " + "─" * 65)

    scoreboard = []
    for sn in strat_names:
        aris = [r["ari"] for r in results[sn] if isinstance(r["ari"], float)]
        exact_k = sum(
            1
            for r in results[sn]
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]
        )
        countable = sum(1 for r in results[sn] if r["true_k"] != "?")
        perfect = sum(1 for r in results[sn] if isinstance(r["ari"], float) and r["ari"] >= 0.999)
        mean_ari = float(np.mean(aris)) if aris else 0.0
        med_ari = float(np.median(aris)) if aris else 0.0
        scoreboard.append((sn, mean_ari, med_ari, exact_k, countable, perfect))
        print(
            f"  {sn:<25} {mean_ari:9.3f} {med_ari:9.3f} {exact_k:5d}/{countable:<3d} {perfect:8d}"
        )

    scoreboard.sort(key=lambda x: (-x[1], -x[3]))
    print("\n═══ FINAL RANKING ═══\n")
    for rank, (name, m_ari, md_ari, ek, cnt, pf) in enumerate(scoreboard, 1):
        marker = " ★" if name == "mp_only" else ""
        print(
            f"  {rank:2d}. {name:<25} ARI={m_ari:.4f}  Med={md_ari:.4f}  K={ek}/{cnt}  Perf={pf}{marker}"
        )

    # ── Per-case detail ─────────────────────────────────────────────────────

    print("\n═══ PER-CASE DETAIL ═══\n")
    header = f"  {'Case':<45} {'TK':>4}"
    for sn in strat_names:
        header += f" │ {sn[:14]:>14}"
    print(header)
    print("  " + "─" * (50 + 17 * n_strat))

    for ci, case_name in enumerate(case_names):
        tk = results[strat_names[0]][ci]["true_k"]
        line = f"  {case_name:<45} {str(tk):>4}"
        for sn in strat_names:
            r = results[sn][ci]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>3} {ari:.3f}{check}"
        print(line)

    # ── Head-to-head: lam12_frac_mp vs mp_only ──────────────────────────────

    best_name = scoreboard[0][0]
    if best_name != "mp_only":
        print(f"\n═══ HEAD-TO-HEAD: {best_name} vs mp_only ═══\n")
        wins = losses = ties = 0
        win_lines = []
        loss_lines = []
        for ci, case_name in enumerate(case_names):
            r_best = results[best_name][ci]
            r_mp = results["mp_only"][ci]
            delta = r_best["ari"] - r_mp["ari"]
            if abs(delta) <= 0.005:
                ties += 1
            elif delta > 0:
                wins += 1
                win_lines.append(
                    f"  WIN  {case_name:<40} TK={r_best['true_k']:>3}  "
                    f"{best_name}: K={r_best['found_k']:>3} {r_best['ari']:.3f}  "
                    f"mp: K={r_mp['found_k']:>3} {r_mp['ari']:.3f}  Δ={delta:+.3f}"
                )
            else:
                losses += 1
                loss_lines.append(
                    f"  LOSS {case_name:<40} TK={r_best['true_k']:>3}  "
                    f"{best_name}: K={r_best['found_k']:>3} {r_best['ari']:.3f}  "
                    f"mp: K={r_mp['found_k']:>3} {r_mp['ari']:.3f}  Δ={delta:+.3f}"
                )
        for ln in sorted(win_lines, key=lambda x: float(x.split("Δ=")[1]), reverse=True):
            print(ln)
        for ln in sorted(loss_lines, key=lambda x: float(x.split("Δ=")[1])):
            print(ln)
        print(f"\n  {best_name} wins: {wins}, mp_only wins: {losses}, ties: {ties}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run Phase B (full 93-case suite)")
    args = parser.parse_args()

    if args.full:
        run_phase_b()
        sys.exit(0)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print("        Gate 3 fixed to: jl_floor_qrt")
    print()

    eq_names = list(ALL_EQUATIONS.keys())
    n_eq = len(eq_names)
    n_cases = len(CASES)
    print(f"═══ PHASE A: {n_cases} sentinel cases × {n_eq} Gate 2 equations ═══\n")

    # results[eq_name] = list of {true_k, found_k, ari} per case
    results: dict[str, list[dict]] = {name: [] for name in eq_names}
    t0 = time.time()

    for i, case_name in enumerate(CASES, 1):
        print(f"  [{i:2d}/{n_cases}] {case_name:<40}", end="", flush=True)
        t1 = time.time()
        for eq_name in eq_names:
            r = run_case_strategy(case_name, eq_name, ALL_EQUATIONS[eq_name])
            results[eq_name].append(r)
        dt = time.time() - t1
        # Show ARI for mp_only baseline vs top candidates
        mp_ari = results["mp_only"][-1]["ari"]
        lam12_ari = results["lam12_frac_jl"][-1]["ari"]
        gap_ari = results["max_gap_jl_qrt"][-1]["ari"]
        print(f" {dt:5.1f}s  MP={mp_ari:.3f}  lam12={lam12_ari:.3f}  gap={gap_ari:.3f}")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ── Aggregate metrics ───────────────────────────────────────────────────

    print("═══ AGGREGATE RESULTS (15 sentinels) ═══\n")
    print(f"  {'Strategy':<25} {'Mean ARI':>9} {'Med ARI':>9} {'Exact K':>9} {'Perfect':>8}")
    print("  " + "─" * 65)

    scoreboard = []
    for eq_name in eq_names:
        aris = [r["ari"] for r in results[eq_name] if isinstance(r["ari"], float)]
        exact_k = sum(
            1
            for r in results[eq_name]
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]
        )
        countable = sum(1 for r in results[eq_name] if r["true_k"] != "?")
        perfect = sum(
            1 for r in results[eq_name] if isinstance(r["ari"], float) and r["ari"] >= 0.999
        )
        mean_ari = float(np.mean(aris)) if aris else 0.0
        med_ari = float(np.median(aris)) if aris else 0.0
        scoreboard.append((eq_name, mean_ari, med_ari, exact_k, countable, perfect))
        print(
            f"  {eq_name:<25} {mean_ari:9.3f} {med_ari:9.3f} {exact_k:5d}/{countable:<3d} {perfect:8d}"
        )

    # Sort by mean ARI descending
    scoreboard.sort(key=lambda x: (-x[1], -x[3]))
    print("\n═══ RANKING (by Mean ARI) ═══\n")
    for rank, (name, m_ari, md_ari, ek, cnt, pf) in enumerate(scoreboard, 1):
        marker = " ★" if name == "mp_only" else ""
        print(
            f"  {rank:2d}. {name:<25} ARI={m_ari:.4f}  Med={md_ari:.4f}  K={ek}/{cnt}  Perf={pf}{marker}"
        )

    # ── Per-case detail for top 5 + baselines ───────────────────────────────

    top5 = [s[0] for s in scoreboard[:5]]
    baselines = ["mp_only", "jl_only"]
    show = list(dict.fromkeys(top5 + baselines))  # deduplicate, preserve order

    print("\n═══ PER-CASE DETAIL (top 5 + baselines) ═══\n")
    header = f"  {'Case':<40} {'TK':>3}"
    for s in show:
        header += f" │ {s:>14}"
    print(header)
    print("  " + "─" * (45 + 17 * len(show)))

    for ci, case_name in enumerate(CASES):
        line = f"  {case_name:<40}"
        tk = results[show[0]][ci]["true_k"]
        line += f" {tk:>3}"
        for s in show:
            r = results[s][ci]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>3} {ari:.3f}{check}"
        print(line)

    # ── Head-to-head: best vs mp_only ───────────────────────────────────────

    best_name = scoreboard[0][0]
    if best_name != "mp_only":
        print(f"\n═══ HEAD-TO-HEAD: {best_name} vs mp_only ═══\n")
        wins = losses = ties = 0
        for ci, case_name in enumerate(CASES):
            r_best = results[best_name][ci]
            r_mp = results["mp_only"][ci]
            delta = r_best["ari"] - r_mp["ari"]
            if abs(delta) <= 0.005:
                ties += 1
            elif delta > 0:
                wins += 1
                print(
                    f"  WIN  {case_name:<40} {best_name}: {r_best['ari']:.3f}  mp: {r_mp['ari']:.3f}  Δ={delta:+.3f}"
                )
            else:
                losses += 1
                print(
                    f"  LOSS {case_name:<40} {best_name}: {r_best['ari']:.3f}  mp: {r_mp['ari']:.3f}  Δ={delta:+.3f}"
                )
        print(f"\n  {best_name} wins: {wins}, mp_only wins: {losses}, ties: {ties}")
