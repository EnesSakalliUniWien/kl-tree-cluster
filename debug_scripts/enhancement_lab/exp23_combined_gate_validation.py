"""Lab exp23: Combined Gate 2 + Gate 3 lam12 validation.

Previous experiments changed one gate at a time:
  - exp21/22 Phase B: Gate 3 strategies with Gate 2 = MP (default)
  - exp22 Phase B: Gate 2 strategies with Gate 3 = jl_floor_qrt

This experiment tests all 4 combinations on the full 93-case suite:
  1. production   — Gate 2 = MP,           Gate 3 = min_child (current default)
  2. gate2_only   — Gate 2 = lam12_frac_mp, Gate 3 = jl_floor_qrt
  3. gate3_only   — Gate 2 = MP,           Gate 3 = lam12_frac_jl
  4. combined     — Gate 2 = lam12_frac_mp, Gate 3 = lam12_frac_jl
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
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    marchenko_pastur as mp_module,
)

# ═════════════════════════════════════════════════════════════════════════════
# Gate 2 helpers (from exp22)
# ═════════════════════════════════════════════════════════════════════════════


def _mp_upper_bound_g2(eigenvalues, n_samples, n_features):
    pos = eigenvalues[eigenvalues > 0]
    sigma2 = float(np.median(pos)) if len(pos) > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    gamma = n_features / max(n_samples, 1)
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _clamp_k(x, minimum=2):
    return max(minimum, int(round(x)))


def _gate2_lam12_frac_mp(eigenvalues, *, n_samples, n_features, minimum_projection_dimension=1):
    """Gate 2: (λ₁+λ₂)/trace × k_MP."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    k_mp = marchenko_pastur_signal_count(ev, n_samples, n_features)
    trace = float(np.sum(ev))
    lam1 = float(ev[0]) if len(ev) > 0 else 0.0
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    frac = (lam1 + lam2) / max(trace, 1e-12)
    k = _clamp_k(frac * k_mp)
    k = max(k, int(minimum_projection_dimension))
    k = min(k, int(n_features))
    return k


# ═════════════════════════════════════════════════════════════════════════════
# Gate 3 helpers (from exp21)
# ═════════════════════════════════════════════════════════════════════════════

_leaf_data_cache: dict = {}


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


def _mp_upper_bound_g3(eigenvalues, n_samples, d_active):
    med = float(np.median(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    sigma2 = max(med, 1e-12)
    gamma = d_active / max(n_samples, 1)
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def _max_consecutive_gap_index(eigenvalues):
    if len(eigenvalues) < 2:
        return 1
    gaps_desc = eigenvalues[:-1] - eigenvalues[1:]
    if np.max(gaps_desc) <= 0:
        return 1
    return int(np.argmax(gaps_desc)) + 1


# ── Gate 3 strategy: min_child (production default) ─────────────────────────


def _derive_min_child(tree, annotated_df):
    """Production default: returns None → orchestrator falls back to min-child."""
    return None


# ── Gate 3 strategy: jl_floor_qrt ──────────────────────────────────────────


def _derive_jl_floor_qrt(tree, annotated_df):
    """max(spectral_min, JL/4)."""
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


# ── Gate 3 strategy: lam12_frac_jl ─────────────────────────────────────────


def _make_gate3_lam12_frac_jl():
    """Create lam12_frac_jl Gate 3 strategy."""

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
            trace_P = float(np.sum(ev_P))
            lam1 = float(ev_P[0]) if len(ev_P) > 0 else 0.0
            lam2 = float(ev_P[1]) if len(ev_P) > 1 else 0.0

            frac = (lam1 + lam2) / max(trace_P, 1e-12)
            k = _clamp_k(frac * jl_k_val)
            sibling_dims[parent] = k

        return sibling_dims if sibling_dims else None

    return _derive


_derive_gate3_lam12 = _make_gate3_lam12_frac_jl()


# ═════════════════════════════════════════════════════════════════════════════
# 4 configurations to test
# ═════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "production": {
        "gate2": None,  # default MP
        "gate3": _derive_min_child,
    },
    "gate2_only": {
        "gate2": _gate2_lam12_frac_mp,
        "gate3": _derive_jl_floor_qrt,
    },
    "gate3_only": {
        "gate2": None,  # default MP
        "gate3": _derive_gate3_lam12,
    },
    "combined": {
        "gate2": _gate2_lam12_frac_mp,
        "gate3": _derive_gate3_lam12,
    },
}

CONFIG_NAMES = list(CONFIGS.keys())
N_CONFIGS = len(CONFIG_NAMES)


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
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    n_cases = len(case_names)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print(f"\n═══ EXP23: Combined Gate Validation — {n_cases} cases × {N_CONFIGS} configs ═══")
    print("    1. production   = Gate2:MP          + Gate3:min_child")
    print("    2. gate2_only   = Gate2:lam12_frac  + Gate3:jl_floor_qrt")
    print("    3. gate3_only   = Gate2:MP          + Gate3:lam12_frac")
    print("    4. combined     = Gate2:lam12_frac  + Gate3:lam12_frac")
    print()

    results: dict[str, list[dict]] = {c: [] for c in CONFIG_NAMES}
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        print(f"  [{i:3d}/{n_cases}] {case_name:<45}", end="", flush=True)
        t1 = time.time()
        for cn in CONFIG_NAMES:
            r = run_case_config(case_name, cn)
            results[cn].append(r)
        dt = time.time() - t1
        prod_ari = results["production"][-1]["ari"]
        comb_ari = results["combined"][-1]["ari"]
        print(f" {dt:5.1f}s  prod={prod_ari:.3f}  comb={comb_ari:.3f}")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ── Aggregate ───────────────────────────────────────────────────────────

    print(f"═══ AGGREGATE ({n_cases} cases) ═══\n")
    print(
        f"  {'Config':<18} {'Mean ARI':>9} {'Med ARI':>9} "
        f"{'Exact K':>9} {'Perfect':>8} {'K=1':>5}"
    )
    print("  " + "─" * 72)

    scoreboard = []
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
        scoreboard.append((cn, mean_ari, med_ari, exact_k, countable, perfect, k1))
        print(
            f"  {cn:<18} {mean_ari:9.3f} {med_ari:9.3f} "
            f"{exact_k:5d}/{countable:<3d} {perfect:8d} {k1:5d}"
        )

    # ── Per-case detail ─────────────────────────────────────────────────────

    print("\n═══ PER-CASE DETAIL ═══\n")
    header = f"  {'Case':<45} {'TK':>4}"
    for cn in CONFIG_NAMES:
        header += f" │ {cn:>14}"
    print(header)
    print("  " + "─" * (50 + 17 * N_CONFIGS))

    for ci, case_name in enumerate(case_names):
        tk = results[CONFIG_NAMES[0]][ci]["true_k"]
        line = f"  {case_name:<45} {str(tk):>4}"
        for cn in CONFIG_NAMES:
            r = results[cn][ci]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>3} {ari:.3f}{check}"
        print(line)

    # ── Head-to-head: combined vs production ────────────────────────────────

    print("\n═══ HEAD-TO-HEAD: combined vs production ═══\n")
    wins = losses = ties = 0
    win_lines = []
    loss_lines = []
    for ci, case_name in enumerate(case_names):
        r_comb = results["combined"][ci]
        r_prod = results["production"][ci]
        delta = r_comb["ari"] - r_prod["ari"]
        if abs(delta) <= 0.005:
            ties += 1
        elif delta > 0:
            wins += 1
            win_lines.append(
                f"  WIN  {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"prod: K={r_prod['found_k']:>3} {r_prod['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
        else:
            losses += 1
            loss_lines.append(
                f"  LOSS {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"prod: K={r_prod['found_k']:>3} {r_prod['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
    for ln in sorted(win_lines, key=lambda x: float(x.split("Δ=")[1]), reverse=True):
        print(ln)
    for ln in sorted(loss_lines, key=lambda x: float(x.split("Δ=")[1])):
        print(ln)
    print(f"\n  combined wins: {wins}, production wins: {losses}, ties: {ties}")

    # ── Head-to-head: combined vs gate2_only ────────────────────────────────

    print("\n═══ HEAD-TO-HEAD: combined vs gate2_only ═══\n")
    wins2 = losses2 = ties2 = 0
    win_lines2 = []
    loss_lines2 = []
    for ci, case_name in enumerate(case_names):
        r_comb = results["combined"][ci]
        r_g2 = results["gate2_only"][ci]
        delta = r_comb["ari"] - r_g2["ari"]
        if abs(delta) <= 0.005:
            ties2 += 1
        elif delta > 0:
            wins2 += 1
            win_lines2.append(
                f"  WIN  {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"g2:   K={r_g2['found_k']:>3} {r_g2['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
        else:
            losses2 += 1
            loss_lines2.append(
                f"  LOSS {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"g2:   K={r_g2['found_k']:>3} {r_g2['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
    for ln in sorted(win_lines2, key=lambda x: float(x.split("Δ=")[1]), reverse=True):
        print(ln)
    for ln in sorted(loss_lines2, key=lambda x: float(x.split("Δ=")[1])):
        print(ln)
    print(f"\n  combined wins: {wins2}, gate2_only wins: {losses2}, ties: {ties2}")

    # ── Head-to-head: combined vs gate3_only ────────────────────────────────

    print("\n═══ HEAD-TO-HEAD: combined vs gate3_only ═══\n")
    wins3 = losses3 = ties3 = 0
    win_lines3 = []
    loss_lines3 = []
    for ci, case_name in enumerate(case_names):
        r_comb = results["combined"][ci]
        r_g3 = results["gate3_only"][ci]
        delta = r_comb["ari"] - r_g3["ari"]
        if abs(delta) <= 0.005:
            ties3 += 1
        elif delta > 0:
            wins3 += 1
            win_lines3.append(
                f"  WIN  {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"g3:   K={r_g3['found_k']:>3} {r_g3['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
        else:
            losses3 += 1
            loss_lines3.append(
                f"  LOSS {case_name:<40} TK={r_comb['true_k']:>3}  "
                f"comb: K={r_comb['found_k']:>3} {r_comb['ari']:.3f}  "
                f"g3:   K={r_g3['found_k']:>3} {r_g3['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )
    for ln in sorted(win_lines3, key=lambda x: float(x.split("Δ=")[1]), reverse=True):
        print(ln)
    for ln in sorted(loss_lines3, key=lambda x: float(x.split("Δ=")[1])):
        print(ln)
    print(f"\n  combined wins: {wins3}, gate3_only wins: {losses3}, ties: {ties3}")

    # ── Interaction effect ──────────────────────────────────────────────────

    print("\n═══ INTERACTION ANALYSIS ═══\n")
    print("  Is combined > max(gate2_only, gate3_only) for any case?")
    print("  (Positive interaction = gates amplify each other)\n")

    synergy_cases = []
    conflict_cases = []
    for ci, case_name in enumerate(case_names):
        a_comb = results["combined"][ci]["ari"]
        a_g2 = results["gate2_only"][ci]["ari"]
        a_g3 = results["gate3_only"][ci]["ari"]
        best_single = max(a_g2, a_g3)
        delta = a_comb - best_single
        if delta > 0.005:
            synergy_cases.append((case_name, a_comb, a_g2, a_g3, delta))
        elif delta < -0.005:
            conflict_cases.append((case_name, a_comb, a_g2, a_g3, delta))

    if synergy_cases:
        print("  SYNERGY (combined beats best single-gate):")
        for name, ac, a2, a3, d in sorted(synergy_cases, key=lambda x: -x[4]):
            print(f"    {name:<40} comb={ac:.3f}  g2={a2:.3f}  g3={a3:.3f}  Δ={d:+.3f}")
    else:
        print("  No synergy cases found.")

    if conflict_cases:
        print("\n  CONFLICT (combined worse than best single-gate):")
        for name, ac, a2, a3, d in sorted(conflict_cases, key=lambda x: x[4]):
            print(f"    {name:<40} comb={ac:.3f}  g2={a2:.3f}  g3={a3:.3f}  Δ={d:+.3f}")
    else:
        print("\n  No conflict cases found.")

    n_syn = len(synergy_cases)
    n_con = len(conflict_cases)
    n_neut = n_cases - n_syn - n_con
    print(f"\n  Summary: {n_syn} synergy, {n_con} conflict, {n_neut} neutral")
