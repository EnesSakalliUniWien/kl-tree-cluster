"""Lab exp21: Full 93-case benchmark validation.

Validates the top equations from exp20 across ALL 93 benchmark cases.
Compares against jl_floor_qrt baseline and the current production strategy
(min-child spectral — the regression).

Top strategies under test:
  1. lam12_frac_jl  — (λ₁+λ₂)/trace × JL       [Eigenvalue]  exp20 ARI=0.995
  2. max_gap_idx_jl — max(gap_idx, JL/4)          [Gap-based]   exp20 ARI=0.993
  3. kP_minus_kmax_jl — kP − max(kL,kR) + JL/4   [Arithmetic]  exp20 ARI=0.991
  4. sin_lam_ratio  — sin(π/2·λ₁/max_λ)×JL/4     [Trig]        exp20 ARI=0.991
  5. post_lam_evid  — JL/4 + sigmoid(λ₁/mp−1)×kP  [Bayesian]    exp20 ARI=0.989
  + baselines: jl_floor_qrt, min_child (production)
"""

from __future__ import annotations

import sys
import time
import traceback
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

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers (subset from exp20 — only what the top 5 equations need)
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


def _mp_upper_bound(eigenvalues, n_samples, d_active):
    med = float(np.median(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    sigma2 = max(med, 1e-12)
    gamma = d_active / max(n_samples, 1)
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def _max_consecutive_gap_index(eigenvalues):
    if len(eigenvalues) < 2:
        return 1
    gaps = np.diff(eigenvalues[::-1])  # ascending
    if len(gaps) == 0:
        return 1
    gaps_desc = eigenvalues[:-1] - eigenvalues[1:]  # descending diffs
    if np.max(gaps_desc) <= 0:
        return 1
    return int(np.argmax(gaps_desc)) + 1


def _safe_div(a, b, default=0.0):
    return a / b if b > 1e-12 else default


def _clamp_k(x):
    return max(2, int(round(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Top 5 equation functions
# ═════════════════════════════════════════════════════════════════════════════


def eq_lam12_frac_jl(row):
    """(λ₁ + λ₂)/trace × JL: top-2 dominance fraction → scale JL."""
    frac = (row["lam1_P"] + row["lam2_P"]) / max(row["trace_P"], 1e-12)
    return _clamp_k(frac * row["jl_k"])


def eq_max_gap_idx_jl(row):
    """max(max_gap_idx_P, JL/4): gap-index with JL floor."""
    return max(2, row["max_gap_idx_P"], row["jl_qrt"])


def eq_kP_minus_kmax_jl(row):
    """k_P − max(k_L, k_R) + JL/4: novel dimensions in parent + floor."""
    novel = row["k_P"] - row["k_max"]
    return _clamp_k(novel + row["jl_qrt"])


def eq_sin_lam_ratio(row):
    """sin(π/2 × λ₁/max_lam) × JL/4."""
    max_lam = max(row["lam1_P"], 1.0)
    ratio = min(row["lam1_P"] / max_lam, 1.0)
    return _clamp_k(np.sin(np.pi / 2 * ratio) * row["jl_qrt"])


def eq_post_lam_evid(row):
    """Posterior k from eigenvalue evidence via sigmoid."""
    evidence = _safe_div(row["lam1_P"], row["mp_bound_P"], 0.0) - 1.0
    sigmoid = 1.0 / (1.0 + np.exp(-2 * evidence))
    return _clamp_k(row["jl_qrt"] + sigmoid * max(row["k_P"], 0))


# ═════════════════════════════════════════════════════════════════════════════
# Strategy factories
# ═════════════════════════════════════════════════════════════════════════════


def _make_strategy(eq_fn):
    """Create a _derive_sibling_spectral_dims function from an equation."""

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
            eig_L = _eigendecompose(data_L)
            eig_R = _eigendecompose(data_R)
            ev_L = eig_L[0] if eig_L else np.array([0.0])
            ns_L = eig_L[1] if eig_L else 1
            da_L = eig_L[2] if eig_L else 1
            ev_R = eig_R[0] if eig_R else np.array([0.0])
            ns_R = eig_R[1] if eig_R else 1
            da_R = eig_R[2] if eig_R else 1

            n_features = leaf_data.shape[1]
            n_L = len(_descendant_leaves(tree, left))
            n_R = len(_descendant_leaves(tree, right))
            n_P = n_L + n_R

            k_P = marchenko_pastur_signal_count(ev_P, ns_P, da_P)
            k_L = marchenko_pastur_signal_count(ev_L, ns_L, da_L) if eig_L else 0
            k_R = marchenko_pastur_signal_count(ev_R, ns_R, da_R) if eig_R else 0

            jl_k_val = compute_jl_dim(n_P, n_features)
            trace_P = float(np.sum(ev_P))

            row = {
                "n_P": n_P,
                "k_P": k_P,
                "k_L": k_L,
                "k_R": k_R,
                "k_max": max(k_L, k_R),
                "jl_k": jl_k_val,
                "jl_qrt": max(1, jl_k_val // 4),
                "lam1_P": float(ev_P[0]) if len(ev_P) > 0 else 0.0,
                "lam2_P": float(ev_P[1]) if len(ev_P) > 1 else 0.0,
                "trace_P": trace_P,
                "mp_bound_P": _mp_upper_bound(ev_P, ns_P, da_P),
                "max_gap_idx_P": _max_consecutive_gap_index(ev_P),
            }

            k = eq_fn(row)
            sibling_dims[parent] = k

        return sibling_dims if sibling_dims else None

    return _derive


def _derive_none(tree, annotated_df):
    """Production default: returns None → falls back to min-child spectral."""
    return None


def _derive_jl_floor_qrt(tree, annotated_df):
    """jl_floor_qrt: max(spectral_min, JL/4)."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Strategies to test
# ═════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "min_child": _derive_none,
    "jl_floor_qrt": _derive_jl_floor_qrt,
    "lam12_frac_jl": _make_strategy(eq_lam12_frac_jl),
    "max_gap_idx_jl": _make_strategy(eq_max_gap_idx_jl),
    "kP_minus_kmax_jl": _make_strategy(eq_kP_minus_kmax_jl),
    "sin_lam_ratio": _make_strategy(eq_sin_lam_ratio),
    "post_lam_evid": _make_strategy(eq_post_lam_evid),
}

STRATEGY_NAMES = list(STRATEGIES.keys())
N_STRAT = len(STRATEGY_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


def run_case(case_name: str) -> dict[str, dict]:
    """Run all strategies on one case. Returns {strategy_name: {true_k, found_k, ari}}."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")
    if "n_clusters" not in tc and "sizes" in tc:
        true_k = len(tc["sizes"])

    results = {}
    for strat_name, strat_fn in STRATEGIES.items():
        try:
            with temporary_experiment_overrides(
                leaf_data_cache=_leaf_data_cache,
                leaf_data=data_df,
                sibling_dims=strat_fn,
            ):
                decomp = tree.decompose(
                    leaf_data=data_df,
                    alpha_local=config.SIBLING_ALPHA,
                    sibling_alpha=config.SIBLING_ALPHA,
                )
                ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
                results[strat_name] = {
                    "true_k": true_k,
                    "found_k": decomp["num_clusters"],
                    "ari": round(ari, 3),
                }
        except Exception as exc:
            results[strat_name] = {
                "true_k": true_k,
                "found_k": "ERR",
                "ari": 0.0,
                "error": str(exc)[:80],
            }

        # Force fresh tree for next strategy (re-build since decompose mutates stats_df)
        tree, data_df, y_true, tc = build_tree_and_data(case_name)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print(f"        Strategies: {N_STRAT} ({', '.join(STRATEGY_NAMES)})")
    print()

    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    n_cases = len(case_names)
    print(f"═══ FULL BENCHMARK: {n_cases} cases × {N_STRAT} strategies ═══\n")

    # Collect all results: list of (case_name, {strat: {true_k, found_k, ari}})
    all_results: list[tuple[str, dict]] = []
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        print(f"  [{i:3d}/{n_cases}] {case_name:<45}", end="", flush=True)
        t1 = time.time()
        try:
            res = run_case(case_name)
        except Exception:
            print(f" FATAL: {traceback.format_exc()[-80:]}")
            res = {s: {"true_k": "?", "found_k": "ERR", "ari": 0.0} for s in STRATEGY_NAMES}
        elapsed = time.time() - t1
        # Quick per-case summary: show ARI for top 3
        aris = [f"{res[s]['ari']:.3f}" for s in STRATEGY_NAMES[:3]]
        print(f" {elapsed:5.1f}s  ARI: {'/'.join(aris)}")
        all_results.append((case_name, res))

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ═══ AGGREGATE RESULTS ═══
    print("═══ AGGREGATE RESULTS ═══\n")

    # Compute per-strategy summary
    header = f"  {'Strategy':<20} │ {'Mean ARI':>8} │ {'Med ARI':>7} │ {'Exact K':>7} │ {'K=1':>3} │ {'Perfect':>7} │ {'Wins':>4}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    strat_aris = {s: [] for s in STRATEGY_NAMES}
    strat_exact = {s: 0 for s in STRATEGY_NAMES}
    strat_k1 = {s: 0 for s in STRATEGY_NAMES}
    strat_perfect = {s: 0 for s in STRATEGY_NAMES}

    for case_name, res_dict in all_results:
        for strat in STRATEGY_NAMES:
            r = res_dict[strat]
            ari = r["ari"]
            strat_aris[strat].append(ari)
            if r["found_k"] == r["true_k"]:
                strat_exact[strat] += 1
            if r["found_k"] == 1:
                strat_k1[strat] += 1
            if ari >= 0.999:
                strat_perfect[strat] += 1

    # Count per-case wins (best ARI)
    strat_wins = {s: 0 for s in STRATEGY_NAMES}
    for case_name, res_dict in all_results:
        best_ari = max(res_dict[s]["ari"] for s in STRATEGY_NAMES)
        for s in STRATEGY_NAMES:
            if res_dict[s]["ari"] >= best_ari - 1e-6:
                strat_wins[s] += 1

    for strat in STRATEGY_NAMES:
        aris = strat_aris[strat]
        mean_a = np.mean(aris)
        med_a = np.median(aris)
        print(
            f"  {strat:<20} │ {mean_a:8.3f} │ {med_a:7.3f} │ {strat_exact[strat]:>3}/{n_cases:<3} │ {strat_k1[strat]:>3} │ {strat_perfect[strat]:>3}/{n_cases:<3} │ {strat_wins[strat]:>4}"
        )

    # ═══ CATEGORY BREAKDOWN ═══
    print("\n═══ CATEGORY BREAKDOWN ═══\n")
    categories: dict[str, list[tuple[str, dict]]] = {}
    for case_name, res_dict in all_results:
        tc = next((c for c in all_cases if c["name"] == case_name), {})
        cat = tc.get("category", "unknown")
        # Group into macro-categories
        if "gaussian" in cat or "gauss" in cat:
            macro = "Gaussian"
        elif "binary" in cat or "sparse" in cat:
            macro = "Binary"
        elif "sbm" in cat:
            macro = "SBM"
        elif "cat" in cat:
            macro = "Categorical"
        elif "phylo" in cat:
            macro = "Phylogenetic"
        elif "overlap" in cat:
            macro = "Overlapping"
        elif "real" in cat:
            macro = "Real Data"
        else:
            macro = cat
        categories.setdefault(macro, []).append((case_name, res_dict))

    for macro in sorted(categories.keys()):
        cases_in_cat = categories[macro]
        n_cat = len(cases_in_cat)
        print(f"  {macro} ({n_cat} cases):")
        for strat in STRATEGY_NAMES:
            aris = [r[strat]["ari"] for _, r in cases_in_cat]
            mean_a = np.mean(aris)
            exact = sum(1 for _, r in cases_in_cat if r[strat]["found_k"] == r[strat]["true_k"])
            print(f"    {strat:<20} ARI={mean_a:.3f}  exact_K={exact}/{n_cat}")
        print()

    # ═══ DETAILED PER-CASE COMPARISON (top strategies) ═══
    print("═══ PER-CASE DETAIL ═══\n")

    top_strats = ["min_child", "jl_floor_qrt", "lam12_frac_jl", "max_gap_idx_jl"]
    hdr = f"  {'Case':<45} {'TK':>2}"
    for s in top_strats:
        hdr += f" │ {s[:13]:>13}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for case_name, res_dict in all_results:
        true_k = res_dict[STRATEGY_NAMES[0]]["true_k"]
        line = f"  {case_name:<45} {str(true_k):>2}"
        for s in top_strats:
            r = res_dict[s]
            fk = r["found_k"]
            ari = r["ari"]
            marker = "✓" if fk == true_k else " "
            line += f" │ K={str(fk):>3} {ari:.3f}{marker}"
        print(line)

    # ═══ HEAD-TO-HEAD: lam12_frac_jl vs jl_floor_qrt ═══
    print("\n═══ HEAD-TO-HEAD: lam12_frac_jl vs jl_floor_qrt ═══\n")

    lam12_better = []
    jlqrt_better = []
    ties = []
    for case_name, res_dict in all_results:
        a_lam = res_dict["lam12_frac_jl"]["ari"]
        a_jl = res_dict["jl_floor_qrt"]["ari"]
        if a_lam > a_jl + 0.005:
            lam12_better.append((case_name, a_lam, a_jl, res_dict))
        elif a_jl > a_lam + 0.005:
            jlqrt_better.append((case_name, a_lam, a_jl, res_dict))
        else:
            ties.append(case_name)

    print(f"  lam12_frac_jl wins: {len(lam12_better)}")
    for cn, al, aj, rd in sorted(lam12_better, key=lambda x: x[1] - x[2], reverse=True):
        tk = rd["lam12_frac_jl"]["true_k"]
        kl = rd["lam12_frac_jl"]["found_k"]
        kj = rd["jl_floor_qrt"]["found_k"]
        print(
            f"    {cn:<40} TK={tk:>2}  lam12: K={kl:>3} ARI={al:.3f}  jl_qrt: K={kj:>3} ARI={aj:.3f}  Δ={al-aj:+.3f}"
        )

    print(f"\n  jl_floor_qrt wins: {len(jlqrt_better)}")
    for cn, al, aj, rd in sorted(jlqrt_better, key=lambda x: x[2] - x[1], reverse=True):
        tk = rd["lam12_frac_jl"]["true_k"]
        kl = rd["lam12_frac_jl"]["found_k"]
        kj = rd["jl_floor_qrt"]["found_k"]
        print(
            f"    {cn:<40} TK={tk:>2}  lam12: K={kl:>3} ARI={al:.3f}  jl_qrt: K={kj:>3} ARI={aj:.3f}  Δ={aj-al:+.3f}"
        )

    print(f"\n  Ties (|ΔARI| ≤ 0.005): {len(ties)}")

    # ═══ FINAL RANKING ═══
    print("\n═══ FINAL RANKING ═══\n")
    ranking = sorted(
        STRATEGY_NAMES, key=lambda s: (np.mean(strat_aris[s]), strat_exact[s]), reverse=True
    )
    for rank, strat in enumerate(ranking, 1):
        mean_a = np.mean(strat_aris[strat])
        med_a = np.median(strat_aris[strat])
        print(
            f"  {rank}. {strat:<20}  Mean ARI={mean_a:.3f}  Med ARI={med_a:.3f}  Exact K={strat_exact[strat]}/{n_cases}  Perfect={strat_perfect[strat]}"
        )
