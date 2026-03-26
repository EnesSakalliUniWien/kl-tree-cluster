"""Lab exp25: Floored lam12 hybrid — best of both worlds.

Key insight from exp21–24:
  - lam12_frac_jl wins on peaked spectra (22 wins vs jl_floor_qrt's 6)
  - lam12_frac_jl loses catastrophically on diffuse spectra (K=1 collapses)
  - jl_floor_qrt never collapses but misses lam12's adaptive scaling wins

Hypothesis: max(lam12_frac × JL, floor) captures lam12's wins while
preventing collapses. The floor prevents k from dropping too low on
diffuse spectra where (λ₁+λ₂)/trace ≈ 2/d.

Configurations (all Gate 2 = production MP, Gate 3 variants):
  1. production     — min_child (current default)
  2. jl_floor_qrt   — max(spectral_min, JL/4)
  3. lam12_raw       — (λ₁+λ₂)/trace × JL (no floor)
  4. lam12_floor_qrt — max(lam12, JL/4)  ← primary hypothesis
  5. lam12_floor_8th — max(lam12, JL/8)
  6. lam12_floor_3   — max(lam12, 3)     ← minimal absolute floor
  7. lam12_floor_min — max(lam12, min_child_spectral) ← spectral floor
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

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
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
    result = eigendecompose_correlation_backend(data_matrix, compute_eigenvectors=False)
    if result is None:
        return None
    return result.eigenvalues, data_matrix.shape[0], result.active_feature_count


def _clamp_k(x, minimum=2):
    return max(minimum, int(round(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Gate 3 strategies
# ═════════════════════════════════════════════════════════════════════════════


def _derive_min_child(tree, annotated_df):
    """Production default: returns None → orchestrator falls back to min-child."""
    return None


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


def _compute_lam12_k(tree, parent, left, right, leaf_data):
    """Compute raw lam12_frac × JL for a parent node. Returns (k, jl_k, frac) or None."""
    try:
        data_L = _node_data(tree, left, leaf_data)
        data_R = _node_data(tree, right, leaf_data)
    except (KeyError, IndexError):
        return None

    data_P = np.vstack([data_L, data_R])
    eig_P = _eigendecompose(data_P)
    if eig_P is None:
        return None

    ev_P, ns_P, da_P = eig_P
    n_features = leaf_data.shape[1]
    n_L = len(_descendant_leaves(tree, left))
    n_R = len(_descendant_leaves(tree, right))
    n_P = n_L + n_R

    jl_k = compute_jl_dim(n_P, n_features)
    trace_P = float(np.sum(ev_P))
    lam1 = float(ev_P[0]) if len(ev_P) > 0 else 0.0
    lam2 = float(ev_P[1]) if len(ev_P) > 1 else 0.0

    frac = (lam1 + lam2) / max(trace_P, 1e-12)
    k = _clamp_k(frac * jl_k)
    return k, jl_k, frac


def _make_lam12_strategy(floor_fn):
    """Create a Gate 3 strategy: max(lam12_frac × JL, floor_fn(context))."""

    def _derive(tree, annotated_df):
        leaf_data = _leaf_data_cache.get("leaf_data")
        if leaf_data is None:
            return None

        edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")

        sibling_dims = {}
        for parent in tree.nodes:
            children = list(tree.successors(parent))
            if len(children) != 2:
                continue
            left, right = children

            result = _compute_lam12_k(tree, parent, left, right, leaf_data)
            if result is None:
                continue

            k_lam12, jl_k, frac = result

            # Compute floor
            ctx = {
                "jl_k": jl_k,
                "tree": tree,
                "left": left,
                "right": right,
                "edge_spectral_dims": edge_spectral_dims,
            }
            floor_k = floor_fn(ctx)
            sibling_dims[parent] = max(k_lam12, floor_k)

        return sibling_dims if sibling_dims else None

    return _derive


# ── Floor functions ─────────────────────────────────────────────────────────


def _floor_none(ctx):
    """No floor — pure lam12."""
    return 2


def _floor_jl_qrt(ctx):
    """JL/4 floor."""
    return max(2, ctx["jl_k"] // 4)


def _floor_jl_8th(ctx):
    """JL/8 floor."""
    return max(2, ctx["jl_k"] // 8)


def _floor_absolute_3(ctx):
    """Absolute floor of 3."""
    return 3


def _floor_spectral_min(ctx):
    """min(k_left, k_right) spectral floor (the production default)."""
    edge_spectral_dims = ctx.get("edge_spectral_dims")
    if not edge_spectral_dims:
        return 2
    k_left = edge_spectral_dims.get(ctx["left"], 0)
    k_right = edge_spectral_dims.get(ctx["right"], 0)
    positive = [k for k in (k_left, k_right) if k > 0]
    return min(positive) if positive else 2


# ═════════════════════════════════════════════════════════════════════════════
# Configurations
# ═════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "production": _derive_min_child,
    "jl_floor_qrt": _derive_jl_floor_qrt,
    "lam12_raw": _make_lam12_strategy(_floor_none),
    "lam12_floor_qrt": _make_lam12_strategy(_floor_jl_qrt),
    "lam12_floor_8th": _make_lam12_strategy(_floor_jl_8th),
    "lam12_floor_3": _make_lam12_strategy(_floor_absolute_3),
    "lam12_floor_min": _make_lam12_strategy(_floor_spectral_min),
}

CONFIG_NAMES = list(CONFIGS.keys())
N_CONFIGS = len(CONFIG_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


def run_case(case_name: str) -> dict[str, dict]:
    """Run all configs on one case."""
    results = {}
    for cfg_name in CONFIG_NAMES:
        tree, data_df, y_true, tc = build_tree_and_data(case_name)
        true_k = tc.get("n_clusters", "?")

        try:
            with temporary_experiment_overrides(
                leaf_data_cache=_leaf_data_cache,
                leaf_data=data_df,
                sibling_dims=CONFIGS[cfg_name],
            ):
                decomp = tree.decompose(
                    leaf_data=data_df,
                    alpha_local=config.SIBLING_ALPHA,
                    sibling_alpha=config.SIBLING_ALPHA,
                )
            ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
            results[cfg_name] = {
                "true_k": true_k,
                "found_k": decomp["num_clusters"],
                "ari": round(ari, 4),
            }
        except Exception as exc:
            results[cfg_name] = {
                "true_k": true_k,
                "found_k": "ERR",
                "ari": 0.0,
                "error": str(exc)[:80],
            }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    n_cases = len(case_names)

    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print("        SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)")
    print(f"\n═══ EXP25: Floored lam12 Hybrid — {n_cases} cases × {N_CONFIGS} configs ═══")
    print("  Strategies:")
    for i, cn in enumerate(CONFIG_NAMES, 1):
        print(f"    {i}. {cn}")
    print()

    all_results: list[tuple[str, dict]] = []
    t0 = time.time()

    for i, case_name in enumerate(case_names, 1):
        print(f"  [{i:3d}/{n_cases}] {case_name:<45}", end="", flush=True)
        t1 = time.time()
        try:
            res = run_case(case_name)
        except Exception as exc:
            print(f" FATAL: {str(exc)[:60]}")
            res = {cn: {"true_k": "?", "found_k": "ERR", "ari": 0.0} for cn in CONFIG_NAMES}
        dt = time.time() - t1
        prod = res["production"]["ari"]
        hybrid = res["lam12_floor_qrt"]["ari"]
        print(f" {dt:5.1f}s  prod={prod:.3f}  hybrid={hybrid:.3f}")
        all_results.append((case_name, res))

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Aggregate
    # ═════════════════════════════════════════════════════════════════════════

    print(f"═══ AGGREGATE ({n_cases} cases) ═══\n")
    print(
        f"  {'Config':<18} │ {'Mean ARI':>9} │ {'Med ARI':>9} │ "
        f"{'Exact K':>9} │ {'Perfect':>8} │ {'K=1':>5} │ {'Wins':>5}"
    )
    print("  " + "─" * 82)

    strat_aris = {cn: [] for cn in CONFIG_NAMES}
    strat_exact = {cn: 0 for cn in CONFIG_NAMES}
    strat_k1 = {cn: 0 for cn in CONFIG_NAMES}
    strat_perfect = {cn: 0 for cn in CONFIG_NAMES}

    for _, res in all_results:
        for cn in CONFIG_NAMES:
            r = res[cn]
            strat_aris[cn].append(r["ari"])
            if isinstance(r["found_k"], int) and r["true_k"] != "?" and r["found_k"] == r["true_k"]:
                strat_exact[cn] += 1
            if isinstance(r["found_k"], int) and r["found_k"] == 1:
                strat_k1[cn] += 1
            if isinstance(r["ari"], float) and r["ari"] >= 0.999:
                strat_perfect[cn] += 1

    countable = sum(1 for _, res in all_results if res[CONFIG_NAMES[0]]["true_k"] != "?")

    # Per-case wins
    strat_wins = {cn: 0 for cn in CONFIG_NAMES}
    for _, res in all_results:
        best_ari = max(res[cn]["ari"] for cn in CONFIG_NAMES)
        for cn in CONFIG_NAMES:
            if res[cn]["ari"] >= best_ari - 1e-6:
                strat_wins[cn] += 1

    for cn in CONFIG_NAMES:
        aris = strat_aris[cn]
        mean_a = float(np.mean(aris))
        med_a = float(np.median(aris))
        print(
            f"  {cn:<18} │ {mean_a:9.3f} │ {med_a:9.3f} │ "
            f"{strat_exact[cn]:5d}/{countable:<3d} │ {strat_perfect[cn]:8d} │ "
            f"{strat_k1[cn]:5d} │ {strat_wins[cn]:5d}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Category breakdown
    # ═════════════════════════════════════════════════════════════════════════

    print("\n═══ CATEGORY BREAKDOWN ═══\n")
    categories: dict[str, list[tuple[str, dict]]] = {}
    for case_name, res in all_results:
        tc = next((c for c in all_cases if c["name"] == case_name), {})
        cat = tc.get("category", "unknown")
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
        categories.setdefault(macro, []).append((case_name, res))

    # Only show key strategies for readability
    key_strats = ["production", "jl_floor_qrt", "lam12_raw", "lam12_floor_qrt"]
    for macro in sorted(categories.keys()):
        cases_in_cat = categories[macro]
        n_cat = len(cases_in_cat)
        print(f"  {macro} ({n_cat} cases):")
        for cn in key_strats:
            aris = [r[cn]["ari"] for _, r in cases_in_cat]
            exact = sum(
                1
                for _, r in cases_in_cat
                if isinstance(r[cn]["found_k"], int)
                and r[cn]["true_k"] != "?"
                and r[cn]["found_k"] == r[cn]["true_k"]
            )
            k1 = sum(
                1
                for _, r in cases_in_cat
                if isinstance(r[cn]["found_k"], int) and r[cn]["found_k"] == 1
            )
            print(f"    {cn:<18} ARI={np.mean(aris):.3f}  exact_K={exact}/{n_cat}  K=1:{k1}")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # Head-to-head: lam12_floor_qrt vs each baseline
    # ═════════════════════════════════════════════════════════════════════════

    for baseline in ["production", "jl_floor_qrt", "lam12_raw"]:
        print(f"═══ H2H: lam12_floor_qrt vs {baseline} ═══\n")
        wins = losses = ties = 0
        win_lines = []
        loss_lines = []
        for case_name, res in all_results:
            r_hyb = res["lam12_floor_qrt"]
            r_base = res[baseline]
            delta = r_hyb["ari"] - r_base["ari"]
            tk = r_hyb["true_k"]
            if abs(delta) <= 0.005:
                ties += 1
            elif delta > 0:
                wins += 1
                win_lines.append(
                    f"    {case_name:<40} TK={tk:>3}  "
                    f"hyb: K={r_hyb['found_k']:>3} {r_hyb['ari']:.3f}  "
                    f"base: K={r_base['found_k']:>3} {r_base['ari']:.3f}  "
                    f"Δ={delta:+.3f}"
                )
            else:
                losses += 1
                loss_lines.append(
                    f"    {case_name:<40} TK={tk:>3}  "
                    f"hyb: K={r_hyb['found_k']:>3} {r_hyb['ari']:.3f}  "
                    f"base: K={r_base['found_k']:>3} {r_base['ari']:.3f}  "
                    f"Δ={delta:+.3f}"
                )
        print(f"  Wins: {wins}")
        for ln in sorted(win_lines, key=lambda x: float(x.split("Δ=")[1]), reverse=True):
            print(ln)
        print(f"\n  Losses: {losses}")
        for ln in sorted(loss_lines, key=lambda x: float(x.split("Δ=")[1])):
            print(ln)
        print(f"\n  Score: {wins}-{losses}-{ties} (win-loss-tie)\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Per-case detail
    # ═════════════════════════════════════════════════════════════════════════

    print("═══ PER-CASE DETAIL ═══\n")
    detail_strats = [
        "production",
        "jl_floor_qrt",
        "lam12_raw",
        "lam12_floor_qrt",
        "lam12_floor_8th",
        "lam12_floor_3",
    ]
    header = f"  {'Case':<40} {'TK':>3}"
    for s in detail_strats:
        header += f" │ {s[:14]:>14}"
    print(header)
    print("  " + "─" * (44 + 17 * len(detail_strats)))

    for case_name, res in all_results:
        tk = res[CONFIG_NAMES[0]]["true_k"]
        line = f"  {case_name:<40} {str(tk):>3}"
        for cn in detail_strats:
            r = res[cn]
            fk = r["found_k"]
            ari = r["ari"]
            check = "✓" if isinstance(fk, int) and tk != "?" and fk == tk else " "
            line += f" │ K={fk:>3} {ari:.3f}{check}"
        print(line)

    # ═════════════════════════════════════════════════════════════════════════
    # Floor comparison: which floor is best?
    # ═════════════════════════════════════════════════════════════════════════

    print("\n═══ FLOOR COMPARISON ═══\n")
    floor_strats = [
        "lam12_raw",
        "lam12_floor_3",
        "lam12_floor_8th",
        "lam12_floor_qrt",
        "lam12_floor_min",
    ]
    print(
        f"  {'Floor variant':<18} │ {'Mean ARI':>9} │ {'Exact K':>9} │ "
        f"{'K=1':>5} │ {'Perfect':>8} │ {'Wins':>5}"
    )
    print("  " + "─" * 70)
    for cn in floor_strats:
        aris = strat_aris[cn]
        print(
            f"  {cn:<18} │ {np.mean(aris):9.3f} │ "
            f"{strat_exact[cn]:5d}/{countable:<3d} │ "
            f"{strat_k1[cn]:5d} │ {strat_perfect[cn]:8d} │ {strat_wins[cn]:5d}"
        )

    # Pairwise: which cases differ between floor variants?
    print("\n  Cases where floor matters (lam12_floor_qrt ≠ lam12_raw):\n")
    for case_name, res in all_results:
        r_raw = res["lam12_raw"]
        r_qrt = res["lam12_floor_qrt"]
        if abs(r_raw["ari"] - r_qrt["ari"]) > 0.005:
            tk = r_raw["true_k"]
            delta = r_qrt["ari"] - r_raw["ari"]
            saved = "SAVED" if delta > 0 else "HURT"
            print(
                f"    {saved:5s} {case_name:<40} TK={tk:>3}  "
                f"raw: K={r_raw['found_k']:>3} {r_raw['ari']:.3f}  "
                f"qrt: K={r_qrt['found_k']:>3} {r_qrt['ari']:.3f}  "
                f"Δ={delta:+.3f}"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # Final ranking
    # ═════════════════════════════════════════════════════════════════════════

    print("\n═══ FINAL RANKING ═══\n")
    ranking = sorted(
        CONFIG_NAMES,
        key=lambda cn: (float(np.mean(strat_aris[cn])), strat_exact[cn]),
        reverse=True,
    )
    for rank, cn in enumerate(ranking, 1):
        aris = strat_aris[cn]
        print(
            f"  {rank}. {cn:<18}  Mean ARI={np.mean(aris):.3f}  "
            f"Med={np.median(aris):.3f}  Exact K={strat_exact[cn]}/{countable}  "
            f"K=1={strat_k1[cn]}  Perfect={strat_perfect[cn]}  Wins={strat_wins[cn]}"
        )
