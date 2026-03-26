"""Lab: compare multiple sibling spectral-dim derivation strategies.

Tests 8 strategies for how Gate 3 (sibling test) picks its projection
dimension, ranging from pure JL to various spectral combinations.

Strategies:
  1. none          — JL fallback only (no spectral override)
  2. min_child     — min(k_L, k_R)  [current HEAD, causes regression]
  3. max_child     — max(k_L, k_R)
  4. sum_child     — k_L + k_R
  5. max_child_2x  — 2 × max(k_L, k_R)
  6. sum_child_2x  — 2 × (k_L + k_R)
  7. jl_floor_half — max(spectral_min, JL_dim // 2)
  8. jl_floor_qrt  — max(spectral_min, JL_dim // 4)
"""

from __future__ import annotations

import sys
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
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_spectral_dims as current_derive_sibling_spectral_dims,
)

# Sentinel cases: 5 improving, 2 going the other direction
CASES = [
    "binary_balanced_low_noise__2",  # ARI 0.000 → 1.000 (worst)
    "gauss_clear_small",  # ARI 0.554 → 1.000
    "binary_low_noise_12c",  # ARI 0.614 → 1.000
    "binary_perfect_8c",  # ARI 0.757 → 1.000
    "binary_hard_4c",  # ARI 0.708 → 0.950
    "gauss_noisy_3c",  # ARI 1.000 → 0.927 (reverse)
    "gauss_overlap_4c_med",  # ARI 1.000 → 0.850 (reverse)
    "gauss_moderate_3c",  # guard
    "gauss_moderate_5c",  # guard
    "binary_low_noise_4c",  # guard
    "binary_perfect_4c",  # guard
    "gauss_clear_large",  # guard
    "binary_multiscale_4c",  # intermediate
    "binary_many_features",  # intermediate
    "gauss_noisy_many",  # intermediate
]

orig_derive = current_derive_sibling_spectral_dims


# ── Strategy derivation functions ──────────────────────────────────────────


def _derive_none(tree, annotated_df):
    """Strategy: none — pure JL fallback."""
    return None


def _derive_min_child(tree, annotated_df):
    """Strategy: min_child — min(k_L, k_R). Current HEAD."""
    return orig_derive(tree, annotated_df)


def _make_combiner(combine_fn):
    """Factory for child-k combination strategies."""

    def _derive(tree, annotated_df):
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
            k = combine_fn(k_left, k_right, positive)
            if k > 0:
                sibling_dims[parent] = k
        return sibling_dims if sibling_dims else None

    return _derive


def _derive_max_child():
    return _make_combiner(lambda kl, kr, pos: max(pos))


def _derive_sum_child():
    return _make_combiner(lambda kl, kr, pos: sum(pos))


def _derive_max_child_2x():
    return _make_combiner(lambda kl, kr, pos: 2 * max(pos))


def _derive_sum_child_2x():
    return _make_combiner(lambda kl, kr, pos: 2 * sum(pos))


def _make_jl_floor_strategy(fraction):
    """Factory: use spectral min but floor at fraction × JL dim."""

    def _derive(tree, annotated_df):
        edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
        if not edge_spectral_dims:
            return None

        # We need to estimate JL dim for each parent based on its leaf count
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

            # Get leaf counts for JL estimate
            n_left = tree.nodes[left].get("leaf_count", 1)
            n_right = tree.nodes[right].get("leaf_count", 1)
            n_parent = n_left + n_right
            # Get feature dimension from distribution
            dist = tree.nodes[left].get("distribution")
            n_features = len(dist) if dist is not None else 100
            jl_k = compute_jl_dim(n_parent, n_features)
            floor_k = max(1, int(jl_k * fraction))

            k_final = max(spectral_k, floor_k)
            sibling_dims[parent] = k_final
        return sibling_dims if sibling_dims else None

    return _derive


STRATEGIES = {
    "none": _derive_none,
    "min_child": _derive_min_child,
    "max_child": _derive_max_child(),
    "sum_child": _derive_sum_child(),
    "max_child_2x": _derive_max_child_2x(),
    "sum_child_2x": _derive_sum_child_2x(),
    "jl_floor_half": _make_jl_floor_strategy(0.5),
    "jl_floor_qrt": _make_jl_floor_strategy(0.25),
}


# ── Runner ─────────────────────────────────────────────────────────────────


def run_case_strategy(case_name: str, strategy_fn) -> dict:
    """Run one case with a given derivation strategy."""
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)
    with temporary_experiment_overrides(sibling_dims=strategy_fn):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    ari = compute_ari(decomp, data_df, true_labels) if true_labels is not None else float("nan")
    return {
        "true_k": tc.get("n_clusters", "?"),
        "found_k": decomp["num_clusters"],
        "ari": round(ari, 3),
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print("        SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)")
    print()

    strat_names = list(STRATEGIES.keys())
    # Header
    header = f"{'Case':<30} {'TK':>3}"
    for s in strat_names:
        header += f" | {'K':>3} {'ARI':>5}"
    print(header)
    header2 = f"{'':<30} {'':>3}"
    for s in strat_names:
        header2 += f" | {s:>9}"
    print(header2)
    print("-" * len(header))

    # Collect all results
    all_results = {}
    for name in CASES:
        row = f"{name:<30}"
        true_k = None
        for sname in strat_names:
            try:
                r = run_case_strategy(name, STRATEGIES[sname])
                if true_k is None:
                    true_k = r["true_k"]
                    row = f"{name:<30} {true_k:>3}"
                all_results.setdefault(name, {})[sname] = r
                row += f" | {r['found_k']:>3} {r['ari']:>5.3f}"
            except Exception as e:
                row += " |  ERR   "
                print(f"  [ERROR] {name}/{sname}: {e}", file=sys.stderr)
        print(row)

    # Summary: mean ARI per strategy
    print()
    print(f"{'Mean ARI':<30} {'':>3}", end="")
    for sname in strat_names:
        aris = [
            all_results[c][sname]["ari"]
            for c in CASES
            if c in all_results and sname in all_results[c]
        ]
        mean_ari = np.mean(aris) if aris else float("nan")
        print(f" | {'':>3} {mean_ari:>5.3f}", end="")
    print()

    # Best strategy per case
    print()
    print("Best strategy per case:")
    for name in CASES:
        if name not in all_results:
            continue
        results = all_results[name]
        best_s = max(results, key=lambda s: results[s]["ari"])
        best_ari = results[best_s]["ari"]
        print(f"  {name:<30} → {best_s:<15} (ARI={best_ari:.3f})")

    # Wins count
    print()
    print("Strategy win counts (best ARI per case):")
    wins = {s: 0 for s in strat_names}
    for name in CASES:
        if name not in all_results:
            continue
        results = all_results[name]
        best_ari = max(results[s]["ari"] for s in results)
        for s in results:
            if abs(results[s]["ari"] - best_ari) < 0.001:
                wins[s] += 1
    for s in strat_names:
        print(f"  {s:<15}: {wins[s]:>2} / {len(CASES)}")
