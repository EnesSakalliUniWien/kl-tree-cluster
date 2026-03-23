"""Lab: compare conditional deflation strategies for the adjusted Wald test.

Tests multiple strategies for computing a per-node deflation factor
c_i = 1 + (c_global - 1) * r_i, where r_i ∈ [0,1] controls how much
of the global deflation is applied to each node.

The bounded interpolation guarantees 1 ≤ c_i ≤ c_global, so these
strategies can only REDUCE over-deflation — they cannot cause new K=1
collapses compared to the global-constant baseline.

Strategies:
  1. global           — current production: c_i = c_global for all nodes
  2. no_deflation     — c_i = 1.0 (raw Wald, maximum splitting power)
  3. half_global      — c_i = 1 + (c_global - 1) * 0.5 (constant 50% reduction)
  4. bounded_df_mismatch — r_i = clip(1 - α·|log(k_i / k_pool_median)|, 0, 1)
  5. bounded_ratio_cap   — r_i = clip(raw_ratio / c_global, 0, 1)
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple
from unittest.mock import patch

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data, compute_ari

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    adjusted_wald_annotation as awa_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    CalibrationModel,
    fit_inflation_model,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    SiblingPairRecord,
    deflate_focal_pairs,
)

# ── Pool statistics computed once per case ─────────────────────────────────


@dataclass(frozen=True)
class PoolStats:
    """Summary statistics from the calibration pool for strategy use."""

    c_global: float
    median_log_df: float  # weighted median of log(df) across all valid records
    median_df: float  # exp(median_log_df)
    n_records: int


def compute_pool_stats(records: List[SiblingPairRecord], model: CalibrationModel) -> PoolStats:
    """Compute pool-level statistics from all sibling pair records."""
    valid = [r for r in records if np.isfinite(r.stat) and r.degrees_of_freedom > 0]
    if not valid:
        return PoolStats(
            c_global=model.global_inflation_factor,
            median_log_df=0.0,
            median_df=1.0,
            n_records=0,
        )

    log_dfs = np.array([np.log(max(r.degrees_of_freedom, 1)) for r in valid])
    weights = np.array([r.edge_weight for r in valid])

    # Weighted median via sorted cumulative weights
    sort_idx = np.argsort(log_dfs)
    sorted_log_dfs = log_dfs[sort_idx]
    sorted_weights = weights[sort_idx]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    if total_weight > 0:
        median_idx = np.searchsorted(cum_weights, total_weight / 2.0)
        median_idx = min(median_idx, len(sorted_log_dfs) - 1)
        median_log_df = float(sorted_log_dfs[median_idx])
    else:
        median_log_df = float(np.median(log_dfs))

    return PoolStats(
        c_global=model.global_inflation_factor,
        median_log_df=median_log_df,
        median_df=float(np.exp(median_log_df)),
        n_records=len(valid),
    )


# ── Strategy definitions ───────────────────────────────────────────────────

StrategyFn = Callable[[SiblingPairRecord, CalibrationModel, PoolStats], float]
"""Strategy signature: (record, model, pool_stats) → c_i."""


def _strategy_global(rec: SiblingPairRecord, model: CalibrationModel, pool: PoolStats) -> float:
    """Baseline: global constant c_hat for all nodes."""
    return model.global_inflation_factor


def _strategy_no_deflation(
    rec: SiblingPairRecord, model: CalibrationModel, pool: PoolStats
) -> float:
    """No deflation at all — raw Wald statistic."""
    return 1.0


def _strategy_half_global(
    rec: SiblingPairRecord, model: CalibrationModel, pool: PoolStats
) -> float:
    """Constant 50% interpolation: c_i = 1 + (c_global - 1) * 0.5."""
    return 1.0 + (pool.c_global - 1.0) * 0.5


def _make_df_mismatch_strategy(alpha: float) -> StrategyFn:
    """Factory: r_i = clip(1 - alpha * |log(k_i / k_pool_median)|, 0, 1).

    Nodes whose df matches the calibration pool median get full deflation.
    Nodes whose df is far from the pool median get reduced deflation.
    """

    def _strategy(rec: SiblingPairRecord, model: CalibrationModel, pool: PoolStats) -> float:
        k_i = max(rec.degrees_of_freedom, 1)
        k_median = max(pool.median_df, 1.0)
        df_mismatch = abs(np.log(k_i / k_median))
        r_i = float(np.clip(1.0 - alpha * df_mismatch, 0.0, 1.0))
        return 1.0 + (pool.c_global - 1.0) * r_i

    _strategy.__doc__ = f"df-mismatch (α={alpha})"
    _strategy.__name__ = f"df_mismatch_a{alpha}"
    return _strategy


def _strategy_ratio_cap(rec: SiblingPairRecord, model: CalibrationModel, pool: PoolStats) -> float:
    """Self-anchored: r_i = clip(raw_ratio / c_global, 0, 1).

    Nodes whose raw T/k is already below c_global get proportionally
    less deflation. Nodes at or above c_global get full deflation.
    """
    if rec.degrees_of_freedom <= 0 or pool.c_global <= 1.0:
        return pool.c_global
    raw_ratio = rec.stat / rec.degrees_of_freedom
    r_i = float(np.clip(raw_ratio / pool.c_global, 0.0, 1.0))
    return 1.0 + (pool.c_global - 1.0) * r_i


STRATEGIES: dict[str, StrategyFn] = {
    "global": _strategy_global,
    "no_deflation": _strategy_no_deflation,
    "half_global": _strategy_half_global,
    "df_mis_0.3": _make_df_mismatch_strategy(0.3),
    "df_mis_0.5": _make_df_mismatch_strategy(0.5),
    "df_mis_0.7": _make_df_mismatch_strategy(0.7),
    "ratio_cap": _strategy_ratio_cap,
}


# ── Monkey-patch mechanism ─────────────────────────────────────────────────

# Module-level stash for records + model captured mid-pipeline.
_captured: dict[str, object] = {}


def _capturing_fit_inflation_model(records: List[SiblingPairRecord]) -> CalibrationModel:
    """Wrapper around fit_inflation_model that stashes records + model."""
    model = fit_inflation_model(records)
    _captured["records"] = records
    _captured["model"] = model
    _captured["pool_stats"] = compute_pool_stats(records, model)
    return model


def _make_patched_deflate_and_test(
    strategy_fn: StrategyFn,
) -> Callable:
    """Build a replacement for _deflate_and_test that uses the given strategy."""

    def _patched_deflate_and_test(
        records: List[SiblingPairRecord],
        model: CalibrationModel,
    ) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
        pool = _captured.get("pool_stats")
        if pool is None:
            pool = compute_pool_stats(records, model)

        def _resolve(rec: SiblingPairRecord) -> tuple[float, str]:
            c_i = strategy_fn(rec, model, pool)
            return c_i, "conditional_experiment"

        return deflate_focal_pairs(records, calibration_resolver=_resolve)

    return _patched_deflate_and_test


@contextmanager
def apply_strategy(strategy_fn: StrategyFn):
    """Context manager that patches the deflation pipeline for one strategy."""
    patched_deflate = _make_patched_deflate_and_test(strategy_fn)
    _captured.clear()
    with (
        patch.object(awa_module, "_deflate_and_test", patched_deflate),
        patch(
            "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence"
            ".adjusted_wald_annotation.fit_inflation_model",
            _capturing_fit_inflation_model,
        ),
    ):
        yield


# ── Case list ──────────────────────────────────────────────────────────────

CASES = [
    # Primary target — K=1 failure due to over-deflation
    "overlap_unbal_4c_small",  # true K=4, found K=1, ARI=0.000
    # Must-not-regress guards (ARI=1.0 in baseline)
    "binary_perfect_2c",  # K=2/2
    "binary_perfect_4c",  # K=4/4
    "binary_low_noise_4c",  # K=4/4
    "gauss_clear_large",  # K=5/5
    "gauss_null_small",  # K=1/1 (null case)
    "gauss_noisy_3c",  # K=3/3
    # Regressed in df-proximity experiment (must stay stable)
    "binary_balanced_low_noise",  # K=3/4, ARI=0.705
    "cat_clear_4cat_4c",  # K=1/4, ARI=0.000
    "cat_highcard_20cat_4c",  # K=1/4, ARI=0.000
    "overlap_unbal_8c_large",  # K=6/8, ARI=0.714
    # Intermediate cases
    "overlap_mod_4c_small",  # K=14/4, ARI=0.672
    "sparse_features_72x72",  # K=3/4, ARI=0.705
    "gauss_clear_small",  # K=2/3, ARI=0.554
]

# ── Runner ─────────────────────────────────────────────────────────────────


def run_case_strategy(case_name: str, strategy_fn: StrategyFn) -> dict:
    """Run one case with a given deflation strategy.

    Rebuilds tree from scratch (decompose mutates internal state).
    """
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    with apply_strategy(strategy_fn):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")

    # Collect strategy diagnostics
    pool = _captured.get("pool_stats")
    model = _captured.get("model")

    return {
        "true_k": tc.get("n_clusters", "?"),
        "found_k": decomp["num_clusters"],
        "ari": round(ari, 3),
        "c_global": round(model.global_inflation_factor, 2) if model else None,
        "pool_median_df": round(pool.median_df, 1) if pool else None,
        "pool_n": pool.n_records if pool else None,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print()

    strat_names = list(STRATEGIES.keys())
    n_strats = len(strat_names)

    # Header
    col_w = 10
    header = f"{'Case':<35} {'TK':>3}"
    for s in strat_names:
        header += f" | {'K':>3} {'ARI':>5}"
    print(header)
    header2 = f"{'':>35} {'':>3}"
    for s in strat_names:
        header2 += f" | {s:>{col_w}}"
    print(header2)
    print("-" * len(header))

    # Collect results
    all_results: dict[str, dict[str, dict]] = {}

    for name in CASES:
        true_k = None
        row = f"{name:<35}"
        case_diag_printed = False
        for sname in strat_names:
            try:
                r = run_case_strategy(name, STRATEGIES[sname])
                if true_k is None:
                    true_k = r["true_k"]
                    row = f"{name:<35} {true_k:>3}"
                all_results.setdefault(name, {})[sname] = r
                row += f" | {r['found_k']:>3} {r['ari']:>5.3f}"

                # Print diagnostics once per case (from baseline strategy)
                if not case_diag_printed and sname == "global" and r["c_global"] is not None:
                    print(
                        f"  [{name}] c_global={r['c_global']}, "
                        f"pool_median_df={r['pool_median_df']}, "
                        f"pool_n={r['pool_n']}",
                        file=sys.stderr,
                    )
                    case_diag_printed = True
            except Exception as e:
                row += " |  ERR   "
                print(f"  [ERROR] {name}/{sname}: {e}", file=sys.stderr)
        print(row)

    # Summary: mean ARI per strategy
    print()
    print(f"{'Mean ARI':<35} {'':>3}", end="")
    for sname in strat_names:
        aris = [
            all_results[c][sname]["ari"]
            for c in CASES
            if c in all_results and sname in all_results[c]
        ]
        mean_ari = np.mean(aris) if aris else float("nan")
        print(f" | {'':>3} {mean_ari:>5.3f}", end="")
    print()

    # Exact K count
    print(f"{'Exact K':<35} {'':>3}", end="")
    for sname in strat_names:
        exact = sum(
            1
            for c in CASES
            if c in all_results
            and sname in all_results[c]
            and all_results[c][sname]["found_k"] == all_results[c][sname]["true_k"]
        )
        total = sum(1 for c in CASES if c in all_results and sname in all_results[c])
        print(f" | {exact:>3}/{total:<5}", end="")
    print()

    # K=1 count (over-deflation indicator)
    print(f"{'K=1 count':<35} {'':>3}", end="")
    for sname in strat_names:
        k1 = sum(
            1
            for c in CASES
            if c in all_results
            and sname in all_results[c]
            and all_results[c][sname]["found_k"] == 1
            and all_results[c][sname]["true_k"] != 1
        )
        print(f" | {k1:>3}{'':>6}", end="")
    print()

    # Regressions vs baseline (global strategy)
    print()
    print("Regressions vs global baseline (ARI drop > 0.01):")
    for sname in strat_names:
        if sname == "global":
            continue
        regressions = []
        for c in CASES:
            if c not in all_results:
                continue
            if sname not in all_results[c] or "global" not in all_results[c]:
                continue
            ari_base = all_results[c]["global"]["ari"]
            ari_new = all_results[c][sname]["ari"]
            if ari_base - ari_new > 0.01:
                regressions.append(
                    f"    {c}: {ari_base:.3f} → {ari_new:.3f} "
                    f"(K: {all_results[c]['global']['found_k']} → {all_results[c][sname]['found_k']})"
                )
        if regressions:
            print(f"  {sname}: {len(regressions)} regressions")
            for line in regressions:
                print(line)
        else:
            print(f"  {sname}: NO regressions")

    # Improvements vs baseline
    print()
    print("Improvements vs global baseline (ARI gain > 0.01):")
    for sname in strat_names:
        if sname == "global":
            continue
        improvements = []
        for c in CASES:
            if c not in all_results:
                continue
            if sname not in all_results[c] or "global" not in all_results[c]:
                continue
            ari_base = all_results[c]["global"]["ari"]
            ari_new = all_results[c][sname]["ari"]
            if ari_new - ari_base > 0.01:
                improvements.append(
                    f"    {c}: {ari_base:.3f} → {ari_new:.3f} "
                    f"(K: {all_results[c]['global']['found_k']} → {all_results[c][sname]['found_k']})"
                )
        if improvements:
            print(f"  {sname}: {len(improvements)} improvements")
            for line in improvements:
                print(line)
        else:
            print(f"  {sname}: no improvements")

    # Per-case diagnostics for primary target
    print()
    print("=== Primary target: overlap_unbal_4c_small ===")
    if "overlap_unbal_4c_small" in all_results:
        for sname in strat_names:
            if sname in all_results["overlap_unbal_4c_small"]:
                r = all_results["overlap_unbal_4c_small"][sname]
                print(f"  {sname:<20}: K={r['found_k']}, ARI={r['ari']:.3f}")


if __name__ == "__main__":
    main()
