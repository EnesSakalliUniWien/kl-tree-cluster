"""Historical benchmark: pre-local-kernel conditional deflation sweep.

This benchmark compares legacy df-mismatch heuristics only. The current
production runtime uses local structural-k kernel deflation instead.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np

# Import the shared infrastructure from the Phase 0 experiment
from exp_conditional_deflation import (
    StrategyFn,
    _captured,
    _make_df_mismatch_strategy,
    _strategy_global,
    _strategy_half_global,
    apply_strategy,
)
from lab_helpers import build_tree_and_data, compute_ari, get_case

from benchmarks.shared.cases import get_default_test_cases
from kl_clustering_analysis import config

# Only two strategies for speed — add more if needed
STRATEGIES: dict[str, StrategyFn] = {
    "global": _strategy_global,
    "legacy_df_mis_0.3": _make_df_mismatch_strategy(0.3),
    "half_global": _strategy_half_global,
}

OUTPUT_CSV = Path("/tmp/full_benchmark_conditional_deflation.csv")


def run_case_strategy(case_name: str, strategy_fn: StrategyFn) -> dict:
    """Run one case with a given deflation strategy."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    with apply_strategy(strategy_fn):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
    pool = _captured.get("pool_stats")
    model = _captured.get("model")

    return {
        "true_k": tc.get("n_clusters", "?"),
        "found_k": decomp["num_clusters"],
        "ari": round(ari, 4),
        "c_global": round(model.global_inflation_factor, 2) if model else None,
        "pool_median_df": round(pool.median_df, 1) if pool else None,
    }


def main() -> None:
    all_cases = get_default_test_cases()
    case_names = [c["name"] for c in all_cases]
    strat_names = list(STRATEGIES.keys())
    n_total = len(case_names)

    print(f"Full benchmark: {n_total} cases × {len(strat_names)} strategies")
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    # CSV writer
    csv_rows: list[dict] = []

    # Collect results
    all_results: dict[str, dict[str, dict]] = {}
    t0 = time.time()

    for idx, name in enumerate(case_names, 1):
        tc = get_case(name)
        true_k = tc.get("n_clusters") or "?"
        tk_str = str(true_k) if true_k != "?" else "?"
        row_parts = []

        for sname in strat_names:
            try:
                r = run_case_strategy(name, STRATEGIES[sname])
                all_results.setdefault(name, {})[sname] = r
                row_parts.append(f"{r['found_k']:>3} {r['ari']:>6.3f}")

                csv_rows.append(
                    {
                        "case": name,
                        "strategy": sname,
                        "true_k": r["true_k"],
                        "found_k": r["found_k"],
                        "ari": r["ari"],
                        "c_global": r["c_global"],
                        "pool_median_df": r["pool_median_df"],
                    }
                )
            except Exception as e:
                row_parts.append("  ERR    ")
                print(f"  [ERROR] {name}/{sname}: {e}", file=sys.stderr)

        elapsed = time.time() - t0
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (n_total - idx) / rate if rate > 0 else 0
        print(
            f"[{idx:>3}/{n_total}] {name:<40} TK={tk_str:>3}  "
            + "  ".join(f"{s}={p}" for s, p in zip(strat_names, row_parts))
            + f"  ({elapsed:.0f}s, ETA {eta:.0f}s)"
        )

    # Write CSV
    if csv_rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV written to {OUTPUT_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for sname in strat_names:
        aris = [
            all_results[c][sname]["ari"]
            for c in case_names
            if c in all_results and sname in all_results[c]
        ]
        exact_k = sum(
            1
            for c in case_names
            if c in all_results
            and sname in all_results[c]
            and all_results[c][sname]["found_k"] == all_results[c][sname]["true_k"]
        )
        k1_false = sum(
            1
            for c in case_names
            if c in all_results
            and sname in all_results[c]
            and all_results[c][sname]["found_k"] == 1
            and all_results[c][sname]["true_k"] != 1
        )
        total = len(aris)
        mean_ari = np.mean(aris) if aris else float("nan")
        median_ari = np.median(aris) if aris else float("nan")
        print(
            f"  {sname:<15}: Mean ARI={mean_ari:.4f}, Median ARI={median_ari:.4f}, "
            f"Exact K={exact_k}/{total}, K=1 false={k1_false}"
        )

    # Pairwise diff: legacy_df_mis_0.3 vs global
    if "global" in STRATEGIES and "legacy_df_mis_0.3" in STRATEGIES:
        print("\n--- legacy_df_mis_0.3 vs global ---")
        improvements = []
        regressions = []
        for c in case_names:
            if c not in all_results:
                continue
            if "global" not in all_results[c] or "legacy_df_mis_0.3" not in all_results[c]:
                continue
            a_base = all_results[c]["global"]["ari"]
            a_new = all_results[c]["legacy_df_mis_0.3"]["ari"]
            diff = a_new - a_base
            if diff > 0.005:
                improvements.append((c, a_base, a_new, diff))
            elif diff < -0.005:
                regressions.append((c, a_base, a_new, diff))

        print(f"  Improvements ({len(improvements)}):")
        for c, a_base, a_new, diff in sorted(improvements, key=lambda x: -x[3]):
            k_base = all_results[c]["global"]["found_k"]
            k_new = all_results[c]["legacy_df_mis_0.3"]["found_k"]
            k_true = all_results[c]["global"]["true_k"]
            print(
                f"    {c:<40} ARI {a_base:.3f}→{a_new:.3f} (+{diff:.3f})  K: {k_base}→{k_new} (true={k_true})"
            )

        print(f"  Regressions ({len(regressions)}):")
        for c, a_base, a_new, diff in sorted(regressions, key=lambda x: x[3]):
            k_base = all_results[c]["global"]["found_k"]
            k_new = all_results[c]["legacy_df_mis_0.3"]["found_k"]
            k_true = all_results[c]["global"]["true_k"]
            print(
                f"    {c:<40} ARI {a_base:.3f}→{a_new:.3f} ({diff:.3f})  K: {k_base}→{k_new} (true={k_true})"
            )

    # Pairwise diff: half_global vs global
    if "global" in STRATEGIES and "half_global" in STRATEGIES:
        print("\n--- half_global vs global ---")
        improvements = []
        regressions = []
        for c in case_names:
            if c not in all_results:
                continue
            if "global" not in all_results[c] or "half_global" not in all_results[c]:
                continue
            a_base = all_results[c]["global"]["ari"]
            a_new = all_results[c]["half_global"]["ari"]
            diff = a_new - a_base
            if diff > 0.005:
                improvements.append((c, a_base, a_new, diff))
            elif diff < -0.005:
                regressions.append((c, a_base, a_new, diff))

        print(f"  Improvements ({len(improvements)}):")
        for c, a_base, a_new, diff in sorted(improvements, key=lambda x: -x[3]):
            k_base = all_results[c]["global"]["found_k"]
            k_new = all_results[c]["half_global"]["found_k"]
            k_true = all_results[c]["global"]["true_k"]
            print(
                f"    {c:<40} ARI {a_base:.3f}→{a_new:.3f} (+{diff:.3f})  K: {k_base}→{k_new} (true={k_true})"
            )

        print(f"  Regressions ({len(regressions)}):")
        for c, a_base, a_new, diff in sorted(regressions, key=lambda x: x[3]):
            k_base = all_results[c]["global"]["found_k"]
            k_new = all_results[c]["half_global"]["found_k"]
            k_true = all_results[c]["global"]["true_k"]
            print(
                f"    {c:<40} ARI {a_base:.3f}→{a_new:.3f} ({diff:.3f})  K: {k_base}→{k_new} (true={k_true})"
            )

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
