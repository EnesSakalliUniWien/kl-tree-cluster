"""Compare sibling test methods on the full benchmark suite.

Runs each test case in a subprocess to isolate segfaults,
comparing cousin_adjusted_wald vs cousin_weighted_wald.

Usage:
    python benchmarks/compare_sibling_methods.py
    python benchmarks/compare_sibling_methods.py --case 5   # run single case
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

warnings.filterwarnings("ignore")

METHODS = ["cousin_adjusted_wald", "cousin_weighted_wald"]

MAX_SAMPLES_THRESH = 700


def _run_single_case_index(case_idx: int) -> list[dict]:
    """Run both methods on case at index case_idx. Called in subprocess."""
    cases = get_default_test_cases()
    case = cases[case_idx]
    case_name = case.get("name", f"case_{case_idx}")

    data_df, true_labels, _x_original, meta = generate_case_data(case)
    n_true = int(case.get("n_clusters", len(np.unique(true_labels))))

    if len(data_df) > MAX_SAMPLES_THRESH:
        return [
            {
                "case": case_name,
                "method": m,
                "k_true": n_true,
                "k_found": -1,
                "ari": -1.0,
                "n_samples": len(data_df),
                "error": "skipped_large",
            }
            for m in METHODS
        ]

    precomputed_distance_condensed = meta.get("precomputed_distance_condensed")
    precomputed_distance_matrix = meta.get("precomputed_distance_matrix")
    if precomputed_distance_condensed is not None:
        dist = np.asarray(precomputed_distance_condensed, dtype=float)
    elif precomputed_distance_matrix is not None:
        distance_matrix = np.asarray(precomputed_distance_matrix, dtype=float)
        np.fill_diagonal(distance_matrix, 0.0)
        dist = squareform(distance_matrix, checks=False)
    else:
        dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)

    rows = []
    for method in METHODS:
        original = config.SIBLING_TEST_METHOD
        try:
            config.SIBLING_TEST_METHOD = method
            Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
            tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
            result = tree.decompose(
                leaf_data=data_df,
                alpha_local=config.ALPHA_LOCAL,
                sibling_alpha=config.SIBLING_ALPHA,
            )
            k_found = int(result.get("num_clusters", 1))
            pred_labels = np.array(_labels_from_decomposition(result, data_df.index.tolist()))
            ari = adjusted_rand_score(true_labels, pred_labels)
            rows.append(
                {
                    "case": case_name,
                    "method": method,
                    "k_true": n_true,
                    "k_found": k_found,
                    "ari": round(float(ari), 3),
                    "n_samples": len(data_df),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "case": case_name,
                    "method": method,
                    "k_true": n_true,
                    "k_found": -1,
                    "ari": -1.0,
                    "n_samples": len(data_df),
                    "error": str(e),
                }
            )
        finally:
            config.SIBLING_TEST_METHOD = original
    return rows


def _run_case_in_subprocess(case_idx: int, timeout: int = 120) -> list[dict]:
    """Run a single case in a subprocess for crash isolation."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    cmd = [sys.executable, str(script_path), "--case", str(case_idx)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )
        if proc.returncode != 0:
            cases = get_default_test_cases()
            name = cases[case_idx].get("name", f"case_{case_idx}")
            return [
                {
                    "case": name,
                    "method": m,
                    "k_true": -1,
                    "k_found": -1,
                    "ari": -1.0,
                    "n_samples": 0,
                    "error": f"crash(rc={proc.returncode})",
                }
                for m in METHODS
            ]
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        cases = get_default_test_cases()
        name = cases[case_idx].get("name", f"case_{case_idx}")
        return [
            {
                "case": name,
                "method": m,
                "k_true": -1,
                "k_found": -1,
                "ari": -1.0,
                "n_samples": 0,
                "error": "timeout",
            }
            for m in METHODS
        ]


def main():
    # Subprocess mode: run single case and print JSON
    if "--case" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--case") + 1])
        rows = _run_single_case_index(idx)
        print(json.dumps(rows))
        return

    test_cases = get_default_test_cases()
    n_cases = len(test_cases)
    print(
        f"Running {n_cases} test cases × {len(METHODS)} methods = "
        f"{n_cases * len(METHODS)} runs (subprocess-isolated)\n"
    )

    rows = []
    t0 = time.time()
    crashed = []

    for i in range(n_cases):
        case_name = test_cases[i].get("name", f"case_{i}")
        case_rows = _run_case_in_subprocess(i)
        for r in case_rows:
            if r.get("error", "").startswith("crash") or r.get("error") == "timeout":
                if case_name not in crashed:
                    crashed.append(case_name)
        rows.extend(case_rows)

        if (i + 1) % 10 == 0 or i == n_cases - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_cases}] {elapsed:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    valid_df = df[df["ari"] >= 0]
    skipped = (
        df[df.get("error", pd.Series(dtype=str)).isin(["skipped_large"])]["case"].unique().tolist()
        if "error" in df.columns
        else []
    )

    # Pivot for side-by-side comparison
    print("\n" + "=" * 100)
    print("SIBLING METHOD COMPARISON: cousin_adjusted_wald vs cousin_weighted_wald")
    print("=" * 100)

    for method in METHODS:
        sub = df[df["method"] == method]
        valid = sub[sub["ari"] >= 0]
        exact_k = (valid["k_true"] == valid["k_found"]).sum()
        k1_cases = (valid["k_found"] == 1).sum()
        print(f"\n  {method}:")
        print(f"    Mean ARI:  {valid['ari'].mean():.3f}")
        print(f"    Median ARI: {valid['ari'].median():.3f}")
        print(f"    Exact K:   {exact_k}/{len(valid)}")
        print(f"    K=1 cases: {k1_cases}")
        print(f"    Mean K:    {valid['k_found'].mean():.1f}")
        print(f"    Errors:    {(sub['ari'] < 0).sum()}")

    # Per-case comparison
    pivot = df.pivot(index="case", columns="method", values="ari")
    if "cousin_adjusted_wald" in pivot.columns and "cousin_weighted_wald" in pivot.columns:
        pivot["diff"] = pivot["cousin_weighted_wald"] - pivot["cousin_adjusted_wald"]
        pivot_k = df.pivot(index="case", columns="method", values="k_found")

        # Sort by diff to show where weighted does better/worse
        pivot = pivot.sort_values("diff", ascending=True)

        print(f"\n{'─' * 100}")
        print(
            f"{'Case':<40s} {'ARI_adj':>8s} {'ARI_wt':>8s} {'Δ':>7s} │ "
            f"{'K_adj':>5s} {'K_wt':>5s} {'K_true':>6s}"
        )
        print(f"{'─' * 100}")

        for case_name in pivot.index:
            ari_adj = pivot.loc[case_name, "cousin_adjusted_wald"]
            ari_wt = pivot.loc[case_name, "cousin_weighted_wald"]
            diff = pivot.loc[case_name, "diff"]
            k_adj = (
                pivot_k.loc[case_name, "cousin_adjusted_wald"]
                if case_name in pivot_k.index
                else "?"
            )
            k_wt = (
                pivot_k.loc[case_name, "cousin_weighted_wald"]
                if case_name in pivot_k.index
                else "?"
            )

            k_true_row = df[(df["case"] == case_name) & (df["method"] == "cousin_adjusted_wald")]
            k_true = int(k_true_row["k_true"].iloc[0]) if len(k_true_row) > 0 else "?"

            marker = ""
            if isinstance(diff, float):
                if diff > 0.05:
                    marker = " ◄ weighted better"
                elif diff < -0.05:
                    marker = " ◄ adjusted better"

            print(
                f"{case_name:<40s} {ari_adj:>8.3f} {ari_wt:>8.3f} {diff:>+7.3f} │ "
                f"{k_adj:>5} {k_wt:>5} {k_true:>6}{marker}"
            )

        # Count wins
        better = (pivot["diff"] > 0.01).sum()
        worse = (pivot["diff"] < -0.01).sum()
        tied = len(pivot) - better - worse
        print(f"\n{'─' * 100}")
        print(f"  Weighted better: {better}  |  Adjusted better: {worse}  |  Tied: {tied}")
        print(f"  Total cases: {len(pivot)}  |  Crashed: {len(crashed)}")
        if crashed:
            print(f"  Crashed cases: {', '.join(crashed)}")
        if skipped:
            print(f"  Skipped (large): {', '.join(skipped)}")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
