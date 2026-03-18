#!/usr/bin/env python
"""Command-line interface for running KL-TE benchmarks.

Usage:
    python run_benchmark.py [OPTIONS]

Examples:
    # Run full benchmark (95 cases, 2 methods)
    python run_benchmark.py --full

    # Run quick sanity check (5 cases)
    python run_benchmark.py --quick

    # Run specific test case
    python run_benchmark.py --case binary_perfect_4c

    # Run with custom config
    python run_benchmark.py --full --sibling-method wald --edge-alpha 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def run_single_case(
    case_name: str,
    alpha_local: float = None,
    sibling_alpha: float = None,
    sibling_method: str = None,
    spectral_method: str = None,
    verbose: bool = True,
) -> dict:
    """Run KL-TE pipeline on a single test case.

    Parameters
    ----------
    case_name : str
        Name of the test case from get_default_test_cases().
    alpha_local : float, optional
        Gate 2 significance level (default: config.EDGE_ALPHA).
    sibling_alpha : float, optional
        Gate 3 significance level (default: config.SIBLING_ALPHA).
    sibling_method : str, optional
        Sibling test method (default: config.SIBLING_TEST_METHOD).
    spectral_method : str, optional
        Spectral dimension method (default: config.SPECTRAL_METHOD).
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict
        Results: {case, true_k, found_k, ari, elapsed_sec, config}
    """
    # Override config if specified
    orig_edge_alpha = config.EDGE_ALPHA
    orig_sibling_alpha = config.SIBLING_ALPHA
    orig_sibling_method = config.SIBLING_TEST_METHOD
    orig_spectral_method = config.SPECTRAL_METHOD

    if alpha_local is not None:
        config.EDGE_ALPHA = alpha_local
    if sibling_alpha is not None:
        config.SIBLING_ALPHA = sibling_alpha
    if sibling_method is not None:
        config.SIBLING_TEST_METHOD = sibling_method
    if spectral_method is not None:
        config.SPECTRAL_METHOD = spectral_method

    try:
        # Get test case
        cases = get_default_test_cases()
        tc = next((c for c in cases if c["name"] == case_name), None)
        if tc is None:
            available = [c["name"] for c in cases[:20]]
            raise ValueError(f"Case '{case_name}' not found. Available: {available}...")

        t0 = time.time()

        # Generate data
        data, y_true, _, _ = generate_case_data(tc)

        # Build tree
        Z = linkage(
            pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

        # Run decomposition
        results = tree.decompose(
            leaf_data=data,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

        # Compute ARI
        n = len(data)
        y_pred = np.full(n, -1, dtype=int)
        for cid, cinfo in results["cluster_assignments"].items():
            for leaf in cinfo["leaves"]:
                idx = data.index.get_loc(leaf)
                y_pred[idx] = cid
        ari = adjusted_rand_score(y_true, y_pred) if y_true is not None else float("nan")

        elapsed = time.time() - t0

        result = {
            "case": case_name,
            "true_k": tc.get("n_clusters", "?"),
            "found_k": results["num_clusters"],
            "ari": ari,
            "elapsed_sec": elapsed,
            "config": {
                "edge_alpha": config.EDGE_ALPHA,
                "sibling_alpha": config.SIBLING_ALPHA,
                "sibling_method": config.SIBLING_TEST_METHOD,
                "spectral_method": config.SPECTRAL_METHOD,
            },
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Case: {case_name}")
            print(f"{'='*60}")
            print(f"True K:      {tc.get('n_clusters', '?')}")
            print(f"Found K:     {results['num_clusters']}")
            print(f"ARI:         {ari:.4f}")
            print(f"Elapsed:     {elapsed:.2f}s")
            print("\nConfig:")
            print(f"  EDGE_ALPHA={config.EDGE_ALPHA}")
            print(f"  SIBLING_ALPHA={config.SIBLING_ALPHA}")
            print(f"  SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")
            print(f"  SPECTRAL_METHOD={config.SPECTRAL_METHOD}")

        return result

    finally:
        # Restore config
        config.EDGE_ALPHA = orig_edge_alpha
        config.SIBLING_ALPHA = orig_sibling_alpha
        config.SIBLING_TEST_METHOD = orig_sibling_method
        config.SPECTRAL_METHOD = orig_spectral_method


def run_quick_benchmark(
    alpha_local: float = None,
    sibling_alpha: float = None,
    sibling_method: str = None,
) -> list[dict]:
    """Run quick sanity check on 5 representative cases."""
    quick_cases = [
        "binary_perfect_4c",
        "gauss_clear_medium",
        "sparse_72x72",
        "sbm_moderate",
        "overlap_mod_4c_small",
    ]

    results = []
    for case_name in quick_cases:
        try:
            result = run_single_case(
                case_name,
                alpha_local=alpha_local,
                sibling_alpha=sibling_alpha,
                sibling_method=sibling_method,
                verbose=True,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR {case_name}: {e}")
            results.append({"case": case_name, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("QUICK BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Case':<35} {'True K':>7} {'Found K':>9} {'ARI':>8}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['case']:<35} {'ERR':>7} {'ERR':>9} {'ERR':>8}")
        else:
            print(f"{r['case']:<35} {str(r['true_k']):>7} {str(r['found_k']):>9} {r['ari']:>8.3f}")

    valid_aris = [r["ari"] for r in results if "ari" in r and not isinstance(r["ari"], str)]
    if valid_aris:
        print(f"\nMean ARI: {sum(valid_aris)/len(valid_aris):.3f}")
        print(f"Median ARI: {sorted(valid_aris)[len(valid_aris)//2]:.3f}")

    return results


def run_full_benchmark(
    alpha_local: float = None,
    sibling_alpha: float = None,
    sibling_method: str = None,
    spectral_method: str = None,
) -> list[dict]:
    """Run full benchmark suite (95 cases)."""
    print("Running full benchmark (95 cases)...")
    print(
        f"Config: EDGE_ALPHA={alpha_local or config.EDGE_ALPHA}, "
        f"SIBLING_ALPHA={sibling_alpha or config.SIBLING_ALPHA}, "
        f"SIBLING_METHOD={sibling_method or config.SIBLING_TEST_METHOD}"
    )
    print()

    # Delegate to full benchmark runner
    from benchmarks.full.run import run_full_benchmark as _run_full

    return _run_full(
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
        sibling_method=sibling_method,
        spectral_method=spectral_method,
    )


def main():
    parser = argparse.ArgumentParser(
        description="KL-TE Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (95 cases, ~15 min)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity check (5 cases, ~30 sec)",
    )

    parser.add_argument(
        "--case",
        type=str,
        help="Run a single test case by name",
    )

    parser.add_argument(
        "--edge-alpha",
        type=float,
        default=None,
        help=f"Gate 2 significance level (default: {config.EDGE_ALPHA})",
    )

    parser.add_argument(
        "--sibling-alpha",
        type=float,
        default=None,
        help=f"Gate 3 significance level (default: {config.SIBLING_ALPHA})",
    )

    parser.add_argument(
        "--sibling-method",
        type=str,
        choices=["wald", "cousin_adjusted_wald", "cousin_weighted_wald"],
        default=None,
        help=f"Sibling test method (default: {config.SIBLING_TEST_METHOD})",
    )

    parser.add_argument(
        "--spectral-method",
        type=str,
        choices=["marchenko_pastur", "none"],
        default=None,
        help=f"Spectral dimension method (default: {config.SPECTRAL_METHOD})",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: benchmarks/results/)",
    )

    args = parser.parse_args()

    # Default to quick if no mode specified
    if not args.full and not args.case:
        args.quick = True

    if args.case:
        run_single_case(
            args.case,
            alpha_local=args.edge_alpha,
            sibling_alpha=args.sibling_alpha,
            sibling_method=args.sibling_method,
            spectral_method=args.spectral_method,
            verbose=True,
        )
    elif args.quick:
        run_quick_benchmark(
            alpha_local=args.edge_alpha,
            sibling_alpha=args.sibling_alpha,
            sibling_method=args.sibling_method,
        )
    elif args.full:
        run_full_benchmark(
            alpha_local=args.edge_alpha,
            sibling_alpha=args.sibling_alpha,
            sibling_method=args.sibling_method,
            spectral_method=args.spectral_method,
        )


if __name__ == "__main__":
    main()
