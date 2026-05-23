#!/usr/bin/env python3
"""Compare current MP k-selection against a branch-corrected median rule.

This is an experimental benchmark script. It does not modify production code.

The experimental estimator uses the same positive-eigenvalue median idea as the
current implementation, but corrects it in two ways:

1. It applies a median-to-edge correction factor derived from the MP law.
2. It uses the aspect ratio of the matrix that is actually diagonalized:
   - primal branch: rho = d / n
   - dual branch:   rho = n / d

Because the Gate 2 worker imports the estimator symbol at module import time,
this script forces ``KL_TE_N_JOBS=1`` so the monkeypatch is guaranteed to apply
in-process.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force sequential execution so monkeypatched spectral workers stay in-process.
os.environ.setdefault("KL_TE_N_JOBS", "1")

import kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.marchenko_pastur as mp_worker
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation import (
    projection_dimension_estimators as k_estimators,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current Marchenko-Pastur k estimator against an "
            "experimental branch-corrected median rule on benchmark cases."
        )
    )
    parser.add_argument(
        "--case-names",
        type=str,
        default="",
        help="Comma-separated benchmark case names. Default: all default cases.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of cases after filtering.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV path for the per-case comparison table.",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Print unchanged cases in addition to changed ones.",
    )
    return parser.parse_args()


def _mp_density(x: float, rho: float) -> float:
    """Marchenko-Pastur density for 0 < rho <= 1 and unit noise variance."""
    lambda_minus = (1.0 - np.sqrt(rho)) ** 2
    lambda_plus = (1.0 + np.sqrt(rho)) ** 2
    if x <= lambda_minus or x >= lambda_plus:
        return 0.0
    return np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (
        2.0 * np.pi * rho * x
    )


@lru_cache(maxsize=None)
def _mp_positive_median(rho: float) -> float:
    """Median of the positive MP spectrum for 0 < rho <= 1."""
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"Expected aspect ratio in (0, 1], got rho={rho!r}.")

    lambda_minus = (1.0 - np.sqrt(rho)) ** 2
    lambda_plus = (1.0 + np.sqrt(rho)) ** 2
    lo, hi = lambda_minus, lambda_plus

    for _ in range(64):
        mid = (lo + hi) / 2.0
        cdf_at_mid, _ = quad(_mp_density, lambda_minus, mid, args=(rho,))
        if cdf_at_mid < 0.5:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def _branch_corrected_mp_signal_count(
    eigenvalues: np.ndarray,
    n_samples: int,
    n_features: int,
) -> int:
    """Count MP spikes using the scale of the matrix actually diagonalized."""
    if n_samples <= 0 or n_features <= 0:
        return 1

    sorted_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positive_eigenvalues = sorted_eigenvalues[sorted_eigenvalues > 0]
    if positive_eigenvalues.size == 0:
        return 1

    observed_positive_median = float(np.median(positive_eigenvalues))
    if observed_positive_median <= 0.0:
        return 1

    # Match the shape of the matrix the backend actually diagonalized.
    # primal: rho = d / n when n >= d
    # dual:   rho = n / d when n < d
    rho = (
        float(n_features) / float(n_samples)
        if n_samples >= n_features
        else float(n_samples) / float(n_features)
    )

    mp_positive_median = _mp_positive_median(rho)
    lambda_plus = (1.0 + np.sqrt(rho)) ** 2
    threshold = observed_positive_median * (lambda_plus / mp_positive_median)

    n_signal_components = int(np.sum(sorted_eigenvalues > threshold))
    return max(n_signal_components, 1)


def _branch_corrected_estimate_k(
    eigenvalues: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    minimum_projection_dimension: int = 1,
) -> int:
    projection_dimension = _branch_corrected_mp_signal_count(
        eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
    )
    projection_dimension = max(int(projection_dimension), int(minimum_projection_dimension))
    projection_dimension = min(int(projection_dimension), int(n_features))
    return projection_dimension


@contextmanager
def _patched_branch_corrected_mp() -> Iterator[None]:
    """Temporarily patch the spectral worker to use the experimental estimator."""
    original_signal_count = k_estimators.marchenko_pastur_signal_count
    original_estimate_k = k_estimators.estimate_k_marchenko_pastur
    original_worker_estimate_k = mp_worker.estimate_k_marchenko_pastur

    k_estimators.marchenko_pastur_signal_count = _branch_corrected_mp_signal_count
    k_estimators.estimate_k_marchenko_pastur = _branch_corrected_estimate_k
    mp_worker.estimate_k_marchenko_pastur = _branch_corrected_estimate_k
    try:
        yield
    finally:
        k_estimators.marchenko_pastur_signal_count = original_signal_count
        k_estimators.estimate_k_marchenko_pastur = original_estimate_k
        mp_worker.estimate_k_marchenko_pastur = original_worker_estimate_k


def _resolve_cases(case_names_arg: str, max_cases: int | None) -> list[dict[str, object]]:
    all_cases = get_default_test_cases()
    if case_names_arg.strip():
        requested_names = [name.strip() for name in case_names_arg.split(",") if name.strip()]
        requested_set = set(requested_names)
        cases = [case for case in all_cases if str(case["name"]) in requested_set]
        found_names = {str(case["name"]) for case in cases}
        missing = [name for name in requested_names if name not in found_names]
        if missing:
            raise ValueError(f"Unknown case names: {', '.join(missing)}")
    else:
        cases = list(all_cases)

    if max_cases is not None:
        cases = cases[: max(max_cases, 0)]
    return cases


def _safe_ari(y_true: object, y_pred: np.ndarray) -> float:
    if y_true is None:
        return float("nan")
    y_true_array = np.asarray(y_true)
    if y_true_array.shape[0] != y_pred.shape[0]:
        return float("nan")
    try:
        return float(adjusted_rand_score(y_true_array, y_pred))
    except ValueError:
        return float("nan")


def _run_case(case: dict[str, object], *, experimental: bool) -> dict[str, object]:
    data_t, y_true, _, meta, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])
    context = _patched_branch_corrected_mp() if experimental else nullcontext()
    with context:
        result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    annotations = result.extra["annotations"]
    audit = annotations.attrs.get("sibling_divergence_audit", {})
    return {
        "case_name": str(case["name"]),
        "case_category": str(case.get("category", "")),
        "true_clusters": (
            int(case["n_clusters"])
            if isinstance(case.get("n_clusters"), (int, np.integer))
            else (len(np.unique(np.asarray(y_true))) if y_true is not None else 0)
        ),
        "ari": _safe_ari(y_true, result.labels),
        "found_clusters": int(result.found_clusters),
        "projection_sources": dict(audit.get("projection_dimension_source_counts", {})),
        "baseline_empirical_scale_factor": float(
            audit["baseline_empirical_scale_factor"]
        ) if "baseline_empirical_scale_factor" in audit else float("nan"),
    }


def _compare_cases(cases: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        current = _run_case(case, experimental=False)
        experimental = _run_case(case, experimental=True)
        rows.append(
            {
                "case_name": current["case_name"],
                "case_category": current["case_category"],
                "true_clusters": current["true_clusters"],
                "ari_current": current["ari"],
                "ari_branch_corrected": experimental["ari"],
                "ari_delta": experimental["ari"] - current["ari"],
                "clusters_current": current["found_clusters"],
                "clusters_branch_corrected": experimental["found_clusters"],
                "cluster_delta": (
                    experimental["found_clusters"] - current["found_clusters"]
                ),
                "projection_sources_current": current["projection_sources"],
                "projection_sources_branch_corrected": experimental["projection_sources"],
                "baseline_empirical_scale_current": current[
                    "baseline_empirical_scale_factor"
                ],
                "baseline_empirical_scale_branch_corrected": experimental[
                    "baseline_empirical_scale_factor"
                ],
            }
        )
        print(
            f"[{index:>3d}/{len(cases):>3d}] {current['case_name']}: "
            f"clusters {current['found_clusters']} -> {experimental['found_clusters']}, "
            f"ARI {current['ari']:.3f} -> {experimental['ari']:.3f}"
        )
    return pd.DataFrame(rows)


def _print_summary(comparison_df: pd.DataFrame, *, show_unchanged: bool) -> None:
    changed = comparison_df[
        (comparison_df["cluster_delta"] != 0)
        | np.abs(comparison_df["ari_delta"]).fillna(0.0).gt(1e-12)
    ].copy()
    with_truth = comparison_df[comparison_df["true_clusters"] > 0].copy()
    if with_truth.empty:
        exact_k_current = 0
        exact_k_branch_corrected = 0
        oversplit_current = 0
        oversplit_branch_corrected = 0
        undersplit_current = 0
        undersplit_branch_corrected = 0
    else:
        exact_k_current = int(
            (with_truth["clusters_current"] == with_truth["true_clusters"]).sum()
        )
        exact_k_branch_corrected = int(
            (with_truth["clusters_branch_corrected"] == with_truth["true_clusters"]).sum()
        )
        oversplit_current = int(
            (with_truth["clusters_current"] > with_truth["true_clusters"]).sum()
        )
        oversplit_branch_corrected = int(
            (with_truth["clusters_branch_corrected"] > with_truth["true_clusters"]).sum()
        )
        undersplit_current = int(
            (with_truth["clusters_current"] < with_truth["true_clusters"]).sum()
        )
        undersplit_branch_corrected = int(
            (with_truth["clusters_branch_corrected"] < with_truth["true_clusters"]).sum()
        )

    explosion_ratio = (
        comparison_df["clusters_branch_corrected"] / comparison_df["clusters_current"].clip(lower=1)
    )
    max_explosion_index = explosion_ratio.idxmax()
    max_explosion_case = str(comparison_df.loc[max_explosion_index, "case_name"])
    max_explosion_ratio = float(explosion_ratio.loc[max_explosion_index])

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Cases compared: {len(comparison_df)}")
    print(f"Changed cases:   {len(changed)}")
    print(
        "Mean ARI:        "
        f"{comparison_df['ari_current'].mean():.3f} -> "
        f"{comparison_df['ari_branch_corrected'].mean():.3f}"
    )
    print(
        "K=1 cases:       "
        f"{int((comparison_df['clusters_current'] == 1).sum())} -> "
        f"{int((comparison_df['clusters_branch_corrected'] == 1).sum())}"
    )
    print(
        "Exact-K cases:   "
        f"{exact_k_current} -> {exact_k_branch_corrected}"
    )
    print(
        "Over-split:      "
        f"{oversplit_current} -> {oversplit_branch_corrected}"
    )
    print(
        "Under-split:     "
        f"{undersplit_current} -> {undersplit_branch_corrected}"
    )
    print(
        "Max explosion:   "
        f"{max_explosion_ratio:.2f}x ({max_explosion_case})"
    )

    display_df = comparison_df if show_unchanged else changed
    if display_df.empty:
        print("\nNo changed cases.")
        return

    display_df = display_df.sort_values(
        by=["ari_delta", "cluster_delta", "case_name"],
        ascending=[True, True, True],
    )
    top_cluster_delta = changed.reindex(
        changed["cluster_delta"].abs().sort_values(ascending=False).index
    )

    print()
    print("=" * 100)
    print("Per-case comparison")
    print("=" * 100)
    columns = [
        "case_name",
        "case_category",
        "true_clusters",
        "clusters_current",
        "clusters_branch_corrected",
        "cluster_delta",
        "ari_current",
        "ari_branch_corrected",
        "ari_delta",
    ]
    print(display_df[columns].to_string(index=False))

    if not top_cluster_delta.empty:
        print()
        print("=" * 100)
        print("Top Changed Cases By |cluster_delta|")
        print("=" * 100)
        top_columns = [
            "case_name",
            "case_category",
            "true_clusters",
            "clusters_current",
            "clusters_branch_corrected",
            "cluster_delta",
            "ari_current",
            "ari_branch_corrected",
            "ari_delta",
        ]
        print(top_cluster_delta.head(10)[top_columns].to_string(index=False))


def main() -> None:
    args = _parse_args()
    cases = _resolve_cases(args.case_names, args.max_cases)
    comparison_df = _compare_cases(cases)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nWrote comparison CSV to {csv_path}")

    _print_summary(comparison_df, show_unchanged=args.show_unchanged)


if __name__ == "__main__":
    main()
