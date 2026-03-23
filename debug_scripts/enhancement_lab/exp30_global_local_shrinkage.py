"""Experiment 30 — Global/local shrinkage calibration on low-ESS cases.

This experiment tests whether a conservative partial-pooling estimator can
improve over the current global c-hat without inheriting the instability of
fully local permutation calibration.

Setup:
1. Reuse the cleaned low-ESS case search from exp29, restricted to cases with
   both null-like and focal pairs.
2. Reuse node-wise permutation c estimates as an offline local target.
3. Compare log-space shrinkage rules of the form:

      log(c_shrunk) = (1 - w) * log(c_global) + w * log(c_local)

   where w is a reliability weight based on case-level ESS, null-pool size,
   and local mean/median agreement.

This is an offline benchmark only. It does not modify production code.

Usage:
    python debug_scripts/enhancement_lab/exp30_global_local_shrinkage.py
    python debug_scripts/enhancement_lab/exp30_global_local_shrinkage.py --top-k 5 --n-permutations 199
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp28_f_uncertainty_deflation import CaseCalibration, _bh_count  # noqa: E402
from exp29_low_ess_permutation_benchmark import (  # noqa: E402
    PermutationPair,
    collect_permutation_pairs,
    discover_low_ess_cases,
)


from kl_clustering_analysis import config  # noqa: E402


@dataclass(frozen=True)
class MethodCounts:
    raw_null: int
    bh_null: int
    raw_focal: int
    bh_focal: int


@dataclass(frozen=True)
class MethodSummary:
    counts: MethodCounts
    median_weight: float
    median_c_hat: float


def _predict_p_chi2(stat: float, k: int, c_hat: float) -> float:
    adjusted_stat = stat / max(c_hat, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else math.nan


def _stability_from_mean_median(c_mean: float, c_median: float) -> float:
    if c_mean <= 0 or c_median <= 0:
        return 0.0
    log_gap = abs(math.log(c_mean / c_median))
    return float(1.0 / (1.0 + log_gap))


def _clip_weight(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _gap_factor(c_global: float, c_local: float, *, scale_log: float = math.log(4.0)) -> float:
    safe_global = max(c_global, 1.0)
    safe_local = max(c_local, 1.0)
    if scale_log <= 0.0:
        raise ValueError("scale_log must be positive.")
    return _clip_weight(abs(math.log(safe_global / safe_local)) / scale_log)


def _log_shrink(c_global: float, c_local: float, weight: float) -> float:
    safe_global = max(c_global, 1.0)
    safe_local = max(c_local, 1.0)
    clipped_weight = _clip_weight(weight)
    log_c = (1.0 - clipped_weight) * math.log(safe_global) + clipped_weight * math.log(safe_local)
    return float(max(math.exp(log_c), 1.0))


def _case_reliability_ess(effective_n: float, tau: float) -> float:
    return _clip_weight(effective_n / (effective_n + tau))


def _case_reliability_joint(
    effective_n: float, n_null: int, *, ess_tau: float, null_tau: float
) -> float:
    ess_weight = _case_reliability_ess(effective_n, ess_tau)
    null_weight = _clip_weight(n_null / (n_null + null_tau))
    return float(ess_weight * null_weight)


def _compute_c_hat(
    method_name: str, pair: PermutationPair, calibration: CaseCalibration
) -> tuple[float, float]:
    if method_name == "global_chi2":
        return pair.c_global, 0.0
    if method_name == "perm_mean_chi2":
        return pair.c_perm_mean, 1.0
    if method_name == "shrink_ess10":
        weight = _case_reliability_ess(calibration.effective_n, tau=10.0)
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    if method_name == "shrink_joint20":
        weight = _case_reliability_joint(
            calibration.effective_n,
            calibration.n_null,
            ess_tau=20.0,
            null_tau=20.0,
        )
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    if method_name == "shrink_joint_stable":
        case_weight = _case_reliability_joint(
            calibration.effective_n,
            calibration.n_null,
            ess_tau=20.0,
            null_tau=20.0,
        )
        stability = _stability_from_mean_median(pair.c_perm_mean, pair.c_perm_median)
        weight = case_weight * stability
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    if method_name == "shrink_joint5":
        weight = _case_reliability_joint(
            calibration.effective_n,
            calibration.n_null,
            ess_tau=5.0,
            null_tau=5.0,
        )
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    if method_name == "shrink_sqrt_joint":
        base_weight = _case_reliability_joint(
            calibration.effective_n,
            calibration.n_null,
            ess_tau=20.0,
            null_tau=20.0,
        )
        weight = math.sqrt(base_weight)
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    if method_name == "shrink_gap_stable":
        base_weight = _case_reliability_joint(
            calibration.effective_n,
            calibration.n_null,
            ess_tau=10.0,
            null_tau=10.0,
        )
        stability = _stability_from_mean_median(pair.c_perm_mean, pair.c_perm_median)
        gap = _gap_factor(pair.c_global, pair.c_perm_mean)
        weight = math.sqrt(base_weight) * stability * gap
        return _log_shrink(pair.c_global, pair.c_perm_mean, weight), weight
    raise ValueError(f"Unsupported method_name: {method_name}")


def _evaluate_method(
    pairs: list[PermutationPair],
    calibration: CaseCalibration,
    *,
    method_name: str,
) -> MethodSummary:
    null_p_values: list[float] = []
    focal_p_values: list[float] = []
    weights: list[float] = []
    c_hats: list[float] = []

    for pair in pairs:
        c_hat, weight = _compute_c_hat(method_name, pair, calibration)
        p_value = _predict_p_chi2(pair.stat, pair.k, c_hat)
        weights.append(weight)
        c_hats.append(c_hat)
        if pair.is_null_like:
            null_p_values.append(p_value)
        else:
            focal_p_values.append(p_value)

    null_arr = np.array(null_p_values, dtype=np.float64)
    focal_arr = np.array(focal_p_values, dtype=np.float64)
    weight_arr = np.array(weights, dtype=np.float64)
    c_hat_arr = np.array(c_hats, dtype=np.float64)

    return MethodSummary(
        counts=MethodCounts(
            raw_null=int(np.sum(null_arr < config.SIBLING_ALPHA)),
            bh_null=_bh_count(null_arr),
            raw_focal=int(np.sum(focal_arr < config.SIBLING_ALPHA)),
            bh_focal=_bh_count(focal_arr),
        ),
        median_weight=float(np.median(weight_arr)) if len(weight_arr) > 0 else math.nan,
        median_c_hat=float(np.median(c_hat_arr)) if len(c_hat_arr) > 0 else math.nan,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=199)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_ess_cases = discover_low_ess_cases(
        top_k=args.top_k,
        require_focal=True,
        require_null_like=True,
    )
    if not low_ess_cases:
        raise ValueError("No low-ESS cases found with both null-like and focal pairs.")

    method_names = (
        "global_chi2",
        "perm_mean_chi2",
        "shrink_ess10",
        "shrink_joint20",
        "shrink_joint_stable",
        "shrink_joint5",
        "shrink_sqrt_joint",
        "shrink_gap_stable",
    )
    aggregate: dict[str, dict[str, float]] = {
        method_name: {
            "raw_null": 0.0,
            "bh_null": 0.0,
            "raw_focal": 0.0,
            "bh_focal": 0.0,
        }
        for method_name in method_names
    }
    total_null = 0
    total_focal = 0

    print("=" * 116)
    print("Experiment 30: global/local shrinkage on cleaned low-ESS cases")
    print("=" * 116)
    print(f"Top-k low-ESS cases with both null-like and focal pairs: {args.top_k}")
    print(
        "Methods: global_chi2, perm_mean_chi2, shrink_ess10, shrink_joint20, "
        "shrink_joint_stable, shrink_joint5, shrink_sqrt_joint, shrink_gap_stable\n"
    )

    for rank, case in enumerate(low_ess_cases, start=1):
        print(
            f"{rank:>2}. {case.name:<28} ESS={case.effective_n:>7.2f}  pairs={case.n_pairs:>4}  "
            f"null={case.n_null:>4}  focal={case.n_focal:>4}  c_global={case.c_global:>8.3f}"
        )

    print("\nRunning shrinkage benchmark...\n")

    for case in low_ess_cases:
        pairs, calibration = collect_permutation_pairs(
            case.name,
            n_permutations=args.n_permutations,
        )
        total_null += calibration.n_null
        total_focal += calibration.n_focal
        summaries = {
            method_name: _evaluate_method(pairs, calibration, method_name=method_name)
            for method_name in method_names
        }
        print(
            f"{case.name:<28} ESS={calibration.effective_n:>7.2f}  null={calibration.n_null:>4}  "
            f"focal={calibration.n_focal:>4}  n_perm={args.n_permutations:>3}"
        )
        for method_name, summary in summaries.items():
            counts = summary.counts
            aggregate[method_name]["raw_null"] += counts.raw_null
            aggregate[method_name]["bh_null"] += counts.bh_null
            aggregate[method_name]["raw_focal"] += counts.raw_focal
            aggregate[method_name]["bh_focal"] += counts.bh_focal
            print(
                f"  {method_name:<19} null raw/BH={counts.raw_null:>3}/{counts.bh_null:>3}  "
                f"focal raw/BH={counts.raw_focal:>3}/{counts.bh_focal:>3}  "
                f"med_w={summary.median_weight:>5.2f}  med_c={summary.median_c_hat:>8.3f}"
            )
        print()

    print("-" * 116)
    print(f"Aggregate over cleaned low-ESS cases: null={total_null}, focal={total_focal}\n")
    for method_name in method_names:
        counts = aggregate[method_name]
        print(
            f"{method_name:<19} "
            f"null raw={int(counts['raw_null']):>4} ({_rate(int(counts['raw_null']), total_null):6.1%})  "
            f"null BH={int(counts['bh_null']):>4} ({_rate(int(counts['bh_null']), total_null):6.1%})  "
            f"focal raw={int(counts['raw_focal']):>4} ({_rate(int(counts['raw_focal']), total_focal):6.1%})  "
            f"focal BH={int(counts['bh_focal']):>4} ({_rate(int(counts['bh_focal']), total_focal):6.1%})"
        )


if __name__ == "__main__":
    main()
