"""Experiment 29 — Low-ESS search + permutation-calibrated offline benchmark.

This script does two things:

1. Search all default benchmark cases for the smallest effective calibration size
   (ESS) under the current weighted-mean calibration.
2. On the lowest-ESS cases, compare:
      - global_chi2   : current weighted-mean c-hat + chi-square tail
      - global_f_ess  : weighted-mean c-hat + approximate F tail
      - perm_mean_*   : node-wise permutation mean c-hat + chi-square / F tail
      - perm_median_* : node-wise permutation median c-hat + chi-square / F tail

This is an offline benchmark only. It does not modify production code.

Usage:
    python debug_scripts/enhancement_lab/exp29_low_ess_permutation_benchmark.py
    python debug_scripts/enhancement_lab/exp29_low_ess_permutation_benchmark.py --top-k 5 --n-permutations 199
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as f_dist

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp28_f_uncertainty_deflation import (  # noqa: E402
    CaseCalibration,
    _bh_count,
    _collect_case_pairs,
)
from exp_parametric_inflation import permutation_c  # noqa: E402
from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_child_pca_projections,
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)


@dataclass(frozen=True)
class LowEssCase:
    name: str
    effective_n: float
    n_pairs: int
    n_null: int
    n_focal: int
    c_global: float
    c_median: float


@dataclass(frozen=True)
class PermutationPair:
    parent: str
    stat: float
    k: int
    is_null_like: bool
    c_global: float
    effective_n: float
    c_perm_mean: float
    c_perm_median: float


@dataclass(frozen=True)
class MethodCounts:
    raw_null: int
    bh_null: int
    raw_focal: int
    bh_focal: int


def _predict_p_chi2(stat: float, k: int, c_hat: float) -> float:
    adjusted_stat = stat / max(c_hat, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _predict_p_f_ess(stat: float, k: int, c_hat: float, effective_n: float) -> float:
    nu = max(0.0, 2.0 * float(effective_n) - 2.0)
    if nu <= 0.0:
        return 1.0
    scaled_ratio = (stat / float(k)) / max(c_hat, 1.0)
    return float(f_dist.sf(scaled_ratio, dfn=float(k), dfd=nu))


def discover_low_ess_cases(
    *,
    top_k: int,
    require_focal: bool = True,
    require_null_like: bool = True,
) -> list[LowEssCase]:
    discovered: list[LowEssCase] = []
    for case in get_default_test_cases():
        case_name = str(case["name"])
        try:
            _pairs, calibration = _collect_case_pairs(case_name)
        except Exception:
            continue
        if require_focal and calibration.n_focal <= 0:
            continue
        if require_null_like and calibration.n_null <= 0:
            continue
        discovered.append(
            LowEssCase(
                name=case_name,
                effective_n=calibration.effective_n,
                n_pairs=calibration.n_pairs,
                n_null=calibration.n_null,
                n_focal=calibration.n_focal,
                c_global=calibration.c_global,
                c_median=calibration.c_median,
            )
        )
    return sorted(discovered, key=lambda row: (row.effective_n, row.n_focal, row.n_pairs))[:top_k]


def collect_permutation_pairs(
    case_name: str, *, n_permutations: int
) -> tuple[list[PermutationPair], CaseCalibration]:
    tree, data_df, _y_true, _tc = build_tree_and_data(case_name)
    run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df
    if annotations_df is None:
        raise ValueError(f"annotations_df not populated for case {case_name}")

    audit = annotations_df.attrs.get("sibling_divergence_audit", {})
    diagnostics = audit.get("diagnostics", {})
    c_global = float(audit.get("global_inflation_factor", 1.0))
    effective_n = float(diagnostics.get("effective_n", 0.0))

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    sibling_dims = derive_sibling_spectral_dims(tree, annotations_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(annotations_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, annotations_df, sibling_dims)
    records, _ = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca,
        pca_eigenvalues=sibling_eig,
        child_pca_projections=sibling_child_pca,
        whitening=config.SIBLING_WHITENING,
    )

    pairs: list[PermutationPair] = []
    for record in records:
        if record.degrees_of_freedom <= 0 or not np.isfinite(record.stat):
            continue
        children = list(tree.successors(record.parent))
        if len(children) != 2:
            continue
        left, right = children
        pca_proj = sibling_pca.get(record.parent) if sibling_pca else None
        pca_eig = sibling_eig.get(record.parent) if sibling_eig else None
        child_pca = sibling_child_pca.get(record.parent) if sibling_child_pca else None

        bl_left = tree.edges[record.parent, left].get("branch_length")
        bl_right = tree.edges[record.parent, right].get("branch_length")
        bl_sum = None
        if mean_bl is not None and bl_left is not None and bl_right is not None:
            candidate = float(bl_left) + float(bl_right)
            bl_sum = candidate if candidate > 0 else None

        c_perm_mean, c_perm_median, _p_perm = permutation_c(
            tree,
            record.parent,
            left,
            right,
            data_df,
            mean_branch_length=mean_bl,
            branch_length_sum=bl_sum,
            spectral_k=int(record.degrees_of_freedom),
            pca_projection=pca_proj,
            pca_eigenvalues=pca_eig,
            child_pca_projections=child_pca,
            whitening=config.SIBLING_WHITENING,
            n_permutations=n_permutations,
        )

        pairs.append(
            PermutationPair(
                parent=record.parent,
                stat=float(record.stat),
                k=int(record.degrees_of_freedom),
                is_null_like=bool(record.is_null_like),
                c_global=c_global,
                effective_n=effective_n,
                c_perm_mean=max(float(c_perm_mean), 1.0),
                c_perm_median=max(float(c_perm_median), 1.0),
            )
        )

    calibration = CaseCalibration(
        case_name=case_name,
        c_global=c_global,
        c_median=float("nan"),
        effective_n=effective_n,
        n_pairs=len(pairs),
        n_null=sum(1 for pair in pairs if pair.is_null_like),
        n_focal=sum(1 for pair in pairs if not pair.is_null_like),
    )
    return pairs, calibration


def _evaluate_pairs(
    pairs: list[PermutationPair],
    *,
    c_selector: str,
    use_f_tail: bool,
) -> MethodCounts:
    null_p_values: list[float] = []
    focal_p_values: list[float] = []
    for pair in pairs:
        if c_selector == "global":
            c_hat = pair.c_global
        elif c_selector == "perm_mean":
            c_hat = pair.c_perm_mean
        elif c_selector == "perm_median":
            c_hat = pair.c_perm_median
        else:
            raise ValueError(f"Unsupported c_selector: {c_selector}")

        if use_f_tail:
            p_value = _predict_p_f_ess(pair.stat, pair.k, c_hat, pair.effective_n)
        else:
            p_value = _predict_p_chi2(pair.stat, pair.k, c_hat)

        if pair.is_null_like:
            null_p_values.append(p_value)
        else:
            focal_p_values.append(p_value)

    null_arr = np.array(null_p_values, dtype=np.float64)
    focal_arr = np.array(focal_p_values, dtype=np.float64)
    return MethodCounts(
        raw_null=int(np.sum(null_arr < config.SIBLING_ALPHA)),
        bh_null=_bh_count(null_arr),
        raw_focal=int(np.sum(focal_arr < config.SIBLING_ALPHA)),
        bh_focal=_bh_count(focal_arr),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else math.nan


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

    print("=" * 108)
    print("Experiment 29: targeted low-ESS search + permutation-calibrated offline benchmark")
    print("=" * 108)
    print(f"Top-k low-ESS cases with both null-like and focal pairs: {args.top_k}\n")
    for rank, case in enumerate(low_ess_cases, start=1):
        print(
            f"{rank:>2}. {case.name:<28} ESS={case.effective_n:>7.2f}  "
            f"pairs={case.n_pairs:>4}  null={case.n_null:>4}  focal={case.n_focal:>4}  "
            f"c_global={case.c_global:>8.3f}"
        )

    print("\nRunning targeted permutation benchmark...\n")

    aggregate: dict[str, dict[str, int]] = {
        name: {"raw_null": 0, "bh_null": 0, "raw_focal": 0, "bh_focal": 0}
        for name in (
            "global_chi2",
            "global_f_ess",
            "perm_mean_chi2",
            "perm_mean_f_ess",
            "perm_median_chi2",
            "perm_median_f_ess",
        )
    }
    total_null = 0
    total_focal = 0

    for case in low_ess_cases:
        pairs, calibration = collect_permutation_pairs(
            case.name,
            n_permutations=args.n_permutations,
        )
        total_null += calibration.n_null
        total_focal += calibration.n_focal
        results = {
            "global_chi2": _evaluate_pairs(pairs, c_selector="global", use_f_tail=False),
            "global_f_ess": _evaluate_pairs(pairs, c_selector="global", use_f_tail=True),
            "perm_mean_chi2": _evaluate_pairs(pairs, c_selector="perm_mean", use_f_tail=False),
            "perm_mean_f_ess": _evaluate_pairs(pairs, c_selector="perm_mean", use_f_tail=True),
            "perm_median_chi2": _evaluate_pairs(pairs, c_selector="perm_median", use_f_tail=False),
            "perm_median_f_ess": _evaluate_pairs(pairs, c_selector="perm_median", use_f_tail=True),
        }

        perm_mean_values = np.array([pair.c_perm_mean for pair in pairs], dtype=np.float64)
        perm_median_values = np.array([pair.c_perm_median for pair in pairs], dtype=np.float64)
        print(
            f"{case.name:<28} ESS={calibration.effective_n:>7.2f}  null={calibration.n_null:>4}  focal={calibration.n_focal:>4}  "
            f"perm_mean_med={np.median(perm_mean_values):>8.3f}  perm_median_med={np.median(perm_median_values):>8.3f}"
        )
        for method_name, counts in results.items():
            aggregate[method_name]["raw_null"] += counts.raw_null
            aggregate[method_name]["bh_null"] += counts.bh_null
            aggregate[method_name]["raw_focal"] += counts.raw_focal
            aggregate[method_name]["bh_focal"] += counts.bh_focal
            print(
                f"  {method_name:<16} null raw/BH={counts.raw_null:>3}/{counts.bh_null:>3}  "
                f"focal raw/BH={counts.raw_focal:>3}/{counts.bh_focal:>3}"
            )
        print()

    print("-" * 108)
    print(f"Aggregate over targeted low-ESS cases: null={total_null}, focal={total_focal}\n")
    for method_name, counts in aggregate.items():
        print(
            f"{method_name:<16} "
            f"null raw={counts['raw_null']:>4} ({_rate(counts['raw_null'], total_null):6.1%})  "
            f"null BH={counts['bh_null']:>4} ({_rate(counts['bh_null'], total_null):6.1%})  "
            f"focal raw={counts['raw_focal']:>4} ({_rate(counts['raw_focal'], total_focal):6.1%})  "
            f"focal BH={counts['bh_focal']:>4} ({_rate(counts['bh_focal'], total_focal):6.1%})"
        )


if __name__ == "__main__":
    main()
