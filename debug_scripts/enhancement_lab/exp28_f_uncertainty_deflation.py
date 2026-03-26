"""Experiment 28 — Uncertainty-aware deflation for sibling calibration.

Tests two proposed calibration enhancements without changing production code:

1. Replace the current chi-square tail with an approximate F tail using
   denominator degrees of freedom derived from effective sample size (ESS):

       p = sf_F((T / k) / c_hat; df1=k, df2=max(2*ESS-2, 0))

   This is treated as a heuristic Satterthwaite-style correction, not an
   exact F law, because the calibration denominator is a weighted average of
   heterogeneous T/k ratios.

2. Replace the weighted-mean c-hat with a weighted median c-hat as a
   contamination-robust sensitivity analysis.

We compare four methods on the existing inflation diagnostic cases:
    - global_chi2   : current weighted-mean c-hat + chi-square tail
    - global_f_ess  : weighted-mean c-hat + approximate F tail
    - median_chi2   : weighted-median c-hat + chi-square tail
    - median_f_ess  : weighted-median c-hat + approximate F tail

Metrics:
    - raw null reject rate
    - BH null reject rate
    - raw focal reject rate
    - BH focal reject rate

Usage:
    python debug_scripts/enhancement_lab/exp28_f_uncertainty_deflation.py
    python debug_scripts/enhancement_lab/exp28_f_uncertainty_deflation.py --cases gauss_null_small
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
from scipy.stats import f as f_dist

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp_parametric_inflation import DIAGNOSTIC_CASES  # noqa: E402
from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402

from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_child_pca_projections,
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

EPS = 1e-9
EXTRA_CASES = ["phylo_divergent_8taxa"]
DEFAULT_CASES = DIAGNOSTIC_CASES + EXTRA_CASES


@dataclass(frozen=True)
class PairOutcome:
    case_name: str
    parent: str
    stat: float
    k: int
    is_null_like: bool
    sibling_null_prior: float
    n_parent: int


@dataclass(frozen=True)
class CaseCalibration:
    case_name: str
    c_global: float
    c_median: float
    effective_n: float
    n_pairs: int
    n_null: int
    n_focal: int


@dataclass(frozen=True)
class MethodCounts:
    raw_null: int
    bh_null: int
    raw_focal: int
    bh_focal: int


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("Cannot compute weighted quantile of empty array.")
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.clip(weights[order], 0.0, None)
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float(np.quantile(sorted_values, quantile))
    cumulative = np.cumsum(sorted_weights) / total_weight
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _bh_count(p_values: np.ndarray) -> int:
    if len(p_values) == 0:
        return 0
    rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
    return int(np.sum(rejected))


def _predict_p_chi2(stat: float, k: int, c_hat: float) -> float:
    adjusted_stat = stat / max(c_hat, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _predict_p_f_ess(stat: float, k: int, c_hat: float, effective_n: float) -> float:
    nu = max(0.0, 2.0 * float(effective_n) - 2.0)
    if nu <= 0.0:
        return 1.0
    scaled_ratio = (stat / float(k)) / max(c_hat, 1.0)
    return float(f_dist.sf(scaled_ratio, dfn=float(k), dfd=nu))


def _collect_case_pairs(case_name: str) -> tuple[list[PairOutcome], CaseCalibration]:
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

    valid_records = [
        record for record in records if np.isfinite(record.stat) and record.degrees_of_freedom > 0
    ]
    ratios = np.array(
        [record.stat / record.degrees_of_freedom for record in valid_records], dtype=np.float64
    )
    weights = np.array([record.sibling_null_prior_from_edge_pvalue for record in valid_records], dtype=np.float64)
    positive_mask = ratios > 0
    ratios = ratios[positive_mask]
    weights = weights[positive_mask]
    c_median = max(_weighted_quantile(ratios, weights, 0.5), 1.0) if len(ratios) > 0 else 1.0

    pairs = [
        PairOutcome(
            case_name=case_name,
            parent=record.parent,
            stat=float(record.stat),
            k=int(record.degrees_of_freedom),
            is_null_like=bool(record.is_null_like),
            sibling_null_prior=float(record.sibling_null_prior_from_edge_pvalue),
            n_parent=int(record.n_parent),
        )
        for record in valid_records
    ]

    calibration = CaseCalibration(
        case_name=case_name,
        c_global=c_global,
        c_median=float(c_median),
        effective_n=effective_n,
        n_pairs=len(pairs),
        n_null=sum(1 for pair in pairs if pair.is_null_like),
        n_focal=sum(1 for pair in pairs if not pair.is_null_like),
    )
    return pairs, calibration


def _evaluate_method(
    pairs: list[PairOutcome],
    *,
    c_hat: float,
    effective_n: float,
    use_f_tail: bool,
) -> MethodCounts:
    null_p_values: list[float] = []
    focal_p_values: list[float] = []
    for pair in pairs:
        if use_f_tail:
            p_value = _predict_p_f_ess(pair.stat, pair.k, c_hat, effective_n)
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


def run_case(case_name: str) -> tuple[CaseCalibration, dict[str, MethodCounts]]:
    pairs, calibration = _collect_case_pairs(case_name)
    results = {
        "global_chi2": _evaluate_method(
            pairs,
            c_hat=calibration.c_global,
            effective_n=calibration.effective_n,
            use_f_tail=False,
        ),
        "global_f_ess": _evaluate_method(
            pairs,
            c_hat=calibration.c_global,
            effective_n=calibration.effective_n,
            use_f_tail=True,
        ),
        "median_chi2": _evaluate_method(
            pairs,
            c_hat=calibration.c_median,
            effective_n=calibration.effective_n,
            use_f_tail=False,
        ),
        "median_f_ess": _evaluate_method(
            pairs,
            c_hat=calibration.c_median,
            effective_n=calibration.effective_n,
            use_f_tail=True,
        ),
    }
    return calibration, results


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_names: list[str] = list(args.cases)

    print("=" * 96)
    print("Experiment 28: uncertainty-aware deflation (approximate F + median c-hat)")
    print("=" * 96)
    print(f"Cases: {len(case_names)}")
    print(
        "Methods: global_chi2, global_f_ess, median_chi2, median_f_ess\n"
        "  global_chi2  = current weighted-mean c-hat + chi-square tail\n"
        "  global_f_ess = weighted-mean c-hat + approximate F(df2=2*ESS-2)\n"
        "  median_chi2  = weighted-median c-hat + chi-square tail\n"
        "  median_f_ess = weighted-median c-hat + approximate F(df2=2*ESS-2)\n"
    )

    aggregate: dict[str, dict[str, int]] = {
        name: {"raw_null": 0, "bh_null": 0, "raw_focal": 0, "bh_focal": 0}
        for name in ("global_chi2", "global_f_ess", "median_chi2", "median_f_ess")
    }
    total_null = 0
    total_focal = 0

    for case_name in case_names:
        try:
            calibration, results = run_case(case_name)
        except Exception as err:
            print(f"[SKIP] {case_name}: {err}")
            continue

        total_null += calibration.n_null
        total_focal += calibration.n_focal
        print(
            f"{case_name:<28} pairs={calibration.n_pairs:>4}  null={calibration.n_null:>4}  "
            f"focal={calibration.n_focal:>4}  ESS={calibration.effective_n:>6.2f}  "
            f"c_mean={calibration.c_global:>7.3f}  c_med={calibration.c_median:>7.3f}"
        )
        for method_name, counts in results.items():
            aggregate[method_name]["raw_null"] += counts.raw_null
            aggregate[method_name]["bh_null"] += counts.bh_null
            aggregate[method_name]["raw_focal"] += counts.raw_focal
            aggregate[method_name]["bh_focal"] += counts.bh_focal
            print(
                f"  {method_name:<12}  "
                f"null raw/BH={counts.raw_null:>3}/{counts.bh_null:>3}  "
                f"focal raw/BH={counts.raw_focal:>3}/{counts.bh_focal:>3}"
            )
        print()

    print("-" * 96)
    print(f"Aggregate totals: null={total_null}, focal={total_focal}\n")
    for method_name, counts in aggregate.items():
        print(
            f"{method_name:<12}  "
            f"null raw={counts['raw_null']:>4} ({_rate(counts['raw_null'], total_null):6.1%})  "
            f"null BH={counts['bh_null']:>4} ({_rate(counts['bh_null'], total_null):6.1%})  "
            f"focal raw={counts['raw_focal']:>4} ({_rate(counts['raw_focal'], total_focal):6.1%})  "
            f"focal BH={counts['bh_focal']:>4} ({_rate(counts['bh_focal'], total_focal):6.1%})"
        )


if __name__ == "__main__":
    main()
