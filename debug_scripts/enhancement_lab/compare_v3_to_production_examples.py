"""Compare analytic v3 calibration to current production calibration examples.

This script evaluates a compact benchmark set in two ways:
1. Runtime pipeline behavior for current methods:
   - cousin_adjusted_wald
   - parametric_wald
2. Row-level calibration behavior on the same sibling-pair records for:
   - production global calibration helper
   - production parametric calibration helper
   - leave-one-case-out analytic v3

The goal is to compare v3 against the current production-style code paths
without integrating v3 into the runtime gate yet.

Usage:
    python debug_scripts/enhancement_lab/compare_v3_to_production_examples.py
    python debug_scripts/enhancement_lab/compare_v3_to_production_examples.py --limit 2
    python debug_scripts/enhancement_lab/compare_v3_to_production_examples.py --cases gauss_moderate_3c gauss_null_small
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp_parametric_inflation_v3 import PairRow, select_v3_model  # noqa: E402
from lab_helpers import quick_eval, temporary_config  # noqa: E402

from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (  # noqa: E402
    fit_inflation_model,
    fit_parametric_inflation_model,
    predict_inflation_factor,
    predict_parametric_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    SiblingPairRecord,
    collect_sibling_pair_records,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

DEFAULT_CASES = [
    "gauss_moderate_3c",
    "gauss_clear_small",
    "binary_hard_4c",
    "gauss_null_small",
    "binary_balanced_low_noise",
]


@dataclass(frozen=True)
class PairExample:
    record: SiblingPairRecord
    row: PairRow


@dataclass(frozen=True)
class CaseContext:
    case_name: str
    true_k: int | None
    found_default: int
    ari_default: float
    found_parametric_runtime: int
    ari_parametric_runtime: float
    pairs: tuple[PairExample, ...]


@dataclass(frozen=True)
class CaseComparison:
    case_name: str
    n_null: int
    n_focal: int
    raw_null_global: int
    raw_null_parametric: int
    raw_null_v3: int
    bh_null_global: int
    bh_null_parametric: int
    bh_null_v3: int
    raw_focal_global: int
    raw_focal_parametric: int
    raw_focal_v3: int
    bh_focal_global: int
    bh_focal_parametric: int
    bh_focal_v3: int


def _case_flags(case_name: str) -> tuple[float, float]:
    is_binary_case = 1.0 if case_name.startswith("binary_") else 0.0
    is_null_case = 1.0 if "_null_" in case_name else 0.0
    return is_binary_case, is_null_case


def _predict_p_value(t_obs: float, k: int, c_value: float) -> float:
    adjusted_stat = t_obs / max(c_value, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _bh_reject_flags(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return np.array([], dtype=bool)
    rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
    return rejected.astype(bool)


def _runtime_summary(case_name: str, sibling_method: str) -> tuple[int | None, int, float]:
    with temporary_config(SIBLING_TEST_METHOD=sibling_method):
        result = quick_eval(case_name)
    return result["true_k"], result["found_k"], float(result["ari"])


def collect_case_context(case_name: str) -> CaseContext:
    runtime_default = quick_eval(case_name)
    runtime_parametric = _runtime_summary(case_name, "parametric_wald")

    tree = runtime_default["tree"]
    annotations_df = runtime_default["annotations_df"]
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
        whitening=config.SIBLING_WHITENING,
    )

    root = tree.root()
    c_global = float(annotations_df.attrs.get("sibling_divergence_audit", {}).get("global_inflation_factor", 1.0))
    is_binary_case, is_null_case = _case_flags(case_name)
    pairs: list[PairExample] = []

    for record in records:
        if record.degrees_of_freedom <= 0 or not np.isfinite(record.stat):
            continue
        children = list(tree.successors(record.parent))
        if len(children) != 2:
            continue
        left, right = children
        n_left = len(tree.get_leaves(left, return_labels=True))
        n_right = len(tree.get_leaves(right, return_labels=True))
        row = PairRow(
            case_name=case_name,
            parent=record.parent,
            n_parent=int(record.n_parent),
            n_left=n_left,
            n_right=n_right,
            k=int(record.degrees_of_freedom),
            depth=int(nx.shortest_path_length(tree, root, record.parent)),
            branch_length_sum=float(max(record.branch_length_sum, 0.0)),
            sibling_null_prior=float(record.sibling_null_prior_from_edge_pvalue),
            is_null_like=bool(record.is_null_like),
            is_binary_case=is_binary_case,
            is_null_case=is_null_case,
            t_obs=float(record.stat),
            c_global=c_global,
            p_global=_predict_p_value(float(record.stat), int(record.degrees_of_freedom), c_global),
            true_k=runtime_default["true_k"],
            found_k=runtime_default["found_k"],
            ari=float(runtime_default["ari"]),
        )
        pairs.append(PairExample(record=record, row=row))

    return CaseContext(
        case_name=case_name,
        true_k=runtime_default["true_k"],
        found_default=runtime_default["found_k"],
        ari_default=float(runtime_default["ari"]),
        found_parametric_runtime=runtime_parametric[1],
        ari_parametric_runtime=float(runtime_parametric[2]),
        pairs=tuple(pairs),
    )


def compare_case(
    context: CaseContext,
    train_null_rows: list[PairRow],
    train_eval_rows: list[PairRow],
) -> CaseComparison:
    records = [pair.record for pair in context.pairs]
    global_model = fit_inflation_model(records)
    parametric_model = fit_parametric_inflation_model(records)
    v3_model = select_v3_model(train_null_rows, train_eval_rows=train_eval_rows)

    null_global: list[float] = []
    null_parametric: list[float] = []
    null_v3: list[float] = []
    focal_global: list[float] = []
    focal_parametric: list[float] = []
    focal_v3: list[float] = []

    for pair in context.pairs:
        record = pair.record
        row = pair.row
        global_c = predict_inflation_factor(
            global_model,
            branch_length_sum=float(max(record.branch_length_sum, 0.0)),
            n_reference=int(record.n_parent),
        )
        parametric_c = predict_parametric_inflation_factor(
            parametric_model,
            n_reference=int(record.n_parent),
        )
        v3_c = v3_model.predict(row)

        global_p = _predict_p_value(float(record.stat), int(record.degrees_of_freedom), global_c)
        parametric_p = _predict_p_value(float(record.stat), int(record.degrees_of_freedom), parametric_c)
        v3_p = _predict_p_value(float(record.stat), int(record.degrees_of_freedom), v3_c)

        if row.is_null_like:
            null_global.append(global_p)
            null_parametric.append(parametric_p)
            null_v3.append(v3_p)
        else:
            focal_global.append(global_p)
            focal_parametric.append(parametric_p)
            focal_v3.append(v3_p)

    null_global_arr = np.array(null_global, dtype=np.float64)
    null_parametric_arr = np.array(null_parametric, dtype=np.float64)
    null_v3_arr = np.array(null_v3, dtype=np.float64)
    focal_global_arr = np.array(focal_global, dtype=np.float64)
    focal_parametric_arr = np.array(focal_parametric, dtype=np.float64)
    focal_v3_arr = np.array(focal_v3, dtype=np.float64)

    return CaseComparison(
        case_name=context.case_name,
        n_null=len(null_global_arr),
        n_focal=len(focal_global_arr),
        raw_null_global=int(np.sum(null_global_arr < config.SIBLING_ALPHA)),
        raw_null_parametric=int(np.sum(null_parametric_arr < config.SIBLING_ALPHA)),
        raw_null_v3=int(np.sum(null_v3_arr < config.SIBLING_ALPHA)),
        bh_null_global=int(np.sum(_bh_reject_flags(null_global_arr))),
        bh_null_parametric=int(np.sum(_bh_reject_flags(null_parametric_arr))),
        bh_null_v3=int(np.sum(_bh_reject_flags(null_v3_arr))),
        raw_focal_global=int(np.sum(focal_global_arr < config.SIBLING_ALPHA)),
        raw_focal_parametric=int(np.sum(focal_parametric_arr < config.SIBLING_ALPHA)),
        raw_focal_v3=int(np.sum(focal_v3_arr < config.SIBLING_ALPHA)),
        bh_focal_global=int(np.sum(_bh_reject_flags(focal_global_arr))),
        bh_focal_parametric=int(np.sum(_bh_reject_flags(focal_parametric_arr))),
        bh_focal_v3=int(np.sum(_bh_reject_flags(focal_v3_arr))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_names = list(args.cases) if args.cases else list(DEFAULT_CASES)
    if args.limit is not None:
        case_names = case_names[: args.limit]
    if not case_names:
        raise ValueError("At least one case is required.")

    print(
        f"Config: METHOD={config.SIBLING_TEST_METHOD}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, EDGE_ALPHA={config.EDGE_ALPHA}"
    )
    print("Comparison: current production examples vs analytic v3")
    print(f"Cases: {len(case_names)}\n")

    contexts = [collect_case_context(case_name) for case_name in case_names]

    print("=" * 110)
    print("PIPELINE RUNTIME SUMMARY")
    print("=" * 110)
    for context in contexts:
        print(
            f"{context.case_name:<28} trueK={context.true_k!s:>3} | "
            f"adjusted K/ARI={context.found_default:>3}/{context.ari_default:>5.3f} | "
            f"parametric K/ARI={context.found_parametric_runtime:>3}/{context.ari_parametric_runtime:>5.3f} | "
            f"pairs={len(context.pairs):>3}"
        )

    print("\n" + "=" * 110)
    print("ROW-LEVEL CALIBRATION SUMMARY")
    print("=" * 110)

    comparisons: list[CaseComparison] = []
    total_null = 0
    total_focal = 0
    totals = {
        "raw_null_global": 0,
        "raw_null_parametric": 0,
        "raw_null_v3": 0,
        "bh_null_global": 0,
        "bh_null_parametric": 0,
        "bh_null_v3": 0,
        "raw_focal_global": 0,
        "raw_focal_parametric": 0,
        "raw_focal_v3": 0,
        "bh_focal_global": 0,
        "bh_focal_parametric": 0,
        "bh_focal_v3": 0,
    }

    all_rows = [pair.row for context in contexts for pair in context.pairs]
    for context in contexts:
        train_null_rows = [row for row in all_rows if row.case_name != context.case_name and row.is_null_like]
        train_eval_rows = [row for row in all_rows if row.case_name != context.case_name]
        if not train_null_rows:
            train_null_rows = [pair.row for pair in context.pairs if pair.row.is_null_like]
            train_eval_rows = [pair.row for pair in context.pairs]
        if not train_null_rows:
            raise ValueError(f"No null-like rows available for case {context.case_name!r}.")

        comparison = compare_case(context, train_null_rows, train_eval_rows)
        comparisons.append(comparison)
        total_null += comparison.n_null
        total_focal += comparison.n_focal
        for key in totals:
            totals[key] += getattr(comparison, key)

        print(
            f"{comparison.case_name:<28} null={comparison.n_null:>3} focal={comparison.n_focal:>3} | "
            f"null raw G/P/V3={comparison.raw_null_global:>2}/{comparison.raw_null_parametric:>2}/{comparison.raw_null_v3:>2} | "
            f"null BH G/P/V3={comparison.bh_null_global:>2}/{comparison.bh_null_parametric:>2}/{comparison.bh_null_v3:>2} | "
            f"focal BH G/P/V3={comparison.bh_focal_global:>2}/{comparison.bh_focal_parametric:>2}/{comparison.bh_focal_v3:>2}"
        )

    print("\n" + "-" * 110)
    print(f"Total null-like rows: {total_null}")
    if total_null > 0:
        print(
            "Approx-null raw rejection rate: "
            f"global={totals['raw_null_global'] / total_null:.1%}, "
            f"parametric={totals['raw_null_parametric'] / total_null:.1%}, "
            f"v3={totals['raw_null_v3'] / total_null:.1%}"
        )
        print(
            "Approx-null BH rejection rate: "
            f"global={totals['bh_null_global'] / total_null:.1%}, "
            f"parametric={totals['bh_null_parametric'] / total_null:.1%}, "
            f"v3={totals['bh_null_v3'] / total_null:.1%}"
        )

    print(f"\nTotal focal rows: {total_focal}")
    if total_focal > 0:
        print(
            "Focal raw rejection rate: "
            f"global={totals['raw_focal_global'] / total_focal:.1%}, "
            f"parametric={totals['raw_focal_parametric'] / total_focal:.1%}, "
            f"v3={totals['raw_focal_v3'] / total_focal:.1%}"
        )
        print(
            "Focal BH rejection rate: "
            f"global={totals['bh_focal_global'] / total_focal:.1%}, "
            f"parametric={totals['bh_focal_parametric'] / total_focal:.1%}, "
            f"v3={totals['bh_focal_v3'] / total_focal:.1%}"
        )


if __name__ == "__main__":
    main()
