#!/usr/bin/env python3
"""Compare explicit leaf-only Marchenko-Pastur dimension contracts.

This diagnostic does not add production configuration. It monkeypatches the
spectral worker in-process and forces ``TBS_N_JOBS=1`` so each variant has a
clear mathematical meaning:

- leaf_only_floor*: PCA directions and MP threshold use leaf rows only.
- leaf_only_finite_null_floor2: optional simulated finite-sample upper edge for
  the leaf-count threshold.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

os.environ.setdefault("TBS_N_JOBS", "1")

import tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator as gate_orchestrator
import tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_context_module
import tree_break_selection.hierarchy_analysis.statistics.projection.spectral.marchenko_pastur as mp_worker
from scipy.cluster.hierarchy import linkage
from tree_break_selection import config
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.decomposition import (
    eigendecompose_covariance,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    MarchenkoPasturDimensionEstimate,
    estimate_marchenko_pastur_dimension,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import labels_and_report_from_decomposition


@dataclass(frozen=True)
class DimensionContractVariant:
    """One diagnostic spectral contract variant."""

    name: str
    minimum_projection_dimension: int
    threshold_policy: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Marchenko-Pastur dimension-contract variants."
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Comma-separated benchmark case names. Default: representative subset.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--csv", default="")
    parser.add_argument(
        "--include-finite-null",
        action="store_true",
        help="Include simulated finite-null upper-edge variants.",
    )
    parser.add_argument("--finite-null-reps", type=int, default=100)
    parser.add_argument("--finite-null-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def _representative_case_names() -> set[str]:
    return {
        "gauss_clear_small",
        "gauss_clear_large",
        "gauss_moderate_3c",
        "binary_perfect_4c",
        "binary_low_noise_4c",
        "binary_moderate_6c",
        "sparse_features_72x72",
        "cat_clear_3cat_4c",
        "sbm_clear_small",
        "overlap_mod_4c_small",
        "overlap_heavy_4c_small_feat",
        "gauss_overlap_3c_small",
        "binary_2clusters",
        "binary_many_features",
    }


def _resolve_cases(case_names_arg: str, max_cases: int | None) -> list[dict[str, object]]:
    all_cases = get_default_test_cases()
    if case_names_arg.strip():
        requested_names = [name.strip() for name in case_names_arg.split(",") if name.strip()]
        requested_set = set(requested_names)
    else:
        requested_set = _representative_case_names()
    cases = [case for case in all_cases if str(case["name"]) in requested_set]
    found_names = {str(case["name"]) for case in cases}
    missing = sorted(requested_set - found_names)
    if missing:
        raise ValueError(f"Unknown case names: {', '.join(missing)}")
    if max_cases is not None:
        cases = cases[: max(max_cases, 0)]
    return cases


def _variants(include_finite_null: bool) -> list[DimensionContractVariant]:
    variants: list[DimensionContractVariant] = []
    for floor in (0, 1, 2):
        variants.append(
            DimensionContractVariant(
                name=f"leaf_only_floor{floor}",
                minimum_projection_dimension=floor,
                threshold_policy="leaf",
            )
        )
    if include_finite_null:
        variants.append(
            DimensionContractVariant(
                name="leaf_only_finite_null_floor2",
                minimum_projection_dimension=2,
                threshold_policy="finite_null",
            )
        )
    return variants


def _as_ari(y_true: object, labels: np.ndarray) -> float:
    if y_true is None:
        return float("nan")
    y_true_array = np.asarray(y_true)
    if y_true_array.shape[0] != labels.shape[0]:
        return float("nan")
    return float(adjusted_rand_score(y_true_array, labels))


@lru_cache(maxsize=None)
def _finite_null_upper_edge(
    n_rows: int,
    n_features: int,
    reps: int,
    quantile: float,
    seed: int,
) -> float:
    if n_rows <= 1 or n_features <= 0:
        raise ValueError(
            "Finite-null MP edge requires at least two rows and one feature; "
            f"got n_rows={n_rows}, n_features={n_features}."
        )
    rng = np.random.default_rng(seed + 1009 * n_rows + 9176 * n_features)
    top_eigenvalues = np.empty(reps, dtype=np.float64)
    for rep_index in range(reps):
        null_matrix = rng.normal(size=(n_rows, n_features))
        eig = eigendecompose_covariance(null_matrix, compute_eigenvectors=False)
        if eig is None or eig.eigenvalues.size == 0:
            top_eigenvalues[rep_index] = 0.0
        else:
            top_eigenvalues[rep_index] = float(eig.eigenvalues[0])
    return float(np.quantile(top_eigenvalues, quantile))


def _finite_null_dimension_estimate(
    eigenvalues: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    effective_independent_rows: int | None,
    minimum_projection_dimension: int,
    reps: int,
    quantile: float,
    seed: int,
) -> MarchenkoPasturDimensionEstimate:
    row_count = int(effective_independent_rows if effective_independent_rows is not None else n_samples)
    threshold = _finite_null_upper_edge(row_count, int(n_features), reps, quantile, seed)
    raw_signal_count = int(np.sum(np.asarray(eigenvalues, dtype=np.float64) > threshold))
    test_projection_dimension = min(
        max(raw_signal_count, int(minimum_projection_dimension)),
        int(n_features),
    )
    return MarchenkoPasturDimensionEstimate(
        raw_mp_signal_count=raw_signal_count,
        test_projection_dimension=test_projection_dimension,
        effective_independent_rows=row_count,
        mp_threshold_rows=row_count,
    )


@contextmanager
def _patched_variant(
    variant: DimensionContractVariant,
    *,
    finite_null_reps: int,
    finite_null_quantile: float,
    seed: int,
) -> Iterator[None]:
    original_floor_context = spectral_context_module.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION
    original_floor_orchestrator = gate_orchestrator.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION
    original_estimator = mp_worker.estimate_marchenko_pastur_dimension

    def _estimate_leaf_threshold(
        eigenvalues: np.ndarray,
        *,
        n_samples: int,
        n_features: int,
        effective_independent_rows: int | None = None,
        mp_threshold_rows: int | None = None,
        minimum_projection_dimension: int = 1,
    ) -> MarchenkoPasturDimensionEstimate:
        row_count = int(effective_independent_rows if effective_independent_rows is not None else n_samples)
        return estimate_marchenko_pastur_dimension(
            eigenvalues,
            n_samples=n_samples,
            n_features=n_features,
            effective_independent_rows=row_count,
            mp_threshold_rows=row_count,
            minimum_projection_dimension=minimum_projection_dimension,
        )

    def _estimate_finite_null_threshold(
        eigenvalues: np.ndarray,
        *,
        n_samples: int,
        n_features: int,
        effective_independent_rows: int | None = None,
        mp_threshold_rows: int | None = None,
        minimum_projection_dimension: int = 1,
    ) -> MarchenkoPasturDimensionEstimate:
        return _finite_null_dimension_estimate(
            eigenvalues,
            n_samples=n_samples,
            n_features=n_features,
            effective_independent_rows=effective_independent_rows,
            minimum_projection_dimension=minimum_projection_dimension,
            reps=finite_null_reps,
            quantile=finite_null_quantile,
            seed=seed,
        )

    spectral_context_module.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = int(
        variant.minimum_projection_dimension
    )
    gate_orchestrator.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = int(
        variant.minimum_projection_dimension
    )
    if variant.threshold_policy == "leaf":
        mp_worker.estimate_marchenko_pastur_dimension = _estimate_leaf_threshold
    elif variant.threshold_policy == "finite_null":
        mp_worker.estimate_marchenko_pastur_dimension = _estimate_finite_null_threshold
    else:
        raise ValueError(f"Unknown threshold policy: {variant.threshold_policy!r}.")

    try:
        yield
    finally:
        spectral_context_module.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = (
            original_floor_context
        )
        gate_orchestrator.EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = (
            original_floor_orchestrator
        )
        mp_worker.estimate_marchenko_pastur_dimension = original_estimator


def _run_tbs_with_gate_bundle(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    *,
    feature_space: FeatureSpace | None,
) -> MethodRunResult:
    linkage_matrix = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df, feature_space=feature_space)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=data_df,
        feature_space=feature_space,
    )
    decomposition = tree.decompose(
        gate_annotation_bundle=gate_bundle,
        leaf_data=data_df,
        feature_space=feature_space,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
    )
    labels, report_df = labels_and_report_from_decomposition(
        decomposition,
        data_df.index.tolist(),
    )
    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomposition["num_clusters"]),
        report_df=report_df,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree,
            "gate_bundle": gate_bundle,
            "decomposition": decomposition,
            "linkage_matrix": linkage_matrix,
        },
    )


def _spectral_summary(result_extra: dict[str, object]) -> dict[str, float]:
    gate_bundle = result_extra["gate_bundle"]
    spectral_context = gate_bundle.edge_gate_result.spectral_context
    test_dimensions = np.asarray(
        list(spectral_context.test_projection_dimensions_by_node.values()),
        dtype=np.float64,
    )
    raw_counts = np.asarray(
        list(spectral_context.raw_mp_signal_counts_by_node.values()),
        dtype=np.float64,
    )
    effective_rows = np.asarray(
        list(spectral_context.effective_independent_rows_by_node.values()),
        dtype=np.float64,
    )
    threshold_rows = np.asarray(
        list(spectral_context.mp_threshold_rows_by_node.values()),
        dtype=np.float64,
    )
    internal_mask = test_dimensions > 0
    if internal_mask.any():
        test_dimensions = test_dimensions[internal_mask]
        raw_counts = raw_counts[internal_mask]
        effective_rows = effective_rows[internal_mask]
        threshold_rows = threshold_rows[internal_mask]
    return {
        "median_test_projection_dimension": float(np.median(test_dimensions)),
        "median_raw_mp_signal_count": float(np.median(raw_counts)),
        "raw_zero_fraction": float(np.mean(raw_counts == 0.0)),
        "median_effective_independent_rows": float(np.median(effective_rows)),
        "median_mp_threshold_rows": float(np.median(threshold_rows)),
    }


def _run_case_variant(
    case: dict[str, object],
    variant: DimensionContractVariant,
    *,
    finite_null_reps: int,
    finite_null_quantile: float,
    seed: int,
) -> dict[str, object]:
    context = build_tbs_tree_context(case, populate_node_distributions=False)
    with _patched_variant(
        variant,
        finite_null_reps=finite_null_reps,
        finite_null_quantile=finite_null_quantile,
        seed=seed,
    ):
        result = _run_tbs_with_gate_bundle(
            context.data,
            context.distance_condensed,
            feature_space=context.feature_space,
        )
    row = {
        "case_name": str(case["name"]),
        "case_category": str(case.get("category", "")),
        "variant": variant.name,
        "threshold_policy": variant.threshold_policy,
        "minimum_projection_dimension": int(variant.minimum_projection_dimension),
        "status": result.status,
        "true_clusters": int(case.get("n_clusters", 0) or 0),
        "found_clusters": int(result.found_clusters),
        "ari": _as_ari(context.true_labels, result.labels),
    }
    row.update(_spectral_summary(result.extra))
    return row


def _run_comparison(
    cases: list[dict[str, object]],
    variants: list[DimensionContractVariant],
    *,
    finite_null_reps: int,
    finite_null_quantile: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(cases) * len(variants)
    run_index = 0
    for case in cases:
        for variant in variants:
            run_index += 1
            try:
                row = _run_case_variant(
                    case,
                    variant,
                    finite_null_reps=finite_null_reps,
                    finite_null_quantile=finite_null_quantile,
                    seed=seed,
                )
                print(
                    f"[{run_index:>3d}/{total:>3d}] {case['name']} {variant.name}: "
                    f"K={row['found_clusters']} ARI={row['ari']:.3f}"
                )
            except Exception as exc:
                row = {
                    "case_name": str(case["name"]),
                    "case_category": str(case.get("category", "")),
                    "variant": variant.name,
                    "threshold_policy": variant.threshold_policy,
                    "minimum_projection_dimension": int(variant.minimum_projection_dimension),
                    "status": "error",
                    "error": str(exc),
                    "true_clusters": int(case.get("n_clusters", 0) or 0),
                    "found_clusters": np.nan,
                    "ari": np.nan,
                    "median_test_projection_dimension": np.nan,
                    "median_raw_mp_signal_count": np.nan,
                    "raw_zero_fraction": np.nan,
                    "median_effective_independent_rows": np.nan,
                    "median_mp_threshold_rows": np.nan,
                }
                print(
                    f"[{run_index:>3d}/{total:>3d}] {case['name']} {variant.name}: "
                    f"ERROR {exc}"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _print_summary(comparison: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("Variant summary")
    print("=" * 110)
    summary = (
        comparison.groupby("variant", dropna=False)
        .agg(
            runs=("case_name", "count"),
            errors=("status", lambda values: int((values == "error").sum())),
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            exact_k=(
                "found_clusters",
                lambda values: int(
                    (
                        values
                        == comparison.loc[values.index, "true_clusters"].astype(float)
                    ).sum()
                ),
            ),
            median_test_projection_dimension=(
                "median_test_projection_dimension",
                "median",
            ),
            median_raw_mp_signal_count=("median_raw_mp_signal_count", "median"),
            raw_zero_fraction=("raw_zero_fraction", "mean"),
            median_mp_threshold_rows=("median_mp_threshold_rows", "median"),
        )
        .reset_index()
        .sort_values(["mean_ari", "exact_k"], ascending=[False, False])
    )
    print(summary.to_string(index=False))


def main() -> None:
    args = _parse_args()
    cases = _resolve_cases(args.case_names, args.max_cases)
    variants = _variants(include_finite_null=bool(args.include_finite_null))
    comparison = _run_comparison(
        cases,
        variants,
        finite_null_reps=int(args.finite_null_reps),
        finite_null_quantile=float(args.finite_null_quantile),
        seed=int(args.seed),
    )
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(csv_path, index=False)
        print(f"\nWrote CSV to {csv_path}")
    _print_summary(comparison)


if __name__ == "__main__":
    main()
