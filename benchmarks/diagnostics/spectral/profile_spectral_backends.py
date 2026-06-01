#!/usr/bin/env python3
"""Profile exact spectral backend variants on production KL node matrices.

This diagnostic answers a narrow performance question: how much runtime is
spent in the local eigensolver, and whether an exact two-stage SciPy path
(`eigvalsh` for MP dimension, then top-k `eigh` for projection vectors) is
worth moving into production.

The script intentionally does not change production configuration. It extracts
the same null-whitened tangent matrices used by the active Marchenko--Pastur
pipeline and profiles backend variants on those matrices.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import linalg

os.environ.setdefault("KL_TE_N_JOBS", "1")

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.decomposition import (
    eigendecompose_covariance,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.operators import (
    build_dual_covariance_gram_matrix,
    build_primal_covariance_matrix,
    center_active_data,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.preparation import (
    PreparedCovarianceData,
    prepare_covariance_data,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.projection import (
    build_pca_projection,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.representation import (
    CovarianceRepresentation,
    select_covariance_representation,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.eigen_result import (
    EigenResult,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    build_null_whitened_tangent_matrix,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    estimate_marchenko_pastur_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.marchenko_pastur import (
    _process_node,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    _build_spectral_tasks,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_helpers import (
    precompute_descendants,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    resolve_feature_space,
    validate_feature_matrix,
)

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.kl_tree_context import build_kl_tree_context

DEFAULT_CASE_NAMES = (
    "binary_many_features",
    "cat_highcard_20cat_4c",
    "dim_consolidated_4c_72f_continuous",
    "gauss_extreme_noise_highd_continuous",
)
EDGE_GATE_MINIMUM_PROJECTION_DIMENSION = 2


@dataclass(frozen=True)
class SpectralMatrixRecord:
    """One node-local matrix extracted from the production spectral context."""

    node_id: str
    matrix: np.ndarray
    descendant_leaf_rows: int
    feature_count: int


@dataclass(frozen=True)
class VariantResult:
    """Aggregate timing and dimension output for one backend variant."""

    backend: str
    seconds: float
    nodes: int
    projected_nodes: int
    total_test_dimension: int
    max_test_dimension: int
    median_test_dimension: float
    total_raw_signal_count: int
    max_raw_signal_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile exact spectral backend variants on KL benchmark cases."
    )
    parser.add_argument(
        "--case-names",
        default=",".join(DEFAULT_CASE_NAMES),
        help="Comma-separated benchmark case names.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--csv", default="")
    return parser.parse_args()


def _resolve_cases(case_names_arg: str, max_cases: int | None) -> list[dict[str, object]]:
    requested_names = [name.strip() for name in case_names_arg.split(",") if name.strip()]
    if not requested_names:
        raise ValueError("At least one case name is required.")
    requested_set = set(requested_names)
    cases = [case for case in get_default_test_cases() if str(case["name"]) in requested_set]
    found_names = {str(case["name"]) for case in cases}
    missing = sorted(requested_set - found_names)
    if missing:
        raise ValueError(f"Unknown case names: {', '.join(missing)}")
    ordered_cases = sorted(cases, key=lambda case: requested_names.index(str(case["name"])))
    if max_cases is not None:
        ordered_cases = ordered_cases[: max(max_cases, 0)]
    return ordered_cases


def _materialize_task_matrix(
    task: object,
    full_feature_matrix: np.ndarray,
) -> np.ndarray:
    row_indices = task.row_indices
    if len(row_indices) < 2:
        return np.zeros((0, task.feature_space.contrast_dimension), dtype=np.float64)

    descendant_leaf_rows = full_feature_matrix[row_indices, :]
    descendant_leaf_rows = build_null_whitened_tangent_matrix(
        descendant_leaf_rows,
        task.null_distribution,
        feature_space=task.feature_space,
        continuous_covariance_by_block=task.continuous_covariance_by_block,
    )

    if not task.internal_distributions:
        return descendant_leaf_rows

    internal_rows = build_null_whitened_tangent_matrix(
        np.asarray(task.internal_distributions, dtype=np.float64),
        task.null_distribution,
        feature_space=task.feature_space,
        continuous_covariance_by_block=task.continuous_covariance_by_block,
    )
    return np.vstack([descendant_leaf_rows, internal_rows])


def _build_matrix_records(
    case: dict[str, object],
    *,
    max_nodes: int | None,
) -> tuple[
    list[SpectralMatrixRecord],
    np.ndarray,
    list[object],
    FeatureSpace,
    float,
    float,
    float,
    float,
    str,
    str,
]:
    tree_start = time.perf_counter()
    context = build_kl_tree_context(case, populate_node_distributions=True)
    tree_seconds = time.perf_counter() - tree_start

    feature_space = resolve_feature_space(tuple(context.data.columns), context.feature_space)
    feature_count = feature_space.contrast_dimension
    leaf_feature_matrix = validate_feature_matrix(
        context.data.to_numpy(dtype=np.float64, copy=False),
        feature_space,
        value_name="leaf_data",
    )
    leaf_label_to_index = {label: i for i, label in enumerate(context.data.index)}
    descendant_leaf_indices_by_node, descendant_internal_nodes_by_node = (
        precompute_descendants(context.tree, leaf_label_to_index)
    )
    internal_node_ids = [
        node_id
        for node_id in context.tree.nodes
        if len(descendant_leaf_indices_by_node[node_id]) > 1
    ]
    if max_nodes is not None:
        internal_node_ids = internal_node_ids[: max(max_nodes, 0)]

    tasks = _build_spectral_tasks(
        context.tree,
        internal_node_ids,
        descendant_leaf_indices_by_node,
        descendant_internal_nodes_by_node,
        include_internal=bool(config.INCLUDE_INTERNAL_IN_SPECTRAL),
        feature_count=feature_count,
        feature_space=feature_space,
    )

    materialize_start = time.perf_counter()
    records = [
        SpectralMatrixRecord(
            node_id=task.node_id,
            matrix=_materialize_task_matrix(task, leaf_feature_matrix),
            descendant_leaf_rows=len(task.row_indices),
            feature_count=feature_count,
        )
        for task in tasks
    ]
    materialize_seconds = time.perf_counter() - materialize_start
    vectorized_materialize_seconds, vectorized_max_abs_diff = (
        _profile_vectorized_bernoulli_materialization(
            tasks,
            records,
            leaf_feature_matrix,
            feature_space,
        )
    )
    return (
        records,
        leaf_feature_matrix,
        tasks,
        feature_space,
        tree_seconds,
        materialize_seconds,
        vectorized_materialize_seconds,
        vectorized_max_abs_diff,
        context.tree_distance_metric,
        context.tree_distance_source,
    )


def _profile_vectorized_bernoulli_materialization(
    tasks: list[object],
    records: list[SpectralMatrixRecord],
    leaf_feature_matrix: np.ndarray,
    feature_space: FeatureSpace,
) -> tuple[float, float]:
    if not _has_vectorizable_bernoulli_blocks(feature_space):
        return float("nan"), float("nan")

    max_abs_diff = 0.0
    start = time.perf_counter()
    for task, record in zip(tasks, records, strict=True):
        vectorized_matrix = _materialize_task_matrix_vectorized_bernoulli(
            task,
            leaf_feature_matrix,
            feature_space,
        )
        if vectorized_matrix.shape != record.matrix.shape:
            raise ValueError(
                "Vectorized materialization produced a different shape for node "
                f"{task.node_id!r}: {vectorized_matrix.shape} vs {record.matrix.shape}."
            )
        if vectorized_matrix.size:
            node_diff = float(np.max(np.abs(vectorized_matrix - record.matrix)))
            max_abs_diff = max(max_abs_diff, node_diff)
    return time.perf_counter() - start, max_abs_diff


def _has_vectorizable_bernoulli_blocks(feature_space: FeatureSpace) -> bool:
    return feature_space.family_label == "bernoulli" and all(
        block.raw_dimension == 1 for block in feature_space.blocks
    )


def _materialize_task_matrix_vectorized_bernoulli(
    task: object,
    full_feature_matrix: np.ndarray,
    feature_space: FeatureSpace,
) -> np.ndarray:
    row_indices = task.row_indices
    if len(row_indices) < 2:
        return np.zeros((0, feature_space.contrast_dimension), dtype=np.float64)

    descendant_leaf_rows = _vectorized_bernoulli_null_whiten(
        full_feature_matrix[row_indices, :],
        task.null_distribution,
        feature_space,
    )

    if not task.internal_distributions:
        return descendant_leaf_rows

    internal_rows = _vectorized_bernoulli_null_whiten(
        np.asarray(task.internal_distributions, dtype=np.float64),
        task.null_distribution,
        feature_space,
    )
    return np.vstack([descendant_leaf_rows, internal_rows])


def _vectorized_bernoulli_null_whiten(
    distributions: np.ndarray,
    null_distribution: np.ndarray,
    feature_space: FeatureSpace,
) -> np.ndarray:
    rows = np.asarray(distributions, dtype=np.float64)
    null = np.asarray(null_distribution, dtype=np.float64)
    if not _has_vectorizable_bernoulli_blocks(feature_space):
        raise ValueError(
            "Vectorized Bernoulli materialization requires pure Bernoulli blocks; "
            f"got {feature_space.family_label!r}."
        )
    variance = null * (1.0 - null) + 1e-12
    return (rows - null) / np.sqrt(variance)


def _estimate_dimension(
    eigenvalues: np.ndarray,
    *,
    active_feature_count: int,
    matrix_rows: int,
    descendant_leaf_rows: int,
) -> tuple[int, int]:
    estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=matrix_rows,
        n_features=active_feature_count,
        effective_independent_rows=descendant_leaf_rows,
        mp_threshold_rows=matrix_rows,
        minimum_projection_dimension=EDGE_GATE_MINIMUM_PROJECTION_DIMENSION,
    )
    return estimate.test_projection_dimension, estimate.raw_mp_signal_count


def _profile_full_eigh_projection(records: list[SpectralMatrixRecord]) -> VariantResult:
    dimensions: list[int] = []
    raw_counts: list[int] = []
    projected_nodes = 0
    start = time.perf_counter()
    for record in records:
        eig = eigendecompose_covariance(record.matrix, compute_eigenvectors=True)
        if eig is None:
            dimensions.append(0)
            raw_counts.append(0)
            continue
        dimension, raw_count = _estimate_dimension(
            eig.eigenvalues,
            active_feature_count=eig.active_feature_count,
            matrix_rows=record.matrix.shape[0],
            descendant_leaf_rows=record.descendant_leaf_rows,
        )
        if dimension > 0:
            projection_matrix, _ = build_pca_projection(
                eig,
                projection_dimension=dimension,
                n_features_total=record.feature_count,
            )
            dimension = int(projection_matrix.shape[0])
            projected_nodes += 1
        dimensions.append(dimension)
        raw_counts.append(raw_count)
    seconds = time.perf_counter() - start
    return _variant_result("full_eigh_projection", seconds, dimensions, raw_counts, projected_nodes)


def _profile_current_process_node(
    tasks: list[object],
    leaf_feature_matrix: np.ndarray,
    *,
    feature_count: int,
) -> VariantResult:
    dimensions: list[int] = []
    raw_counts: list[int] = []
    projected_nodes = 0
    start = time.perf_counter()
    for task in tasks:
        result = _process_node(
            task,
            leaf_feature_matrix,
            EDGE_GATE_MINIMUM_PROJECTION_DIMENSION,
            feature_count,
            True,
        )
        dimensions.append(int(result.test_projection_dimension))
        raw_counts.append(int(result.raw_mp_signal_count))
        if result.test_projection_dimension > 0:
            projected_nodes += 1
    seconds = time.perf_counter() - start
    return _variant_result("current_process_node", seconds, dimensions, raw_counts, projected_nodes)


def _profile_eigvalsh_dimension_only(records: list[SpectralMatrixRecord]) -> VariantResult:
    dimensions: list[int] = []
    raw_counts: list[int] = []
    start = time.perf_counter()
    for record in records:
        eig = eigendecompose_covariance(record.matrix, compute_eigenvectors=False)
        if eig is None:
            dimensions.append(0)
            raw_counts.append(0)
            continue
        dimension, raw_count = _estimate_dimension(
            eig.eigenvalues,
            active_feature_count=eig.active_feature_count,
            matrix_rows=record.matrix.shape[0],
            descendant_leaf_rows=record.descendant_leaf_rows,
        )
        dimension = _cap_to_projectable_dimension(
            eig.eigenvalues,
            active_feature_count=eig.active_feature_count,
            matrix_rows=record.matrix.shape[0],
            use_dual=bool(eig.use_dual),
            requested_dimension=dimension,
        )
        dimensions.append(dimension)
        raw_counts.append(raw_count)
    seconds = time.perf_counter() - start
    return _variant_result("eigvalsh_dimension_only", seconds, dimensions, raw_counts, 0)


def _profile_eigvalsh_then_subset_projection(
    records: list[SpectralMatrixRecord],
) -> VariantResult:
    dimensions: list[int] = []
    raw_counts: list[int] = []
    projected_nodes = 0
    start = time.perf_counter()
    for record in records:
        prepared = prepare_covariance_data(record.matrix)
        if prepared.n_active_features == 0:
            dimensions.append(0)
            raw_counts.append(0)
            continue
        representation = select_covariance_representation(prepared)
        if representation is CovarianceRepresentation.DUAL:
            eigenvalues = _eigvalsh_dual(prepared)
        else:
            eigenvalues = _eigvalsh_primal(prepared)
        dimension, raw_count = _estimate_dimension(
            eigenvalues,
            active_feature_count=prepared.n_active_features,
            matrix_rows=record.matrix.shape[0],
            descendant_leaf_rows=record.descendant_leaf_rows,
        )
        if dimension > 0:
            eig = _subset_eigh(prepared, representation, dimension)
            projection_matrix, _ = build_pca_projection(
                eig,
                projection_dimension=dimension,
                n_features_total=record.feature_count,
            )
            dimension = int(projection_matrix.shape[0])
            projected_nodes += 1
        dimensions.append(dimension)
        raw_counts.append(raw_count)
    seconds = time.perf_counter() - start
    return _variant_result(
        "eigvalsh_then_subset_projection",
        seconds,
        dimensions,
        raw_counts,
        projected_nodes,
    )


def _eigvalsh_dual(prepared: PreparedCovarianceData) -> np.ndarray:
    centered_active_data = center_active_data(prepared.active_data)
    matrix = build_dual_covariance_gram_matrix(centered_active_data)
    return np.maximum(linalg.eigvalsh(matrix, check_finite=False)[::-1], 0.0)


def _cap_to_projectable_dimension(
    eigenvalues: np.ndarray,
    *,
    active_feature_count: int,
    matrix_rows: int,
    use_dual: bool,
    requested_dimension: int,
) -> int:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.size == 0:
        return 0
    tolerance = (
        np.finfo(np.float64).eps
        * max(eigenvalues.shape[0], 1)
        * max(float(np.max(eigenvalues)), 1.0)
    )
    positive_eigenvalues = int(np.count_nonzero(eigenvalues > tolerance))
    available = min(eigenvalues.shape[0], int(active_feature_count), positive_eigenvalues)
    if use_dual:
        available = min(available, max(int(matrix_rows) - 1, 0))
    return min(int(requested_dimension), available)


def _eigvalsh_primal(prepared: PreparedCovarianceData) -> np.ndarray:
    matrix = build_primal_covariance_matrix(prepared.active_data)
    return np.maximum(linalg.eigvalsh(matrix, check_finite=False)[::-1], 0.0)


def _subset_eigh(
    prepared: PreparedCovarianceData,
    representation: CovarianceRepresentation,
    dimension: int,
) -> EigenResult:
    if representation is CovarianceRepresentation.DUAL:
        centered_active_data = center_active_data(prepared.active_data)
        matrix = build_dual_covariance_gram_matrix(centered_active_data)
        top_eigenvalues, top_eigenvectors = _top_eigh(matrix, dimension)
        return EigenResult(
            eigenvalues=top_eigenvalues,
            is_active_feature=prepared.is_active_feature,
            active_feature_count=prepared.n_active_features,
            use_dual=True,
            dual_sample_eigenvectors=top_eigenvectors,
            centered_data_active=centered_active_data,
        )

    matrix = build_primal_covariance_matrix(prepared.active_data)
    top_eigenvalues, top_eigenvectors = _top_eigh(matrix, dimension)
    return EigenResult(
        eigenvalues=top_eigenvalues,
        is_active_feature=prepared.is_active_feature,
        active_feature_count=prepared.n_active_features,
        use_dual=False,
        eigenvectors_active=top_eigenvectors,
    )


def _top_eigh(matrix: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    matrix_size = matrix.shape[0]
    top_dimension = min(int(dimension), matrix_size)
    if top_dimension <= 0:
        raise ValueError("Top-k eigendecomposition requires a positive dimension.")
    lower_index = max(matrix_size - top_dimension, 0)
    eigenvalues, eigenvectors = linalg.eigh(
        matrix,
        subset_by_index=(lower_index, matrix_size - 1),
        check_finite=False,
    )
    return np.maximum(eigenvalues[::-1], 0.0), eigenvectors[:, ::-1]


def _variant_result(
    backend: str,
    seconds: float,
    dimensions: list[int],
    raw_counts: list[int],
    projected_nodes: int,
) -> VariantResult:
    dimension_array = np.asarray(dimensions, dtype=np.float64)
    raw_count_array = np.asarray(raw_counts, dtype=np.float64)
    return VariantResult(
        backend=backend,
        seconds=float(seconds),
        nodes=len(dimensions),
        projected_nodes=int(projected_nodes),
        total_test_dimension=int(np.sum(dimension_array)),
        max_test_dimension=int(np.max(dimension_array)) if dimension_array.size else 0,
        median_test_dimension=float(np.median(dimension_array)) if dimension_array.size else 0.0,
        total_raw_signal_count=int(np.sum(raw_count_array)),
        max_raw_signal_count=int(np.max(raw_count_array)) if raw_count_array.size else 0,
    )


def _run_repeated(
    profiler: Callable[[], VariantResult],
    *,
    repeat: int,
) -> VariantResult:
    results = [profiler() for _ in range(max(int(repeat), 1))]
    fastest = min(results, key=lambda result: result.seconds)
    return fastest


def _profile_case(
    case: dict[str, object],
    *,
    max_nodes: int | None,
    repeat: int,
) -> list[dict[str, object]]:
    (
        records,
        leaf_feature_matrix,
        tasks,
        feature_space,
        tree_seconds,
        materialize_seconds,
        vectorized_materialize_seconds,
        vectorized_max_abs_diff,
        tree_distance_metric,
        tree_distance_source,
    ) = _build_matrix_records(case, max_nodes=max_nodes)
    node_rows = np.asarray([record.matrix.shape[0] for record in records], dtype=np.float64)
    active_features = np.asarray(
        [
            prepare_covariance_data(record.matrix).n_active_features
            for record in records
            if record.matrix.shape[0] > 0
        ],
        dtype=np.float64,
    )
    profilers: tuple[Callable[[], VariantResult], ...] = (
        lambda: _profile_current_process_node(
            tasks,
            leaf_feature_matrix,
            feature_count=feature_space.contrast_dimension,
        ),
        lambda: _profile_full_eigh_projection(records),
        lambda: _profile_eigvalsh_dimension_only(records),
        lambda: _profile_eigvalsh_then_subset_projection(records),
    )
    rows: list[dict[str, object]] = []
    full_projection_seconds: float | None = None
    for profiler in profilers:
        result = _run_repeated(profiler, repeat=repeat)
        if result.backend == "full_eigh_projection":
            full_projection_seconds = result.seconds
        speedup_vs_full = (
            float(full_projection_seconds / result.seconds)
            if full_projection_seconds is not None and result.seconds > 0
            else np.nan
        )
        rows.append(
            {
                "case_name": str(case["name"]),
                "case_category": str(case.get("category", "")),
                "generator": str(case.get("generator", "")),
                "backend": result.backend,
                "seconds": result.seconds,
                "speedup_vs_full_eigh_projection": speedup_vs_full,
                "tree_build_and_populate_seconds": tree_seconds,
                "matrix_materialization_seconds": materialize_seconds,
                "vectorized_bernoulli_materialization_seconds": (
                    vectorized_materialize_seconds
                ),
                "vectorized_bernoulli_materialization_speedup": (
                    float(materialize_seconds / vectorized_materialize_seconds)
                    if vectorized_materialize_seconds > 0
                    else np.nan
                ),
                "vectorized_bernoulli_materialization_max_abs_diff": (
                    vectorized_max_abs_diff
                ),
                "nodes": result.nodes,
                "projected_nodes": result.projected_nodes,
                "total_test_dimension": result.total_test_dimension,
                "max_test_dimension": result.max_test_dimension,
                "median_test_dimension": result.median_test_dimension,
                "total_raw_signal_count": result.total_raw_signal_count,
                "max_raw_signal_count": result.max_raw_signal_count,
                "median_node_rows": float(np.median(node_rows)) if node_rows.size else 0.0,
                "max_node_rows": int(np.max(node_rows)) if node_rows.size else 0,
                "median_active_features": (
                    float(np.median(active_features)) if active_features.size else 0.0
                ),
                "max_active_features": (
                    int(np.max(active_features)) if active_features.size else 0
                ),
                "contrast_dimension": int(feature_space.contrast_dimension),
                "tree_distance_metric": tree_distance_metric,
                "tree_distance_source": tree_distance_source,
            }
        )
    return rows


def _print_summary(rows: list[dict[str, object]]) -> None:
    table = pd.DataFrame(rows)
    display_columns = [
        "case_name",
        "backend",
        "seconds",
        "speedup_vs_full_eigh_projection",
        "matrix_materialization_seconds",
        "vectorized_bernoulli_materialization_seconds",
        "vectorized_bernoulli_materialization_speedup",
        "vectorized_bernoulli_materialization_max_abs_diff",
        "nodes",
        "median_test_dimension",
        "max_test_dimension",
        "median_active_features",
        "max_active_features",
    ]
    print()
    print("=" * 120)
    print("Spectral backend timing")
    print("=" * 120)
    print(table[display_columns].to_string(index=False))


def main() -> None:
    args = _parse_args()
    cases = _resolve_cases(args.case_names, args.max_cases)
    rows: list[dict[str, object]] = []
    for case in cases:
        print(f"Profiling {case['name']}...")
        rows.extend(
            _profile_case(
                case,
                max_nodes=args.max_nodes,
                repeat=args.repeat,
            )
        )
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Wrote CSV to {csv_path}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
