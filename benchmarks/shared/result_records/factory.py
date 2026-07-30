"""Factory helpers for benchmark result rows."""

from __future__ import annotations

import math

from benchmarks.shared.result_records.models import BenchmarkResultRow, BenchmarkRunStatus
from benchmarks.shared.util.params import format_params_for_display
from benchmarks.shared.util.time import normalize_stage_timings


def _normalize_status(status: str | BenchmarkRunStatus) -> BenchmarkRunStatus:
    if isinstance(status, BenchmarkRunStatus):
        return status
    raw = str(status).strip().lower()
    if raw == BenchmarkRunStatus.OK.value:
        return BenchmarkRunStatus.OK
    if raw == BenchmarkRunStatus.SKIP.value:
        return BenchmarkRunStatus.SKIP
    raise ValueError(
        f"Invalid benchmark status: {status!r}. "
        f"Expected one of: {BenchmarkRunStatus.OK.value!r}, {BenchmarkRunStatus.SKIP.value!r}."
    )


def build_benchmark_result_row(
    *,
    test_case: int,
    case_id: str,
    case_category: str,
    source_family: str,
    feature_representation: str,
    method: str,
    run_params: dict[str, object],
    true_clusters: int,
    found_clusters: int,
    samples: int,
    features: int,
    noise: float,
    ari: float,
    nmi: float,
    purity: float,
    macro_recall: float,
    macro_f1: float,
    worst_cluster_recall: float,
    outlier_precision: float,
    outlier_recall: float,
    outlier_f1: float,
    singleton_outlier_isolated: float,
    grouped_outlier_cluster_recovered: float,
    cluster_count_abs_error: float,
    over_split: float,
    under_split: float,
    status: str | BenchmarkRunStatus,
    skip_reason: str | None,
    labels_length: int,
    run_id: str | None = None,
    benchmark_class: str = "unclassified",
    benchmark_grid: str = "default",
    benchmark_repeat: int = 0,
    stage_timings: dict[str, object] | None = None,
    ami: float = math.nan,
    homogeneity: float = math.nan,
    completeness: float = math.nan,
    v_measure: float = math.nan,
    fowlkes_mallows: float = math.nan,
    n_singleton_clusters: float = math.nan,
    singleton_fraction: float = math.nan,
    median_cluster_size: float = math.nan,
    largest_cluster_fraction: float = math.nan,
    effective_cluster_count: float = math.nan,
    cluster_size_entropy: float = math.nan,
    cluster_size_gini: float = math.nan,
    noise_label_fraction: float = math.nan,
    silhouette_score: float = math.nan,
    davies_bouldin_index: float = math.nan,
    calinski_harabasz_index: float = math.nan,
    simulation_model: str = "unspecified",
    observation_model: str = "unspecified",
    benchmark_intent: str = "unspecified",
    scientific_caution: str = "unspecified",
    recommended_simulation_family: str = "unspecified",
) -> BenchmarkResultRow:
    """Build a typed benchmark row with normalized output fields."""
    if true_clusters is None:
        raise ValueError("true_clusters must be an integer; use 0 when cluster truth is unknown.")
    if noise is None:
        raise ValueError("noise must be a float; use NaN when noise metadata is unavailable.")
    normalized_stage_timings = normalize_stage_timings(stage_timings)
    return BenchmarkResultRow(
        test_case=int(test_case),
        case_id=str(case_id),
        case_category=str(case_category),
        source_family=str(source_family),
        feature_representation=str(feature_representation),
        simulation_model=str(simulation_model),
        observation_model=str(observation_model),
        benchmark_intent=str(benchmark_intent),
        scientific_caution=str(scientific_caution),
        recommended_simulation_family=str(recommended_simulation_family),
        method=str(method),
        run_id=str(run_id or method),
        benchmark_class=str(benchmark_class),
        benchmark_grid=str(benchmark_grid),
        benchmark_repeat=int(benchmark_repeat),
        params_raw=dict(run_params),
        params_display=format_params_for_display(run_params),
        true_clusters=int(true_clusters),
        found_clusters=int(found_clusters),
        samples=int(samples),
        features=int(features),
        noise=float(noise),
        ari=float(ari),
        nmi=float(nmi),
        ami=float(ami),
        purity=float(purity),
        homogeneity=float(homogeneity),
        completeness=float(completeness),
        v_measure=float(v_measure),
        fowlkes_mallows=float(fowlkes_mallows),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        worst_cluster_recall=float(worst_cluster_recall),
        n_singleton_clusters=float(n_singleton_clusters),
        singleton_fraction=float(singleton_fraction),
        median_cluster_size=float(median_cluster_size),
        largest_cluster_fraction=float(largest_cluster_fraction),
        effective_cluster_count=float(effective_cluster_count),
        cluster_size_entropy=float(cluster_size_entropy),
        cluster_size_gini=float(cluster_size_gini),
        noise_label_fraction=float(noise_label_fraction),
        silhouette_score=float(silhouette_score),
        davies_bouldin_index=float(davies_bouldin_index),
        calinski_harabasz_index=float(calinski_harabasz_index),
        outlier_precision=float(outlier_precision),
        outlier_recall=float(outlier_recall),
        outlier_f1=float(outlier_f1),
        singleton_outlier_isolated=float(singleton_outlier_isolated),
        grouped_outlier_cluster_recovered=float(grouped_outlier_cluster_recovered),
        cluster_count_abs_error=float(cluster_count_abs_error),
        over_split=float(over_split),
        under_split=float(under_split),
        tree_build_sec=normalized_stage_timings["tree_build_sec"],
        populate_divergences_sec=normalized_stage_timings["populate_divergences_sec"],
        edge_gate_sec=normalized_stage_timings["edge_gate_sec"],
        edge_gate_contrast_covariance_sec=normalized_stage_timings[
            "edge_gate_contrast_covariance_sec"
        ],
        edge_gate_projection_sec=normalized_stage_timings["edge_gate_projection_sec"],
        edge_gate_wald_statistic_sec=normalized_stage_timings["edge_gate_wald_statistic_sec"],
        edge_gate_tree_bh_sec=normalized_stage_timings["edge_gate_tree_bh_sec"],
        spectral_context_sec=normalized_stage_timings["spectral_context_sec"],
        tangent_whitening_sec=normalized_stage_timings["tangent_whitening_sec"],
        eigensolve_sec=normalized_stage_timings["eigensolve_sec"],
        pca_projection_sec=normalized_stage_timings["pca_projection_sec"],
        sibling_gate_sec=normalized_stage_timings["sibling_gate_sec"],
        sibling_gate_pair_record_collection_sec=normalized_stage_timings[
            "sibling_gate_pair_record_collection_sec"
        ],
        sibling_gate_inflation_fit_sec=normalized_stage_timings["sibling_gate_inflation_fit_sec"],
        sibling_gate_adjusted_tests_sec=normalized_stage_timings["sibling_gate_adjusted_tests_sec"],
        sibling_gate_fdr_sec=normalized_stage_timings["sibling_gate_fdr_sec"],
        traversal_sec=normalized_stage_timings["traversal_sec"],
        status=_normalize_status(status),
        skip_reason=(skip_reason or ""),
        labels_length=int(labels_length),
    )


__all__ = ["build_benchmark_result_row"]
