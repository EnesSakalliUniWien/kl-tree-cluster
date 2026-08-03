"""Typed result-row models for benchmark pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.shared.types.run_status import BenchmarkRunStatus


@dataclass(frozen=True)
class BenchmarkResultRow:
    """Typed representation of a single benchmark output row."""

    test_case: int
    case_id: str
    case_category: str
    source_family: str
    feature_representation: str
    simulation_model: str
    observation_model: str
    benchmark_intent: str
    scientific_caution: str
    recommended_simulation_family: str
    method: str
    run_id: str
    benchmark_class: str
    benchmark_grid: str
    benchmark_repeat: int
    params_raw: dict[str, object]
    params_display: str
    true_clusters: int
    found_clusters: int
    samples: int
    features: int
    noise: float
    ari: float
    nmi: float
    ami: float
    purity: float
    homogeneity: float
    completeness: float
    v_measure: float
    fowlkes_mallows: float
    macro_recall: float
    macro_f1: float
    worst_cluster_recall: float
    n_singleton_clusters: float
    singleton_fraction: float
    median_cluster_size: float
    largest_cluster_fraction: float
    effective_cluster_count: float
    cluster_size_entropy: float
    cluster_size_gini: float
    noise_label_fraction: float
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    outlier_precision: float
    outlier_recall: float
    outlier_f1: float
    singleton_outlier_isolated: float
    grouped_outlier_cluster_recovered: float
    cluster_count_abs_error: float
    over_split: float
    under_split: float
    tree_build_sec: float
    populate_divergences_sec: float
    edge_gate_sec: float
    edge_gate_contrast_covariance_sec: float
    edge_gate_projection_sec: float
    edge_gate_wald_statistic_sec: float
    edge_gate_tree_bh_sec: float
    spectral_context_sec: float
    tangent_whitening_sec: float
    eigensolve_sec: float
    pca_projection_sec: float
    sibling_gate_sec: float
    sibling_gate_pair_record_collection_sec: float
    sibling_gate_inflation_fit_sec: float
    sibling_gate_adjusted_tests_sec: float
    sibling_gate_fdr_sec: float
    traversal_sec: float
    status: BenchmarkRunStatus
    skip_reason: str
    unsupported_reason_code: str
    unsupported_stage: str
    unsupported_reason: str
    unsupported_focal_record_count: float
    unsupported_admissible_support_count: float
    unsupported_invalid_record_count: float
    unsupported_upstream_tested_count: float
    unsupported_upstream_rejected_count: float
    labels_length: int


__all__ = [
    "BenchmarkResultRow",
]
