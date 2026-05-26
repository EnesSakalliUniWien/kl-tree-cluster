"""Factory helpers for benchmark result rows."""

from __future__ import annotations

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
    stage_timings: dict[str, object] | None = None,
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
        method=str(method),
        params_raw=dict(run_params),
        params_display=format_params_for_display(run_params),
        true_clusters=int(true_clusters),
        found_clusters=int(found_clusters),
        samples=int(samples),
        features=int(features),
        noise=float(noise),
        ari=float(ari),
        nmi=float(nmi),
        purity=float(purity),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        worst_cluster_recall=float(worst_cluster_recall),
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
        gate2_sec=normalized_stage_timings["gate2_sec"],
        gate2_contrast_covariance_sec=normalized_stage_timings[
            "gate2_contrast_covariance_sec"
        ],
        gate2_projection_sec=normalized_stage_timings["gate2_projection_sec"],
        gate2_wald_statistic_sec=normalized_stage_timings[
            "gate2_wald_statistic_sec"
        ],
        gate2_tree_bh_sec=normalized_stage_timings["gate2_tree_bh_sec"],
        spectral_context_sec=normalized_stage_timings["spectral_context_sec"],
        tangent_whitening_sec=normalized_stage_timings["tangent_whitening_sec"],
        eigensolve_sec=normalized_stage_timings["eigensolve_sec"],
        pca_projection_sec=normalized_stage_timings["pca_projection_sec"],
        gate3_sec=normalized_stage_timings["gate3_sec"],
        gate3_pair_record_collection_sec=normalized_stage_timings[
            "gate3_pair_record_collection_sec"
        ],
        gate3_inflation_fit_sec=normalized_stage_timings["gate3_inflation_fit_sec"],
        gate3_adjusted_tests_sec=normalized_stage_timings["gate3_adjusted_tests_sec"],
        gate3_sibling_fdr_sec=normalized_stage_timings["gate3_sibling_fdr_sec"],
        traversal_sec=normalized_stage_timings["traversal_sec"],
        status=_normalize_status(status),
        skip_reason=(skip_reason or ""),
        labels_length=int(labels_length),
    )


__all__ = ["build_benchmark_result_row"]
