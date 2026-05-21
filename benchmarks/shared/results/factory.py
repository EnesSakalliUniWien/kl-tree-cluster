"""Factory helpers for benchmark result rows."""

from __future__ import annotations

from benchmarks.shared.results.models import BenchmarkResultRow, BenchmarkRunStatus
from benchmarks.shared.util.params import format_params_for_display


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
) -> BenchmarkResultRow:
    """Build a typed benchmark row with normalized output fields."""
    _true = true_clusters if true_clusters is not None else 0
    _noise = (
        noise if noise is not None and not (isinstance(noise, float) and noise != noise) else 0.0
    )
    return BenchmarkResultRow(
        test_case=int(test_case),
        case_id=str(case_id),
        case_category=str(case_category),
        method=str(method),
        params_raw=dict(run_params),
        params_display=format_params_for_display(run_params),
        true_clusters=int(_true),
        found_clusters=int(found_clusters),
        samples=int(samples),
        features=int(features),
        noise=float(_noise),
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
        status=_normalize_status(status),
        skip_reason=(skip_reason or ""),
        labels_length=int(labels_length),
    )


__all__ = ["build_benchmark_result_row"]
