"""Per-method execution helper for benchmark pipeline runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.shared.metrics import _calculate_ari_nmi_purity_metrics
from benchmarks.shared.result_records import (
    BenchmarkResultRow,
    ComputedResultRecord,
    build_benchmark_result_row,
    build_computed_result_record,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.types import MethodSpec
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels
from scipy.spatial.distance import pdist


KL_TREE_DISTANCE_SOURCE_KEY = "tree_distance_source"
KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC = "feature_metric"
KL_TREE_DISTANCE_SOURCE_PRECOMPUTED = "precomputed"


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _require_precomputed_kl_distance_metric(
    *,
    meta: dict[str, object],
    case_name: str,
) -> str:
    """Return the named metric/source for a required precomputed KL distance."""
    if "distance_metric" not in meta:
        raise ValueError(
            f"Case '{case_name}' requires precomputed KL distance metadata but "
            "does not define 'distance_metric'."
        )
    distance_metric = str(meta["distance_metric"])
    if not distance_metric:
        raise ValueError(
            f"Case '{case_name}' requires non-empty precomputed KL distance metadata."
        )
    return distance_metric


def _report_for_metric_evaluation(
    report_df: pd.DataFrame | None,
    labels: np.ndarray,
    sample_index: pd.Index,
) -> pd.DataFrame:
    """Return a report table aligned to the sample index used for metrics."""
    if report_df is None:
        return _create_report_dataframe_from_labels(labels, sample_index)
    if "cluster_id" not in report_df.columns:
        raise ValueError("Runner report_df must include a 'cluster_id' column.")
    if len(report_df.index) != len(sample_index):
        raise ValueError(
            "Runner report_df must have one row per sample. "
            f"Got {len(report_df.index)} rows for {len(sample_index)} samples."
        )
    if report_df.index.has_duplicates:
        raise ValueError("Runner report_df index contains duplicate sample ids.")
    if sample_index.has_duplicates:
        raise ValueError("Sample index contains duplicate sample ids.")
    if report_df.index.equals(sample_index):
        return report_df

    missing = sample_index.difference(report_df.index)
    extras = report_df.index.difference(sample_index)
    if missing.empty and extras.empty:
        aligned_report = report_df.loc[sample_index].copy()
        aligned_report.index.name = "sample_id"
        return aligned_report
    raise ValueError(
        "Runner report_df index must match sample ids. "
        f"Missing={list(missing[:5])}, extra={list(extras[:5])}."
    )


def _build_method_failure_row(
    *,
    method_id: str,
    recorded_run_params: dict[str, object],
    case_idx: int,
    case_name: str,
    meta: dict[str, object],
    error: Exception,
) -> BenchmarkResultRow:
    """Represent a method runtime failure as a benchmark skip row."""
    return build_benchmark_result_row(
        test_case=case_idx,
        case_id=case_name,
        case_category=meta["category"],
        method=method_id,
        run_params=recorded_run_params,
        true_clusters=int(meta["n_clusters"]),
        found_clusters=0,
        samples=int(meta["n_samples"]),
        features=int(meta["n_features"]),
        noise=float(meta["noise"]),
        ari=np.nan,
        nmi=np.nan,
        purity=np.nan,
        macro_recall=np.nan,
        macro_f1=np.nan,
        worst_cluster_recall=np.nan,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
        cluster_count_abs_error=np.nan,
        over_split=np.nan,
        under_split=np.nan,
        status="skip",
        skip_reason=str(error),
        labels_length=0,
    )


def run_single_method_once(
    *,
    method_id: str,
    spec: MethodSpec,
    params: dict[str, object],
    case_idx: int,
    case_name: str,
    tc_seed: object,
    significance_level: float,
    data_t: object,
    y_t: object,
    x_original: object,
    meta: dict[str, object],
    distance_matrix: np.ndarray | None,
    distance_condensed: np.ndarray | None,
    precomputed_distance_condensed: object,
    matrix_audit: bool,
) -> tuple[BenchmarkResultRow, ComputedResultRecord | None, tuple[str, dict[str, object]] | None]:
    """Execute one method+params run and return typed outputs."""
    run_params = dict(params)
    if method_id in {"kmeans", "spectral"}:
        raw_k = run_params["n_clusters"]
        if str(raw_k).strip().lower() in {"true", "expected", "auto"}:
            run_params["n_clusters"] = int(meta["n_clusters"])

    meta_run = meta.copy()
    feature_space = meta.get("feature_space")
    distance_condensed_for_run = None
    recorded_run_params = dict(run_params)
    if method_id in {"kl", "kl_complete", "kl_single"}:
        metric = str(run_params["tree_distance_metric"])
        requires_precomputed_kl_distance = bool(meta["requires_precomputed_kl_distance"])
        if requires_precomputed_kl_distance:
            if distance_condensed is None:
                raise ValueError(
                    f"Case '{meta['name']}' requires "
                    "'precomputed_distance_condensed' for KL but it is missing."
                )
            distance_condensed_for_run = distance_condensed
            recorded_run_params["tree_distance_metric"] = _require_precomputed_kl_distance_metric(
                meta=meta,
                case_name=str(meta["name"]),
            )
            recorded_run_params[KL_TREE_DISTANCE_SOURCE_KEY] = (
                KL_TREE_DISTANCE_SOURCE_PRECOMPUTED
            )
        elif precomputed_distance_condensed is not None:
            if distance_condensed is None:
                raise ValueError(
                    f"Case '{meta['name']}' provides "
                    "'precomputed_distance_condensed' but it was not loaded."
                )
            distance_condensed_for_run = distance_condensed
            recorded_run_params["tree_distance_metric"] = _require_precomputed_kl_distance_metric(
                meta=meta,
                case_name=str(meta["name"]),
            )
            recorded_run_params[KL_TREE_DISTANCE_SOURCE_KEY] = (
                KL_TREE_DISTANCE_SOURCE_PRECOMPUTED
            )
        else:
            distance_condensed_for_run = pdist(data_t.values, metric=metric)
            recorded_run_params["tree_distance_metric"] = metric
            recorded_run_params[KL_TREE_DISTANCE_SOURCE_KEY] = (
                KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC
            )

    try:
        result = run_clustering_result(
            data_df=data_t,
            method_id=method_id,
            params=run_params,
            seed=tc_seed,
            significance_level=significance_level,
            distance_matrix=distance_matrix,
            distance_condensed=distance_condensed_for_run,
            feature_space=feature_space,
        )
    except Exception as exc:
        return (
            _build_method_failure_row(
                method_id=method_id,
                recorded_run_params=recorded_run_params,
                case_idx=case_idx,
                case_name=case_name,
                meta=meta,
                error=exc,
            ),
            None,
            None,
        )

    true_clusters_raw = meta["n_clusters"]
    true_clusters = int(true_clusters_raw)

    if result.status == "ok" and result.labels is not None:
        labels = result.labels
        report_df = _report_for_metric_evaluation(result.report_df, labels, data_t.index)
        found_clusters = result.found_clusters
        labels_len = len(labels)
        metrics = _calculate_ari_nmi_purity_metrics(report_df, data_t.index, y_t, meta)
        ari = metrics.ari
        nmi = metrics.nmi
        purity = metrics.purity
        macro_recall = metrics.macro_recall
        macro_f1 = metrics.macro_f1
        worst_cluster_recall = metrics.worst_cluster_recall
        outlier_precision = metrics.outlier_precision
        outlier_recall = metrics.outlier_recall
        outlier_f1 = metrics.outlier_f1
        singleton_outlier_isolated = metrics.singleton_outlier_isolated
        grouped_outlier_cluster_recovered = metrics.grouped_outlier_cluster_recovered
    else:
        labels_len = 0
        found_clusters = 0
        ari, nmi, purity = np.nan, np.nan, np.nan
        macro_recall, macro_f1, worst_cluster_recall = np.nan, np.nan, np.nan
        outlier_precision, outlier_recall, outlier_f1 = np.nan, np.nan, np.nan
        singleton_outlier_isolated = np.nan
        grouped_outlier_cluster_recovered = np.nan

    if result.status == "ok":
        cluster_count_abs_error = float(abs(int(found_clusters) - true_clusters))
        over_split = float(int(found_clusters > true_clusters))
        under_split = float(int(found_clusters < true_clusters))
    else:
        cluster_count_abs_error = np.nan
        over_split = np.nan
        under_split = np.nan

    result_row = build_benchmark_result_row(
        test_case=case_idx,
        case_id=case_name,
        case_category=meta["category"],
        method=method_id,
        run_params=recorded_run_params,
        true_clusters=true_clusters,
        found_clusters=found_clusters,
        samples=meta["n_samples"],
        features=meta["n_features"],
        noise=meta["noise"],
        ari=ari,
        nmi=nmi,
        purity=purity,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        worst_cluster_recall=worst_cluster_recall,
        outlier_precision=outlier_precision,
        outlier_recall=outlier_recall,
        outlier_f1=outlier_f1,
        singleton_outlier_isolated=singleton_outlier_isolated,
        grouped_outlier_cluster_recovered=grouped_outlier_cluster_recovered,
        cluster_count_abs_error=cluster_count_abs_error,
        over_split=over_split,
        under_split=under_split,
        status=result.status,
        skip_reason=result.skip_reason,
        labels_length=labels_len,
    )

    computed_result = None
    if result.status == "ok" and result.labels is not None:
        meta_run["found_clusters"] = found_clusters
        computed_result = build_computed_result_record(
            test_case_num=case_idx,
            method=method_id,
            method_name=spec.name,
            params=recorded_run_params,
            ari=float(ari) if np.isfinite(ari) else np.nan,
            nmi=float(nmi) if np.isfinite(nmi) else np.nan,
            purity=float(purity) if np.isfinite(purity) else np.nan,
            outlier_precision=float(outlier_precision) if np.isfinite(outlier_precision) else np.nan,
            outlier_recall=float(outlier_recall) if np.isfinite(outlier_recall) else np.nan,
            outlier_f1=float(outlier_f1) if np.isfinite(outlier_f1) else np.nan,
            singleton_outlier_isolated=(
                float(singleton_outlier_isolated)
                if np.isfinite(singleton_outlier_isolated)
                else np.nan
            ),
            grouped_outlier_cluster_recovered=(
                float(grouped_outlier_cluster_recovered)
                if np.isfinite(grouped_outlier_cluster_recovered)
                else np.nan
            ),
            labels=result.labels,
            data=data_t,
            meta=meta_run,
            x_original=x_original,
            y_true=y_t,
            tree=result.extra.get("tree") if result.extra else None,
            decomposition=result.extra.get("decomposition") if result.extra else None,
            annotations=result.extra.get("annotations") if result.extra else None,
        )

    method_audit = None
    if matrix_audit:
        method_name = _slugify(method_id)
        params_slug = _slugify(result_row.params_display)
        method_tag = method_name if not params_slug else f"{method_name}__{params_slug}"
        matrices: dict[str, object] = {}
        if method_id.startswith("kl"):
            matrices["distance_condensed"] = distance_condensed_for_run
        if result.extra and result.extra.get("linkage_matrix") is not None:
            matrices["linkage_matrix"] = result.extra.get("linkage_matrix")
        if matrices:
            method_audit = (method_tag, matrices)

    return result_row, computed_result, method_audit


__all__ = ["run_single_method_once"]
