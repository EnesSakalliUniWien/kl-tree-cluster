"""Per-method execution helper for benchmark pipeline runs."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from benchmarks.shared.metrics import _calculate_ari_nmi_purity_metrics
from benchmarks.shared.results import (
    BenchmarkResultRow,
    ComputedResultRecord,
    build_benchmark_result_row,
    build_computed_result_record,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.types import MethodSpec
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels
from kl_clustering_analysis import config


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


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
        raw_k = run_params.get("n_clusters")
        if raw_k is None or str(raw_k).strip().lower() in {"true", "expected", "auto"}:
            true_k = meta.get("n_clusters")
            if true_k is not None:
                run_params["n_clusters"] = int(true_k)

    meta_run = meta.copy()
    distance_condensed_for_run = None
    if method_id.startswith("kl"):
        metric = run_params.get("tree_distance_metric", config.TREE_DISTANCE_METRIC)
        requires_precomputed_kl_distance = bool(meta.get("requires_precomputed_kl_distance", False))
        if requires_precomputed_kl_distance:
            if distance_condensed is None:
                raise ValueError(
                    f"Case '{meta.get('name', '?')}' requires "
                    "'precomputed_distance_condensed' for KL but it is missing."
                )
            distance_condensed_for_run = distance_condensed
        elif precomputed_distance_condensed is not None:
            if distance_condensed is None:
                raise ValueError(
                    f"Case '{meta.get('name', '?')}' provides "
                    "'precomputed_distance_condensed' but it was not loaded."
                )
            distance_condensed_for_run = distance_condensed
        else:
            distance_condensed_for_run = pdist(data_t.values, metric=metric)

    result = run_clustering_result(
        data_df=data_t,
        method_id=method_id,
        params=run_params,
        seed=tc_seed,
        significance_level=significance_level,
        distance_matrix=distance_matrix,
        distance_condensed=distance_condensed_for_run,
    )

    true_clusters_raw = meta.get("n_clusters")
    true_clusters = int(true_clusters_raw) if true_clusters_raw is not None else 0

    if result.status == "ok" and result.labels is not None:
        labels = result.labels
        report_df = result.report_df
        if report_df is None or "cluster_id" not in report_df.columns:
            report_df = _create_report_dataframe_from_labels(labels, data_t.index)
        elif len(report_df.index) != len(data_t.index):
            report_df = _create_report_dataframe_from_labels(labels, data_t.index)
        elif report_df.index.has_duplicates or data_t.index.has_duplicates:
            report_df = _create_report_dataframe_from_labels(labels, data_t.index)
        elif not report_df.index.equals(data_t.index):
            # Reuse runner-provided cluster columns only when row labels are alignable.
            missing = data_t.index.difference(report_df.index)
            extras = report_df.index.difference(data_t.index)
            if missing.empty and extras.empty:
                report_df = report_df.loc[data_t.index].copy()
                report_df.index.name = "sample_id"
            else:
                report_df = _create_report_dataframe_from_labels(labels, data_t.index)
        found_clusters = result.found_clusters
        labels_len = len(labels)
        metrics = _calculate_ari_nmi_purity_metrics(report_df, data_t.index, y_t)
        ari = metrics.ari
        nmi = metrics.nmi
        purity = metrics.purity
        macro_recall = metrics.macro_recall
        macro_f1 = metrics.macro_f1
        worst_cluster_recall = metrics.worst_cluster_recall
    else:
        labels_len = 0
        found_clusters = 0
        ari, nmi, purity = np.nan, np.nan, np.nan
        macro_recall, macro_f1, worst_cluster_recall = np.nan, np.nan, np.nan

    if result.status == "ok" and true_clusters_raw is not None:
        cluster_count_abs_error = float(abs(int(found_clusters) - true_clusters))
        over_split = float(int(found_clusters > true_clusters))
        under_split = float(int(found_clusters < true_clusters))
    else:
        cluster_count_abs_error = np.nan
        over_split = np.nan
        under_split = np.nan

    result_row = build_benchmark_result_row(
        case_idx=case_idx,
        case_name=case_name,
        case_category=meta.get("category", "unknown"),
        method_name=spec.name,
        run_params=run_params,
        true_clusters=true_clusters,
        found_clusters=found_clusters,
        samples=meta["n_samples"],
        features=meta["n_features"],
        noise=meta.get("noise", np.nan),
        ari=ari,
        nmi=nmi,
        purity=purity,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        worst_cluster_recall=worst_cluster_recall,
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
            method_name=spec.name,
            params=run_params,
            ari=float(ari) if np.isfinite(ari) else np.nan,
            nmi=float(nmi) if np.isfinite(nmi) else np.nan,
            purity=float(purity) if np.isfinite(purity) else np.nan,
            labels=result.labels,
            data=data_t,
            meta=meta_run,
            x_original=x_original,
            y_true=y_t,
            tree=result.extra.get("tree") if result.extra else None,
            decomposition=result.extra.get("decomposition") if result.extra else None,
            stats=result.extra.get("stats") if result.extra else None,
            posthoc_merge_audit=result.extra.get("posthoc_merge_audit") if result.extra else None,
        )

    method_audit = None
    if matrix_audit:
        method_name = _slugify(spec.name)
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
