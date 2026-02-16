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
            run_params["n_clusters"] = int(meta["n_clusters"])

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

    if result.status == "ok" and result.labels is not None:
        labels = result.labels
        report_df = result.report_df
        if report_df is None or "cluster_id" not in report_df.columns:
            report_df = _create_report_dataframe_from_labels(labels, data_t.index)
        elif len(report_df.index) == len(data_t.index) and not report_df.index.equals(data_t.index):
            # Reuse runner-provided cluster columns but align index to sample ids.
            report_df = report_df.copy()
            report_df.index = data_t.index
            report_df.index.name = "sample_id"
        found_clusters = result.found_clusters
        labels_len = len(labels)
        ari, nmi, purity = _calculate_ari_nmi_purity_metrics(report_df, data_t.index, y_t)
    else:
        labels_len = 0
        found_clusters = 0
        ari, nmi, purity = np.nan, np.nan, np.nan

    result_row = build_benchmark_result_row(
        case_idx=case_idx,
        case_name=case_name,
        case_category=meta.get("category", "unknown"),
        method_name=spec.name,
        run_params=run_params,
        true_clusters=meta["n_clusters"],
        found_clusters=found_clusters,
        samples=meta["n_samples"],
        features=meta["n_features"],
        noise=meta["noise"],
        ari=ari,
        nmi=nmi,
        purity=purity,
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
