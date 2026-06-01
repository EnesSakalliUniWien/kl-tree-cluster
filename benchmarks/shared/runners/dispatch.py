"""Shared dispatch helper to run a registered clustering method."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.feature_space import FeatureSpace
from scipy.spatial.distance import pdist, squareform

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels


def _normalize_method_result(
    result: MethodRunResult,
    sample_index: pd.Index,
) -> MethodRunResult:
    """Normalize method outputs to the stable ``ok/skip`` runner contract."""
    if result.status == "ok" and result.labels is not None:
        labels = np.asarray(result.labels)
        if len(labels) != len(sample_index):
            raise ValueError(
                "Runner labels must align to input samples. "
                f"Got {len(labels)} labels for {len(sample_index)} samples."
            )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(result.found_clusters),
            report_df=_create_report_dataframe_from_labels(labels, sample_index),
            status="ok",
            skip_reason=None,
            extra=result.extra,
        )

    skip_reason = result.skip_reason
    if not skip_reason:
        if result.status == "ok":
            skip_reason = "Runner returned status=ok without labels."
        else:
            skip_reason = f"Runner returned status={result.status!r}."

    return MethodRunResult(
        labels=None,
        found_clusters=int(result.found_clusters),
        report_df=None,
        status="skip",
        skip_reason=str(skip_reason),
        extra=result.extra,
    )


def _method_failure_result(error: Exception) -> MethodRunResult:
    return MethodRunResult(
        labels=None,
        found_clusters=0,
        report_df=None,
        status="skip",
        skip_reason=str(error),
        extra={},
    )


def run_clustering_result(
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    seed: Optional[int] = None,
    *,
    significance_level: float | None = None,
    distance_matrix: Optional[np.ndarray] = None,
    distance_condensed: Optional[np.ndarray] = None,
    feature_space: FeatureSpace | None = None,
) -> MethodRunResult:
    """Run one benchmark method and return a normalized ``MethodRunResult``.

    This is the canonical method dispatcher used by pipeline and benchmark helpers.
    """
    spec = METHOD_SPECS[method_id]
    alpha = config.SIBLING_ALPHA if significance_level is None else float(significance_level)
    if method_id == "kl_diffusion":
        try:
            result = spec.runner(
                data_df,
                alpha,
                k_neighbors=int(params["k_neighbors"]),
                diffusion_time=int(params["diffusion_time"]),
                feature_space=feature_space,
            )
        except Exception as exc:
            return _method_failure_result(exc)
        return _normalize_method_result(result, data_df.index)
    if method_id == "kl_diffusion_adaptive":
        try:
            result = spec.runner(
                data_df,
                alpha,
                k_neighbors=int(params["k_neighbors"]),
                diffusion_time=int(params["diffusion_time"]),
                n_components=int(params["n_components"]),
                metric=str(params["metric"]),
                bandwidth_type=params["bandwidth_type"],
                epsilon=params["epsilon"],
                feature_space=feature_space,
            )
        except Exception as exc:
            return _method_failure_result(exc)
        return _normalize_method_result(result, data_df.index)

    if method_id in {"kl", "kl_complete", "kl_single"}:
        metric = str(params["tree_distance_metric"])
        if distance_condensed is not None:
            # Use precomputed distance (e.g. SBM modularity distance).
            kl_distance_condensed = np.asarray(distance_condensed, dtype=float)
        else:
            kl_distance_condensed = pdist(data_df.values, metric=metric)
        try:
            result = spec.runner(
                data_df,
                kl_distance_condensed,
                alpha,
                tree_linkage_method=str(params["tree_linkage_method"]),
                feature_space=feature_space,
            )
        except Exception as exc:
            return _method_failure_result(exc)
        return _normalize_method_result(result, data_df.index)

    if method_id in {"kmeans", "spectral"}:
        int(params["n_clusters"])
        try:
            result = spec.runner(data_df.values, params, seed)
        except Exception as exc:
            return _method_failure_result(exc)
        return _normalize_method_result(result, data_df.index)

    if distance_matrix is None:
        if distance_condensed is None:
            dm_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        else:
            dm_condensed = np.asarray(distance_condensed, dtype=float)
        dm_square = squareform(dm_condensed)
    else:
        dm_square = np.asarray(distance_matrix, dtype=float)

    if method_id in {"leiden", "louvain", "optics"}:
        try:
            result = spec.runner(dm_square, params, seed)
        except Exception as exc:
            return _method_failure_result(exc)
    else:
        try:
            result = spec.runner(dm_square, params)
        except Exception as exc:
            return _method_failure_result(exc)
    return _normalize_method_result(result, data_df.index)


__all__ = ["run_clustering_result"]
