"""Shared dispatch helper to run a registered clustering method."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import MethodRunResult
from kl_clustering_analysis import config

logger = logging.getLogger(__name__)


def _normalize_method_result(
    result: MethodRunResult,
) -> MethodRunResult:
    """Normalize method outputs to the stable ``ok/skip`` runner contract."""
    if result.status == "ok" and result.labels is not None:
        labels = np.asarray(result.labels)
        return MethodRunResult(
            labels=labels,
            found_clusters=int(result.found_clusters),
            report_df=result.report_df,
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
        found_clusters=int(getattr(result, "found_clusters", 0) or 0),
        report_df=None,
        status="skip",
        skip_reason=str(skip_reason),
        extra=result.extra,
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
) -> MethodRunResult:
    """Run one benchmark method and return a normalized ``MethodRunResult``.

    This is the canonical method dispatcher used by pipeline and benchmark helpers.
    """
    spec = METHOD_SPECS[method_id]
    alpha = config.SIBLING_ALPHA if significance_level is None else float(significance_level)

    try:
        if method_id in {"kl", "kl_rogerstanimoto"}:
            if distance_condensed is None:
                metric = params.get("tree_distance_metric", config.TREE_DISTANCE_METRIC)
                kl_distance_condensed = pdist(data_df.values, metric=metric)
            else:
                kl_distance_condensed = np.asarray(distance_condensed, dtype=float)
            result = spec.runner(
                data_df,
                kl_distance_condensed,
                alpha,
                tree_linkage_method=params.get("tree_linkage_method", config.TREE_LINKAGE_METHOD),
            )
            return _normalize_method_result(result)

        if method_id in {"kmeans", "spectral"}:
            result = spec.runner(data_df.values, params, seed)
            return _normalize_method_result(result)

        if distance_matrix is None:
            if distance_condensed is None:
                dm_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
            else:
                dm_condensed = np.asarray(distance_condensed, dtype=float)
            dm_square = squareform(dm_condensed)
        else:
            dm_square = np.asarray(distance_matrix, dtype=float)

        if method_id in {"leiden", "louvain", "optics"}:
            result = spec.runner(dm_square, params, seed)
        else:
            result = spec.runner(dm_square, params)
        return _normalize_method_result(result)
    except Exception as exc:
        skip_reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "run_clustering_result failed for method=%s; returning skip. reason=%s: %s",
            method_id,
            type(exc).__name__,
            exc,
        )
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=skip_reason,
            extra={},
        )


__all__ = ["run_clustering_result"]
