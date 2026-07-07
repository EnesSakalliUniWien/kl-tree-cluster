"""Typed computed-result records for successful method runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComputedResultRecord:
    """Structured record for successful per-method benchmark outputs."""

    test_case_num: int
    method: str
    method_name: str
    params: dict[str, object]
    ari: float
    nmi: float
    purity: float
    outlier_precision: float
    outlier_recall: float
    outlier_f1: float
    singleton_outlier_isolated: float
    grouped_outlier_cluster_recovered: float
    labels: np.ndarray
    data: pd.DataFrame
    meta: dict[str, Any]
    x_original: np.ndarray
    y_true: np.ndarray
    tree: Any
    decomposition: Any
    annotations: Any
    run_id: str = ""
    benchmark_class: str = "unclassified"
    benchmark_grid: str = "default"
    benchmark_repeat: int = 0


def build_computed_result_record(
    *,
    test_case_num: int,
    method: str,
    method_name: str,
    params: dict[str, object],
    ari: float,
    nmi: float,
    purity: float,
    outlier_precision: float,
    outlier_recall: float,
    outlier_f1: float,
    singleton_outlier_isolated: float,
    grouped_outlier_cluster_recovered: float,
    labels: np.ndarray,
    data: pd.DataFrame,
    meta: dict[str, Any],
    x_original: np.ndarray,
    y_true: np.ndarray,
    tree: Any,
    decomposition: Any,
    annotations: Any,
    run_id: str | None = None,
    benchmark_class: str = "unclassified",
    benchmark_grid: str = "default",
    benchmark_repeat: int = 0,
) -> ComputedResultRecord:
    """Build a typed computed-result record."""
    return ComputedResultRecord(
        test_case_num=int(test_case_num),
        method=str(method),
        method_name=str(method_name),
        run_id=str(run_id or method),
        benchmark_class=str(benchmark_class),
        benchmark_grid=str(benchmark_grid),
        benchmark_repeat=int(benchmark_repeat),
        params=dict(params),
        ari=float(ari),
        nmi=float(nmi),
        purity=float(purity),
        outlier_precision=float(outlier_precision),
        outlier_recall=float(outlier_recall),
        outlier_f1=float(outlier_f1),
        singleton_outlier_isolated=float(singleton_outlier_isolated),
        grouped_outlier_cluster_recovered=float(grouped_outlier_cluster_recovered),
        labels=np.asarray(labels),
        data=data,
        meta=dict(meta),
        x_original=np.asarray(x_original),
        y_true=np.asarray(y_true),
        tree=tree,
        decomposition=decomposition,
        annotations=annotations,
    )


__all__ = [
    "ComputedResultRecord",
    "build_computed_result_record",
]
