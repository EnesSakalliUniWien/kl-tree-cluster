"""Typed result-row models for benchmark pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BenchmarkRunStatus(str, Enum):
    """Canonical run statuses for benchmark rows."""

    OK = "ok"
    SKIP = "skip"


@dataclass(frozen=True)
class BenchmarkResultRow:
    """Typed representation of a single benchmark output row."""

    test_case: int
    case_id: str
    case_category: str
    source_family: str
    feature_representation: str
    method: str
    params_raw: dict[str, object]
    params_display: str
    true_clusters: int
    found_clusters: int
    samples: int
    features: int
    noise: float
    ari: float
    nmi: float
    purity: float
    macro_recall: float
    macro_f1: float
    worst_cluster_recall: float
    outlier_precision: float
    outlier_recall: float
    outlier_f1: float
    singleton_outlier_isolated: float
    grouped_outlier_cluster_recovered: float
    cluster_count_abs_error: float
    over_split: float
    under_split: float
    status: BenchmarkRunStatus
    skip_reason: str
    labels_length: int


__all__ = [
    "BenchmarkRunStatus",
    "BenchmarkResultRow",
]
