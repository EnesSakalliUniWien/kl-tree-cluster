"""Time helpers shared by benchmark entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

BENCHMARK_STAGE_TIMING_KEYS = (
    "tree_build_sec",
    "populate_divergences_sec",
    "edge_gate_sec",
    "edge_gate_contrast_covariance_sec",
    "edge_gate_projection_sec",
    "edge_gate_wald_statistic_sec",
    "edge_gate_tree_bh_sec",
    "spectral_context_sec",
    "tangent_whitening_sec",
    "eigensolve_sec",
    "pca_projection_sec",
    "sibling_gate_sec",
    "sibling_gate_pair_record_collection_sec",
    "sibling_gate_inflation_fit_sec",
    "sibling_gate_adjusted_tests_sec",
    "sibling_gate_fdr_sec",
    "traversal_sec",
)


def format_timestamp_utc(dt: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp like 20250101_235959Z."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def elapsed_since(start_sec: float) -> float:
    """Return elapsed wall-clock seconds since a ``perf_counter`` reading."""
    return float(perf_counter() - start_sec)


def normalize_stage_timings(
    stage_timings: Mapping[str, object] | None,
) -> dict[str, float]:
    """Return canonical benchmark stage timing fields.

    Missing fields are represented as ``NaN`` because non-TBS methods do not
    execute TBS-specific stages.
    """
    normalized = {key: np.nan for key in BENCHMARK_STAGE_TIMING_KEYS}
    if stage_timings is None:
        return normalized
    for key in BENCHMARK_STAGE_TIMING_KEYS:
        if key not in stage_timings:
            continue
        value = float(stage_timings[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Stage timing {key!r} must be finite and non-negative; got {value}."
            )
        normalized[key] = value
    return normalized


__all__ = [
    "BENCHMARK_STAGE_TIMING_KEYS",
    "elapsed_since",
    "format_timestamp_utc",
    "normalize_stage_timings",
]
