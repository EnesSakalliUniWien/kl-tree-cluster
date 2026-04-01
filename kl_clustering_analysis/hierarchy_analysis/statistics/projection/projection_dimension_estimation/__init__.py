"""Projection-dimension estimation utilities."""

from .projection_dimension_estimators import (
    effective_rank,
    estimate_k_marchenko_pastur,
    marchenko_pastur_signal_count,
)

__all__ = [
    "effective_rank",
    "estimate_k_marchenko_pastur",
    "marchenko_pastur_signal_count",
]
