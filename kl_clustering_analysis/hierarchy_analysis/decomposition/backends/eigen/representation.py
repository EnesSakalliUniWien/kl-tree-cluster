"""Representation selection for local correlation eigendecomposition."""

from __future__ import annotations

from enum import Enum

from .preparation import PreparedCorrelationData


class CorrelationRepresentation(Enum):
    """Matrix space used to solve the same feature-correlation spectrum."""

    PRIMAL = "primal"
    DUAL = "dual"


def select_correlation_representation(
    prepared_data: PreparedCorrelationData,
) -> CorrelationRepresentation:
    """Choose the smaller matrix that preserves the feature-correlation spectrum."""
    if prepared_data.n_samples < prepared_data.n_active_features:
        return CorrelationRepresentation.DUAL
    return CorrelationRepresentation.PRIMAL


__all__ = ["CorrelationRepresentation", "select_correlation_representation"]
