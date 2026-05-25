"""Representation selection for local covariance eigendecomposition."""

from __future__ import annotations

from enum import Enum

from .preparation import PreparedCovarianceData


class CovarianceRepresentation(Enum):
    """Matrix space used to solve the same feature-covariance spectrum."""

    PRIMAL = "primal"
    DUAL = "dual"


def select_covariance_representation(
    prepared_data: PreparedCovarianceData,
) -> CovarianceRepresentation:
    """Choose the smaller matrix that preserves the feature-covariance spectrum."""
    if prepared_data.n_samples < prepared_data.n_active_features:
        return CovarianceRepresentation.DUAL
    return CovarianceRepresentation.PRIMAL


__all__ = ["CovarianceRepresentation", "select_covariance_representation"]
