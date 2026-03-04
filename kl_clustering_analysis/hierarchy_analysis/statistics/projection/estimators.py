"""Compatibility wrappers for projection estimators.

Canonical implementations now live in
``hierarchy_analysis.decomposition.methods.k_estimators``.
"""

from __future__ import annotations

import numpy as np

from ...decomposition.methods.k_estimators import (
    count_active_features as _count_active_features_method,
)
from ...decomposition.methods.k_estimators import (
    effective_rank as _effective_rank_method,
)
from ...decomposition.methods.k_estimators import (
    marchenko_pastur_signal_count as _marchenko_pastur_signal_count_method,
)


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Continuous effective rank via Shannon entropy."""
    return _effective_rank_method(np.asarray(eigenvalues, dtype=np.float64))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n: int,
    d: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound."""
    return _marchenko_pastur_signal_count_method(
        np.asarray(eigenvalues, dtype=np.float64),
        n_desc=int(n),
        d_active=int(d),
    )


def count_active_features(data_sub: np.ndarray) -> int:
    """Count features with non-zero variance."""
    return _count_active_features_method(np.asarray(data_sub, dtype=np.float64))


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
]
