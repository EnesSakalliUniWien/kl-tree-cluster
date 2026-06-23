"""JL dimension resolution for random projection."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection import config

from .floor import estimate_projection_dimension_floor

logger = logging.getLogger(__name__)

_RESOLVED_MINIMUM_PROJECTION_DIMENSION: int | None = None


def resolve_minimum_projection_dimension(
    minimum_projection_dimension_config: int | str,
    *,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Resolve the configured minimum projection dimension to an integer."""
    if isinstance(minimum_projection_dimension_config, int):
        set_resolved_minimum_projection_dimension(minimum_projection_dimension_config)
        return minimum_projection_dimension_config

    if minimum_projection_dimension_config == "auto":
        if leaf_data is None:
            logger.info(
                "PROJECTION_MINIMUM_DIMENSION='auto' but leaf_data is None; falling back to 2."
            )
            set_resolved_minimum_projection_dimension(2)
            return 2

        resolved = estimate_projection_dimension_floor(leaf_data)
        set_resolved_minimum_projection_dimension(resolved)
        logger.info(
            "Adaptive PROJECTION_MINIMUM_DIMENSION: minimum_projection_dimension=%d (n=%d, d=%d, d_active=%d)",
            resolved,
            leaf_data.shape[0],
            leaf_data.shape[1],
            int((leaf_data.var(axis=0) > 0).sum()),
        )
        return resolved

    raise ValueError(
        "PROJECTION_MINIMUM_DIMENSION must be an int or 'auto', "
        f"got {minimum_projection_dimension_config!r}"
    )


def compute_projection_dimension(
    n_samples: int,
    n_features: int,
    *,
    eps: float | None = None,
    minimum_projection_dimension: int | str | None = None,
) -> int:
    """Compute the JL projection dimension for a specific test.

    Starts from the Johnson-Lindenstrauss minimum dimension for ``n_samples``
    and ``eps``, then applies the globally resolved minimum projection floor
    and the ambient feature cap.
    """
    if eps is None:
        eps = config.PROJECTION_EPS

    if minimum_projection_dimension is None:
        if _RESOLVED_MINIMUM_PROJECTION_DIMENSION is not None:
            minimum_projection_dimension = _RESOLVED_MINIMUM_PROJECTION_DIMENSION
        else:
            configured_value = config.PROJECTION_MINIMUM_DIMENSION
            minimum_projection_dimension = (
                configured_value if isinstance(configured_value, int) else 2
            )
    elif isinstance(minimum_projection_dimension, str):
        minimum_projection_dimension = 2

    n_samples = max(int(n_samples), 1)
    projection_dimension = int(johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps))
    if n_features >= 4 * n_samples:
        projection_dimension = min(projection_dimension, n_samples)
    projection_dimension = max(projection_dimension, int(minimum_projection_dimension))
    projection_dimension = min(projection_dimension, int(n_features))
    return projection_dimension


def set_resolved_minimum_projection_dimension(value: int | None) -> None:
    """Cache the resolved minimum projection dimension."""
    global _RESOLVED_MINIMUM_PROJECTION_DIMENSION
    _RESOLVED_MINIMUM_PROJECTION_DIMENSION = value


def get_resolved_minimum_projection_dimension() -> int | None:
    """Return the cached resolved minimum projection dimension."""
    return _RESOLVED_MINIMUM_PROJECTION_DIMENSION
