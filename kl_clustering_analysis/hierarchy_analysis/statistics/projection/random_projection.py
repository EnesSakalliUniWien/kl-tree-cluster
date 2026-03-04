"""Compatibility wrappers for random/JL projection utilities.

Numerical implementations now live in
``hierarchy_analysis.decomposition.backends.random_projection_backend``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ...decomposition.backends import random_projection_backend as _backend

# Keep cache object identity for tests/tools importing this symbol directly.
_PROJECTION_CACHE = _backend._PROJECTION_CACHE

# Read-only mirror of backend resolved floor (updated after backend calls).
_RESOLVED_MIN_K: int | None = _backend.get_resolved_min_k_backend()


def _sync_resolved_from_backend() -> None:
    """Sync backend global state back into local compatibility variable."""
    global _RESOLVED_MIN_K
    _RESOLVED_MIN_K = _backend.get_resolved_min_k_backend()


def estimate_min_projection_dimension(
    leaf_data: pd.DataFrame,
    *,
    hard_floor: int = 2,
    hard_cap: int = 20,
) -> int:
    """Estimate minimum projection dimension from global effective rank."""
    return _backend.estimate_min_projection_dimension_backend(
        leaf_data,
        hard_floor=hard_floor,
        hard_cap=hard_cap,
    )


def resolve_min_k(
    min_k_config: int | str,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Resolve ``PROJECTION_MIN_K`` to an integer and cache the value."""
    result = _backend.resolve_min_k_backend(min_k_config, leaf_data=leaf_data)
    _sync_resolved_from_backend()
    return result


def compute_projection_dimension(
    n_samples: int,
    n_features: int,
    eps: float = config.PROJECTION_EPS,
    min_k: int | str | None = None,
) -> int:
    """Compute target projection dimension, capped by data rank."""
    result = _backend.compute_projection_dimension_backend(
        n_samples=n_samples,
        n_features=n_features,
        eps=eps,
        min_k=min_k,
    )
    _sync_resolved_from_backend()
    return result


def generate_projection_matrix(
    n_features: int,
    k: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate an orthonormal random projection matrix."""
    return _backend.generate_projection_matrix_backend(
        n_features=n_features,
        k=k,
        random_state=random_state,
        use_cache=use_cache,
    )


def derive_projection_seed(base_seed: int | None, test_id: str) -> int:
    """Derive deterministic per-test projection seed."""
    return _backend.derive_projection_seed_backend(base_seed, test_id)


__all__ = [
    "compute_projection_dimension",
    "derive_projection_seed",
    "estimate_min_projection_dimension",
    "generate_projection_matrix",
    "resolve_min_k",
    "_PROJECTION_CACHE",
    "_RESOLVED_MIN_K",
]
