"""Random projection backend wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...statistics.projection.random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
    resolve_min_k,
)


def compute_projection_dimension_backend(
    n_samples: int,
    n_features: int,
    *,
    eps: float,
    min_k: int | str | None = None,
) -> int:
    """Backend wrapper for JL projection-dimension calculation."""
    return compute_projection_dimension(
        n_samples=n_samples,
        n_features=n_features,
        eps=eps,
        min_k=min_k,
    )


def generate_projection_matrix_backend(
    n_features: int,
    k: int,
    *,
    random_state: int | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Backend wrapper for random orthonormal projection generation."""
    return generate_projection_matrix(
        n_features=n_features,
        k=k,
        random_state=random_state,
        use_cache=use_cache,
    )


def derive_projection_seed_backend(base_seed: int | None, test_id: str) -> int:
    """Backend wrapper for deterministic per-test seed derivation."""
    return derive_projection_seed(base_seed, test_id)


def resolve_min_k_backend(
    min_k_config: int | str,
    *,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Backend wrapper for adaptive minimum projection dimension."""
    return resolve_min_k(min_k_config, leaf_data=leaf_data)


__all__ = [
    "compute_projection_dimension_backend",
    "generate_projection_matrix_backend",
    "derive_projection_seed_backend",
    "resolve_min_k_backend",
]

