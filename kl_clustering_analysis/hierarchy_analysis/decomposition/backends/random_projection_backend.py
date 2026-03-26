"""Numerical backend for random/JL projection internals."""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from kl_clustering_analysis import config

from ...statistics.projection.k_estimators import effective_rank as _effective_rank

logger = logging.getLogger(__name__)

_RESOLVED_MINIMUM_PROJECTION_DIMENSION: int | None = None


def _generate_structured_orthonormal_rows_backend(
    n_features: int,
    n_components: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate sparse signed-coordinate orthonormal rows."""
    selected_columns = rng.permutation(n_features)[:n_components]
    random_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_components)
    projection_matrix = np.zeros((n_components, n_features), dtype=np.float64)
    projection_matrix[np.arange(n_components), selected_columns] = random_signs
    return projection_matrix


def set_resolved_minimum_projection_dimension_backend(value: int | None) -> None:
    """Cache the resolved minimum projection dimension."""
    global _RESOLVED_MINIMUM_PROJECTION_DIMENSION
    _RESOLVED_MINIMUM_PROJECTION_DIMENSION = value


def get_resolved_minimum_projection_dimension_backend() -> int | None:
    """Return the cached resolved minimum projection dimension."""
    return _RESOLVED_MINIMUM_PROJECTION_DIMENSION


def estimate_projection_dimension_floor_backend(
    leaf_feature_matrix: pd.DataFrame,
    *,
    minimum_dimension_floor: int = 2,
    maximum_dimension_cap: int = 20,
) -> int:
    """Estimate an adaptive JL floor from the effective rank of the full data."""
    leaf_feature_values = leaf_feature_matrix.values.astype(np.float64)
    n_samples, n_features = leaf_feature_values.shape

    if n_samples < 2 or n_features < 2:
        return minimum_dimension_floor

    feature_variances = np.var(leaf_feature_values, axis=0)
    nonconstant_feature_mask = feature_variances > 0
    n_nonconstant_features = int(np.sum(nonconstant_feature_mask))
    if n_nonconstant_features < 2:
        return minimum_dimension_floor

    nonconstant_feature_matrix = leaf_feature_values[:, nonconstant_feature_mask]
    if n_samples < n_nonconstant_features:
        feature_means = nonconstant_feature_matrix.mean(axis=0)
        feature_standard_deviations = nonconstant_feature_matrix.std(axis=0, ddof=0)
        feature_standard_deviations[feature_standard_deviations == 0] = 1.0
        standardized_feature_matrix = (
            nonconstant_feature_matrix - feature_means
        ) / feature_standard_deviations
        gram_matrix = standardized_feature_matrix @ standardized_feature_matrix.T
        gram_matrix /= n_nonconstant_features
        spectrum_eigenvalues = np.sort(np.linalg.eigvalsh(gram_matrix))[::-1]
    else:
        correlation_matrix = np.corrcoef(nonconstant_feature_matrix.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)
        spectrum_eigenvalues = np.sort(np.linalg.eigvalsh(correlation_matrix))[::-1]

    estimated_projection_dimension_floor = int(
        np.ceil(_effective_rank(np.maximum(spectrum_eigenvalues, 0.0)))
    )
    estimated_projection_dimension_floor = max(
        estimated_projection_dimension_floor,
        minimum_dimension_floor,
    )
    estimated_projection_dimension_floor = min(
        estimated_projection_dimension_floor,
        maximum_dimension_cap,
    )

    logger.info(
        "Adaptive PROJECTION_MINIMUM_DIMENSION: minimum_projection_dimension=%d (n=%d, d=%d, d_active=%d)",
        estimated_projection_dimension_floor,
        n_samples,
        n_features,
        n_nonconstant_features,
    )
    return estimated_projection_dimension_floor


def resolve_minimum_projection_dimension_backend(
    minimum_projection_dimension_config: int | str,
    *,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Resolve the configured minimum projection dimension to an integer."""
    if isinstance(minimum_projection_dimension_config, int):
        set_resolved_minimum_projection_dimension_backend(minimum_projection_dimension_config)
        return minimum_projection_dimension_config

    if minimum_projection_dimension_config == "auto":
        if leaf_data is None:
            logger.info(
                "PROJECTION_MINIMUM_DIMENSION='auto' but leaf_data is None; falling back to 2."
            )
            set_resolved_minimum_projection_dimension_backend(2)
            return 2
        resolved = estimate_projection_dimension_floor_backend(leaf_data)
        set_resolved_minimum_projection_dimension_backend(resolved)
        return resolved

    raise ValueError(
        "PROJECTION_MINIMUM_DIMENSION must be an int or 'auto', "
        f"got {minimum_projection_dimension_config!r}"
    )


def compute_projection_dimension_backend(
    n_samples: int,
    n_features: int,
    *,
    eps: float | None = None,
    minimum_projection_dimension: int | str | None = None,
) -> int:
    """Compute a JL projection dimension with a configurable floor."""
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


def generate_projection_matrix_backend(
    n_features: int,
    n_components: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate an orthonormal projection matrix."""
    del use_cache

    if n_components <= 0 or n_features <= 0:
        return np.zeros((max(n_components, 0), max(n_features, 0)), dtype=np.float64)

    rng = np.random.RandomState(random_state)
    use_structured = n_components == n_features or (
        n_features >= 512 and (n_components / float(n_features)) >= 0.8
    )

    if use_structured:
        return _generate_structured_orthonormal_rows_backend(
            int(n_features),
            int(n_components),
            rng,
        )

    gaussian_matrix = rng.standard_normal((int(n_components), int(n_features)))
    orthogonal_basis, _ = np.linalg.qr(gaussian_matrix.T, mode="reduced")
    return orthogonal_basis.T


def derive_projection_seed_backend(base_seed: int | None, test_id: str) -> int:
    """Derive deterministic per-test seed from base seed and test id."""
    if not test_id:
        raise ValueError("test_id must be a non-empty string.")
    base = "none" if base_seed is None else str(int(base_seed))
    seed_payload = f"{base}|{test_id}".encode("utf-8")
    seed_digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
    return int.from_bytes(seed_digest, byteorder="big", signed=False) & 0xFFFFFFFF


__all__ = [
    "compute_projection_dimension_backend",
    "derive_projection_seed_backend",
    "estimate_projection_dimension_floor_backend",
    "generate_projection_matrix_backend",
    "get_resolved_minimum_projection_dimension_backend",
    "resolve_minimum_projection_dimension_backend",
    "set_resolved_minimum_projection_dimension_backend",
]
