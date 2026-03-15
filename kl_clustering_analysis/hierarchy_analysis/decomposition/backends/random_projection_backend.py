"""Numerical backend for random/JL projection internals."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from kl_clustering_analysis import config

from ...statistics.projection.k_estimators import effective_rank as _effective_rank

logger = logging.getLogger(__name__)

# Global projection cache.
_PROJECTION_CACHE: Dict[Tuple[str, int, int, int | None], np.ndarray] = {}
_AUDITED_PROJECTIONS: set[Tuple[str, int, int, int | None]] = set()

# Resolved adaptive floor.
_RESOLVED_MINIMUM_PROJECTION_DIMENSION: int | None = None


def set_resolved_minimum_projection_dimension_backend(value: int | None) -> None:
    """Set backend resolved minimum dimension cache."""
    global _RESOLVED_MINIMUM_PROJECTION_DIMENSION
    _RESOLVED_MINIMUM_PROJECTION_DIMENSION = value


def get_resolved_minimum_projection_dimension_backend() -> int | None:
    """Get backend resolved minimum dimension cache."""
    return _RESOLVED_MINIMUM_PROJECTION_DIMENSION


def _generate_structured_orthonormal_rows_backend(
    n_features: int,
    n_components: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate sparse signed-coordinate orthonormal rows."""
    if n_components <= 0 or n_features <= 0:
        return np.zeros((max(n_components, 0), max(n_features, 0)), dtype=np.float64)

    selected_columns = rng.permutation(n_features)[:n_components]
    random_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_components)
    projection_matrix = np.zeros((n_components, n_features), dtype=np.float64)
    projection_matrix[np.arange(n_components), selected_columns] = random_signs
    return projection_matrix


def estimate_min_projection_dimension_backend(
    leaf_data: pd.DataFrame,
    *,
    hard_floor: int = 2,
    hard_cap: int = 20,
) -> int:
    """Estimate adaptive projection floor from global effective rank."""
    feature_matrix = leaf_data.values.astype(np.float64)
    n_samples, n_features = feature_matrix.shape

    if n_samples < 2 or n_features < 2:
        return hard_floor

    column_variance = np.var(feature_matrix, axis=0)
    active_feature_mask = column_variance > 0
    n_active_features = int(np.sum(active_feature_mask))

    if n_active_features < 2:
        return hard_floor

    active_feature_matrix = feature_matrix[:, active_feature_mask]
    use_dual = n_samples < n_active_features
    if use_dual:
        column_means = active_feature_matrix.mean(axis=0)
        column_stds = active_feature_matrix.std(axis=0, ddof=0)
        column_stds[column_stds == 0] = 1.0
        standardized_features = (active_feature_matrix - column_means) / column_stds
        gram_matrix = standardized_features @ standardized_features.T / n_active_features
        eigenvalues = np.sort(np.linalg.eigvalsh(gram_matrix))[::-1]
    else:
        correlation_matrix = np.corrcoef(active_feature_matrix.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)
        eigenvalues = np.sort(np.linalg.eigvalsh(correlation_matrix))[::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    estimated_effective_rank = _effective_rank(eigenvalues)
    minimum_projection_dimension = int(np.ceil(estimated_effective_rank))
    minimum_projection_dimension = max(minimum_projection_dimension, hard_floor)
    minimum_projection_dimension = min(minimum_projection_dimension, hard_cap)

    logger.info(
        "Adaptive PROJECTION_MINIMUM_DIMENSION: effective_rank=%.1f -> minimum_projection_dimension=%d (n=%d, d=%d, d_active=%d)",
        estimated_effective_rank,
        minimum_projection_dimension,
        n_samples,
        n_features,
        n_active_features,
    )

    return minimum_projection_dimension


def resolve_minimum_projection_dimension_backend(
    minimum_projection_dimension_config: int | str,
    *,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Resolve configured minimum k to an integer and cache the result."""
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
        resolved = estimate_min_projection_dimension_backend(leaf_data)
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
    eps: float = config.PROJECTION_EPS,
    minimum_projection_dimension: int | str | None = None,
) -> int:
    """Compute target projection dimension with JL + information cap."""
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

    n_samples = max(n_samples, 1)
    projection_dimension: int = int(johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps))

    if n_features >= 4 * n_samples:
        projection_dimension = min(projection_dimension, n_samples)
    projection_dimension = max(projection_dimension, minimum_projection_dimension)
    projection_dimension = min(projection_dimension, n_features)
    return projection_dimension


def _maybe_audit_projection_backend(
    cache_key: Tuple[str, int, int, int | None],
    matrix: np.ndarray,
) -> None:
    """Optionally export projection matrix audits."""
    audit_root = os.getenv("KL_TE_MATRIX_AUDIT_ROOT")
    if not audit_root:
        return

    if cache_key in _AUDITED_PROJECTIONS:
        return

    max_audit_logs_env = os.getenv("KL_TE_MATRIX_AUDIT_MAX", "50")
    try:
        max_audit_logs = int(max_audit_logs_env)
    except ValueError:
        max_audit_logs = 50
    if len(_AUDITED_PROJECTIONS) >= max_audit_logs:
        return

    try:
        from benchmarks.shared.audit_utils import export_matrix_audit
    except Exception:
        return

    _AUDITED_PROJECTIONS.add(cache_key)
    method_name, n_features, n_components, random_state = cache_key
    audit_tag = f"random_projection/{method_name}_d{n_features}_k{n_components}_seed{random_state}"

    export_matrix_audit(
        matrices={"projection_matrix": np.asarray(matrix)},
        output_root=Path(audit_root),
        tag_prefix=audit_tag,
        step=0,
        include_products=True,
        verbose=False,
    )


def _generate_orthonormal_projection_backend(
    n_features: int,
    n_components: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate orthonormal projection via QR or structured rows."""
    cache_key = ("orthonormal", n_features, n_components, random_state)

    if use_cache:
        if cache_key not in _PROJECTION_CACHE:

            rng = np.random.RandomState(random_state)

            use_structured = n_components == n_features or (
                n_features >= 512 and (n_components / float(n_features)) >= 0.8
            )

            if use_structured:
                projection_matrix = _generate_structured_orthonormal_rows_backend(
                    n_features, n_components, rng
                )
            else:
                gaussian_matrix = rng.standard_normal((n_components, n_features))
                orthogonal_basis, _ = np.linalg.qr(gaussian_matrix.T, mode="reduced")
                projection_matrix = orthogonal_basis.T

            _PROJECTION_CACHE[cache_key] = projection_matrix

        _maybe_audit_projection_backend(cache_key, _PROJECTION_CACHE[cache_key])
        return _PROJECTION_CACHE[cache_key]

    rng = np.random.RandomState(random_state)

    use_structured = n_components == n_features or (
        n_features >= 512 and (n_components / float(n_features)) >= 0.8
    )

    if use_structured:
        projection_matrix = _generate_structured_orthonormal_rows_backend(
            n_features, n_components, rng
        )
    else:
        gaussian_matrix = rng.standard_normal((n_components, n_features))
        orthogonal_basis, _ = np.linalg.qr(gaussian_matrix.T, mode="reduced")
        projection_matrix = orthogonal_basis.T

    _maybe_audit_projection_backend(cache_key, projection_matrix)

    return projection_matrix


def generate_projection_matrix_backend(
    n_features: int,
    n_components: int,
    *,
    random_state: int | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate orthonormal random projection matrix."""
    return _generate_orthonormal_projection_backend(
        n_features=n_features,
        n_components=n_components,
        random_state=random_state,
        use_cache=use_cache,
    )


def derive_projection_seed_backend(base_seed: int | None, test_id: str) -> int:
    """Derive deterministic per-test seed from base seed and test id."""
    if not test_id:
        raise ValueError("test_id must be a non-empty string.")
    base = "none" if base_seed is None else str(int(base_seed))
    seed_payload = f"{base}|{test_id}".encode("utf-8")
    seed_digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
    return int.from_bytes(seed_digest, byteorder="big", signed=False) & 0xFFFFFFFF


__all__ = [
    "_PROJECTION_CACHE",
    "_RESOLVED_MINIMUM_PROJECTION_DIMENSION",
    "get_resolved_minimum_projection_dimension_backend",
    "set_resolved_minimum_projection_dimension_backend",
    "estimate_min_projection_dimension_backend",
    "resolve_minimum_projection_dimension_backend",
    "compute_projection_dimension_backend",
    "generate_projection_matrix_backend",
    "derive_projection_seed_backend",
]
