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

from ..methods.k_estimators import effective_rank as _effective_rank

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
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate sparse signed-coordinate orthonormal rows."""
    if k <= 0 or n_features <= 0:
        return np.zeros((max(k, 0), max(n_features, 0)), dtype=np.float64)

    cols = rng.permutation(n_features)[:k]
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=k)
    R = np.zeros((k, n_features), dtype=np.float64)
    R[np.arange(k), cols] = signs
    return R


def estimate_min_projection_dimension_backend(
    leaf_data: pd.DataFrame,
    *,
    hard_floor: int = 2,
    hard_cap: int = 20,
) -> int:
    """Estimate adaptive projection floor from global effective rank."""
    X = leaf_data.values.astype(np.float64)
    n, d = X.shape

    if n < 2 or d < 2:
        return hard_floor

    col_var = np.var(X, axis=0)
    active_mask = col_var > 0
    d_active = int(np.sum(active_mask))

    if d_active < 2:
        return hard_floor

    X_active = X[:, active_mask]
    use_dual = n < d_active
    if use_dual:
        col_means = X_active.mean(axis=0)
        col_stds = X_active.std(axis=0, ddof=0)
        col_stds[col_stds == 0] = 1.0
        X_std = (X_active - col_means) / col_stds
        gram = X_std @ X_std.T / d_active
        eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
    else:
        corr = np.corrcoef(X_active.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    erank = _effective_rank(eigenvalues)
    minimum_projection_dimension = int(np.ceil(erank))
    minimum_projection_dimension = max(minimum_projection_dimension, hard_floor)
    minimum_projection_dimension = min(minimum_projection_dimension, hard_cap)

    logger.info(
        "Adaptive PROJECTION_MINIMUM_DIMENSION: effective_rank=%.1f -> minimum_projection_dimension=%d (n=%d, d=%d, d_active=%d)",
        erank,
        minimum_projection_dimension,
        n,
        d,
        d_active,
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
            cfg_val = config.PROJECTION_MINIMUM_DIMENSION
            minimum_projection_dimension = cfg_val if isinstance(cfg_val, int) else 2
    elif isinstance(minimum_projection_dimension, str):
        minimum_projection_dimension = 2

    n_samples = max(n_samples, 1)
    k: int = int(johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps))

    if n_features >= 4 * n_samples:
        k = min(k, n_samples)
    k = max(k, minimum_projection_dimension)
    k = min(k, n_features)
    return k


def _maybe_audit_projection_backend(
    cache_key: Tuple[str, int, int, int | None],
    matrix: np.ndarray,
) -> None:
    """Optionally export projection matrix audits."""
    root = os.getenv("KL_TE_MATRIX_AUDIT_ROOT")
    if not root:
        return

    if cache_key in _AUDITED_PROJECTIONS:
        return

    max_logs_env = os.getenv("KL_TE_MATRIX_AUDIT_MAX", "50")
    try:
        max_logs = int(max_logs_env)
    except ValueError:
        max_logs = 50
    if len(_AUDITED_PROJECTIONS) >= max_logs:
        return

    try:
        from benchmarks.shared.audit_utils import export_matrix_audit
    except Exception:
        return

    _AUDITED_PROJECTIONS.add(cache_key)
    method, n_features, k, random_state = cache_key
    tag = f"random_projection/{method}_d{n_features}_k{k}_seed{random_state}"

    export_matrix_audit(
        matrices={"projection_matrix": np.asarray(matrix)},
        output_root=Path(root),
        tag_prefix=tag,
        step=0,
        include_products=True,
        verbose=False,
    )


def _generate_orthonormal_projection_backend(
    n_features: int,
    k: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate orthonormal projection via QR or structured rows."""
    cache_key = ("orthonormal", n_features, k, random_state)

    if use_cache:
        if cache_key not in _PROJECTION_CACHE:
            rng = np.random.RandomState(random_state)
            use_structured = k == n_features or (
                n_features >= 512 and (k / float(n_features)) >= 0.8
            )
            if use_structured:
                R = _generate_structured_orthonormal_rows_backend(n_features, k, rng)
            else:
                G = rng.standard_normal((k, n_features))
                Q, _ = np.linalg.qr(G.T, mode="reduced")
                R = Q.T
            _PROJECTION_CACHE[cache_key] = R

        _maybe_audit_projection_backend(cache_key, _PROJECTION_CACHE[cache_key])
        return _PROJECTION_CACHE[cache_key]

    rng = np.random.RandomState(random_state)
    use_structured = k == n_features or (n_features >= 512 and (k / float(n_features)) >= 0.8)
    if use_structured:
        R = _generate_structured_orthonormal_rows_backend(n_features, k, rng)
    else:
        G = rng.standard_normal((k, n_features))
        Q, _ = np.linalg.qr(G.T, mode="reduced")
        R = Q.T
    _maybe_audit_projection_backend(cache_key, R)
    return R


def generate_projection_matrix_backend(
    n_features: int,
    k: int,
    *,
    random_state: int | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate orthonormal random projection matrix."""
    return _generate_orthonormal_projection_backend(
        n_features=n_features,
        k=k,
        random_state=random_state,
        use_cache=use_cache,
    )


def derive_projection_seed_backend(base_seed: int | None, test_id: str) -> int:
    """Derive deterministic per-test seed from base seed and test id."""
    if not test_id:
        raise ValueError("test_id must be a non-empty string.")
    base = "none" if base_seed is None else str(int(base_seed))
    payload = f"{base}|{test_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFF


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
