"""Random projection utilities for dimensionality reduction.

Supports two projection methods:
1. **Sparse (default for speed)**: Uses SparseRandomProjection based on the
   Johnson-Lindenstrauss (JL) lemma. Fast and memory-efficient, but the
   projected test statistic follows χ² only approximately.

2. **Orthonormal (for exactness)**: Uses QR decomposition of a Gaussian random
   matrix to produce an orthonormal projection. Slower but guarantees that
   T = ||R·z||² ~ χ²(k) exactly under H₀.

The projection matrix is created ONCE and cached for reuse.

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
Achlioptas, D. (2003). Database-friendly random projections.
    Journal of Computer and System Sciences, 66(4), 671-687.
"""

from __future__ import annotations

from typing import Dict, Tuple
import os
from pathlib import Path

import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from kl_clustering_analysis import config


# Global cache for projection matrices - fitted once, reused everywhere
# Key: (method, n_features, k, random_state) -> projection matrix (ndarray)
_PROJECTION_CACHE: Dict[Tuple[str, int, int, int | None], np.ndarray] = {}

# Audit cache to avoid logging the same projection repeatedly.
_AUDITED_PROJECTIONS: set[Tuple[str, int, int, int | None]] = set()

# Legacy cache for sparse projectors (kept for backward compatibility)
_PROJECTOR_CACHE: Dict[Tuple[int, int, int | None], SparseRandomProjection] = {}


def compute_projection_dimension(
    n_samples: int,
    n_features: int,
    eps: float = config.PROJECTION_EPS,
    min_k: int = config.PROJECTION_MIN_K,
) -> int:
    """Compute target dimension for random projection using Johnson-Lindenstrauss lemma.

    Uses sklearn's johnson_lindenstrauss_min_dim for a theoretically-grounded
    dimension that guarantees pairwise distances are preserved within (1 ± eps).

    The JL lemma formula: k >= 4 * ln(n) / (eps^2/2 - eps^3/3)
    For small eps, this simplifies to approximately: k ≈ 8 * ln(n) / eps^2

    Parameters
    ----------
    n_samples : int
        Effective sample size (number of points whose pairwise distances
        need to be preserved).
    n_features : int
        Original number of features.
    eps : float
        Distortion tolerance. Distances are preserved within (1 ± eps).
        - eps=0.1: Conservative, many dimensions, ±10% distortion
        - eps=0.3: Balanced, moderate dimensions, ±30% distortion (recommended)
        - eps=0.5: Aggressive, few dimensions, ±50% distortion
    min_k : int
        Minimum projected dimension (floor).

    Returns
    -------
    int
        Target dimension k satisfying JL guarantee.

    References
    ----------
    Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
        mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
    sklearn.random_projection.johnson_lindenstrauss_min_dim
    """

    # Use sklearn's theoretically-grounded JL formula
    # Ensure n_samples >= 1 to avoid edge cases
    n_samples = max(n_samples, 1)
    k = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)

    k = max(k, min_k)
    k = min(k, n_features)  # Never exceed original dimension
    return k


def _get_cached_projector(
    n_features: int,
    k: int,
    random_state: int | None = config.PROJECTION_RANDOM_SEED,
) -> SparseRandomProjection:
    """Get or create a cached SparseRandomProjection (internal helper)."""
    cache_key = (n_features, k, random_state)

    if cache_key not in _PROJECTOR_CACHE:
        # Use SparseRandomProjection for performance with high-dimensional data.
        # It's much faster than GaussianRandomProjection when n_features is large.
        projector = SparseRandomProjection(
            n_components=k,
            random_state=random_state,
        )
        # Fit on dummy data to initialize components
        dummy = np.zeros((1, n_features))
        projector.fit(dummy)
        _PROJECTOR_CACHE[cache_key] = projector

    return _PROJECTOR_CACHE[cache_key]


def _generate_orthonormal_projection(
    n_features: int,
    k: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate an orthonormal projection matrix via QR decomposition.

    Creates R such that R @ R.T = I_k (identity), guaranteeing that
    T = ||R @ z||² ~ χ²(k) exactly when z ~ N(0, I_d).

    Parameters
    ----------
    n_features : int
        Original dimension (d).
    k : int
        Target dimension. Must be <= n_features.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Orthonormal projection matrix of shape (k, n_features).
    """
    cache_key = ("orthonormal", n_features, k, random_state)

    if cache_key not in _PROJECTION_CACHE:
        rng = np.random.default_rng(random_state)

        # Generate k×d Gaussian random matrix
        G = rng.standard_normal((k, n_features))

        # QR decomposition: G.T = Q @ R_qr, where Q has orthonormal columns
        # Q is (n_features × k), so Q.T is (k × n_features) with orthonormal rows
        Q, _ = np.linalg.qr(G.T)

        # No scaling needed: R @ R.T = I guarantees ||R @ z||² ~ χ²(k)
        # when z ~ N(0, I). The orthonormal rows preserve the chi-square
        # distribution exactly.
        R = Q.T  # Shape: (k, n_features)

        _PROJECTION_CACHE[cache_key] = R

    _maybe_audit_projection(cache_key, _PROJECTION_CACHE[cache_key])

    return _PROJECTION_CACHE[cache_key]


def _generate_sparse_projection(
    n_features: int,
    k: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate a sparse random projection matrix.

    Uses sklearn's SparseRandomProjection for efficiency.
    Note: R @ R.T ≠ I_k, so χ² distribution is approximate.

    Parameters
    ----------
    n_features : int
        Original dimension (d).
    k : int
        Target dimension.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sparse projection matrix of shape (k, n_features).
    """
    cache_key = ("sparse", n_features, k, random_state)

    if cache_key not in _PROJECTION_CACHE:
        projector = _get_cached_projector(n_features, k, random_state)
        _PROJECTION_CACHE[cache_key] = projector.components_

    _maybe_audit_projection(cache_key, _PROJECTION_CACHE[cache_key])

    return _PROJECTION_CACHE[cache_key]


def generate_projection_matrix(
    n_features: int,
    k: int,
    random_state: int | None = None,
    method: str | None = None,
) -> np.ndarray:
    """Generate a random projection matrix.

    Supports two methods controlled by `method` parameter or config.PROJECTION_METHOD:
    - 'sparse': Fast SparseRandomProjection (JL lemma, approximate χ² distribution)
    - 'orthonormal': QR-based orthonormal projection (exact χ² distribution)

    Parameters
    ----------
    n_features : int
        Original dimension (d).
    k : int
        Target dimension.
    random_state : int, optional
        Random seed for reproducibility.
    method : str, optional
        Projection method: 'sparse' or 'orthonormal'.
        If None, uses config.PROJECTION_METHOD.

    Returns
    -------
    np.ndarray
        Projection matrix of shape (k, n_features).

    Notes
    -----
    For hypothesis testing where exact p-values matter, use 'orthonormal'.
    For speed with very high dimensions (d > 10,000), use 'sparse'.
    """
    if method is None:
        method = getattr(config, "PROJECTION_METHOD", "sparse")

    if method == "orthonormal":
        return _generate_orthonormal_projection(n_features, k, random_state)
    elif method == "sparse":
        return _generate_sparse_projection(n_features, k, random_state)
    else:
        raise ValueError(
            f"Unknown projection method: {method}. Use 'sparse' or 'orthonormal'."
        )


def _maybe_audit_projection(
    cache_key: Tuple[str, int, int, int | None],
    matrix: np.ndarray,
) -> None:
    """Optionally audit projection matrices to TensorBoard.

    Controlled by KL_TE_MATRIX_AUDIT_ROOT env var. Audits only the first
    occurrence of each cache key to limit output volume.
    """
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

    matrices = {"projection_matrix": np.asarray(matrix)}
    export_matrix_audit(
        matrices=matrices,
        output_root=Path(root),
        tag_prefix=tag,
        step=0,
        include_products=True,
        verbose=False,
    )


def should_use_projection(
    n_features: int,
    n_samples: int,
    eps: float = config.PROJECTION_EPS,
) -> bool:
    """Check if random projection will reduce dimensionality.

    Projection is applied when the JL-based target dimension k is smaller
    than the original number of features.

    Parameters
    ----------
    n_features : int
        Original number of features.
    n_samples : int
        Effective sample size.
    eps : float
        Distortion tolerance (default from config).

    Returns
    -------
    bool
        True if projection will reduce dimensionality.
    """
    k = compute_projection_dimension(n_samples, n_features, eps=eps)
    return k < n_features


__all__ = [
    "compute_projection_dimension",
    "generate_projection_matrix",
    "should_use_projection",
]
