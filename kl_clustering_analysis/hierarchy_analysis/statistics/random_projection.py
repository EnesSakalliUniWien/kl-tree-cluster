"""Random projection utilities for dimensionality reduction.

Uses SparseRandomProjection for a fast, memory-efficient, and deterministic
projection. The projection matrix is created ONCE and cached for reuse.

This approach is based on the Johnson-Lindenstrauss (JL) lemma, which
guarantees that pairwise distances are preserved. When the number of features
d >> sample size n, random projection reduces d to a smaller k = O(log n),
making subsequent statistical approximations more reliable.

Sparse random projections are highly efficient and theoretically sound, making
them ideal for high-dimensional data.

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
Achlioptas, D. (2003). Database-friendly random projections.
    Journal of Computer and System Sciences, 66(4), 671-687.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from kl_clustering_analysis import config


# Global cache for projection matrices - fitted once, reused everywhere
# Key: (n_features, k, random_state) -> fitted SparseRandomProjection
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


def generate_projection_matrix(
    n_features: int,
    k: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate a random projection matrix using SparseRandomProjection.

    Uses sklearn's SparseRandomProjection, which is highly efficient for
    high-dimensional data.

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
        Projection matrix of shape (k, n_features).

    References
    ----------
    Achlioptas, D. (2003). "Database-friendly random projections."
    """
    projector = _get_cached_projector(n_features, k, random_state)
    return projector.components_


__all__ = [
    "compute_projection_dimension",
    "generate_projection_matrix",
]
