"""Random projection utilities for dimensionality reduction.

Uses GaussianRandomProjection with a fixed random_state for deterministic,
reproducible projections. The projection matrix is created ONCE and cached
for reuse across all statistical tests.

Based on the Johnson-Lindenstrauss lemma which guarantees that
k = O(log(n) / ε²) dimensions suffice to preserve pairwise
distances within a factor of (1 ± ε).

When the number of features d >> sample size n, the chi-square
approximation 2*n*JSD ~ χ²(d) becomes unreliable. Random projection
reduces d to k = O(log n), making the approximation valid.

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
Achlioptas, D. (2003). Database-friendly random projections.
    Journal of Computer and System Sciences, 66(4), 671-687.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from kl_clustering_analysis import config


# Global cache for projection matrices - fitted once, reused everywhere
# Key: (n_features, k, random_state) -> fitted GaussianRandomProjection
_PROJECTOR_CACHE: Dict[Tuple[int, int, Optional[int]], GaussianRandomProjection] = {}


def should_use_projection(
    n_features: int,
    n_samples: int,
    threshold_ratio: float = config.PROJECTION_THRESHOLD_RATIO,
) -> bool:
    """Determine if random projection should be applied.

    Parameters
    ----------
    n_features : int
        Number of features (d).
    n_samples : int
        Effective sample size (n).
    threshold_ratio : float
        Apply projection if d > threshold_ratio * n.

    Returns
    -------
    bool
        True if projection is recommended.
    """
    if not config.USE_RANDOM_PROJECTION:
        return False
    return n_features > threshold_ratio * n_samples


def compute_projection_dimension(
    n_samples: int,
    n_features: int,
    k_multiplier: float = config.PROJECTION_K_MULTIPLIER,
    min_k: int = config.PROJECTION_MIN_K,
) -> int:
    """Compute target dimension for random projection.

    Uses k = k_multiplier * log(n), capped at original dimension.
    Based on the Johnson-Lindenstrauss lemma which guarantees that
    k = O(log(n) / ε²) dimensions suffice to preserve pairwise
    distances within a factor of (1 ± ε).

    Parameters
    ----------
    n_samples : int
        Effective sample size.
    n_features : int
        Original number of features.
    k_multiplier : float
        Multiplier for log(n).
    min_k : int
        Minimum projected dimension.

    Returns
    -------
    int
        Target dimension k.

    References
    ----------
    Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
        mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
    """
    k = int(np.ceil(k_multiplier * np.log(max(n_samples, 2))))
    k = max(k, min_k)
    k = min(k, n_features)  # Never exceed original dimension
    return k


def get_cached_projector(
    n_features: int,
    k: int,
    random_state: Optional[int] = config.PROJECTION_RANDOM_SEED,
) -> GaussianRandomProjection:
    """Get or create a cached GaussianRandomProjection.

    This ensures the SAME projection matrix is used for all tests
    with the same (n_features, k, random_state), making results
    fully deterministic and reproducible.

    Parameters
    ----------
    n_features : int
        Original dimension (d).
    k : int
        Target dimension.
    random_state : int, optional
        Random seed for reproducibility (default: config.PROJECTION_RANDOM_SEED).

    Returns
    -------
    GaussianRandomProjection
        Fitted projector with cached components_.
    """
    cache_key = (n_features, k, random_state)

    if cache_key not in _PROJECTOR_CACHE:
        projector = GaussianRandomProjection(
            n_components=k,
            random_state=random_state,
        )
        # Fit on dummy data to initialize components
        dummy = np.zeros((1, n_features))
        projector.fit(dummy)
        _PROJECTOR_CACHE[cache_key] = projector

    return _PROJECTOR_CACHE[cache_key]


def clear_projector_cache() -> None:
    """Clear the projector cache.

    Call this when starting a new analysis to ensure fresh projections.
    """
    global _PROJECTOR_CACHE
    _PROJECTOR_CACHE.clear()


def generate_projection_matrix(
    n_features: int,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate a random projection matrix using GaussianRandomProjection.

    Uses sklearn's GaussianRandomProjection which draws components
    from N(0, 1/n_components) - this is the standard JL projection.

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
    Johnson-Lindenstrauss lemma guarantees distance preservation.
    """
    projector = get_cached_projector(n_features, k, random_state)
    return projector.components_


def create_projector(
    n_features: int,
    k: int,
    random_state: Optional[int] = None,
) -> GaussianRandomProjection:
    """Create a fitted sklearn GaussianRandomProjection.

    Uses the cached projector for deterministic results.

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
    GaussianRandomProjection
        Fitted projector ready to transform data.
    """
    return get_cached_projector(n_features, k, random_state)


def project_distribution(
    distribution: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Project a probability distribution to lower dimension.

    For Bernoulli distributions, projects the mean vector θ.
    The projected values may not be in [0,1], so they should be
    interpreted as scores, not probabilities.

    Parameters
    ----------
    distribution : np.ndarray
        Original distribution (shape: (n_features,)).
    projection_matrix : np.ndarray
        Projection matrix (shape: (k, n_features)).

    Returns
    -------
    np.ndarray
        Projected distribution (shape: (k,)).
    """
    return projection_matrix @ distribution


def projected_euclidean_distance_squared(
    p: np.ndarray,
    q: np.ndarray,
    projection_matrix: np.ndarray,
) -> float:
    """Compute squared Euclidean distance in projected space.

    By Johnson-Lindenstrauss, this approximately preserves the
    original squared distance: ||Rp - Rq||² ≈ ||p - q||²

    Parameters
    ----------
    p, q : np.ndarray
        Original distributions (shape: (n_features,)).
    projection_matrix : np.ndarray
        Projection matrix (shape: (k, n_features)).

    Returns
    -------
    float
        Squared Euclidean distance in projected space.
    """
    diff = p - q
    projected_diff = projection_matrix @ diff
    return float(np.sum(projected_diff**2))


def projected_distance_with_projector(
    p: np.ndarray,
    q: np.ndarray,
    projector: GaussianRandomProjection,
) -> float:
    """Compute squared Euclidean distance using sklearn projector.

    Parameters
    ----------
    p, q : np.ndarray
        Original distributions (shape: (n_features,)).
    projector : GaussianRandomProjection
        Fitted sklearn projector.

    Returns
    -------
    float
        Squared Euclidean distance in projected space.
    """
    diff = (p - q).reshape(1, -1)  # sklearn expects 2D
    projected = projector.transform(diff)
    return float(np.sum(projected**2))


def averaged_projected_distance(
    p: np.ndarray,
    q: np.ndarray,
    k: int,
    n_trials: int = config.PROJECTION_N_TRIALS,
    base_seed: Optional[int] = config.PROJECTION_RANDOM_SEED,
) -> Tuple[float, int]:
    """Compute squared distance using a single cached random projection.

    Uses a deterministic projection (same matrix every time) for
    reproducible results. The n_trials parameter is kept for API
    compatibility but is ignored - we now use a single projection.

    Parameters
    ----------
    p, q : np.ndarray
        Original distributions.
    k : int
        Target dimension.
    n_trials : int
        Ignored (kept for API compatibility).
    base_seed : int, optional
        Random seed for the projection (default: config.PROJECTION_RANDOM_SEED).

    Returns
    -------
    Tuple[float, int]
        (squared_distance, projection_dimension)
    """
    n_features = len(p)

    # Use single cached projector for deterministic results
    projector = get_cached_projector(n_features, k, base_seed)
    dist_sq = projected_distance_with_projector(p, q, projector)

    return float(dist_sq), k


def projected_chi_square_statistic(
    p: np.ndarray,
    q: np.ndarray,
    n_effective: float,
    k: int,
    n_trials: int = config.PROJECTION_N_TRIALS,
    base_seed: Optional[int] = config.PROJECTION_RANDOM_SEED,
) -> Tuple[float, int]:
    """Compute chi-square statistic using random projection.

    The statistic is based on the squared Euclidean distance in
    projected space, scaled appropriately for a chi-square test.

    For Bernoulli distributions with means θ, the squared difference
    ||θ_left - θ_right||² relates to the chi-square statistic as:
    χ² ≈ n_eff * ||Δθ||² / σ²

    where σ² is the pooled variance. With projection:
    χ² ≈ n_eff * ||R(Δθ)||²

    Parameters
    ----------
    p, q : np.ndarray
        Distributions (Bernoulli means) for each group.
    n_effective : float
        Effective sample size (e.g., harmonic mean for two-sample).
    k : int
        Target projection dimension.
    n_trials : int
        Number of projections to average.
    base_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, int]
        (chi_square_statistic, degrees_of_freedom)
    """
    avg_dist_sq, proj_dim = averaged_projected_distance(
        p, q, k, n_trials=n_trials, base_seed=base_seed
    )

    # Scaling: n_eff * ||Δθ||² / (pooled_variance)
    # For Bernoulli with unknown θ, use pooled estimate
    pooled = 0.5 * (p + q)
    pooled_var = pooled * (1 - pooled)

    # Use the average variance as scaling
    avg_var = float(np.mean(pooled_var)) + 1e-10

    # Chi-square statistic
    chi_sq = n_effective * avg_dist_sq / avg_var

    return chi_sq, proj_dim


__all__ = [
    "should_use_projection",
    "compute_projection_dimension",
    "generate_projection_matrix",
    "create_projector",
    "get_cached_projector",
    "clear_projector_cache",
    "project_distribution",
    "projected_euclidean_distance_squared",
    "projected_distance_with_projector",
    "averaged_projected_distance",
    "projected_chi_square_statistic",
]
