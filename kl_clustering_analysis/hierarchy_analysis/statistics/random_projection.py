"""Random projection utilities for dimensionality reduction.

Uses orthonormal projection for exact χ² distribution guarantees.

The projection matrix is created ONCE and cached for reuse.

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
"""

from __future__ import annotations

from typing import Dict, Tuple
import os
from pathlib import Path
import hashlib

import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from kl_clustering_analysis import config


# Global cache for projection matrices - fitted once, reused everywhere
# Key: (method, n_features, k, random_state) -> projection matrix (ndarray)
_PROJECTION_CACHE: Dict[Tuple[str, int, int, int | None], np.ndarray] = {}

# Audit cache to avoid logging the same projection repeatedly.
_AUDITED_PROJECTIONS: set[Tuple[str, int, int, int | None]] = set()


def _generate_structured_orthonormal_rows(
    n_features: int,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate a sparse orthonormal projection with signed coordinate rows.

    Each row selects one coordinate with a random sign, and selected coordinates
    are unique across rows. Rows are exactly orthonormal by construction.
    """
    if k <= 0 or n_features <= 0:
        return np.zeros((max(k, 0), max(n_features, 0)), dtype=np.float64)

    cols = rng.permutation(n_features)[:k]
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=k)
    R = np.zeros((k, n_features), dtype=np.float64)
    R[np.arange(k), cols] = signs
    return R


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


def _generate_orthonormal_projection(
    n_features: int,
    k: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
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

    if use_cache:
        if cache_key not in _PROJECTION_CACHE:
            # NOTE:
            # On this benchmark stack (Python 3.13 + NumPy 2.3.x), repeated
            # construction of `default_rng(...)` in tight loops intermittently
            # segfaults inside SeedSequence internals for large runs.
            # Use RandomState here for stability while preserving deterministic
            # behavior from the same integer seed.
            rng = np.random.RandomState(random_state)

            # For near full-dimensional projections, dense QR is very expensive
            # and has shown native instability on some stacks. Use a structured
            # orthonormal projection instead (still exactly orthonormal rows).
            use_structured = (
                k == n_features
                or (n_features >= 512 and (k / float(n_features)) >= 0.8)
            )
            if use_structured:
                R = _generate_structured_orthonormal_rows(n_features, k, rng)
            else:
                # Generate k×d Gaussian random matrix
                G = rng.standard_normal((k, n_features))

                # QR decomposition: G.T = Q @ R_qr, where Q has orthonormal columns
                # Q is (n_features × k), so Q.T is (k × n_features) with orthonormal rows
                Q, _ = np.linalg.qr(G.T, mode="reduced")

                # No scaling needed: R @ R.T = I guarantees ||R @ z||² ~ χ²(k)
                # when z ~ N(0, I). The orthonormal rows preserve the chi-square
                # distribution exactly.
                R = Q.T  # Shape: (k, n_features)

            _PROJECTION_CACHE[cache_key] = R

        _maybe_audit_projection(cache_key, _PROJECTION_CACHE[cache_key])
        return _PROJECTION_CACHE[cache_key]

    # See stability note above about default_rng on this runtime stack.
    rng = np.random.RandomState(random_state)
    use_structured = (
        k == n_features or (n_features >= 512 and (k / float(n_features)) >= 0.8)
    )
    if use_structured:
        R = _generate_structured_orthonormal_rows(n_features, k, rng)
    else:
        G = rng.standard_normal((k, n_features))
        Q, _ = np.linalg.qr(G.T, mode="reduced")
        R = Q.T
    _maybe_audit_projection(cache_key, R)
    return R


def generate_projection_matrix(
    n_features: int,
    k: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate an orthonormal random projection matrix.

    Uses QR decomposition of a Gaussian random matrix to produce an
    orthonormal projection. Guarantees that T = ||R·z||² ~ χ²(k) exactly.

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
        Orthonormal projection matrix of shape (k, n_features).
    """
    return _generate_orthonormal_projection(
        n_features, k, random_state, use_cache=use_cache
    )


def derive_projection_seed(base_seed: int | None, test_id: str) -> int:
    """Derive a deterministic per-test projection seed.

    Parameters
    ----------
    base_seed
        Global base seed from configuration. May be None.
    test_id
        Stable identifier for the hypothesis/test (for example edge or parent id).

    Returns
    -------
    int
        Deterministic 32-bit seed suitable for NumPy RNG initialization.
    """
    if not test_id:
        raise ValueError("test_id must be a non-empty string.")

    base = "none" if base_seed is None else str(int(base_seed))
    payload = f"{base}|{test_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFF


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


__all__ = [
    "compute_projection_dimension",
    "derive_projection_seed",
    "generate_projection_matrix",
]
