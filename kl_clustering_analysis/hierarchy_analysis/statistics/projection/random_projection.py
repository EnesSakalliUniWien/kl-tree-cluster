"""Random projection utilities for dimensionality reduction.

Uses orthonormal projection for exact χ² distribution guarantees.

The projection matrix is created ONCE and cached for reuse.

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
"""

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

logger = logging.getLogger(__name__)

# Global cache for projection matrices - fitted once, reused everywhere
# Key: (method, n_features, k, random_state) -> projection matrix (ndarray)
_PROJECTION_CACHE: Dict[Tuple[str, int, int, int | None], np.ndarray] = {}

# Audit cache to avoid logging the same projection repeatedly.
_AUDITED_PROJECTIONS: set[Tuple[str, int, int, int | None]] = set()

# --- Resolved min_k cache (set once per pipeline run) ---
# When PROJECTION_MIN_K="auto", the resolved integer is stored here
# so that compute_projection_dimension() can read it without needing
# the leaf_data parameter.  Set by resolve_min_k() on first call.
_RESOLVED_MIN_K: int | None = None


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


# --- Adaptive floor estimation ---


def estimate_min_projection_dimension(
    leaf_data: pd.DataFrame,
    *,
    hard_floor: int = 2,
    hard_cap: int = 20,
) -> int:
    """Estimate minimum projection dimension from the global effective rank.

    Computes the effective rank (Shannon entropy of eigenvalue spectrum) of
    the full dataset's correlation matrix and uses it as a data-driven floor
    for projection dimension.  This prevents the fixed ``PROJECTION_MIN_K``
    from adding pure-noise χ² components when the data has low intrinsic
    dimensionality — and avoids under-projecting high-rank data.

    Parameters
    ----------
    leaf_data
        Binary data matrix (samples × features).
    hard_floor
        Absolute minimum — ensures χ²(k) with k ≥ 2 for valid testing.
    hard_cap
        Upper clamp — the floor should not be too large; the JL formula
        and effective-rank estimators determine the actual k above this.

    Returns
    -------
    int
        Data-adaptive minimum projection dimension.

    Notes
    -----
    The computation is O(min(n, d)³) via dual-form eigendecomposition,
    executed once per pipeline run.  For typical binary datasets (n ~ 50-500,
    d ~ 100-2000) this takes < 100 ms.

    The effective rank of the full data characterises the dataset's global
    intrinsic dimensionality:

    - Low-rank data (erank ≈ 3): floor = 3, no wasted noise dimensions
    - Moderate-rank data (erank ≈ 12): floor = 12
    - High-rank data (erank ≈ 50): clamped to hard_cap (default 20)
    """
    from .spectral_dimension import effective_rank as _effective_rank

    X = leaf_data.values.astype(np.float64)
    n, d = X.shape

    if n < 2 or d < 2:
        return hard_floor

    # Identify active features (non-constant columns)
    col_var = np.var(X, axis=0)
    active_mask = col_var > 0
    d_active = int(np.sum(active_mask))

    if d_active < 2:
        return hard_floor

    X_active = X[:, active_mask]

    # Dual-form optimisation: when n < d, compute n×n Gram matrix
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
    min_k = int(np.ceil(erank))
    min_k = max(min_k, hard_floor)
    min_k = min(min_k, hard_cap)

    logger.info(
        "Adaptive PROJECTION_MIN_K: effective_rank=%.1f → min_k=%d " "(n=%d, d=%d, d_active=%d)",
        erank,
        min_k,
        n,
        d,
        d_active,
    )

    return min_k


def resolve_min_k(
    min_k_config: int | str,
    leaf_data: pd.DataFrame | None = None,
) -> int:
    """Resolve ``PROJECTION_MIN_K`` to a concrete integer.

    When ``min_k_config`` is ``"auto"``, the effective rank of the full
    dataset is computed and cached in the module-level ``_RESOLVED_MIN_K``
    so that downstream calls to :func:`compute_projection_dimension` (which
    don't receive ``leaf_data``) can read the resolved value automatically.

    Parameters
    ----------
    min_k_config
        Either an integer (used as-is) or ``"auto"`` (estimated from data).
    leaf_data
        Required when *min_k_config* is ``"auto"``.

    Returns
    -------
    int
        Resolved minimum projection dimension.

    Raises
    ------
    ValueError
        If ``"auto"`` is requested but *leaf_data* is None.
    """
    global _RESOLVED_MIN_K

    if isinstance(min_k_config, int):
        _RESOLVED_MIN_K = min_k_config
        return min_k_config

    if min_k_config == "auto":
        if leaf_data is None:
            # No data available — fall back to a reasonable hard floor.
            # This happens when TreeDecomposition is constructed without
            # leaf_data (e.g. pre-annotated path, or tests).
            logger.info(
                "PROJECTION_MIN_K='auto' but leaf_data is None; " "falling back to hard floor of 2."
            )
            _RESOLVED_MIN_K = 2
            return 2
        resolved = estimate_min_projection_dimension(leaf_data)
        _RESOLVED_MIN_K = resolved
        return resolved

    raise ValueError(f"PROJECTION_MIN_K must be an int or 'auto', got {min_k_config!r}")


def compute_projection_dimension(
    n_samples: int,
    n_features: int,
    eps: float = config.PROJECTION_EPS,
    min_k: int | str | None = None,
) -> int:
    """Compute target projection dimension, capped by data rank.

    The projection dimension is the minimum of three constraints:

    1. **JL lemma**: ``k_JL = O(log(n) / eps²)`` — preserves pairwise
       distances among ``n_samples`` points within ``(1 ± eps)``.
    2. **Information cap** (only when ``d/n ≥ 4``): ``n_samples`` — the
       data matrix has rank ≤ ``n``, so z-components beyond ``n``
       carry only estimation noise.  Only activates when ``d`` is
       at least 4× larger than ``n`` to avoid penalising moderate-
       dimensional cases.
    3. **Feature count**: ``n_features`` — cannot exceed input dimension.

    The final ``k = max(min(k_JL, [n if d≥4n], d), min_k)``.

    Parameters
    ----------
    n_samples : int
        Effective sample size (leaf count for edge tests, harmonic mean
        for sibling tests).  Also determines the rank cap.
    n_features : int
        Original number of features.
    eps : float
        JL distortion tolerance.  Distances are preserved within
        ``(1 ± eps)``.
    min_k : int or None
        Minimum projected dimension (floor).  When ``None``, reads from
        ``config.PROJECTION_MIN_K``.  If the config value is ``"auto"``
        and no resolved int has been threaded through, falls back to 2.

    Returns
    -------
    int
        Target dimension ``k``.

    Notes
    -----
    The information cap is the key change relative to pure-JL dimension
    selection.  For a subtree with ``n`` leaves and ``d`` features
    (``n ≪ d``), each proportion is estimated from ``n`` samples.
    The JL formula ``k ≈ 8 ln(n) / eps²`` ignores this and can return
    ``k ≫ n``, adding ``k - n`` pure-noise χ² components that
    absorb degrees of freedom without contributing signal.

    References
    ----------
    Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
        mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
    """

    # --- Resolve min_k ---
    # When None, check the module-level resolved cache first (set by
    # resolve_min_k() at pipeline entry), then fall back to config.
    # If config is "auto" but hasn't been resolved, use hard floor of 2.
    if min_k is None:
        if _RESOLVED_MIN_K is not None:
            min_k = _RESOLVED_MIN_K
        else:
            _cfg_val = config.PROJECTION_MIN_K
            min_k = _cfg_val if isinstance(_cfg_val, int) else 2
    elif isinstance(min_k, str):
        # "auto" passed directly without resolution — safe fallback
        min_k = 2

    # Use sklearn's theoretically-grounded JL formula
    # Ensure n_samples >= 1 to avoid edge cases
    n_samples = max(n_samples, 1)
    k: int = int(johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps))

    # --- Information cap (only when n ≪ d) ---
    # When n_samples is much smaller than n_features, the data matrix is
    # severely rank-deficient: rank ≤ n_samples while d ≫ n. Most of the
    # d z-components beyond n carry only estimation noise, wasting χ² df.
    # We only apply this cap when the ratio d/n ≥ 4 to avoid affecting
    # cases where n and d are comparable (e.g. n=20, d=30 stays k=30).
    # For n=10, d=2000 (ratio=200) this caps k at 10 instead of 2000.
    if n_features >= 4 * n_samples:
        k = min(k, n_samples)
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
            use_structured = k == n_features or (
                n_features >= 512 and (k / float(n_features)) >= 0.8
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
    use_structured = k == n_features or (n_features >= 512 and (k / float(n_features)) >= 0.8)
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
    return _generate_orthonormal_projection(n_features, k, random_state, use_cache=use_cache)


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
    "estimate_min_projection_dimension",
    "generate_projection_matrix",
    "resolve_min_k",
]
