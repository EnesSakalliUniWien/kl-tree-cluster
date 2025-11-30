"""Permutation tests for conditional independence using CMI.

Provides statistical testing utilities for assessing conditional independence
of binary variables using permutation-based hypothesis tests.
"""

from __future__ import annotations

import numpy as np

from .cmi import _cmi_binary_vec


def _perm_cmi_binary_batch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Produce K permuted CMI values in one vectorized batch.

    Under the null hypothesis X ⟂ Y | Z (conditional independence),
    shuffling Y within each Z stratum preserves both marginals P(X|Z)
    and P(Y|Z) while breaking the X-Y dependence.

    Parameters
    ----------
    x : np.ndarray
        Binary array, shape (F,)
    y : np.ndarray
        Binary array, shape (F,)
    z : np.ndarray
        Binary conditioning array, shape (F,)
    K : int
        Number of permutations to generate
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    np.ndarray
        Shape (K,), permuted CMI values
    """
    F = y.size
    if K <= 0 or F == 0:
        return np.zeros(0, dtype=float)

    idx0 = np.flatnonzero(z == 0)
    idx1 = np.flatnonzero(z == 1)

    # Create K copies of y
    Yp = np.repeat(y[None, :], K, axis=0)

    # Permute within each Z stratum
    try:
        # Newer NumPy (>= 1.22)
        if idx0.size > 1:
            Yp[:, idx0] = rng.permuted(Yp[:, idx0], axis=1)
        if idx1.size > 1:
            Yp[:, idx1] = rng.permuted(Yp[:, idx1], axis=1)
    except AttributeError:
        # Fallback for older NumPy
        for r in range(K):
            if idx0.size > 1:
                Yp[r, idx0] = rng.permutation(Yp[r, idx0])
            if idx1.size > 1:
                Yp[r, idx1] = rng.permutation(Yp[r, idx1])

    return _cmi_binary_vec(x, Yp, z)


def _perm_test_cmi_binary(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    permutations: int = 300,
    random_state: int | None = None,
    batch_size: int = 256,
) -> tuple[float, float]:
    """
    Batched, vectorized permutation test for I(X;Y|Z).

    Tests the null hypothesis H0: X ⟂ Y | Z (conditional independence).
    Uses stratified permutation within Z values to preserve marginals.

    Parameters
    ----------
    x : np.ndarray
        Binary array
    y : np.ndarray
        Binary array
    z : np.ndarray
        Binary conditioning array
    permutations : int, default=300
        Number of permutations for the test
    random_state : int | None, default=None
        Random seed for reproducibility
    batch_size : int, default=256
        Number of permutations per batch (for memory efficiency)

    Returns
    -------
    cmi_obs : float
        Observed CMI value
    p_value : float
        Permutation p-value

    Notes
    -----
    Returns p=1.0 for edge cases (empty data, insufficient stratum sizes)
    as these provide no evidence against conditional independence.
    """
    rng = np.random.default_rng(random_state)
    x = np.ascontiguousarray(x, dtype=np.uint8).ravel()
    y = np.ascontiguousarray(y, dtype=np.uint8).ravel()
    z = np.ascontiguousarray(z, dtype=np.uint8).ravel()

    # Compute observed CMI
    cmi_obs = float(_cmi_binary_vec(x, y.reshape(1, -1), z)[0])

    # Early exits for uninformative cases → p=1.0 (no evidence)
    if permutations <= 0:
        return cmi_obs, 1.0
    idx0 = np.flatnonzero(z == 0)
    idx1 = np.flatnonzero(z == 1)
    if y.size == 0 or ((idx0.size <= 1) and (idx1.size <= 1)):
        return cmi_obs, 1.0

    # Run batched permutations
    ge = 0  # Count of permutations >= observed
    done = 0
    bs = max(1, int(batch_size))
    while done < permutations:
        k = min(bs, permutations - done)
        cmi_perm_vec = _perm_cmi_binary_batch(x, y, z, K=k, rng=rng)
        ge += int(np.sum(cmi_perm_vec >= cmi_obs - 1e-12))
        done += k

    # Compute p-value with continuity correction
    p_val = (1.0 + ge) / (permutations + 1.0)
    return cmi_obs, float(p_val)


def _cmi_perm_from_args(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int],
) -> tuple[float, float]:
    """
    Helper to unpack args for parallel mapping.

    Enables use with multiprocessing.Pool.map() or similar parallel executors.

    Parameters
    ----------
    args : tuple
        (x, y, z, permutations, random_state, batch_size)

    Returns
    -------
    tuple[float, float]
        (cmi_obs, p_value)
    """
    x, y, z, permutations, random_state, batch_size = args
    return _perm_test_cmi_binary(
        x, y, z, int(permutations), random_state, int(batch_size)
    )


__all__ = [
    "_perm_cmi_binary_batch",
    "_perm_test_cmi_binary",
    "_cmi_perm_from_args",
]
