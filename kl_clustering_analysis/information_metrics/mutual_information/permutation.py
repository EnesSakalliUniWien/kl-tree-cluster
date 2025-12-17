"""Permutation tests for conditional independence using CMI.

Provides statistical testing utilities for assessing conditional independence
of binary variables using permutation-based hypothesis tests.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

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
        Discrete array, shape (F,)
    y : np.ndarray
        Discrete array, shape (F,)
    z : np.ndarray
        Discrete conditioning array, shape (F,)
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

    # Create K copies of y
    Yp = np.repeat(y[None, :], K, axis=0)

    # Permute within each Z stratum
    unique_z = np.unique(z)

    try:
        # Newer NumPy (>= 1.22)
        for z_val in unique_z:
            idx = np.flatnonzero(z == z_val)
            if idx.size > 1:
                Yp[:, idx] = rng.permuted(Yp[:, idx], axis=1)
    except AttributeError:
        # Fallback for older NumPy
        for z_val in unique_z:
            idx = np.flatnonzero(z == z_val)
            if idx.size > 1:
                for r in range(K):
                    Yp[r, idx] = rng.permutation(Yp[r, idx])

    return _cmi_binary_vec(x, Yp, z)


def _process_batch(
    k: int,
    seed: np.random.SeedSequence,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    z_flat: np.ndarray,
    observed_cmi: float,
) -> int:
    """Helper function to process a batch of permutations."""
    # Local RNG for this batch
    local_rng = np.random.default_rng(seed)
    permuted_cmi_values = _perm_cmi_binary_batch(
        x_flat, y_flat, z_flat, K=k, rng=local_rng
    )
    return int(np.sum(permuted_cmi_values >= observed_cmi - 1e-12))


def _perm_test_cmi_binary(
    x_vector: np.ndarray,
    y_vector: np.ndarray,
    z_condition: np.ndarray,
    permutations: int = 300,
    random_state: int | None = None,
    batch_size: int = 256,
    n_jobs: int | None = None,
) -> tuple[float, float]:
    """
    Batched, vectorized permutation test for I(X;Y|Z).

    Tests the null hypothesis H0: X ⟂ Y | Z (conditional independence).
    Uses stratified permutation within Z values to preserve marginals.

    Parameters
    ----------
    x_vector : np.ndarray
        Binary array
    y_vector : np.ndarray
        Binary array
    z_condition : np.ndarray
        Binary conditioning array
    permutations : int, default=300
        Number of permutations for the test
    random_state : int | None, default=None
        Random seed for reproducibility
    batch_size : int, default=256
        Number of permutations per batch (for memory efficiency)
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1. -1 means using all processors.

    Returns
    -------
    observed_cmi : float
        Observed CMI value
    p_value : float
        Permutation p-value

    Notes
    -----
    Returns p=1.0 for edge cases (empty data, insufficient stratum sizes)
    as these provide no evidence against conditional independence.
    """
    x_flat = np.ascontiguousarray(x_vector).ravel()
    y_flat = np.ascontiguousarray(y_vector).ravel()
    z_flat = np.ascontiguousarray(z_condition).ravel()

    # Compute observed CMI
    observed_cmi = float(_cmi_binary_vec(x_flat, y_flat.reshape(1, -1), z_flat)[0])

    # Early exits for uninformative cases → p=1.0 (no evidence)
    if permutations <= 0:
        return observed_cmi, 1.0

    if y_flat.size == 0:
        return observed_cmi, 1.0

    # Check if we have enough data in strata to permute
    # We need at least one stratum with > 1 sample to do any permutation
    unique_z, counts = np.unique(z_flat, return_counts=True)
    if np.all(counts <= 1):
        return observed_cmi, 1.0

    # Prepare for parallel execution
    # Use SeedSequence for robust parallel RNG
    seed_seq = np.random.SeedSequence(random_state)

    # Calculate batches
    effective_batch_size = max(1, int(batch_size))
    n_full_batches = permutations // effective_batch_size
    remainder = permutations % effective_batch_size

    batch_sizes = [effective_batch_size] * n_full_batches
    if remainder > 0:
        batch_sizes.append(remainder)

    n_batches = len(batch_sizes)
    batch_seeds = seed_seq.spawn(n_batches)

    # Run parallel batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(k, seed, x_flat, y_flat, z_flat, observed_cmi)
        for k, seed in zip(batch_sizes, batch_seeds)
    )

    count_greater_equal = sum(results)

    # Compute p-value with continuity correction
    p_value = (1.0 + count_greater_equal) / (permutations + 1.0)
    return observed_cmi, float(p_value)


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
