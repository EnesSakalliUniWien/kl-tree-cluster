"""Canonical projection-dimension estimators for hierarchical decomposition.

Provides three estimators used at Gate 2 (child-parent divergence test) to
determine how many dimensions to project onto when computing the Wald χ²
statistic for each internal tree node:

- ``effective_rank``: continuous measure of spectral spread; used to set the
  global minimum projection dimension from the full dataset's covariance.
- ``marchenko_pastur_signal_count``: core per-node estimator; counts
  eigenvalues above the Marchenko-Pastur noise threshold.
- ``estimate_k_marchenko_pastur``: public API wrapper that applies a
  minimum-dimension floor and an upper cap before returning k.

All three operate on eigenvalues produced by the eigen-decomposition backend
in ``spectral/marchenko_pastur.py`` and are called once per internal node.
"""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Estimate how many dimensions carry meaningful information in a dataset.

    Eigenvalues describe how much variance each dimension accounts for. If one
    dimension dominates (e.g. one eigenvalue is 1000, the rest near 0), the
    effective rank approaches 1 — the data is essentially one-dimensional. If
    all dimensions contribute equally, the effective rank equals the total
    number of dimensions.

    The measure is the exponentiated Shannon entropy of the normalized spectrum,
    which quantifies how spread out energy is across dimensions::

        effective_rank = exp( -Σᵢ pᵢ · ln(pᵢ) ),   pᵢ = λᵢ / Σⱼ λⱼ

    Args:
        eigenvalues (np.ndarray): Eigenvalues from a covariance or correlation
            matrix. Negative values (numerical noise from eigen-solvers) are
            clamped to zero before processing.

    Returns:
        float: Continuous value in [1, len(eigenvalues)].

            - ``1.0`` — all variance concentrated in a single dimension, or
              input is degenerate / all-zero.
            - ``len(eigenvalues)`` — all dimensions contribute equally.
            - Values in between reflect partial concentration.

    Examples:
        >>> effective_rank(np.array([100.0, 0.0, 0.0]))  # one dominant dim
        1.0
        >>> effective_rank(np.array([1.0, 1.0, 1.0]))    # three equal dims
        3.0
    """
    # Clamp negatives to zero: eigen-solvers can produce small negative values
    # due to floating-point error; these carry no variance and must be excluded.
    nonneg_eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)

    # Total variance across all dimensions — used to convert to a probability
    # distribution over which Shannon entropy is computed.
    eigenvalue_sum = float(np.sum(nonneg_eigenvalues))
    if eigenvalue_sum <= 0:
        return 1.0

    # Full normalized spectrum: each entry is the fraction of total variance
    # explained by that dimension, i.e. pᵢ = λᵢ / Σⱼ λⱼ.
    normalized_spectrum = nonneg_eigenvalues / eigenvalue_sum

    # Drop zero-weight entries before taking log to avoid -inf in the entropy
    # sum. These are zero eigenvalues that contribute nothing to the spread.
    nonzero_spectrum_weights = normalized_spectrum[normalized_spectrum > 0]
    if nonzero_spectrum_weights.size == 0:
        return 1.0

    # Shannon entropy H = -Σᵢ pᵢ ln(pᵢ): high when energy is evenly spread
    # across many dimensions, low when one dimension dominates.
    shannon_entropy = -float(
        np.sum(nonzero_spectrum_weights * np.log(nonzero_spectrum_weights))
    )

    # exp(H) maps entropy back to a "number of effective dimensions" scale:
    # exp(0) = 1 (single dominant dimension), exp(ln d) = d (all equal).
    return float(np.exp(shannon_entropy))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n_descendant_rows: int,
    n_active_features: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound for a tree node.

    Applies Marchenko-Pastur (MP) random matrix theory to separate signal
    components from noise in a node's eigenvalue spectrum. Called once per
    internal tree node during Gate 2 (child-parent divergence test).

    The MP upper bound λ_max is the highest eigenvalue expected under pure
    noise. Eigenvalues above it indicate genuine signal dimensions. Because
    the aspect ratio c = d/n grows as you descend the tree (fewer descendant
    rows), λ_max inflates near leaf nodes — making signal detection harder
    in small subtrees and causing those nodes to return k=1.

    The upper bound is computed as::

        λ_max = σ₀² (1 + √c)²

    where σ₀² is the median positive eigenvalue (noise-floor estimate) and
    c = n_active_features / n_descendant_rows.

    Args:
        eigenvalues (np.ndarray): Eigenvalues of the node-local correlation
            matrix (descendant leaf rows + internal distribution vectors).
            Negative values (numerical noise from eigen-solvers) are ignored.
        n_descendant_rows (int): Number of rows in the node-local descendant
            feature matrix — one row per descendant leaf sample (e.g. patient
            S42) plus any internal node distribution vectors stacked below.
            Equals the full dataset size at the root; shrinks at deeper nodes.
            Must be positive.
        n_active_features (int): Number of non-constant feature columns after
            zero-variance columns are excluded before eigen-decomposition.
            Must be positive.

    Returns:
        int: Count of eigenvalues exceeding λ_max, floored at 1 to guarantee
            a valid projection dimension. Returns 1 for degenerate inputs or
            when the entire spectrum falls within the noise floor.

    References:
        Marchenko & Pastur (1967). Distribution of eigenvalues for some sets
        of random matrices. Mathematical USSR-Sbornik, 1(4), 457-483.
    """
    if n_descendant_rows <= 0 or n_active_features <= 0:
        return 1

    # Convert to float64 for numerical stability in eigenvalue comparison
    eigenvalues_f64 = np.asarray(eigenvalues, dtype=np.float64)

    # Retain only physically meaningful (positive) eigenvalues
    positive_eigenvalues = eigenvalues_f64[eigenvalues_f64 > 0]

    # Estimate noise-floor variance from the median of positive eigenvalues.
    # Under the null (pure noise), the median is a robust proxy for the
    # Marchenko-Pastur bulk-spectrum center.
    noise_floor_variance = (
        float(np.median(positive_eigenvalues)) if positive_eigenvalues.size > 0 else 0.0
    )
    if noise_floor_variance <= 0:
        return 1

    # Aspect ratio: active features vs descendant rows for this node.
    # Grows as the node gets smaller (fewer descendant rows), raising λ_max.
    node_aspect_ratio = float(n_active_features) / float(n_descendant_rows)

    # Marchenko-Pastur upper bound: highest eigenvalue expected under null
    # Eigenvalues above this threshold indicate true signal.
    mp_upper_bound = noise_floor_variance * (1.0 + np.sqrt(node_aspect_ratio)) ** 2

    # Count eigenvalues exceeding the MP threshold
    n_signal_eigenvalues = int(np.sum(eigenvalues_f64 > mp_upper_bound))

    # Ensure at least 1 dimension is retained (fallback for degenerate cases)
    return max(n_signal_eigenvalues, 1)


def estimate_k_marchenko_pastur(
    eigenvalues: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    minimum_projection_dimension: int = 1,
) -> int:
    """Estimate the projection dimension for a tree node via Marchenko-Pastur thresholding.

    Thin wrapper around ``marchenko_pastur_signal_count`` that enforces a
    minimum-dimension floor and caps the result at ``n_features``.

    Args:
        eigenvalues (np.ndarray): Eigenvalues of the node-local correlation
            matrix, as returned by the eigen-decomposition backend.
        n_samples (int): Number of rows in the node-local descendant feature
            matrix (descendant leaf rows + internal distribution vectors).
            Forwarded to ``marchenko_pastur_signal_count`` as
            ``n_descendant_rows``.
        n_features (int): Number of active (non-constant) feature columns for
            this node. Forwarded as ``n_active_features`` and used as the
            upper cap to prevent over-projection.
        minimum_projection_dimension (int): Floor on the returned dimension.
            Defaults to 1.

    Returns:
        int: Projection dimension clamped to
            [minimum_projection_dimension, n_features].
    """
    eigenvalues_f64 = np.asarray(eigenvalues, dtype=np.float64)
    projection_dimension = int(
        marchenko_pastur_signal_count(
            eigenvalues_f64,
            n_descendant_rows=n_samples,
            n_active_features=n_features,
        )
    )
    # Floor: guarantee at least minimum_projection_dimension dimensions so
    # downstream Wald χ² has a valid degrees-of-freedom value.
    projection_dimension = max(projection_dimension, int(minimum_projection_dimension))
    # Cap: projection cannot exceed the number of available feature columns;
    # over-projection would introduce zero-variance directions.
    projection_dimension = min(projection_dimension, int(n_features))
    return projection_dimension


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "estimate_k_marchenko_pastur",
]
