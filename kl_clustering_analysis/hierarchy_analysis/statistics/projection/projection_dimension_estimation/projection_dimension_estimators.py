"""Canonical projection-dimension estimators for hierarchical decomposition.

Provides estimators used by the local spectral context to determine how many
dimensions to project onto when computing projected Wald statistics for each
internal tree node:

- ``effective_rank``: continuous measure of spectral spread; used to set the
  global minimum projection dimension from the full dataset's covariance.
- ``marchenko_pastur_signal_count``: core per-node estimator; counts
  eigenvalues above the Marchenko-Pastur noise threshold.
- ``estimate_marchenko_pastur_dimension``: canonical MP dimension contract;
  exposes raw signal count, test dimension, effective independent rows, and the
  row count actually used by the MP threshold.

All three operate on eigenvalues produced by the eigendecomposition backend in
``spectral/marchenko_pastur.py`` and are called once per internal node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarchenkoPasturDimensionEstimate:
    """Explicit split between the MP signal count and the Wald test dimension."""

    raw_mp_signal_count: int
    test_projection_dimension: int
    effective_independent_rows: int
    mp_threshold_rows: int


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
    mp_threshold_rows: int,
    n_active_features: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound for a tree node.

    Applies Marchenko-Pastur (MP) random matrix theory to separate signal
    components from noise in a node's eigenvalue spectrum. Called once per
    internal tree node while preparing the shared spectral context.

    The MP upper bound λ_max is the highest eigenvalue expected under pure
    noise. Eigenvalues above it indicate candidate signal dimensions. Because
    the aspect ratio c = d/n grows as you descend the tree (fewer effective
    independent rows), λ_max inflates near leaf nodes and makes signal
    detection harder in small subtrees.

    The upper bound is computed as::

        λ_max = (1 + √c)²

    because the input spectrum comes from a covariance matrix in null-whitened
    tangent coordinates, whose working null variance scale is 1. Here
    c = n_active_features / n_descendant_rows.

    Args:
        eigenvalues (np.ndarray): Eigenvalues of the node-local covariance
            matrix in null-whitened tangent coordinates. Negative values
            (numerical noise from eigensolvers) are ignored.
        mp_threshold_rows (int): Row count used in the MP aspect ratio. This
            can be the independent leaf-row count for diagnostics or the
            augmented spectral row count for the current production contract.
            Must be positive.
        n_active_features (int): Number of non-constant feature columns after
            zero-variance columns are excluded before eigen-decomposition.
            Must be positive.

    Returns:
        int: Raw count of eigenvalues exceeding λ_max. This may be zero under
            a pure-noise local spectrum; downstream test-dimension floors are
            applied by ``estimate_marchenko_pastur_dimension``.

    References:
        Marchenko & Pastur (1967). Distribution of eigenvalues for some sets
        of random matrices. Mathematical USSR-Sbornik, 1(4), 457-483.
    """
    if mp_threshold_rows <= 0 or n_active_features <= 0:
        raise ValueError(
            "Marchenko-Pastur signal count requires positive row and feature counts."
        )

    # Convert to float64 for numerical stability in eigenvalue comparison
    eigenvalues_f64 = np.asarray(eigenvalues, dtype=np.float64)

    # Aspect ratio: active features vs descendant rows for this node.
    # Grows as the node gets smaller (fewer descendant rows), raising λ_max.
    node_aspect_ratio = float(n_active_features) / float(mp_threshold_rows)

    # Marchenko-Pastur upper bound for unit-scale null-whitened covariance.
    # Eigenvalues above this threshold indicate true signal.
    mp_upper_bound = (1.0 + np.sqrt(node_aspect_ratio)) ** 2

    return int(np.sum(eigenvalues_f64 > mp_upper_bound))


def estimate_marchenko_pastur_dimension(
    eigenvalues: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    effective_independent_rows: int | None = None,
    mp_threshold_rows: int | None = None,
    minimum_projection_dimension: int = 1,
) -> MarchenkoPasturDimensionEstimate:
    """Estimate raw MP signal count and final projected-Wald test dimension.

    ``n_samples`` is the number of rows eigendecomposed. When internal
    distribution rows are stacked for PCA stabilization,
    ``effective_independent_rows`` should record the descendant leaf count.
    ``mp_threshold_rows`` records the row count actually used in the MP aspect
    ratio; diagnostics may set it to the leaf count, while the current
    production contract keeps the augmented-row threshold explicit.
    """
    independent_row_count = (
        int(n_samples)
        if effective_independent_rows is None
        else int(effective_independent_rows)
    )
    row_count_for_threshold = (
        independent_row_count if mp_threshold_rows is None else int(mp_threshold_rows)
    )
    if int(n_samples) <= 0 or int(n_features) <= 0:
        raise ValueError(
            "Marchenko-Pastur dimension estimation requires positive sample and "
            f"feature counts. Got n_samples={n_samples!r}, n_features={n_features!r}."
        )
    if independent_row_count <= 0 or row_count_for_threshold <= 0:
        raise ValueError(
            "Marchenko-Pastur dimension estimation requires positive effective "
            "and threshold row counts. "
            f"Got effective_independent_rows={independent_row_count!r}, "
            f"mp_threshold_rows={row_count_for_threshold!r}."
        )
    minimum_dimension = int(minimum_projection_dimension)
    if minimum_dimension < 0:
        raise ValueError(
            "minimum_projection_dimension must be non-negative. "
            f"Got {minimum_projection_dimension!r}."
        )
    raw_mp_signal_count = marchenko_pastur_signal_count(
        np.asarray(eigenvalues, dtype=np.float64),
        mp_threshold_rows=row_count_for_threshold,
        n_active_features=n_features,
    )
    test_projection_dimension = max(
        int(raw_mp_signal_count),
        minimum_dimension,
    )
    test_projection_dimension = min(test_projection_dimension, int(n_features))
    return MarchenkoPasturDimensionEstimate(
        raw_mp_signal_count=int(raw_mp_signal_count),
        test_projection_dimension=int(test_projection_dimension),
        effective_independent_rows=int(independent_row_count),
        mp_threshold_rows=int(row_count_for_threshold),
    )


__all__ = [
    "MarchenkoPasturDimensionEstimate",
    "effective_rank",
    "estimate_marchenko_pastur_dimension",
    "marchenko_pastur_signal_count",
]
