"""Canonical projection-dimension estimators for hierarchical decomposition.

Provides estimators used by the local spectral context to determine how many
dimensions to project onto when computing projected Wald statistics for each
internal tree node:

- ``marchenko_pastur_signal_count``: core per-node estimator; counts
  eigenvalues above the Marchenko-Pastur noise threshold.
- ``estimate_marchenko_pastur_dimension``: canonical MP dimension contract;
  exposes raw signal count, test dimension, effective independent rows, and the
  row count actually used by the MP threshold.

Both operate on eigenvalues produced by the eigendecomposition backend in
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
            is the descendant leaf-row count in the production spectral
            contract. Must be positive.
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
        raise ValueError("Marchenko-Pastur signal count requires positive row and feature counts.")

    # Convert to float64 for numerical stability in eigenvalue comparison
    eigenvalues_f64 = np.asarray(eigenvalues, dtype=np.float64)

    # Aspect ratio: active features vs descendant rows for this node.
    # Grows as the node gets smaller (fewer descendant rows), raising λ_max.
    node_aspect_ratio = float(n_active_features) / float(mp_threshold_rows)

    # Marchenko-Pastur upper bound for unit-scale null-whitened covariance.
    # Eigenvalues above this threshold indicate true signal.
    mp_upper_bound = (1.0 + np.sqrt(node_aspect_ratio)) ** 2

    return int(np.sum(eigenvalues_f64 > mp_upper_bound))


def _positive_eigenvalue_count(eigenvalues: np.ndarray) -> int:
    """Count numerically positive eigenvalues that expose PCA directions."""
    eigenvalues_f64 = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues_f64.size == 0:
        return 0
    tolerance = (
        np.finfo(np.float64).eps
        * max(eigenvalues_f64.shape[0], 1)
        * max(float(np.max(eigenvalues_f64)), 1.0)
    )
    return int(np.count_nonzero(eigenvalues_f64 > tolerance))


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

    ``n_samples`` is the number of rows eigendecomposed.
    ``effective_independent_rows`` records the descendant leaf count.
    ``mp_threshold_rows`` records the row count actually used in the MP aspect
    ratio. In the production leaf-only spectral contract these counts are the
    same; diagnostics may vary them explicitly.
    """
    independent_row_count = (
        int(n_samples) if effective_independent_rows is None else int(effective_independent_rows)
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
    available_projection_directions = min(
        int(n_features),
        _positive_eigenvalue_count(np.asarray(eigenvalues, dtype=np.float64)),
    )
    test_projection_dimension = min(
        test_projection_dimension,
        int(available_projection_directions),
    )
    return MarchenkoPasturDimensionEstimate(
        raw_mp_signal_count=int(raw_mp_signal_count),
        test_projection_dimension=int(test_projection_dimension),
        effective_independent_rows=int(independent_row_count),
        mp_threshold_rows=int(row_count_for_threshold),
    )


__all__ = [
    "MarchenkoPasturDimensionEstimate",
    "estimate_marchenko_pastur_dimension",
    "marchenko_pastur_signal_count",
]
