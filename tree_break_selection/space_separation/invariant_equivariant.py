"""Invariant/equivariant PCA-space separation.

The first spectral axis is treated as the common or invariant direction and
the requested following axes as orthogonal equivariant variation. This is a
PCA/spectral construction, not a formal Cartan decomposition claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InvariantEquivariantSpace:
    """Result of separating a matrix into one invariant and several equivariant axes."""

    coordinates: np.ndarray
    axes: np.ndarray
    singular_values: np.ndarray
    energy_fraction: np.ndarray

    @property
    def invariant_coordinates(self) -> np.ndarray:
        """Coordinates on the leading invariant axis."""

        return self.coordinates[:, 0]

    @property
    def equivariant_coordinates(self) -> np.ndarray:
        """Coordinates on the orthogonal equivariant axes."""

        return self.coordinates[:, 1:]


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Center and scale matrix columns while keeping constant columns finite."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Space separation requires a non-empty matrix.")

    center = np.nanmean(matrix, axis=0, keepdims=True)
    scale = np.nanstd(matrix, axis=0, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    standardized = (matrix - center) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("Matrix contains non-finite values after standardization.")
    return standardized


def decompose_invariant_equivariant_space(
    values: np.ndarray,
    *,
    equivariant_dim: int,
    standardize: bool = True,
) -> InvariantEquivariantSpace:
    """Separate one invariant axis and up to ``equivariant_dim`` following axes."""

    if equivariant_dim < 0:
        raise ValueError("equivariant_dim must be non-negative.")
    matrix = standardize_columns(values) if standardize else np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Space separation requires a non-empty matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Space-separation matrix contains non-finite values.")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, right_axes = np.linalg.svd(centered, full_matrices=False)
    n_axes = min(1 + int(equivariant_dim), right_axes.shape[0])
    axes = right_axes[:n_axes]
    coordinates = centered @ axes.T
    kept_singular_values = singular_values[:n_axes]
    energies = kept_singular_values**2
    total_energy = singular_values.dot(singular_values)
    energy_fraction = energies / total_energy if total_energy > 0.0 else np.zeros_like(energies)
    return InvariantEquivariantSpace(
        coordinates=coordinates,
        axes=axes,
        singular_values=kept_singular_values,
        energy_fraction=energy_fraction,
    )
