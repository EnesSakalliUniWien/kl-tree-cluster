"""Projection-matrix generation for random projection."""

from __future__ import annotations

import numpy as np


def generate_projection_matrix(
    n_features: int,
    n_components: int,
    random_state: int | None = None,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate an orthonormal projection matrix."""
    del use_cache

    if n_components <= 0 or n_features <= 0:
        return np.zeros((max(n_components, 0), max(n_features, 0)), dtype=np.float64)

    rng = np.random.RandomState(random_state)
    use_structured = n_components == n_features or (
        n_features >= 512 and (n_components / float(n_features)) >= 0.8
    )

    if use_structured:
        return _generate_structured_orthonormal_rows(
            int(n_features),
            int(n_components),
            rng,
        )

    gaussian_matrix = rng.standard_normal((int(n_components), int(n_features)))
    orthogonal_basis, _ = np.linalg.qr(gaussian_matrix.T, mode="reduced")
    return orthogonal_basis.T


def _generate_structured_orthonormal_rows(
    n_features: int,
    n_components: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate sparse signed-coordinate orthonormal rows."""
    selected_columns = rng.permutation(n_features)[:n_components]
    random_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_components)
    projection_matrix = np.zeros((n_components, n_features), dtype=np.float64)
    projection_matrix[np.arange(n_components), selected_columns] = random_signs
    return projection_matrix
