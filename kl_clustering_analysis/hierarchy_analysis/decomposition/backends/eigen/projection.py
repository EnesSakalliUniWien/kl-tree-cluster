"""PCA projection recovery from eigendecomposition results."""

from __future__ import annotations

import numpy as np

from ...core.eigen_result import EigenResult


def build_pca_projection(
    eig: EigenResult,
    *,
    projection_dimension: int,
    n_features_total: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a full-width PCA projection matrix and top eigenvalues."""
    effective_dimension = _resolve_effective_dimension(
        eig,
        projection_dimension=projection_dimension,
    )

    if effective_dimension <= 0:
        return None, None

    active_eigenvectors = eig.eigenvectors_active

    if (
        eig.use_dual
        and eig.dual_sample_eigenvectors is not None
        and eig.standardized_data_active is not None
    ):
        active_eigenvectors = _recover_dual_feature_eigenvectors(
            eig,
            effective_dimension=effective_dimension,
        )

    if active_eigenvectors is None:
        return None, None

    projection_matrix = _expand_active_projection_to_full_space(
        active_eigenvectors[:, :effective_dimension],
        is_active_feature=eig.is_active_feature,
        n_features_total=n_features_total,
    )
    top_eigenvalues = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12).astype(np.float64)

    return projection_matrix, top_eigenvalues


def _resolve_effective_dimension(
    eig: EigenResult,
    *,
    projection_dimension: int,
) -> int:
    """Resolve the dimension that can be exposed downstream."""
    if not eig.use_dual:
        return int(projection_dimension)

    n_samples = (
        eig.standardized_data_active.shape[0] if eig.standardized_data_active is not None else 0
    )
    return min(int(projection_dimension), n_samples)


def _recover_dual_feature_eigenvectors(
    eig: EigenResult,
    *,
    effective_dimension: int,
) -> np.ndarray:
    """Recover feature-space eigenvectors from dual sample-space eigenvectors."""
    dual_sample_eigenvectors = eig.dual_sample_eigenvectors
    standardized_data_active = eig.standardized_data_active
    if dual_sample_eigenvectors is None or standardized_data_active is None:
        raise ValueError("Dual projection recovery requires dual sample vectors and data.")

    dual_sample_vectors = dual_sample_eigenvectors[:, :effective_dimension]
    top_eigenvalues_floored = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12)
    recovery_scale = np.sqrt(top_eigenvalues_floored) * np.sqrt(eig.active_feature_count)
    recovered_feature_vectors = standardized_data_active.T @ dual_sample_vectors / recovery_scale

    recovered_norms = np.linalg.norm(recovered_feature_vectors, axis=0)
    recovered_norms[recovered_norms == 0] = 1.0
    return recovered_feature_vectors / recovered_norms


def _expand_active_projection_to_full_space(
    active_eigenvectors: np.ndarray,
    *,
    is_active_feature: np.ndarray,
    n_features_total: int,
) -> np.ndarray:
    """Expand active-feature eigenvectors back to the full feature width."""
    full_projection = np.zeros((n_features_total, active_eigenvectors.shape[1]), dtype=np.float64)
    full_projection[is_active_feature, :] = active_eigenvectors
    return full_projection.T
