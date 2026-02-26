"""Correlation-matrix eigendecomposition and PCA projection construction.

Handles both primal-form (d×d correlation matrix) and dual-form (n×n Gram
matrix) eigendecomposition, dimension estimation from eigenvalues, and
embedding of PCA projections into the full feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .estimators import effective_rank, marchenko_pastur_signal_count


@dataclass
class EigenResult:
    """Container for eigendecomposition outputs."""

    eigenvalues: np.ndarray
    active_mask: np.ndarray
    d_active: int
    use_dual: bool
    eigenvectors_active: Optional[np.ndarray] = None
    gram_vecs: Optional[np.ndarray] = None
    X_std: Optional[np.ndarray] = None


def eigendecompose_correlation(
    data_sub: np.ndarray,
    need_eigh: bool,
) -> Optional[EigenResult]:
    """Eigendecompose the correlation matrix (primal or dual form).

    Uses dual-form (n×n Gram matrix) when n < d_active for O(n²d + n³)
    instead of O(d³).

    Parameters
    ----------
    data_sub
        Data matrix (n × d).
    need_eigh
        If True, also compute eigenvectors (not just eigenvalues).

    Returns
    -------
    EigenResult or None
        *None* when fewer than 2 active features exist.
    """
    col_var = np.var(data_sub, axis=0)
    active_mask = col_var > 0
    d_active = int(np.sum(active_mask))

    if d_active < 2:
        return None

    data_active = data_sub[:, active_mask]
    n_desc = data_sub.shape[0]
    use_dual = n_desc < d_active

    if use_dual:
        col_means = data_active.mean(axis=0)
        col_stds = data_active.std(axis=0, ddof=0)
        col_stds[col_stds == 0] = 1.0
        X_std = (data_active - col_means) / col_stds
        gram = X_std @ X_std.T / d_active

        if need_eigh:
            eigenvalues, gram_vecs = np.linalg.eigh(gram)
            eigenvalues = eigenvalues[::-1]
            gram_vecs = gram_vecs[:, ::-1]
        else:
            eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
            gram_vecs = None

        eigenvalues = np.maximum(eigenvalues, 0.0)
        return EigenResult(
            eigenvalues=eigenvalues,
            active_mask=active_mask,
            d_active=d_active,
            use_dual=True,
            gram_vecs=gram_vecs,
            X_std=X_std,
        )
    else:
        corr = np.corrcoef(data_active.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        if need_eigh:
            eigenvalues, eigenvectors_active = np.linalg.eigh(corr)
            eigenvalues = eigenvalues[::-1]
            eigenvectors_active = eigenvectors_active[:, ::-1]
        else:
            eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
            eigenvectors_active = None

        eigenvalues = np.maximum(eigenvalues, 0.0)
        return EigenResult(
            eigenvalues=eigenvalues,
            active_mask=active_mask,
            d_active=d_active,
            use_dual=False,
            eigenvectors_active=eigenvectors_active,
        )


def estimate_spectral_k(
    eigenvalues: np.ndarray,
    method: str,
    n_desc: int,
    d_active: int,
    min_k: int,
) -> int:
    """Estimate projection dimension from eigenvalues.

    Parameters
    ----------
    eigenvalues
        Descending eigenvalues from the correlation matrix.
    method
        ``"effective_rank"`` or ``"marchenko_pastur"``.
    n_desc
        Number of observations used to build the correlation matrix.
    d_active
        Number of active (non-constant) features.
    min_k
        Floor on the returned dimension.
    """
    if method == "effective_rank":
        k = int(np.round(effective_rank(eigenvalues)))
    else:  # marchenko_pastur
        k = marchenko_pastur_signal_count(eigenvalues, n_desc, d_active)

    k = max(k, min_k)
    k = min(k, d_active)
    return k


def build_pca_projection(
    eig: EigenResult,
    k: int,
    d: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build (k_avail × d) PCA projection matrix and eigenvalue array.

    For dual-form eigendecompositions, recovers d-space eigenvectors from
    the Gram matrix eigenvectors.  Returns *(None, None)* when no usable
    eigenvectors are available.

    Returns
    -------
    projection
        (k_avail × d) projection matrix, or *None*.
    eigenvalues
        (k_avail,) eigenvalue array for whitening, or *None*.
    """
    n_desc = (
        eig.X_std.shape[0]
        if eig.X_std is not None
        else eig.eigenvectors_active.shape[0] if eig.eigenvectors_active is not None else 0
    )
    k_avail = min(k, n_desc) if eig.use_dual else k

    if k_avail <= 0:
        return None, None

    eigenvectors_active = eig.eigenvectors_active

    # Recover d-space eigenvectors for dual form
    if eig.use_dual and eig.gram_vecs is not None and eig.X_std is not None:
        top_eigs = np.maximum(eig.eigenvalues[:k_avail], 1e-12)
        eigenvectors_active = (
            eig.X_std.T @ eig.gram_vecs[:, :k_avail] / (np.sqrt(top_eigs) * np.sqrt(eig.d_active))
        )
        norms = np.linalg.norm(eigenvectors_active, axis=0)
        norms[norms == 0] = 1.0
        eigenvectors_active = eigenvectors_active / norms

    if eigenvectors_active is None:
        return None, None

    full_eigvecs = np.zeros((d, k_avail), dtype=np.float64)
    full_eigvecs[eig.active_mask, :] = eigenvectors_active[:, :k_avail]
    projection = full_eigvecs.T  # (k_avail × d)
    eigenvalues = np.maximum(eig.eigenvalues[:k_avail], 1e-12).astype(np.float64)
    return projection, eigenvalues


__all__ = [
    "EigenResult",
    "eigendecompose_correlation",
    "estimate_spectral_k",
    "build_pca_projection",
]
