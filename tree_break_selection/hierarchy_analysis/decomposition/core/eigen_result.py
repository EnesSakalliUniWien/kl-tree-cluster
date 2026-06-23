"""Eigendecomposition result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EigenResult:
    """Container for eigendecomposition outputs."""

    eigenvalues: np.ndarray
    is_active_feature: np.ndarray
    active_feature_count: int
    use_dual: bool
    eigenvectors_active: Optional[np.ndarray] = None
    dual_sample_eigenvectors: Optional[np.ndarray] = None
    centered_data_active: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        eigenvalues = np.asarray(self.eigenvalues)
        is_active_feature = np.asarray(self.is_active_feature)

        if eigenvalues.ndim != 1:
            raise ValueError("EigenResult.eigenvalues must be a 1D array.")

        if is_active_feature.ndim != 1:
            raise ValueError("EigenResult.is_active_feature must be a 1D mask.")

        active_feature_count = int(np.count_nonzero(is_active_feature))
        if self.active_feature_count != active_feature_count:
            raise ValueError(
                "EigenResult.active_feature_count must match the active-feature mask."
            )

        if self.active_feature_count < 1:
            raise ValueError("EigenResult requires at least one active feature.")

        if self.use_dual:
            if self.eigenvectors_active is not None:
                raise ValueError("Dual eigendecompositions cannot carry primal eigenvectors.")

            if self.dual_sample_eigenvectors is None:
                if self.centered_data_active is not None:
                    raise ValueError(
                        "Spectrum-only dual eigendecompositions cannot carry "
                        "projection recovery data."
                    )
                return

            if self.centered_data_active is None:
                raise ValueError(
                    "Dual projection recovery requires centered active data."
                )

            centered_data_active = np.asarray(self.centered_data_active)
            if centered_data_active.ndim != 2:
                raise ValueError(
                    "EigenResult.centered_data_active must be a 2D array in dual mode."
                )
            if centered_data_active.shape[1] != self.active_feature_count:
                raise ValueError(
                    "Dual centered data must have one column per active feature."
                )

            dual_sample_eigenvectors = np.asarray(self.dual_sample_eigenvectors)
            if dual_sample_eigenvectors.ndim != 2:
                raise ValueError("EigenResult.dual_sample_eigenvectors must be a 2D array.")
            if dual_sample_eigenvectors.shape[0] != centered_data_active.shape[0]:
                raise ValueError(
                    "Dual sample eigenvectors must align with the centered sample count."
                )
            if dual_sample_eigenvectors.shape[1] != eigenvalues.shape[0]:
                raise ValueError(
                    "Dual sample eigenvectors must align with the eigenvalue count."
                )
            return

        if self.dual_sample_eigenvectors is not None or self.centered_data_active is not None:
            raise ValueError(
                "Primal eigendecompositions cannot carry dual-only sample state."
            )

        if self.eigenvectors_active is not None:
            eigenvectors_active = np.asarray(self.eigenvectors_active)
            if eigenvectors_active.ndim != 2:
                raise ValueError("EigenResult.eigenvectors_active must be a 2D array.")
            if eigenvectors_active.shape[0] != self.active_feature_count:
                raise ValueError(
                    "Primal eigenvectors must have one row per active feature."
                )
            if eigenvectors_active.shape[1] != eigenvalues.shape[0]:
                raise ValueError(
                    "Primal eigenvectors must align with the eigenvalue count."
                )
