"""Sibling-gate annotation wrapper (Gate 3)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..methods.sibling_calibration import apply_sibling_calibration


def annotate_sibling_gate(
    tree,
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Annotate sibling divergence for decomposition gate logic."""
    return apply_sibling_calibration(
        sibling_method,
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )


__all__ = ["annotate_sibling_gate"]

