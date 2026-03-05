"""Sibling calibration controllers exposed through decomposition methods."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ...statistics.sibling_divergence import (
    annotate_sibling_divergence_adjusted,
    annotate_sibling_divergence_cousin,
    annotate_sibling_divergence_tree_guided,
    annotate_sibling_divergence_weighted,
)
from ...statistics.sibling_divergence.sibling_divergence_test import annotate_sibling_divergence


def annotate_wald(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run baseline sibling Wald annotation."""
    _ = pca_eigenvalues
    return annotate_sibling_divergence(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


def annotate_cousin_ftest(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run cousin F-test sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_cousin(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


def annotate_cousin_weighted_wald(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run weighted cousin-Wald sibling annotation."""
    return annotate_sibling_divergence_weighted(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )


def annotate_cousin_adjusted_wald(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run adjusted cousin-Wald sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_adjusted(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


def annotate_cousin_tree_guided(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run tree-guided cousin sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_tree_guided(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


__all__ = [
    "annotate_wald",
    "annotate_cousin_ftest",
    "annotate_cousin_weighted_wald",
    "annotate_cousin_adjusted_wald",
    "annotate_cousin_tree_guided",
]
