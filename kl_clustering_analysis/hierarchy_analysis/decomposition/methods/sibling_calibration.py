"""Sibling calibration controllers exposed through decomposition methods."""

from __future__ import annotations

import warnings

import pandas as pd

from kl_clustering_analysis import config

from ...statistics.sibling_divergence import (
    annotate_sibling_divergence_adjusted,
    annotate_sibling_divergence_cousin,
    annotate_sibling_divergence_tree_guided,
    annotate_sibling_divergence_weighted,
)
from ...statistics.sibling_divergence.sibling_divergence_test import annotate_sibling_divergence
from ..core.enums import SiblingCalibrationMethod
from ..core.errors import DecompositionMethodError

_APPLY_CALIBRATION_WARNING_EMITTED = False


def fit_wald(
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run baseline sibling Wald annotation."""
    _ = pca_eigenvalues
    return annotate_sibling_divergence(
        tree,
        nodes_statistics_dataframe,
        significance_level_alpha=significance_level_alpha,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )


def fit_cousin_ftest(
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run cousin F-test sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_cousin(
        tree,
        nodes_statistics_dataframe,
        significance_level_alpha=significance_level_alpha,
    )


def fit_cousin_weighted_wald(
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run weighted cousin-Wald sibling annotation."""
    return annotate_sibling_divergence_weighted(
        tree,
        nodes_statistics_dataframe,
        significance_level_alpha=significance_level_alpha,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )


def fit_cousin_adjusted_wald(
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run adjusted cousin-Wald sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_adjusted(
        tree,
        nodes_statistics_dataframe,
        significance_level_alpha=significance_level_alpha,
    )


def fit_cousin_tree_guided(
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run tree-guided cousin sibling annotation."""
    _ = (spectral_dims, pca_projections, pca_eigenvalues)
    return annotate_sibling_divergence_tree_guided(
        tree,
        nodes_statistics_dataframe,
        significance_level_alpha=significance_level_alpha,
    )


def apply_sibling_calibration(
    method: SiblingCalibrationMethod | str,
    tree,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Dispatch sibling calibration by method name/enum."""
    global _APPLY_CALIBRATION_WARNING_EMITTED
    if not _APPLY_CALIBRATION_WARNING_EMITTED:
        warnings.warn(
            "apply_sibling_calibration is deprecated; use "
            "decomposition.core.registry.resolve_sibling_calibrator + direct callable execution "
            "or decomposition.gates.sibling_gate.annotate_sibling_gate.",
            DeprecationWarning,
            stacklevel=2,
        )
        _APPLY_CALIBRATION_WARNING_EMITTED = True

    name = method.value if isinstance(method, SiblingCalibrationMethod) else str(method)

    if name == SiblingCalibrationMethod.COUSIN_WEIGHTED_WALD.value:
        return fit_cousin_weighted_wald(
            tree,
            nodes_statistics_dataframe,
            significance_level_alpha=significance_level_alpha,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
            pca_eigenvalues=pca_eigenvalues,
        )
    if name == SiblingCalibrationMethod.COUSIN_FTEST.value:
        return fit_cousin_ftest(
            tree,
            nodes_statistics_dataframe,
            significance_level_alpha=significance_level_alpha,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
            pca_eigenvalues=pca_eigenvalues,
        )
    if name == SiblingCalibrationMethod.COUSIN_ADJUSTED_WALD.value:
        return fit_cousin_adjusted_wald(
            tree,
            nodes_statistics_dataframe,
            significance_level_alpha=significance_level_alpha,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
            pca_eigenvalues=pca_eigenvalues,
        )
    if name == SiblingCalibrationMethod.COUSIN_TREE_GUIDED.value:
        return fit_cousin_tree_guided(
            tree,
            nodes_statistics_dataframe,
            significance_level_alpha=significance_level_alpha,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
            pca_eigenvalues=pca_eigenvalues,
        )
    if name == SiblingCalibrationMethod.WALD.value:
        return fit_wald(
            tree,
            nodes_statistics_dataframe,
            significance_level_alpha=significance_level_alpha,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
            pca_eigenvalues=pca_eigenvalues,
        )

    raise DecompositionMethodError(f"Unknown sibling calibration method: {name!r}.")


__all__ = [
    "fit_wald",
    "fit_cousin_ftest",
    "fit_cousin_weighted_wald",
    "fit_cousin_adjusted_wald",
    "fit_cousin_tree_guided",
    "apply_sibling_calibration",
]
