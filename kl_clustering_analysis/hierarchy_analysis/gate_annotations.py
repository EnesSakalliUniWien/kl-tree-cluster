"""Compute statistical gate annotations for tree decomposition.

This module is kept as a compatibility layer for legacy imports and delegates
to canonical decomposition gate adapters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from .. import config
from .decomposition.gates.edge_gate import annotate_edge_gate as _annotate_edge_gate_bundle
from .decomposition.gates.orchestrator import run_gate_annotation_pipeline
from .decomposition.gates.sibling_gate import annotate_sibling_gate as _annotate_sibling_gate_bundle

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

_WARNED_LEGACY_ENTRYPOINTS: set[str] = set()


def _warn_legacy_entrypoint_once(name: str, replacement: str) -> None:
    if name in _WARNED_LEGACY_ENTRYPOINTS:
        return
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release; "
        f"use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    _WARNED_LEGACY_ENTRYPOINTS.add(name)


def compute_gate_annotations(
    tree: "PosetTree",
    results_df: pd.DataFrame,
    *,
    alpha_local: float = config.ALPHA_LOCAL,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
) -> pd.DataFrame:
    """Deprecated compatibility entrypoint for full gate annotation flow."""
    _warn_legacy_entrypoint_once(
        "hierarchy_analysis.gate_annotations.compute_gate_annotations",
        "hierarchy_analysis.decomposition.gates.orchestrator.run_gate_annotation_pipeline",
    )
    bundle = run_gate_annotation_pipeline(
        tree,
        results_df,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
        sibling_method=config.SIBLING_TEST_METHOD,
        # Preserve legacy Gate 2 correction default.
        fdr_method="tree_bh",
        # Preserve legacy sibling behavior: JL fallback, no PCA siblings.
        sibling_spectral_dims=None,
        sibling_pca_projections=None,
        sibling_pca_eigenvalues=None,
        # Preserve legacy config-controlled edge calibration behavior.
        edge_calibration=None,
    )
    return bundle.annotated_df


def annotate_edge_gate(
    tree: "PosetTree",
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.ALPHA_LOCAL,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
    fdr_method: str = "tree_bh",
) -> pd.DataFrame:
    """Deprecated compatibility shim for edge-gate annotation imports."""
    _warn_legacy_entrypoint_once(
        "hierarchy_analysis.gate_annotations.annotate_edge_gate",
        "hierarchy_analysis.decomposition.gates.edge_gate.annotate_edge_gate",
    )
    return _annotate_edge_gate_bundle(
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
        fdr_method=fdr_method,
    ).annotated_df


def annotate_sibling_gate(
    tree: "PosetTree",
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Deprecated compatibility shim for sibling-gate annotation imports."""
    _warn_legacy_entrypoint_once(
        "hierarchy_analysis.gate_annotations.annotate_sibling_gate",
        "hierarchy_analysis.decomposition.gates.sibling_gate.annotate_sibling_gate",
    )
    return _annotate_sibling_gate_bundle(
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        sibling_method=sibling_method,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    ).annotated_df


__all__ = [
    "compute_gate_annotations",
    "annotate_edge_gate",
    "annotate_sibling_gate",
]
