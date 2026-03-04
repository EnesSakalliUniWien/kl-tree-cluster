"""Sibling-gate annotation wrapper (Gate 3)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import GateAnnotationBundle
from ..core.registry import normalize_sibling_calibration_method, resolve_sibling_calibrator
from .column_contracts import (
    validate_legacy_edge_columns,
    validate_legacy_sibling_columns,
)


def annotate_sibling_gate(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    min_k: int | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> GateAnnotationBundle:
    """Run Gate 3 (sibling divergence) and return typed legacy-compatible output."""
    resolved_method_name = normalize_sibling_calibration_method(sibling_method)
    calibrator = resolve_sibling_calibrator(resolved_method_name)
    annotated_df = calibrator(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        min_k=min_k,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )
    edge_columns = validate_legacy_edge_columns(
        annotated_df,
        error_context="Sibling gate input/output edge columns differ from legacy contract",
    )
    sibling_columns = validate_legacy_sibling_columns(annotated_df)
    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "gate": "sibling",
            "alpha": float(significance_level_alpha),
            "sibling_method": resolved_method_name,
            "min_k": min_k,
            "uses_spectral_dims": spectral_dims is not None,
            "uses_pca_projections": pca_projections is not None,
            "uses_pca_eigenvalues": pca_eigenvalues is not None,
            "column_names": {
                "edge": list(edge_columns),
                "sibling": list(sibling_columns),
            },
        },
    )


__all__ = ["annotate_sibling_gate"]
