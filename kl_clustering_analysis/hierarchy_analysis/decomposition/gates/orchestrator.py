"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import GateAnnotationBundle
from .edge_gate import annotate_edge_gate
from .sibling_gate import annotate_sibling_gate


def run_gate_annotation_pipeline(
    tree,
    annotations_df: pd.DataFrame,
    *,
    alpha_local: float = config.EDGE_ALPHA,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    minimum_projection_dimension: int | None = None,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    fdr_method: str = "tree_bh",
    sibling_spectral_dims: dict[str, int] | None = None,
    sibling_pca_projections: dict[str, np.ndarray] | None = None,
    sibling_pca_eigenvalues: dict[str, np.ndarray] | None = None,
    sibling_whitening: str = config.SIBLING_WHITENING,
) -> GateAnnotationBundle:
    """Run Gate 2 (edge) and Gate 3 (sibling) annotation pipeline."""
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
        derive_sibling_pca_projections,
        derive_sibling_spectral_dims,
    )

    # Run Gate 2: child-parent edge tests
    edge_bundle = annotate_edge_gate(
        tree,
        annotations_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        minimum_projection_dimension=minimum_projection_dimension,
        fdr_method=fdr_method,
    )

    # Auto-derive Gate 3 configuration from Gate 2 output
    if sibling_spectral_dims is None:
        sibling_spectral_dims = derive_sibling_spectral_dims(tree, edge_bundle.annotated_df)

    if sibling_pca_projections is None:
        sibling_pca_projections, sibling_pca_eigenvalues = derive_sibling_pca_projections(
            edge_bundle.annotated_df, sibling_spectral_dims
        )

    # Run Gate 3: sibling divergence tests
    sibling_bundle = annotate_sibling_gate(
        tree,
        edge_bundle.annotated_df,
        significance_level_alpha=sibling_alpha,
        sibling_method=sibling_method,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=sibling_spectral_dims,
        pca_projections=sibling_pca_projections,
        pca_eigenvalues=sibling_pca_eigenvalues,
        whitening=sibling_whitening,
    )

    return GateAnnotationBundle(
        annotated_df=sibling_bundle.annotated_df,
        metadata={
            "pipeline": "gate_annotation",
            "edge": edge_bundle.metadata,
            "sibling": sibling_bundle.metadata,
        },
    )


__all__ = ["run_gate_annotation_pipeline"]
