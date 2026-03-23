"""Edge-gate annotation wrapper (Gate 2)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ...statistics.child_parent_divergence import annotate_child_parent_divergence
from ..core.contracts import GateAnnotationBundle
from .column_contracts import sibling_gate_columns, validate_legacy_edge_columns


def annotate_edge_gate(
    tree,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.EDGE_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    minimum_projection_dimension: int | None = None,
    fdr_method: str = config.EDGE_FDR_METHOD,
) -> GateAnnotationBundle:
    """Run Gate 2 (edge divergence) and return typed legacy-compatible output."""
    annotated_df = annotate_child_parent_divergence(
        tree,
        annotations_df,
        significance_level_alpha=significance_level_alpha,
        fdr_method=fdr_method,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        minimum_projection_dimension=minimum_projection_dimension,
    )
    edge_columns = validate_legacy_edge_columns(annotated_df)
    sibling_columns = sibling_gate_columns(annotated_df)
    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "gate": "edge",
            "alpha": float(significance_level_alpha),
            "fdr_method": str(fdr_method),
            "spectral_method": spectral_method,
            "minimum_projection_dimension": minimum_projection_dimension,
            "column_names": {
                "edge": list(edge_columns),
                "sibling": list(sibling_columns),
            },
        },
    )


__all__ = ["annotate_edge_gate"]
