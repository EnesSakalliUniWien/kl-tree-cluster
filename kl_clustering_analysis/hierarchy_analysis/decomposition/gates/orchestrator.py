"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import GateAnnotationBundle
from .edge_gate import annotate_edge_gate
from .sibling_gate import annotate_sibling_gate

_EDGE_PREFIX = "Child_Parent_"
_SIBLING_PREFIX = "Sibling_"


def _prefix_columns(df: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    return tuple(col for col in df.columns if col.startswith(prefix))


def run_gate_annotation_pipeline(
    tree,
    results_df: pd.DataFrame,
    *,
    alpha_local: float = config.ALPHA_LOCAL,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    fdr_method: str = "tree_bh",
    sibling_spectral_dims: dict[str, int] | None = None,
    sibling_pca_projections: dict[str, object] | None = None,
    sibling_pca_eigenvalues: dict[str, object] | None = None,
    edge_calibration: bool | None = None,
) -> GateAnnotationBundle:
    """Run Gate 2 -> Gate 3 adapters and return typed legacy-compatible output."""
    edge_bundle = annotate_edge_gate(
        tree,
        results_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
        fdr_method=fdr_method,
    )
    sibling_bundle = annotate_sibling_gate(
        tree,
        edge_bundle.annotated_df,
        significance_level_alpha=sibling_alpha,
        sibling_method=sibling_method,
        spectral_dims=sibling_spectral_dims,
        pca_projections=sibling_pca_projections,
        pca_eigenvalues=sibling_pca_eigenvalues,
    )
    annotated_df = sibling_bundle.annotated_df

    should_calibrate_edges = config.EDGE_CALIBRATION if edge_calibration is None else edge_calibration
    if should_calibrate_edges:
        from ...statistics.kl_tests.edge_calibration import calibrate_edges_from_sibling_neighborhood

        annotated_df = calibrate_edges_from_sibling_neighborhood(
            tree,
            annotated_df,
            alpha=alpha_local,
        )

    edge_columns = _prefix_columns(annotated_df, _EDGE_PREFIX)
    sibling_columns = _prefix_columns(annotated_df, _SIBLING_PREFIX)
    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "pipeline": "gate_annotation",
            "column_names": {
                "edge": list(edge_columns),
                "sibling": list(sibling_columns),
            },
            "edge": edge_bundle.metadata,
            "sibling": sibling_bundle.metadata,
            "edge_calibration_enabled": bool(config.EDGE_CALIBRATION),
            "edge_calibration_applied": bool(should_calibrate_edges),
        },
    )


__all__ = ["run_gate_annotation_pipeline"]
