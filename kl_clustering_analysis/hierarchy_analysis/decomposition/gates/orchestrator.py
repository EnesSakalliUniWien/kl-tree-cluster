"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ...statistics.child_parent_divergence import annotate_child_parent_divergence
from ...statistics.projection.chi2_pvalue import WhiteningMode
from ...statistics.sibling_divergence import annotate_sibling_divergence
from ...statistics.sibling_divergence.sibling_config import (
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)
from ..core.contracts import GateAnnotationBundle
from .column_contracts import (
    sibling_gate_columns,
    validate_legacy_edge_columns,
    validate_legacy_sibling_columns,
)


@dataclass(frozen=True)
class _SiblingGateInputs:
    spectral_dims: dict[str, int] | None
    pca_projections: dict[str, np.ndarray] | None
    pca_eigenvalues: dict[str, np.ndarray] | None


def _build_column_names_metadata(
    *,
    edge_columns: tuple[str, ...],
    sibling_columns: tuple[str, ...],
) -> dict[str, list[str]]:
    """Normalize column collections into the legacy metadata shape."""
    return {
        "edge": list(edge_columns),
        "sibling": list(sibling_columns),
    }


def _build_edge_metadata(
    *,
    alpha_local: float,
    edge_columns: tuple[str, ...],
    sibling_columns: tuple[str, ...],
) -> dict[str, object]:
    """Build metadata for Gate 2 output.

    Tree-BH is the only supported FDR method, so not stored in metadata.
    """
    return {
        "gate": "edge",
        "alpha": float(alpha_local),
        "column_names": _build_column_names_metadata(
            edge_columns=edge_columns,
            sibling_columns=sibling_columns,
        ),
    }


def _build_sibling_metadata(
    *,
    sibling_alpha: float,
    sibling_method: str,
    sibling_inputs: _SiblingGateInputs,
    edge_columns: tuple[str, ...],
    sibling_columns: tuple[str, ...],
) -> dict[str, object]:
    """Build legacy metadata for Gate 3 output."""
    return {
        "gate": "sibling",
        "alpha": float(sibling_alpha),
        "sibling_method": sibling_method,
        "uses_spectral_dims": sibling_inputs.spectral_dims is not None,
        "uses_pca_projections": sibling_inputs.pca_projections is not None,
        "uses_pca_eigenvalues": sibling_inputs.pca_eigenvalues is not None,
        "column_names": _build_column_names_metadata(
            edge_columns=edge_columns,
            sibling_columns=sibling_columns,
        ),
    }


def _resolve_sibling_gate_inputs(
    tree,
    edge_annotated_df: pd.DataFrame,
    *,
    sibling_spectral_dims: dict[str, int] | None,
    sibling_pca_projections: dict[str, np.ndarray] | None,
    sibling_pca_eigenvalues: dict[str, np.ndarray] | None,
) -> _SiblingGateInputs:
    """Resolve optional Gate 3 inputs from explicit args or Gate 2 attrs."""
    resolved_spectral_dims = sibling_spectral_dims
    if resolved_spectral_dims is None:
        resolved_spectral_dims = derive_sibling_spectral_dims(tree, edge_annotated_df)

    resolved_pca_projections = sibling_pca_projections
    resolved_pca_eigenvalues = sibling_pca_eigenvalues
    if resolved_pca_projections is None:
        (
            resolved_pca_projections,
            resolved_pca_eigenvalues,
        ) = derive_sibling_pca_projections(edge_annotated_df, resolved_spectral_dims)

    return _SiblingGateInputs(
        spectral_dims=resolved_spectral_dims,
        pca_projections=resolved_pca_projections,
        pca_eigenvalues=resolved_pca_eigenvalues,
    )


def run_gate_annotation_pipeline(
    tree,
    annotations_df: pd.DataFrame,
    *,
    alpha_local: float = config.EDGE_ALPHA,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    sibling_spectral_dims: dict[str, int] | None = None,
    sibling_pca_projections: dict[str, np.ndarray] | None = None,
    sibling_pca_eigenvalues: dict[str, np.ndarray] | None = None,
    sibling_whitening: WhiteningMode = config.SIBLING_WHITENING,
) -> GateAnnotationBundle:
    """Run Gate 2 (edge) and Gate 3 (sibling) annotation pipeline.

    Gate 2 uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR correction.
    This is the only supported multiple-testing method.
    """
    # Run Gate 2: child-parent edge tests
    edge_annotated_df = annotate_child_parent_divergence(
        tree,
        annotations_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
    )
    edge_columns = validate_legacy_edge_columns(edge_annotated_df)
    edge_sibling_columns = sibling_gate_columns(edge_annotated_df)
    edge_metadata = _build_edge_metadata(
        alpha_local=alpha_local,
        edge_columns=edge_columns,
        sibling_columns=edge_sibling_columns,
    )

    sibling_inputs = _resolve_sibling_gate_inputs(
        tree,
        edge_annotated_df,
        sibling_spectral_dims=sibling_spectral_dims,
        sibling_pca_projections=sibling_pca_projections,
        sibling_pca_eigenvalues=sibling_pca_eigenvalues,
    )

    # Run Gate 3: sibling divergence tests
    if sibling_method != config.SIBLING_TEST_METHOD:
        raise ValueError(
            f"unsupported sibling_method={sibling_method!r}; only {config.SIBLING_TEST_METHOD!r} is supported"
        )
    annotated_df = annotate_sibling_divergence(
        tree,
        edge_annotated_df,
        significance_level_alpha=sibling_alpha,
        spectral_dims=sibling_inputs.spectral_dims,
        pca_projections=sibling_inputs.pca_projections,
        pca_eigenvalues=sibling_inputs.pca_eigenvalues,
        whitening=sibling_whitening,
    )
    output_edge_columns = validate_legacy_edge_columns(
        annotated_df,
        error_context="Sibling gate input/output edge columns differ from legacy contract",
    )
    sibling_columns = validate_legacy_sibling_columns(annotated_df)
    sibling_metadata = _build_sibling_metadata(
        sibling_alpha=sibling_alpha,
        sibling_method=sibling_method,
        sibling_inputs=sibling_inputs,
        edge_columns=output_edge_columns,
        sibling_columns=sibling_columns,
    )

    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=output_edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "pipeline": "gate_annotation",
            "edge": edge_metadata,
            "sibling": sibling_metadata,
        },
    )


__all__ = ["run_gate_annotation_pipeline"]
