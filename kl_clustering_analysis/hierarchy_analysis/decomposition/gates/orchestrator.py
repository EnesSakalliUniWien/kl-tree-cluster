"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import GateAnnotationBundle
from ..core.errors import DecompositionMethodError
from .column_contracts import edge_gate_columns, sibling_gate_columns
from .edge_gate import annotate_edge_gate
from .sibling_gate import annotate_sibling_gate

_VALID_SPECTRAL_METHODS = {"marchenko_pastur"}
_VALID_SIBLING_METHODS = {"wald", "cousin_adjusted_wald"}


def _normalize_spectral_method(method: str | None) -> str | None:
    if method is None or method == "none":
        return None
    name = str(method).strip()
    if name not in _VALID_SPECTRAL_METHODS:
        raise DecompositionMethodError(f"Unknown spectral k estimator: {name!r}.")
    return name


def _normalize_sibling_method(method: str) -> str:
    name = str(method).strip()
    if name not in _VALID_SIBLING_METHODS:
        raise DecompositionMethodError(f"Unknown sibling calibration method: {name!r}.")
    return name


def _derive_sibling_spectral_dims(
    tree,
    annotated_df: pd.DataFrame,
) -> dict[str, int] | None:
    """Compute min-child spectral k for each binary parent from Gate 2 output.

    For each binary parent, takes the minimum of the two children's spectral
    dimensions (from Marchenko-Pastur eigendecomposition in the edge test).

    Children with spectral k = 0 (leaves, no signal eigenvalues) are excluded
    from the minimum.  Parents where no child has a positive spectral k are
    omitted — the sibling test falls back to JL-based dimension for those.
    """
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None

    sibling_dims: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        child_ks = [
            k
            for k in (
                edge_spectral_dims.get(left, 0),
                edge_spectral_dims.get(right, 0),
            )
            if k > 0
        ]
        if not child_ks:
            continue
        sibling_k = min(child_ks)
        sibling_dims[parent] = sibling_k

    return sibling_dims if sibling_dims else None


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
    sibling_pca_projections: dict[str, object] | None = None,
    sibling_pca_eigenvalues: dict[str, object] | None = None,
) -> GateAnnotationBundle:
    """Run Gate 2 -> Gate 3 adapters and return typed legacy-compatible output."""
    resolved_spectral_method = _normalize_spectral_method(spectral_method)
    resolved_sibling_method = _normalize_sibling_method(sibling_method)

    edge_bundle = annotate_edge_gate(
        tree,
        annotations_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method=resolved_spectral_method,
        minimum_projection_dimension=minimum_projection_dimension,
        fdr_method=fdr_method,
    )

    # Auto-derive sibling spectral dims from Gate 2 output when not provided.
    # Uses min-child strategy: min(k_left, k_right).
    if sibling_spectral_dims is None:
        sibling_spectral_dims = _derive_sibling_spectral_dims(
            tree, edge_bundle.annotated_df
        )

    sibling_bundle = annotate_sibling_gate(
        tree,
        edge_bundle.annotated_df,
        significance_level_alpha=sibling_alpha,
        sibling_method=resolved_sibling_method,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=sibling_spectral_dims,
        pca_projections=sibling_pca_projections,
        pca_eigenvalues=sibling_pca_eigenvalues,
    )
    annotated_df = sibling_bundle.annotated_df

    edge_columns = edge_gate_columns(annotated_df)
    sibling_columns = sibling_gate_columns(annotated_df)
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
            "resolved_methods": {
                "spectral_method": resolved_spectral_method,
                "sibling_method": resolved_sibling_method,
            },
        },
    )


__all__ = ["run_gate_annotation_pipeline"]
