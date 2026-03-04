"""Edge-gate annotation wrapper (Gate 2)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import (
    LEGACY_EDGE_COLUMNS,
    GateAnnotationBundle,
)
from ..core.errors import DecompositionValidationError
from ...statistics.kl_tests.edge_significance import annotate_child_parent_divergence

_EDGE_PREFIX = "Child_Parent_"
_SIBLING_PREFIX = "Sibling_"


def _prefix_columns(df: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    return tuple(col for col in df.columns if col.startswith(prefix))


def _validate_legacy_edge_columns(df: pd.DataFrame) -> tuple[str, ...]:
    produced = _prefix_columns(df, _EDGE_PREFIX)
    missing = [col for col in LEGACY_EDGE_COLUMNS if col not in produced]
    extras = [col for col in produced if col not in LEGACY_EDGE_COLUMNS]
    if missing or extras:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing={missing}")
        if extras:
            detail_parts.append(f"unexpected={extras}")
        detail = "; ".join(detail_parts)
        raise DecompositionValidationError(f"Edge gate columns differ from legacy contract: {detail}.")
    return produced


def annotate_edge_gate(
    tree,
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.ALPHA_LOCAL,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
    fdr_method: str = "tree_bh",
) -> GateAnnotationBundle:
    """Run Gate 2 (edge divergence) and return typed legacy-compatible output."""
    annotated_df = annotate_child_parent_divergence(
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        fdr_method=fdr_method,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
    )
    edge_columns = _validate_legacy_edge_columns(annotated_df)
    sibling_columns = _prefix_columns(annotated_df, _SIBLING_PREFIX)
    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "gate": "edge",
            "alpha": float(significance_level_alpha),
            "fdr_method": str(fdr_method),
            "spectral_method": spectral_method,
            "min_k": min_k,
            "column_names": {
                "edge": list(edge_columns),
                "sibling": list(sibling_columns),
            },
        },
    )


__all__ = ["annotate_edge_gate"]
