"""Sibling-gate annotation wrapper (Gate 3)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
    LEGACY_SIBLING_OPTIONAL_COLUMNS,
    GateAnnotationBundle,
)
from ..core.registry import normalize_sibling_calibration_method, resolve_sibling_calibrator
from ..core.errors import DecompositionValidationError

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
        raise DecompositionValidationError(
            f"Sibling gate input/output edge columns differ from legacy contract: {detail}."
        )
    return produced


def _validate_legacy_sibling_columns(df: pd.DataFrame) -> tuple[str, ...]:
    produced = _prefix_columns(df, _SIBLING_PREFIX)
    missing = [col for col in LEGACY_SIBLING_COLUMNS if col not in produced]
    allowed = set(LEGACY_SIBLING_COLUMNS) | set(LEGACY_SIBLING_OPTIONAL_COLUMNS)
    extras = [col for col in produced if col not in allowed]
    if missing or extras:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing={missing}")
        if extras:
            detail_parts.append(f"unexpected={extras}")
        detail = "; ".join(detail_parts)
        raise DecompositionValidationError(
            f"Sibling gate columns differ from legacy contract: {detail}."
        )
    return produced


def annotate_sibling_gate(
    tree,
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    sibling_method: str = config.SIBLING_TEST_METHOD,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, object] | None = None,
    pca_eigenvalues: dict[str, object] | None = None,
) -> GateAnnotationBundle:
    """Run Gate 3 (sibling divergence) and return typed legacy-compatible output."""
    resolved_method_name = normalize_sibling_calibration_method(sibling_method)
    calibrator = resolve_sibling_calibrator(resolved_method_name)
    annotated_df = calibrator(
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )
    edge_columns = _validate_legacy_edge_columns(annotated_df)
    sibling_columns = _validate_legacy_sibling_columns(annotated_df)
    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata={
            "gate": "sibling",
            "alpha": float(significance_level_alpha),
            "sibling_method": resolved_method_name,
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
