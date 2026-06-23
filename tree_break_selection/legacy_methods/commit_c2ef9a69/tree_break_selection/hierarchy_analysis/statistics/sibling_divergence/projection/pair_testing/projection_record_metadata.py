"""Projection-source metadata written onto sibling test records."""

from __future__ import annotations

import numpy as np


def determine_projection_metadata_for_sibling_test(
    *,
    projection_diagnostics: dict[str, object],
    projection_dimension_from_edge_comparisons: int | None,
    parent_principal_component_projection: np.ndarray | None,
) -> tuple[str, float, bool]:
    """Resolve projection metadata for one sibling test record."""
    projection_dimension_source = str(projection_diagnostics.get("source", ""))
    resolved_projection_dimension = float(
        projection_diagnostics.get("resolved_projection_dimension", np.nan)
    )
    if not projection_dimension_source and projection_dimension_from_edge_comparisons is not None:
        projection_dimension_source = "derived_from_edge_comparisons"
        resolved_projection_dimension = float(projection_dimension_from_edge_comparisons)
    elif not projection_dimension_source and projection_dimension_from_edge_comparisons is None:
        projection_dimension_source = "johnson_lindenstrauss_fallback"

    used_parent_principal_component_basis = bool(
        projection_dimension_source == "derived_from_edge_comparisons"
        and parent_principal_component_projection is not None
    )
    return (
        projection_dimension_source,
        resolved_projection_dimension,
        used_parent_principal_component_basis,
    )


def resolve_sibling_test_calibration_scale(
    *,
    projection_dimension_from_edge_comparisons: int | None,
    degrees_of_freedom: float,
) -> float:
    """Resolve the calibration scale used for local sibling deflation."""
    if (
        projection_dimension_from_edge_comparisons is not None
        and projection_dimension_from_edge_comparisons > 0
    ):
        return float(projection_dimension_from_edge_comparisons)
    if np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0:
        return float(degrees_of_freedom)
    return 0.0


__all__ = [
    "determine_projection_metadata_for_sibling_test",
    "resolve_sibling_test_calibration_scale",
]
