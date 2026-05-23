"""Projection-source metadata written onto sibling test records."""

from __future__ import annotations

import numpy as np


def determine_projection_metadata_for_sibling_test(
    *,
    projection_dimension_from_edge_comparisons: int | None,
    parent_principal_component_projection: np.ndarray | None,
) -> tuple[str, float, bool]:
    """Resolve projection metadata for one sibling test record."""
    projection_dimension_source = "derived_from_edge_comparisons"
    if projection_dimension_from_edge_comparisons is None:
        raise ValueError(
            "Sibling test metadata requires the Gate 2 projection dimension."
        )
    resolved_projection_dimension = float(projection_dimension_from_edge_comparisons)
    if not np.isfinite(resolved_projection_dimension) or resolved_projection_dimension < 0:
        raise ValueError(
            "Invalid resolved projection dimension for sibling test metadata: "
            f"{resolved_projection_dimension!r}."
        )
    if parent_principal_component_projection is None:
        raise ValueError(
            "Sibling test metadata requires the parent principal-component projection used "
            "by the derived projection dimension."
        )

    used_parent_principal_component_basis = True
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
    del degrees_of_freedom
    if projection_dimension_from_edge_comparisons is None:
        raise ValueError("Sibling calibration scale requires Gate 2 projection dimension.")
    if projection_dimension_from_edge_comparisons < 0:
        raise ValueError(
            "Sibling calibration scale requires a non-negative Gate 2 projection dimension; "
            f"got {projection_dimension_from_edge_comparisons!r}."
        )
    return float(projection_dimension_from_edge_comparisons)


__all__ = [
    "determine_projection_metadata_for_sibling_test",
    "resolve_sibling_test_calibration_scale",
]
