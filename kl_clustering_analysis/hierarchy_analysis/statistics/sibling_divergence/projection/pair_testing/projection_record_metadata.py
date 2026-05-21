"""Projection-source metadata written onto sibling test records."""

from __future__ import annotations

import numpy as np


def determine_projection_metadata_for_sibling_test(
    *,
    projection_diagnostics: dict[str, object],
    parent_principal_component_projection: np.ndarray | None,
) -> tuple[str, float, bool]:
    """Resolve projection metadata for one sibling test record."""
    required_keys = {"source", "resolved_projection_dimension"}
    missing_keys = sorted(required_keys.difference(projection_diagnostics))
    if missing_keys:
        raise ValueError(
            "Missing projection diagnostics fields for sibling test metadata: "
            f"{missing_keys!r}."
        )

    projection_dimension_source = str(projection_diagnostics["source"])
    resolved_projection_dimension = float(
        projection_diagnostics["resolved_projection_dimension"]
    )
    valid_sources = {
        "derived_from_edge_comparisons",
        "johnson_lindenstrauss_projection",
    }
    if projection_dimension_source not in valid_sources:
        raise ValueError(
            "Invalid projection diagnostics source for sibling test metadata: "
            f"{projection_dimension_source!r}."
        )
    if not np.isfinite(resolved_projection_dimension) or resolved_projection_dimension <= 0:
        raise ValueError(
            "Invalid resolved projection dimension for sibling test metadata: "
            f"{resolved_projection_dimension!r}."
        )

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
