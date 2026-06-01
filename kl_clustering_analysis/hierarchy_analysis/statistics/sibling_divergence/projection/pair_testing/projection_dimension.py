"""Projection-dimension resolution used during sibling pair testing."""

from __future__ import annotations


def resolve_sibling_projection_dimension(
    *,
    projection_dimension_from_edge_comparisons: int | None,
    left_sample_size: float,
    right_sample_size: float,
    n_features: int,
) -> tuple[int, str]:
    """Resolve the sibling projection dimension and record its provenance."""
    if projection_dimension_from_edge_comparisons is None:
        raise ValueError("Sibling projection dimension must be supplied by edge-gate context.")

    if projection_dimension_from_edge_comparisons < 0:
        raise ValueError(
            "Invalid projection_dimension_from_edge_comparisons="
            f"{projection_dimension_from_edge_comparisons}; expected a non-negative integer."
        )

    return int(projection_dimension_from_edge_comparisons), "derived_from_edge_comparisons"


__all__ = ["resolve_sibling_projection_dimension"]
