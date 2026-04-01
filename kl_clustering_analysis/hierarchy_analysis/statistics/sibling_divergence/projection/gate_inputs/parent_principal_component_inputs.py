"""Parent principal-component inputs reused by Gate 3 sibling tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def collect_parent_principal_component_inputs_for_sibling_tests(
    annotated_df: pd.DataFrame,
    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] | None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Collect parent principal-component inputs for sibling tests.

    Returns the subset of Gate 2 principal-component projections and eigenvalues
    corresponding to parents that have valid sibling projection dimensions.
    """
    if sibling_projection_dimensions_from_child_edge_comparisons is None:
        return None, None

    principal_component_projections = annotated_df.attrs.get("_pca_projections")
    if not principal_component_projections:
        logger.debug("Gate 3: no _pca_projections found on Gate 2 annotations")
        return None, None

    principal_component_eigenvalues = annotated_df.attrs.get("_pca_eigenvalues")

    parent_principal_component_projections: dict[str, np.ndarray] = {}
    parent_principal_component_eigenvalues: dict[str, np.ndarray] = {}

    for parent in sibling_projection_dimensions_from_child_edge_comparisons:
        projection = principal_component_projections.get(parent)
        if projection is not None:
            parent_principal_component_projections[parent] = projection

        eigenvalues = (
            principal_component_eigenvalues.get(parent)
            if principal_component_eigenvalues
            else None
        )
        if eigenvalues is not None:
            parent_principal_component_eigenvalues[parent] = eigenvalues

    missing_parent_principal_component_inputs = (
        sibling_projection_dimensions_from_child_edge_comparisons.keys()
        - principal_component_projections.keys()
    )
    if missing_parent_principal_component_inputs:
        logger.debug(
            "Gate 3: %d parents have child-edge-derived sibling projection dimensions but no parent principal-component projections: %s",
            len(missing_parent_principal_component_inputs),
            sorted(missing_parent_principal_component_inputs)[:10],
        )

    return (
        parent_principal_component_projections if parent_principal_component_projections else None,
        parent_principal_component_eigenvalues if parent_principal_component_eigenvalues else None,
    )


__all__ = ["collect_parent_principal_component_inputs_for_sibling_tests"]
