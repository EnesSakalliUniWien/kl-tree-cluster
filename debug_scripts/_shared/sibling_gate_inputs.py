"""Shared debug helpers for deriving current sibling-gate projection inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

from .sibling_child_pca import derive_sibling_child_pca_projections

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
        SpectralContext,
    )


@dataclass(frozen=True)
class SiblingGateDebugInputs:
    """Projection inputs reconstructed from the typed Gate 2 spectral context."""

    spectral_context: SpectralContext
    projection_dimensions: dict[str, int] | None
    parent_principal_component_projections: dict[str, np.ndarray] | None
    parent_principal_component_eigenvalues: dict[str, np.ndarray] | None
    child_principal_component_projections: dict[str, list[np.ndarray]] | None


def derive_sibling_gate_debug_inputs(
    tree,
    annotations_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
    *,
    alpha_local: float | None = None,
) -> SiblingGateDebugInputs:
    """Derive sibling-gate inputs from the typed spectral context."""
    _, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df,
        significance_level_alpha=config.SIBLING_ALPHA if alpha_local is None else alpha_local,
        leaf_data=leaf_data,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    (
        parent_principal_component_projections,
        parent_principal_component_eigenvalues,
    ) = collect_parent_principal_component_inputs_for_sibling_tests(
        projection_dimensions,
        spectral_context=spectral_context,
    )
    child_principal_component_projections = derive_sibling_child_pca_projections(
        tree,
        projection_dimensions,
        spectral_context=spectral_context,
    )
    return SiblingGateDebugInputs(
        spectral_context=spectral_context,
        projection_dimensions=projection_dimensions,
        parent_principal_component_projections=parent_principal_component_projections,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
        child_principal_component_projections=child_principal_component_projections,
    )
