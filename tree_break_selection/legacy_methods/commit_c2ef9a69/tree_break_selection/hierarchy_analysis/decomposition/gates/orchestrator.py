"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection import config

from ...statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from ...statistics.sibling_divergence.adjusted_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from ...statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from ...statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from ..backends.random_projection.dimension import (
    get_resolved_minimum_projection_dimension,
    resolve_minimum_projection_dimension,
)
from ..core.contracts import (
    GATE_ANNOTATION_METADATA_ATTR,
    Gate2Result,
    GateAnnotationBundle,
)
from .column_contracts import (
    sibling_gate_columns,
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)


@dataclass(frozen=True)
class _SiblingGateInputs:
    projection_dimensions_from_edge_comparisons: dict[str, int] | None
    parent_principal_component_projections: dict[str, np.ndarray] | None
    parent_principal_component_eigenvalues: dict[str, np.ndarray] | None


def _build_column_names_metadata(
    *,
    edge_columns: tuple[str, ...],
    sibling_columns: tuple[str, ...],
) -> dict[str, list[str]]:
    """Normalize column collections into the pipeline metadata shape."""
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
    sibling_inputs: _SiblingGateInputs,
    edge_columns: tuple[str, ...],
    sibling_columns: tuple[str, ...],
) -> dict[str, object]:
    """Build metadata for Gate 3 output."""
    return {
        "gate": "sibling",
        "alpha": float(sibling_alpha),
        "uses_projection_dimensions_from_edge_comparisons": (
            sibling_inputs.projection_dimensions_from_edge_comparisons is not None
        ),
        "uses_parent_principal_component_projections": (
            sibling_inputs.parent_principal_component_projections is not None
        ),
        "uses_parent_principal_component_eigenvalues": (
            sibling_inputs.parent_principal_component_eigenvalues is not None
        ),
        "column_names": _build_column_names_metadata(
            edge_columns=edge_columns,
            sibling_columns=sibling_columns,
        ),
    }


def build_gate_annotation_config_metadata() -> dict[str, object]:
    """Capture config values that affect gate annotation outputs."""
    return {
        "felsenstein_scaling": bool(config.FELSENSTEIN_SCALING),
        "projection_eps": float(config.PROJECTION_EPS),
        "projection_minimum_dimension": config.PROJECTION_MINIMUM_DIMENSION,
        "resolved_projection_minimum_dimension": get_resolved_minimum_projection_dimension(),
        "spectral_minimum_dimension": int(config.SPECTRAL_MINIMUM_DIMENSION),
        "include_internal_in_spectral": bool(config.INCLUDE_INTERNAL_IN_SPECTRAL),
        "single_feature_subtree_mode": str(config.SINGLE_FEATURE_SUBTREE_MODE),
        "projection_random_seed": config.PROJECTION_RANDOM_SEED,
    }


def build_gate_annotation_leaf_data_metadata(
    leaf_data: pd.DataFrame | None,
) -> dict[str, object]:
    """Capture enough leaf-data identity to validate reusable annotations."""
    if leaf_data is None:
        return {"present": False}

    content_hash = hashlib.sha256()
    content_hash.update(str(tuple(leaf_data.shape)).encode("utf-8"))
    content_hash.update(
        pd.util.hash_pandas_object(pd.Index(leaf_data.index), index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    content_hash.update(
        pd.util.hash_pandas_object(pd.Index(leaf_data.columns), index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    content_hash.update(
        pd.util.hash_pandas_object(leaf_data, index=True)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    return {
        "present": True,
        "shape": [int(leaf_data.shape[0]), int(leaf_data.shape[1])],
        "content_hash": content_hash.hexdigest(),
    }


def _resolve_sibling_gate_inputs(
    tree,
    gate_two_result: Gate2Result,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[str, int] | None,
    parent_principal_component_projections: dict[str, np.ndarray] | None,
    parent_principal_component_eigenvalues: dict[str, np.ndarray] | None,
) -> _SiblingGateInputs:
    """Resolve optional Gate 3 inputs from explicit args or Gate 2 context."""
    resolved_projection_dimensions_from_edge_comparisons = (
        sibling_projection_dimensions_from_edge_comparisons
    )
    if resolved_projection_dimensions_from_edge_comparisons is None:
        resolved_projection_dimensions_from_edge_comparisons = (
            derive_sibling_projection_dimensions_from_child_edge_comparisons(
                tree,
                spectral_context=gate_two_result.spectral_context,
            )
        )

    resolved_parent_principal_component_projections = parent_principal_component_projections
    resolved_parent_principal_component_eigenvalues = parent_principal_component_eigenvalues
    if resolved_parent_principal_component_projections is None:
        (
            resolved_parent_principal_component_projections,
            resolved_parent_principal_component_eigenvalues,
        ) = collect_parent_principal_component_inputs_for_sibling_tests(
            resolved_projection_dimensions_from_edge_comparisons,
            spectral_context=gate_two_result.spectral_context,
        )

    return _SiblingGateInputs(
        projection_dimensions_from_edge_comparisons=(
            resolved_projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=(
            resolved_parent_principal_component_projections
        ),
        parent_principal_component_eigenvalues=(
            resolved_parent_principal_component_eigenvalues
        ),
    )


def run_gate_annotation_pipeline(
    tree,
    annotations_df: pd.DataFrame,
    *,
    alpha_local: float = config.EDGE_ALPHA,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    sibling_projection_dimensions_from_edge_comparisons: dict[str, int] | None = None,
    parent_principal_component_projections: dict[str, np.ndarray] | None = None,
    parent_principal_component_eigenvalues: dict[str, np.ndarray] | None = None,
) -> GateAnnotationBundle:
    """Run Gate 2 (edge) and Gate 3 (sibling) annotation pipeline.

    Gate 2 uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR correction.
    This is the only supported multiple-testing method.
    """
    resolve_minimum_projection_dimension(
        config.PROJECTION_MINIMUM_DIMENSION,
        leaf_data=leaf_data,
    )

    # Run Gate 2: child-parent edge tests
    edge_annotated_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
    )
    edge_columns = validate_edge_gate_columns(edge_annotated_df)
    edge_sibling_columns = sibling_gate_columns(edge_annotated_df)
    edge_metadata = _build_edge_metadata(
        alpha_local=alpha_local,
        edge_columns=edge_columns,
        sibling_columns=edge_sibling_columns,
    )
    gate_two_result = Gate2Result(
        annotated_df=edge_annotated_df,
        spectral_context=spectral_context,
        local_gate_columns=edge_columns,
        metadata=edge_metadata,
    )

    sibling_inputs = _resolve_sibling_gate_inputs(
        tree,
        gate_two_result,
        sibling_projection_dimensions_from_edge_comparisons=(
            sibling_projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=parent_principal_component_projections,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
    )

    # Run Gate 3: sibling divergence tests
    annotated_df = annotate_sibling_divergence(
        tree,
        edge_annotated_df,
        significance_level_alpha=sibling_alpha,
        sibling_projection_dimensions_from_edge_comparisons=(
            sibling_inputs.projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=(
            sibling_inputs.parent_principal_component_projections
        ),
        parent_principal_component_eigenvalues=(
            sibling_inputs.parent_principal_component_eigenvalues
        ),
        edge_projection_dimensions_by_node=(
            gate_two_result.spectral_context.spectral_projection_dimensions_by_node
        ),
    )
    output_edge_columns = validate_edge_gate_columns(
        annotated_df,
        error_context="Sibling gate input/output edge columns differ from required contract",
    )
    sibling_columns = validate_sibling_gate_columns(annotated_df)
    sibling_metadata = _build_sibling_metadata(
        sibling_alpha=sibling_alpha,
        sibling_inputs=sibling_inputs,
        edge_columns=output_edge_columns,
        sibling_columns=sibling_columns,
    )

    metadata = {
        "pipeline": "gate_annotation",
        "edge": edge_metadata,
        "sibling": sibling_metadata,
        "config": build_gate_annotation_config_metadata(),
        "leaf_data": build_gate_annotation_leaf_data_metadata(leaf_data),
    }
    annotated_df.attrs[GATE_ANNOTATION_METADATA_ATTR] = metadata

    return GateAnnotationBundle(
        annotated_df=annotated_df,
        local_gate_columns=output_edge_columns,
        sibling_gate_columns=sibling_columns,
        metadata=metadata,
        gate_two_result=gate_two_result,
    )


__all__ = [
    "build_gate_annotation_config_metadata",
    "build_gate_annotation_leaf_data_metadata",
    "run_gate_annotation_pipeline",
]
