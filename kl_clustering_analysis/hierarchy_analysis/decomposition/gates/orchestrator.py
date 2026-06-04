"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ...statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from ...statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from ...statistics.sibling_divergence.inflated_projected_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from ...statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from ...statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from .annotation_bundle import (
    EdgeGateResult,
    GateAnnotationBundle,
    GateAnnotationConfigMetadata,
    GateAnnotationLeafDataMetadata,
    GateAnnotationMetadata,
    GateMetadata,
)
from .column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)


@dataclass(frozen=True)
class _SiblingGateInputs:
    projection_dimensions_from_edge_comparisons: dict[str, int]
    parent_principal_component_projections: dict[str, np.ndarray]
    parent_principal_component_eigenvalues: dict[str, np.ndarray]


def _build_edge_metadata(
    *,
    edge_alpha: float,
) -> GateMetadata:
    """Build metadata for edge-gate output.

    Tree-BH is the only supported FDR method, so not stored in metadata.
    """
    return GateMetadata(gate="edge", alpha=float(edge_alpha))


def _build_sibling_metadata(
    *,
    sibling_alpha: float,
) -> GateMetadata:
    """Build metadata for sibling-gate output."""
    return GateMetadata(gate="sibling", alpha=float(sibling_alpha))


def build_gate_annotation_config_metadata() -> GateAnnotationConfigMetadata:
    """Capture config values that affect gate annotation outputs."""
    return GateAnnotationConfigMetadata(
        spectral_minimum_dimension=EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    )


def build_gate_annotation_leaf_data_metadata(
    leaf_data: pd.DataFrame | None,
    *,
    feature_space: FeatureSpace | None = None,
) -> GateAnnotationLeafDataMetadata:
    """Capture enough leaf-data identity to validate reusable annotations."""
    if leaf_data is None:
        return GateAnnotationLeafDataMetadata(present=False)

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
    return GateAnnotationLeafDataMetadata(
        present=True,
        shape=(int(leaf_data.shape[0]), int(leaf_data.shape[1])),
        content_hash=content_hash.hexdigest(),
        feature_space_signature=(
            None if feature_space is None else feature_space.signature
        ),
    )


def _resolve_sibling_gate_inputs(
    tree,
    edge_gate_result: EdgeGateResult,
) -> _SiblingGateInputs:
    """Resolve sibling-gate inputs from edge-gate context."""
    resolved_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=edge_gate_result.spectral_context,
        )
    )
    (
        resolved_parent_principal_component_projections,
        resolved_parent_principal_component_eigenvalues,
    ) = collect_parent_principal_component_inputs_for_sibling_tests(
        resolved_projection_dimensions_from_edge_comparisons,
        spectral_context=edge_gate_result.spectral_context,
    )
    expected_parent_keys = set(resolved_projection_dimensions_from_edge_comparisons)
    projection_keys = set(resolved_parent_principal_component_projections)
    eigenvalue_keys = set(resolved_parent_principal_component_eigenvalues)
    if projection_keys != expected_parent_keys or eigenvalue_keys != expected_parent_keys:
        raise ValueError(
            "Sibling-gate parent PCA inputs must be keyed exactly by sibling projection parents. "
            f"expected={sorted(expected_parent_keys)!r}, "
            f"projection_keys={sorted(projection_keys)!r}, "
            f"eigenvalue_keys={sorted(eigenvalue_keys)!r}."
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
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    feature_space: FeatureSpace | None = None,
) -> GateAnnotationBundle:
    """Run the edge-gate and sibling-gate annotation pipeline.

    The edge-divergence gate uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR
    correction. This is the only supported edge multiple-testing method.
    """
    stage_timings = {
        "edge_gate_contrast_covariance_sec": 0.0,
        "edge_gate_projection_sec": 0.0,
        "edge_gate_wald_statistic_sec": 0.0,
        "edge_gate_tree_bh_sec": 0.0,
        "sibling_gate_pair_record_collection_sec": 0.0,
        "sibling_gate_inflation_fit_sec": 0.0,
        "sibling_gate_adjusted_tests_sec": 0.0,
        "sibling_gate_fdr_sec": 0.0,
    }

    # Run edge-divergence gate: child-parent edge tests
    edge_gate_start_sec = perf_counter()
    edge_annotated_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df,
        significance_level_alpha=edge_alpha,
        leaf_data=leaf_data,
        feature_space=feature_space,
        stage_timings=stage_timings,
    )
    edge_gate_sec = float(perf_counter() - edge_gate_start_sec)
    stage_timings.update(spectral_context.stage_timings)
    validate_edge_gate_columns(edge_annotated_df)
    edge_metadata = _build_edge_metadata(
        edge_alpha=edge_alpha,
    )
    edge_gate_result = EdgeGateResult(
        annotated_df=edge_annotated_df,
        spectral_context=spectral_context,
        metadata=edge_metadata,
    )

    sibling_inputs = _resolve_sibling_gate_inputs(
        tree,
        edge_gate_result,
    )

    # Run sibling-divergence gate
    sibling_gate_start_sec = perf_counter()
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
        feature_space=feature_space,
        stage_timings=stage_timings,
    )
    sibling_gate_sec = float(perf_counter() - sibling_gate_start_sec)
    validate_edge_gate_columns(
        annotated_df,
        error_context="Sibling gate input/output edge columns differ from required contract",
    )
    validate_sibling_gate_columns(annotated_df)
    sibling_metadata = _build_sibling_metadata(
        sibling_alpha=sibling_alpha,
    )

    metadata = GateAnnotationMetadata(
        pipeline="gate_annotation",
        edge=edge_metadata,
        sibling=sibling_metadata,
        config=build_gate_annotation_config_metadata(),
        leaf_data=build_gate_annotation_leaf_data_metadata(
            leaf_data,
            feature_space=feature_space,
        ),
    )

    stage_timings["edge_gate_sec"] = edge_gate_sec
    stage_timings["sibling_gate_sec"] = sibling_gate_sec

    return GateAnnotationBundle(
        annotated_df=annotated_df,
        metadata=metadata,
        edge_gate_result=edge_gate_result,
        stage_timings=stage_timings,
    )


__all__ = [
    "build_gate_annotation_config_metadata",
    "build_gate_annotation_leaf_data_metadata",
    "run_gate_annotation_pipeline",
]
