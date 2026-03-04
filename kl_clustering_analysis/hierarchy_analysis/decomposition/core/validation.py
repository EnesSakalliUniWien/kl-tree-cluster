"""Validation helpers for decomposition contracts and inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ProjectionPlan
from .errors import DecompositionValidationError


def validate_distribution_shapes(left: np.ndarray, right: np.ndarray) -> None:
    """Validate that two distribution arrays are shape-compatible."""
    if left.shape != right.shape:
        raise DecompositionValidationError(
            f"Distribution shape mismatch: left={left.shape}, right={right.shape}."
        )
    if left.size == 0:
        raise DecompositionValidationError("Distribution arrays must be non-empty.")


def validate_leaf_data_alignment(tree, leaf_data: pd.DataFrame) -> None:
    """Validate that every leaf label in the tree exists in leaf_data.index."""
    missing = []
    for node_id, attrs in tree.nodes(data=True):
        if attrs.get("is_leaf", False):
            label = attrs.get("label", node_id)
            if label not in leaf_data.index:
                missing.append(label)
    if missing:
        preview = ", ".join(map(repr, missing[:5]))
        raise DecompositionValidationError(
            f"leaf_data is missing {len(missing)} leaf labels. Examples: {preview}."
        )


def validate_projection_plan(plan: ProjectionPlan) -> None:
    """Validate projection dimension and matrix consistency."""
    if plan.k <= 0:
        raise DecompositionValidationError(f"Projection dimension k must be > 0. Got {plan.k}.")
    if plan.projection_matrix is not None:
        if plan.projection_matrix.ndim != 2:
            raise DecompositionValidationError("Projection matrix must be 2D.")
        if plan.projection_matrix.shape[0] != plan.k:
            raise DecompositionValidationError(
                "Projection matrix row count must equal k. "
                f"rows={plan.projection_matrix.shape[0]}, k={plan.k}."
            )

