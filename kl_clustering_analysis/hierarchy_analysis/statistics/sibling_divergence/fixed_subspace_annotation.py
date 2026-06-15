"""Data-independent fixed-subspace sibling gate annotation.

This module provides an opt-in sibling gate that avoids learning parent PCA
projection rows and projection dimension from the same sample being tested.
It evaluates fixed covariance-whitened sibling contrasts with predeclared
coordinate-wise or feature-block BH aggregation.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from .inflated_projected_wald_annotation.fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .pair_testing.collection.pair_observations import identify_binary_sibling_children

FixedSubspaceSiblingGateMethod = Literal[
    "fixed_global_chi_square",
    "fixed_coordinate_bh",
    "fixed_block_bh",
]

FIXED_SUBSPACE_SIBLING_GATE_METHODS: tuple[FixedSubspaceSiblingGateMethod, ...] = (
    "fixed_global_chi_square",
    "fixed_coordinate_bh",
    "fixed_block_bh",
)


def _bh_adjusted_minimum(p_values: np.ndarray) -> float:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional; got {values.shape}.")
    if values.size == 0:
        return 1.0
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1].")
    order = np.argsort(values)
    sorted_p = values[order]
    n_tests = float(sorted_p.size)
    adjusted_sorted = np.minimum.accumulate(
        (n_tests / np.arange(1, sorted_p.size + 1, dtype=float) * sorted_p)[::-1]
    )[::-1]
    return float(np.clip(np.min(adjusted_sorted), 0.0, 1.0))


def fixed_coordinate_bh_p_value(z: np.ndarray) -> float:
    """Return the coordinate-wise BH p-value for a whitened contrast."""
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if vector.size == 0:
        return 1.0
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    coordinate_p = chi2.sf(vector * vector, df=1.0)
    return _bh_adjusted_minimum(coordinate_p)


def _feature_block_p_values(
    z: np.ndarray,
    feature_space: FeatureSpace,
) -> np.ndarray:
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")

    p_values: list[float] = []
    offset = 0
    for block in feature_space.blocks:
        width = int(block.contrast_dimension)
        if width <= 0:
            raise ValueError(
                f"Feature block {block.name!r} has invalid contrast_dimension={width}."
            )
        block_z = vector[offset : offset + width]
        if block_z.shape[0] != width:
            raise ValueError(
                "Feature blocks do not cover the whitened contrast vector: "
                f"block={block.name!r}, offset={offset}, width={width}, "
                f"z_dimension={vector.shape[0]}."
            )
        p_values.append(float(chi2.sf(float(np.dot(block_z, block_z)), df=width)))
        offset += width
    if offset != vector.shape[0]:
        raise ValueError(
            "Feature blocks do not cover the whitened contrast vector: "
            f"covered={offset}, z_dimension={vector.shape[0]}."
        )
    return np.asarray(p_values, dtype=float)


def fixed_subspace_sibling_p_value(
    z: np.ndarray,
    feature_space: FeatureSpace,
    *,
    method: FixedSubspaceSiblingGateMethod,
) -> float:
    """Return a fixed-subspace sibling p-value for a whitened contrast."""
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    if vector.size == 0:
        return 1.0

    if method == "fixed_global_chi_square":
        return float(chi2.sf(float(np.dot(vector, vector)), df=vector.size))
    if method == "fixed_coordinate_bh":
        return fixed_coordinate_bh_p_value(vector)
    if method == "fixed_block_bh":
        return _bh_adjusted_minimum(_feature_block_p_values(vector, feature_space))
    raise ValueError(
        f"Unknown fixed-subspace sibling gate method {method!r}; "
        f"allowed={FIXED_SUBSPACE_SIBLING_GATE_METHODS!r}."
    )


def _fixed_subspace_sibling_results(
    tree: nx.DiGraph,
    feature_space: FeatureSpace,
    *,
    method: FixedSubspaceSiblingGateMethod,
) -> tuple[list[object], list[tuple[float, float, float]], list[object]]:
    parents: list[object] = []
    results: list[tuple[float, float, float]] = []
    skipped: list[object] = []

    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            skipped.append(parent)
            continue
        left, right = children
        z = compute_whitened_wald_contrast(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(tree.nodes[left]["leaf_count"]),
            float(tree.nodes[right]["leaf_count"]),
            comparison="sibling",
            feature_space=feature_space,
        )
        statistic = float(np.dot(z, z))
        degrees_of_freedom = float(feature_space.contrast_dimension)
        p_value = fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method=method,
        )
        parents.append(parent)
        results.append((statistic, degrees_of_freedom, float(p_value)))

    return parents, results, skipped


def annotate_fixed_subspace_sibling_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    feature_space: FeatureSpace,
    method: FixedSubspaceSiblingGateMethod = "fixed_coordinate_bh",
    significance_level_alpha: float = DEFAULT_SIBLING_ALPHA,
) -> pd.DataFrame:
    """Annotate sibling divergence with an opt-in fixed-subspace gate."""
    if method not in FIXED_SUBSPACE_SIBLING_GATE_METHODS:
        raise ValueError(
            f"Unknown fixed-subspace sibling gate method {method!r}; "
            f"allowed={FIXED_SUBSPACE_SIBLING_GATE_METHODS!r}."
        )
    annotations_df = init_sibling_annotation_df(annotations_df)
    parents, results, skipped = _fixed_subspace_sibling_results(
        tree,
        feature_space,
        method=method,
    )
    mark_non_binary_as_skipped(annotations_df, skipped)
    if not results:
        return annotations_df

    return apply_traversal_aligned_sibling_bh_results(
        tree,
        annotations_df,
        parents,
        results,
        float(significance_level_alpha),
        method_labels=[method] * len(parents),
        skipped_parents=skipped,
    )


__all__ = [
    "FIXED_SUBSPACE_SIBLING_GATE_METHODS",
    "FixedSubspaceSiblingGateMethod",
    "annotate_fixed_subspace_sibling_divergence",
    "fixed_coordinate_bh_p_value",
    "fixed_subspace_sibling_p_value",
]
