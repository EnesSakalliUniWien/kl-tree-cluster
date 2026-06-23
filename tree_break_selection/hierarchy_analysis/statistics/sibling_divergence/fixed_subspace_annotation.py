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

from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from .inflated_projected_wald_annotation.fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .pair_testing.collection.pair_observations import identify_binary_sibling_children

FixedSubspaceSiblingGateMethod = Literal[
    "fixed_global_chi_square",
    "fixed_coordinate_bh",
    "fixed_coordinate_by",
    "fixed_coordinate_holm",
    "fixed_coordinate_bonferroni",
    "fixed_block_bh",
    "fixed_block_simes_bh",
]
CoordinateFdrMethod = Literal["bh", "by", "holm", "bonferroni"]

FIXED_SUBSPACE_SIBLING_GATE_METHODS: tuple[FixedSubspaceSiblingGateMethod, ...] = (
    "fixed_global_chi_square",
    "fixed_coordinate_bh",
    "fixed_coordinate_by",
    "fixed_coordinate_holm",
    "fixed_coordinate_bonferroni",
    "fixed_block_bh",
    "fixed_block_simes_bh",
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


def coordinate_fdr_adjusted_minimum(
    p_values: np.ndarray,
    *,
    method: CoordinateFdrMethod = "bh",
) -> float:
    """Return the minimum adjusted p-value for fixed coordinate evidence."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional; got {values.shape}.")
    if values.size == 0:
        return 1.0
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1].")

    method_name = str(method)
    if method_name == "bh":
        return _bh_adjusted_minimum(values)

    sorted_p = np.sort(values)
    n_tests = float(sorted_p.size)
    if method_name == "by":
        harmonic = float(np.sum(1.0 / np.arange(1, sorted_p.size + 1, dtype=float)))
        return float(np.clip(harmonic * _bh_adjusted_minimum(sorted_p), 0.0, 1.0))
    if method_name == "bonferroni":
        return float(np.clip(n_tests * sorted_p[0], 0.0, 1.0))
    if method_name == "holm":
        adjusted_sorted = np.maximum.accumulate(
            (n_tests - np.arange(sorted_p.size, dtype=float)) * sorted_p
        )
        return float(np.clip(np.min(adjusted_sorted), 0.0, 1.0))
    raise ValueError(
        f"Unknown coordinate FDR method {method!r}; "
        "allowed=('bh', 'by', 'holm', 'bonferroni')."
    )


def fixed_coordinate_bh_p_value(z: np.ndarray) -> float:
    """Return the coordinate-wise BH p-value for a whitened contrast."""
    return fixed_coordinate_fdr_p_value(z, method="bh")


def fixed_coordinate_fdr_p_value(
    z: np.ndarray,
    *,
    method: CoordinateFdrMethod = "bh",
) -> float:
    """Return a coordinate-wise adjusted p-value for a whitened contrast."""
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if vector.size == 0:
        return 1.0
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    coordinate_p = chi2.sf(vector * vector, df=1.0)
    return coordinate_fdr_adjusted_minimum(coordinate_p, method=method)


def fixed_block_simes_bh_p_value(
    z: np.ndarray,
    feature_space: FeatureSpace,
) -> float:
    """Return BH-over-blocks using Simes p-values inside feature blocks."""
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    if vector.size == 0:
        return 1.0

    block_p_values: list[float] = []
    offset = 0
    for block in feature_space.blocks:
        width = int(block.contrast_dimension)
        block_z = vector[offset : offset + width]
        if block_z.shape[0] != width:
            raise ValueError(
                "Feature blocks do not cover the whitened contrast vector: "
                f"block={block.name!r}, offset={offset}, width={width}, "
                f"z_dimension={vector.shape[0]}."
            )
        coordinate_p = chi2.sf(block_z * block_z, df=1.0)
        block_p_values.append(_bh_adjusted_minimum(coordinate_p))
        offset += width
    if offset != vector.shape[0]:
        raise ValueError(
            "Feature blocks do not cover the whitened contrast vector: "
            f"covered={offset}, z_dimension={vector.shape[0]}."
        )
    return _bh_adjusted_minimum(np.asarray(block_p_values, dtype=float))



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
    if method == "fixed_coordinate_by":
        return fixed_coordinate_fdr_p_value(vector, method="by")
    if method == "fixed_coordinate_holm":
        return fixed_coordinate_fdr_p_value(vector, method="holm")
    if method == "fixed_coordinate_bonferroni":
        return fixed_coordinate_fdr_p_value(vector, method="bonferroni")
    if method == "fixed_block_bh":
        return _bh_adjusted_minimum(_feature_block_p_values(vector, feature_space))
    if method == "fixed_block_simes_bh":
        return fixed_block_simes_bh_p_value(vector, feature_space)
    raise ValueError(
        f"Unknown fixed-subspace sibling gate method {method!r}; "
        f"allowed={FIXED_SUBSPACE_SIBLING_GATE_METHODS!r}."
    )


def _sibling_fixed_subspace_contrast(
    tree: nx.DiGraph,
    feature_space: FeatureSpace,
    parent: object,
    left: object,
    right: object,
) -> np.ndarray:
    return compute_whitened_wald_contrast(
        np.asarray(tree.nodes[left]["distribution"], dtype=float),
        np.asarray(tree.nodes[right]["distribution"], dtype=float),
        float(tree.nodes[left]["leaf_count"]),
        float(tree.nodes[right]["leaf_count"]),
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=require_node_continuous_covariance_by_block(
            tree,
            parent,
            feature_space,
        ),
    )


def _fixed_subspace_channel_values(
    z: np.ndarray,
    feature_space: FeatureSpace,
) -> dict[str, float]:
    vector = np.asarray(z, dtype=float)
    statistic = float(np.dot(vector, vector))
    degrees_of_freedom = float(feature_space.contrast_dimension)
    return {
        "fixed_coordinate_bh": fixed_coordinate_bh_p_value(vector),
        "fixed_block_bh": fixed_subspace_sibling_p_value(
            vector,
            feature_space,
            method="fixed_block_bh",
        ),
        "fixed_global_chi_square": fixed_subspace_sibling_p_value(
            vector,
            feature_space,
            method="fixed_global_chi_square",
        ),
        "statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
    }


def annotate_fixed_subspace_sibling_evidence_channels(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    feature_space: FeatureSpace,
    sparse_method: FixedSubspaceSiblingGateMethod = "fixed_coordinate_bh",
    dense_method: FixedSubspaceSiblingGateMethod = "fixed_global_chi_square",
) -> pd.DataFrame:
    """Annotate interpretable sparse and dense fixed-subspace sibling channels.

    These columns are auxiliary evidence channels. They do not change the
    active sibling gate p-value unless the caller has explicitly configured the
    gate to use the same fixed-subspace method.
    """
    if sparse_method not in {"fixed_coordinate_bh", "fixed_block_bh"}:
        raise ValueError(
            "sparse_method must be 'fixed_coordinate_bh' or 'fixed_block_bh'; "
            f"got {sparse_method!r}."
        )
    if dense_method != "fixed_global_chi_square":
        raise ValueError(
            "dense_method must currently be 'fixed_global_chi_square'; "
            f"got {dense_method!r}."
        )

    out = annotations_df.copy()
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            continue
        left, right = children
        z = _sibling_fixed_subspace_contrast(tree, feature_space, parent, left, right)
        values = _fixed_subspace_channel_values(z, feature_space)

        out.loc[parent, "Sibling_Fixed_Coordinate_BH_P_Value"] = values[
            "fixed_coordinate_bh"
        ]
        out.loc[parent, "Sibling_Fixed_Block_BH_P_Value"] = values[
            "fixed_block_bh"
        ]
        out.loc[parent, "Sibling_Fixed_Global_P_Value"] = values[
            "fixed_global_chi_square"
        ]
        out.loc[parent, "Sibling_Sparse_Evidence_P_Value"] = values[
            str(sparse_method)
        ]
        out.loc[parent, "Sibling_Sparse_Evidence_Method"] = str(sparse_method)
        out.loc[parent, "Sibling_Sparse_Evidence_Calibration"] = (
            "fixed_subspace_bh"
        )
        out.loc[parent, "Sibling_Dense_Evidence_P_Value"] = values[
            str(dense_method)
        ]
        out.loc[parent, "Sibling_Dense_Evidence_Method"] = str(dense_method)
        out.loc[parent, "Sibling_Dense_Evidence_Calibration"] = (
            "fixed_subspace_chi_square"
        )
        out.loc[parent, "Sibling_Dense_Evidence_Test_Statistic"] = values[
            "statistic"
        ]
        out.loc[parent, "Sibling_Dense_Evidence_Degrees_of_Freedom"] = values[
            "degrees_of_freedom"
        ]

    return out


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
        z = _sibling_fixed_subspace_contrast(tree, feature_space, parent, left, right)
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
    "CoordinateFdrMethod",
    "FIXED_SUBSPACE_SIBLING_GATE_METHODS",
    "FixedSubspaceSiblingGateMethod",
    "annotate_fixed_subspace_sibling_evidence_channels",
    "annotate_fixed_subspace_sibling_divergence",
    "coordinate_fdr_adjusted_minimum",
    "fixed_block_simes_bh_p_value",
    "fixed_coordinate_bh_p_value",
    "fixed_coordinate_fdr_p_value",
    "fixed_subspace_sibling_p_value",
]
