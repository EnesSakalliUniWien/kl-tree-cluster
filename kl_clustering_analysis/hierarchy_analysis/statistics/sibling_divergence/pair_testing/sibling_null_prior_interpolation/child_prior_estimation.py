"""Per-child sibling null prior estimation, including fallback logic."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..types import SiblingPairRecord
from .adaptive_kernel_bandwidths import AdaptiveKernelBandwidths
from .edge_metadata import EdgeLevelMetadata, StoppingEdgeSummary
from .kernel_interpolation import (
    _compute_ancestor_trust_weight,
    _compute_child_log_k,
    _compute_signal_suppression_factor,
    _estimate_null_pvalue_from_stable_neighbors,
    _interpolate_ancestor_and_neighbor_pvalue,
)
from .types import ChildSiblingNullPriorEstimate, NeighborhoodReferenceSet


def _fallback_child_sibling_null_prior(
    child_id: str,
    record: SiblingPairRecord,
    annotations_dataframe: pd.DataFrame,
) -> ChildSiblingNullPriorEstimate:
    """Use the BH-corrected edge p-value when no stopping-edge context is available."""
    if child_id in annotations_dataframe.index:
        bh_corrected_edge_pvalue = float(
            np.asarray(
                annotations_dataframe.at[child_id, "Child_Parent_Divergence_P_Value_BH"],
                dtype=float,
            ).item()
        )
    else:
        bh_corrected_edge_pvalue = float("nan")
    if np.isfinite(bh_corrected_edge_pvalue):
        child_sibling_null_prior = bh_corrected_edge_pvalue
    else:
        child_sibling_null_prior = float(record.sibling_null_prior_from_edge_pvalue)

    return ChildSiblingNullPriorEstimate(
        sibling_null_prior=child_sibling_null_prior,
        neighborhood_estimate=child_sibling_null_prior,
        ancestor_support=child_sibling_null_prior,
        neighborhood_interpolation_weight=0.0,
    )


def _compute_child_sibling_null_prior(
    *,
    child_id: str,
    annotations_dataframe: pd.DataFrame,
    stopping_info: StoppingEdgeSummary,
    edge_metadata: EdgeLevelMetadata,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance: Callable[[str, str], float],
) -> ChildSiblingNullPriorEstimate:
    """Full interpolated estimate for a child with a valid stopping-edge context."""
    ancestor_trust_weight = _compute_ancestor_trust_weight(stopping_info, kernel_bandwidths)
    ancestor_p_value = stopping_info.stopping_edge_p_value
    child_log_k = _compute_child_log_k(
        child_id, annotations_dataframe, edge_metadata.edge_spectral_dims
    )
    trusted_neighbor_weight_sum, neighborhood_p_value_estimate = (
        _estimate_null_pvalue_from_stable_neighbors(
            child_id,
            child_log_k,
            reference_sets,
            kernel_bandwidths,
            tree_distance,
            fallback_p_value=ancestor_p_value,
        )
    )

    interpolated_null_pvalue = _interpolate_ancestor_and_neighbor_pvalue(
        ancestor_trust_weight,
        ancestor_p_value,
        trusted_neighbor_weight_sum,
        neighborhood_p_value_estimate,
    )

    signal_suppression_factor = _compute_signal_suppression_factor(
        child_id,
        reference_sets,
        kernel_bandwidths,
        tree_distance,
    )

    signal_suppressed_null_pvalue = float(
        np.real(np.clip(interpolated_null_pvalue * (1.0 - signal_suppression_factor), 0.0, 1.0))
    )
    neighborhood_interpolation_weight = (
        float(trusted_neighbor_weight_sum / (ancestor_trust_weight + trusted_neighbor_weight_sum))
        if (ancestor_trust_weight + trusted_neighbor_weight_sum) > 0
        else 0.0
    )

    return ChildSiblingNullPriorEstimate(
        sibling_null_prior=signal_suppressed_null_pvalue,
        neighborhood_estimate=neighborhood_p_value_estimate,
        ancestor_support=ancestor_p_value,
        neighborhood_interpolation_weight=neighborhood_interpolation_weight,
    )


def _estimate_sibling_pair_null_priors(
    record: SiblingPairRecord,
    *,
    annotations_dataframe: pd.DataFrame,
    stopping_edge_info_by_child: dict[str, StoppingEdgeSummary],
    edge_metadata: EdgeLevelMetadata,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance: Callable[[str, str], float],
) -> tuple[ChildSiblingNullPriorEstimate, ChildSiblingNullPriorEstimate]:
    """Compute left and right child estimates for a single sibling pair record."""
    estimates = []
    for child in (record.left, record.right):
        child_id = str(child)
        stopping_info = stopping_edge_info_by_child.get(child_id)
        estimate = (
            _fallback_child_sibling_null_prior(child_id, record, annotations_dataframe)
            if stopping_info is None
            else _compute_child_sibling_null_prior(
                child_id=child_id,
                annotations_dataframe=annotations_dataframe,
                stopping_info=stopping_info,
                edge_metadata=edge_metadata,
                reference_sets=reference_sets,
                kernel_bandwidths=kernel_bandwidths,
                tree_distance=tree_distance,
            )
        )
        estimates.append(estimate)
    return estimates[0], estimates[1]


__all__ = [
    "_fallback_child_sibling_null_prior",
    "_compute_child_sibling_null_prior",
    "_estimate_sibling_pair_null_priors",
]
