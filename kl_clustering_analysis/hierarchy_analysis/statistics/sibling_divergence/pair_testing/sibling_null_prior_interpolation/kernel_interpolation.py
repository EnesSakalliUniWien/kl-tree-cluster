"""Pure mathematical kernel functions for tree-neighborhood interpolation.

All functions are stateless and have no side effects — they depend only on
their arguments and standard numeric libraries.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .adaptive_kernel_bandwidths import AdaptiveKernelBandwidths, structural_kernel
from .edge_metadata import StoppingEdgeSummary, edge_scale
from .types import NeighborhoodReferenceSet


def _compute_ancestor_trust_weight(
    stopping_info: StoppingEdgeSummary,
    kernel_bandwidths: AdaptiveKernelBandwidths,
) -> float:
    """Exponential decay weight for the stopping-edge ancestor signal."""
    return float(
        np.exp(-max(stopping_info.distance_to_stopping_edge - 1.0, 0.0) / kernel_bandwidths.tau_b)
    )


def _compute_child_log_k(
    child_id: str,
    annotations_dataframe: pd.DataFrame,
    edge_spectral_dims: dict[str, int] | None,
) -> float:
    """Log of the scale used to match nearby child nodes."""
    scale = edge_scale(child_id, annotations_dataframe, edge_spectral_dims)
    return float(np.real(np.log(max(scale, 1.0))))


def _estimate_null_pvalue_from_stable_neighbors(
    child_id: str,
    child_log_k: float,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance: Callable[[str, str], float],
    fallback_p_value: float,
) -> tuple[float, float]:
    """Kernel-weighted average p-value over stable neighbors.

    Returns
    -------
    (trusted_neighbor_weight_sum, neighborhood_p_value_estimate)
    """
    if not reference_sets.stable_nodes:
        return 0.0, fallback_p_value

    stable_distances = np.asarray(
        [tree_distance(child_id, stable_node) for stable_node in reference_sets.stable_nodes],
        dtype=float,
    )
    tree_kernel = np.exp(-stable_distances / kernel_bandwidths.tau_t)
    structural_weights = structural_kernel(
        reference_sets.stable_log_ks,
        child_log_k,
        kernel_bandwidths.h_k,
    )
    neighbor_weights = tree_kernel * structural_weights
    trusted_neighbor_weight_sum = float(np.real(np.sum(neighbor_weights)))
    neighborhood_p_value_estimate = (
        float(np.real(np.average(reference_sets.stable_p_values, weights=neighbor_weights)))
        if trusted_neighbor_weight_sum > 0
        else fallback_p_value
    )
    return trusted_neighbor_weight_sum, neighborhood_p_value_estimate


def _interpolate_ancestor_and_neighbor_pvalue(
    ancestor_trust_weight: float,
    ancestor_p_value: float,
    trusted_neighbor_weight_sum: float,
    neighborhood_p_value_estimate: float,
) -> float:
    """Convex combination of ancestor and neighborhood p-value estimates."""
    support_denominator = ancestor_trust_weight + trusted_neighbor_weight_sum
    if support_denominator <= 0:
        return ancestor_p_value
    support_numerator = (
        ancestor_trust_weight * ancestor_p_value
        + trusted_neighbor_weight_sum * neighborhood_p_value_estimate
    )
    return float(np.real(support_numerator / support_denominator))


def _compute_signal_suppression_factor(
    child_id: str,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance: Callable[[str, str], float],
) -> float:
    """Max signal-decay suppression from nearby significant edges."""
    if not reference_sets.signal_nodes:
        return 0.0
    signal_distances = np.asarray(
        [tree_distance(child_id, signal_node) for signal_node in reference_sets.signal_nodes],
        dtype=float,
    )
    signal_terms = (1.0 - reference_sets.signal_p_values) * np.exp(
        -signal_distances / kernel_bandwidths.tau_s
    )
    return float(np.real(np.max(signal_terms))) if len(signal_terms) else 0.0


__all__ = [
    "_compute_ancestor_trust_weight",
    "_compute_child_log_k",
    "_estimate_null_pvalue_from_stable_neighbors",
    "_interpolate_ancestor_and_neighbor_pvalue",
    "_compute_signal_suppression_factor",
]
