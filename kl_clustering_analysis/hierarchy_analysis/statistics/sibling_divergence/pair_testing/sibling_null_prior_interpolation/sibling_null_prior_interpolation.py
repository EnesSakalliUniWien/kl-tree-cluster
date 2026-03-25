"""Tree-neighborhood interpolation of sibling null priors for blocked nodes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from ..types import SiblingPairRecord
from .adaptive_kernel_bandwidths import (
    AdaptiveKernelBandwidths,
    compute_adaptive_kernel_bandwidths,
    structural_kernel,
)
from .edge_metadata import (
    EdgeLevelMetadata,
    StoppingEdgeSummary,
    build_tree_distance,
    edge_structural_dimension,
    extract_edge_metadata,
    extract_stopping_edge_info,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeighborhoodReferenceSet:
    """Stable and signal edge neighborhoods used for sibling null prior interpolation."""

    stable_nodes: list[str]
    stable_p_values: np.ndarray
    stable_log_ks: np.ndarray
    signal_nodes: list[str]
    signal_p_values: np.ndarray


@dataclass(frozen=True)
class ChildSiblingNullPriorEstimate:
    """Per-child interpolated sibling null prior from tree-neighborhood estimation."""

    sibling_null_prior: float
    neighborhood_estimate: float
    ancestor_support: float
    stable_share: float


def _collect_reference_sets(
    annotations_df: pd.DataFrame,
    edge_metadata: EdgeLevelMetadata,
) -> NeighborhoodReferenceSet:
    stable_nodes: list[str] = []
    stable_p_values: list[float] = []
    stable_log_ks: list[float] = []
    signal_nodes: list[str] = []
    signal_p_values: list[float] = []

    for node_id, child_was_tested, child_was_significant, child_parent_edge_bh_p_value in zip(
        edge_metadata.edge_child_ids,
        edge_metadata.child_parent_edge_tested,
        edge_metadata.child_parent_edge_significant,
        edge_metadata.child_parent_edge_bh_p_values,
        strict=False,
    ):
        if not child_was_tested or not np.isfinite(child_parent_edge_bh_p_value):
            continue

        if child_was_significant:
            signal_nodes.append(node_id)
            signal_p_values.append(float(child_parent_edge_bh_p_value))
            continue

        stable_nodes.append(node_id)
        stable_p_values.append(float(child_parent_edge_bh_p_value))
        stable_log_ks.append(
            float(
                np.log(
                    max(
                        edge_structural_dimension(
                            node_id,
                            annotations_df,
                            edge_metadata.edge_spectral_dims,
                        ),
                        1.0,
                    )
                )
            )
        )

    return NeighborhoodReferenceSet(
        stable_nodes=stable_nodes,
        stable_p_values=np.asarray(stable_p_values, dtype=float),
        stable_log_ks=np.asarray(stable_log_ks, dtype=float),
        signal_nodes=signal_nodes,
        signal_p_values=np.asarray(signal_p_values, dtype=float),
    )


def _fallback_child_sibling_null_prior(
    child_id: str,
    record: SiblingPairRecord,
    annotations_df: pd.DataFrame,
) -> ChildSiblingNullPriorEstimate:
    if child_id in annotations_df.index and np.isfinite(
        float(annotations_df.at[child_id, "Child_Parent_Divergence_P_Value_BH"])
    ):
        child_sibling_null_prior = float(annotations_df.at[child_id, "Child_Parent_Divergence_P_Value_BH"])
    else:
        child_sibling_null_prior = float(record.sibling_null_prior_from_edge_pvalue)

    return ChildSiblingNullPriorEstimate(
        sibling_null_prior=child_sibling_null_prior,
        neighborhood_estimate=child_sibling_null_prior,
        ancestor_support=child_sibling_null_prior,
        stable_share=0.0,
    )


def _compute_child_sibling_null_prior(
    *,
    child_id: str,
    annotations_df: pd.DataFrame,
    stopping_info: StoppingEdgeSummary,
    edge_metadata: EdgeLevelMetadata,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance,
) -> ChildSiblingNullPriorEstimate:
    stopping_edge_kernel = float(
        np.exp(-max(stopping_info.distance_to_stopping_edge - 1.0, 0.0) / kernel_bandwidths.tau_b)
    )
    ancestor_support = stopping_info.stopping_edge_p_value
    child_log_k = float(
        np.log(
            max(
                edge_structural_dimension(
                    child_id,
                    annotations_df,
                    edge_metadata.edge_spectral_dims,
                ),
                1.0,
            )
        )
    )

    if reference_sets.stable_nodes:
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
        stable_mass = float(np.sum(neighbor_weights))
        stable_support = (
            float(np.average(reference_sets.stable_p_values, weights=neighbor_weights))
            if stable_mass > 0
            else ancestor_support
        )
    else:
        stable_mass = 0.0
        stable_support = ancestor_support

    support_numerator = (
        stopping_edge_kernel * stopping_info.stopping_edge_p_value + stable_mass * stable_support
    )
    support_denominator = stopping_edge_kernel + stable_mass
    support = (
        float(support_numerator / support_denominator)
        if support_denominator > 0
        else float(ancestor_support)
    )

    if reference_sets.signal_nodes:
        signal_distances = np.asarray(
            [tree_distance(child_id, signal_node) for signal_node in reference_sets.signal_nodes],
            dtype=float,
        )
        signal_terms = (1.0 - reference_sets.signal_p_values) * np.exp(
            -signal_distances / kernel_bandwidths.tau_s
        )
        signal_penalty = float(np.max(signal_terms)) if len(signal_terms) else 0.0
    else:
        signal_penalty = 0.0

    child_sibling_null_prior = float(np.clip(support * (1.0 - signal_penalty), 0.0, 1.0))
    stable_share = (
        float(stable_mass / (stopping_edge_kernel + stable_mass))
        if (stopping_edge_kernel + stable_mass) > 0
        else 0.0
    )
    return ChildSiblingNullPriorEstimate(
        sibling_null_prior=child_sibling_null_prior,
        neighborhood_estimate=stable_support,
        ancestor_support=ancestor_support,
        stable_share=stable_share,
    )


def _summarize_record_children(
    record: SiblingPairRecord,
    *,
    annotations_df: pd.DataFrame,
    stopping_edge_info_by_child: dict[str, StoppingEdgeSummary],
    edge_metadata: EdgeLevelMetadata,
    reference_sets: NeighborhoodReferenceSet,
    kernel_bandwidths: AdaptiveKernelBandwidths,
    tree_distance,
) -> tuple[list[float], list[float], list[float], list[float]]:
    child_sibling_null_priors: list[float] = []
    child_neighborhood_estimates: list[float] = []
    child_ancestor_supports: list[float] = []
    child_stable_shares: list[float] = []

    for child in (record.left, record.right):
        child_id = str(child)
        stopping_info = stopping_edge_info_by_child.get(child_id)
        child_estimate = (
            _fallback_child_sibling_null_prior(child_id, record, annotations_df)
            if stopping_info is None
            else _compute_child_sibling_null_prior(
                child_id=child_id,
                annotations_df=annotations_df,
                stopping_info=stopping_info,
                edge_metadata=edge_metadata,
                reference_sets=reference_sets,
                kernel_bandwidths=kernel_bandwidths,
                tree_distance=tree_distance,
            )
        )
        child_sibling_null_priors.append(child_estimate.sibling_null_prior)
        child_neighborhood_estimates.append(child_estimate.neighborhood_estimate)
        child_ancestor_supports.append(child_estimate.ancestor_support)
        child_stable_shares.append(child_estimate.stable_share)

    return child_sibling_null_priors, child_neighborhood_estimates, child_ancestor_supports, child_stable_shares


def interpolate_sibling_null_priors(
    records: list[SiblingPairRecord],
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
) -> list[SiblingPairRecord]:
    """Interpolate sibling null priors for blocked nodes from tree-local neighborhoods."""
    stopping_edge_info_by_child = extract_stopping_edge_info(annotations_df)
    if stopping_edge_info_by_child is None:
        return records

    edge_metadata = extract_edge_metadata(annotations_df)
    reference_sets = _collect_reference_sets(annotations_df, edge_metadata)
    child_ids = sorted(stopping_edge_info_by_child.keys())
    tree_distance = build_tree_distance(tree)
    kernel_bandwidths = compute_adaptive_kernel_bandwidths(
        stopping_edge_info=stopping_edge_info_by_child,
        child_ids=child_ids,
        stable_nodes=reference_sets.stable_nodes,
        signal_nodes=reference_sets.signal_nodes,
        stable_log_ks=reference_sets.stable_log_ks,
        tree_distance=tree_distance,
    )

    for record in records:
        if not record.is_gate2_blocked:
            continue

        (
            child_sibling_null_priors,
            child_neighborhood_estimates,
            child_ancestor_supports,
            child_stable_shares,
        ) = _summarize_record_children(
            record,
            annotations_df=annotations_df,
            stopping_edge_info_by_child=stopping_edge_info_by_child,
            edge_metadata=edge_metadata,
            reference_sets=reference_sets,
            kernel_bandwidths=kernel_bandwidths,
            tree_distance=tree_distance,
        )

        record.sibling_null_prior_from_edge_pvalue = min(child_sibling_null_priors) if child_sibling_null_priors else record.sibling_null_prior_from_edge_pvalue
        record.smoothed_sibling_null_prior = min(child_neighborhood_estimates) if child_neighborhood_estimates else None
        record.ancestor_support = min(child_ancestor_supports) if child_ancestor_supports else None
        record.blend_lambda = max(child_stable_shares) if child_stable_shares else None

    logger.debug(
        "Sibling null prior interpolation: n_children=%d, tau_b=%.3f, "
        "tau_t=%.3f, tau_s=%.3f, h_k=%.6f, n_stable=%d, n_signal=%d.",
        len(child_ids),
        kernel_bandwidths.tau_b,
        kernel_bandwidths.tau_t,
        kernel_bandwidths.tau_s,
        kernel_bandwidths.h_k,
        len(reference_sets.stable_nodes),
        len(reference_sets.signal_nodes),
    )
    return records


__all__ = ["interpolate_sibling_null_priors"]
