"""Tree-neighborhood interpolation of sibling null priors for blocked nodes."""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from ..types import SiblingPairRecord
from .adaptive_kernel_bandwidths import compute_adaptive_kernel_bandwidths
from .child_prior_estimation import _estimate_sibling_pair_null_priors
from .edge_metadata import build_tree_distance, extract_edge_metadata, extract_stopping_edge_info
from .reference_set import _collect_reference_sets

logger = logging.getLogger(__name__)


def interpolate_sibling_null_priors(
    records: list[SiblingPairRecord],
    tree: nx.DiGraph,
    annotations_dataframe: pd.DataFrame,
) -> list[SiblingPairRecord]:
    """Interpolate sibling null priors for blocked nodes from tree-local neighborhoods."""
    stopping_edge_info_by_child = extract_stopping_edge_info(annotations_dataframe)
    if stopping_edge_info_by_child is None:
        return records

    edge_metadata = extract_edge_metadata(annotations_dataframe)

    reference_sets = _collect_reference_sets(annotations_dataframe, edge_metadata)
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

        left_estimate, right_estimate = _estimate_sibling_pair_null_priors(
            record,
            annotations_dataframe=annotations_dataframe,
            stopping_edge_info_by_child=stopping_edge_info_by_child,
            edge_metadata=edge_metadata,
            reference_sets=reference_sets,
            kernel_bandwidths=kernel_bandwidths,
            tree_distance=tree_distance,
        )

        record.sibling_null_prior_from_edge_pvalue = min(
            left_estimate.sibling_null_prior, right_estimate.sibling_null_prior
        )
        record.smoothed_sibling_null_prior = min(
            left_estimate.neighborhood_estimate, right_estimate.neighborhood_estimate
        )
        record.ancestor_support = min(
            left_estimate.ancestor_support, right_estimate.ancestor_support
        )
        record.neighborhood_reliance = max(
            left_estimate.neighborhood_interpolation_weight,
            right_estimate.neighborhood_interpolation_weight,
        )

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
