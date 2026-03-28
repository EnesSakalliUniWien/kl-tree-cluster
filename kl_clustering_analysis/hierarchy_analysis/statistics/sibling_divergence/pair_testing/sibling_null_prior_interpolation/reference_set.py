"""Builds the stable/signal reference neighborhoods from tree annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .edge_metadata import EdgeLevelMetadata, edge_scale
from .types import NeighborhoodReferenceSet


def _collect_reference_sets(
    annotations_dataframe: pd.DataFrame,
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

        edge_dimension = edge_scale(
            node_id,
            annotations_dataframe,
            edge_metadata.edge_spectral_dims,
        )
        edge_dimension_clamped = max(edge_dimension, 1.0)
        log_value = np.log(edge_dimension_clamped)
        stable_log_ks.append(float(np.real(log_value)))

    return NeighborhoodReferenceSet(
        stable_nodes=stable_nodes,
        stable_p_values=np.asarray(stable_p_values, dtype=float),
        stable_log_ks=np.asarray(stable_log_ks, dtype=float),
        signal_nodes=signal_nodes,
        signal_p_values=np.asarray(signal_p_values, dtype=float),
    )


__all__ = ["_collect_reference_sets"]
