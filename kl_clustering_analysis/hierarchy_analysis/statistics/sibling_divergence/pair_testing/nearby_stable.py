"""Nearby-stable enrichment for blocked-edge calibration weights.

This module replaces the legacy ``p=1.0`` blocked-edge fallback with a
child-local neighborhood formula:

1. Blocked children inherit support from their nearest tested blocker.
2. That support is augmented by nearby tested non-significant edges.
3. Nearby tested significant edges apply a multiplicative penalty.

For a blocked child edge ``v`` the runtime weight is:

    support(v) =
        ( blocker_kernel(v) * p_blocker(v)
          + sum_j a_j(v) * p_j ) /
        ( blocker_kernel(v) + sum_j a_j(v) )

    a_j(v) =
        exp(-0.5 * ((log k_j - log k_v) / h_k)^2) *
        exp(-d_tree(j, v) / tau_t)

    signal(v) =
        max_s (1 - p_s) * exp(-d_tree(s, v) / tau_s)

    w_child(v) = clip(support(v) * (1 - signal(v)), 0, 1)

where ``j`` ranges over tested non-significant edge children and ``s`` ranges
over tested significant edge children.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from .types import SiblingPairRecord

logger = logging.getLogger(__name__)

_DISTANCE_INF = 1e12
_EPS = 1e-12


def _extract_blocker_metadata(annotations_df: pd.DataFrame) -> Dict[str, Dict[str, float]] | None:
    """Extract per-child blocker metadata from ``annotations_df.attrs``."""
    metadata = annotations_df.attrs.get("_blocker_metadata")
    if metadata is None:
        return None

    child_ids = metadata["child_ids"]
    result: Dict[str, Dict[str, float]] = {}
    for index, child_id in enumerate(child_ids):
        blocker_p = metadata["blocker_p_values"][index]
        if not np.isfinite(blocker_p):
            continue
        result[str(child_id)] = {
            "blocker_p": float(blocker_p),
            "dist_blocker": float(metadata["distances_to_blocker"][index]),
        }
    return result if result else None


def _extract_edge_level_context(
    annotations_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return edge-level tested/significant masks and BH p-values."""
    tested = (
        annotations_df["Child_Parent_Divergence_Tested"].astype(bool).to_numpy()
        if "Child_Parent_Divergence_Tested" in annotations_df.columns
        else np.ones(len(annotations_df), dtype=bool)
    )
    significant = (
        annotations_df["Child_Parent_Divergence_Significant"].astype(bool).to_numpy()
        if "Child_Parent_Divergence_Significant" in annotations_df.columns
        else np.zeros(len(annotations_df), dtype=bool)
    )
    edge_p_values = (
        annotations_df["Child_Parent_Divergence_P_Value_BH"].astype(float).to_numpy()
        if "Child_Parent_Divergence_P_Value_BH" in annotations_df.columns
        else np.full(len(annotations_df), np.nan, dtype=float)
    )
    return list(map(str, annotations_df.index.tolist())), tested, significant, edge_p_values


def _edge_structural_dimension(
    node_id: str,
    annotations_df: pd.DataFrame,
    edge_spectral_dims: Dict[str, int] | None,
) -> float:
    """Return the edge-level structural dimension used for neighborhood matching."""
    if edge_spectral_dims is not None:
        value = edge_spectral_dims.get(str(node_id))
        if value is not None and np.isfinite(value) and float(value) > 0:
            return float(value)

    if "Child_Parent_Divergence_df" in annotations_df.columns:
        value = float(annotations_df.at[node_id, "Child_Parent_Divergence_df"])
        if np.isfinite(value) and value > 0:
            return float(value)

    return 1.0


def _safe_positive_scale(values: list[float], default: float) -> float:
    """Return a positive finite scale from a sample list."""
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0]
    if not finite:
        return float(default)
    return max(float(np.median(finite)), _EPS)


def _compute_adaptive_scales(
    *,
    blocker_meta: Dict[str, Dict[str, float]],
    blocked_child_ids: list[str],
    stable_nodes: list[str],
    signal_nodes: list[str],
    stable_log_ks: np.ndarray,
    tree_distance,
) -> Dict[str, float]:
    """Infer tree-specific scales from blocked/stable/signal neighborhoods."""
    tau_b = _safe_positive_scale(
        [meta["dist_blocker"] for meta in blocker_meta.values()],
        default=1.0,
    )

    nearest_stable_distances: list[float] = []
    if stable_nodes:
        for blocked_child in blocked_child_ids:
            distances = [tree_distance(blocked_child, stable_child) for stable_child in stable_nodes]
            nearest_stable_distances.append(min(distances))
    tau_t = _safe_positive_scale(nearest_stable_distances, default=1.0)

    nearest_signal_distances: list[float] = []
    if signal_nodes:
        for blocked_child in blocked_child_ids:
            distances = [tree_distance(blocked_child, signal_child) for signal_child in signal_nodes]
            nearest_signal_distances.append(min(distances))
    tau_s = _safe_positive_scale(nearest_signal_distances, default=1.0)

    if len(stable_log_ks) > 1:
        h_k = float(np.std(stable_log_ks))
    else:
        h_k = 0.0

    return {
        "tau_b": tau_b,
        "tau_t": tau_t,
        "tau_s": tau_s,
        "h_k": max(h_k, 0.0),
    }


def _structural_kernel(log_k_source: np.ndarray, log_k_target: float, h_k: float) -> np.ndarray:
    """Return Gaussian-in-log-k weights, with exact-match behavior at zero spread."""
    if h_k <= 0:
        return np.where(np.isclose(log_k_source, log_k_target, atol=1e-12), 1.0, 0.0)
    normalized = (log_k_source - log_k_target) / h_k
    return np.exp(-0.5 * normalized**2)


def enrich_blocked_weights(
    records: List[SiblingPairRecord],
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
) -> List[SiblingPairRecord]:
    """Enrich blocked-edge calibration weights with nearby-stable borrowing."""
    blocker_meta = _extract_blocker_metadata(annotations_df)
    if blocker_meta is None:
        return records

    edge_child_ids, edge_tested, edge_significant, edge_p_values = _extract_edge_level_context(
        annotations_df
    )
    edge_spectral_dims = annotations_df.attrs.get("_spectral_dims")

    stable_nodes: list[str] = []
    stable_p_values: list[float] = []
    stable_log_ks: list[float] = []
    signal_nodes: list[str] = []
    signal_p_values: list[float] = []

    for node_id, tested, significant, p_value in zip(
        edge_child_ids,
        edge_tested,
        edge_significant,
        edge_p_values,
        strict=False,
    ):
        if not tested or not np.isfinite(p_value):
            continue

        if significant:
            signal_nodes.append(node_id)
            signal_p_values.append(float(p_value))
            continue

        stable_nodes.append(node_id)
        stable_p_values.append(float(p_value))
        stable_log_ks.append(
            float(
                np.log(
                    max(
                        _edge_structural_dimension(node_id, annotations_df, edge_spectral_dims),
                        1.0,
                    )
                )
            )
        )

    stable_p_values_arr = np.asarray(stable_p_values, dtype=float)
    stable_log_ks_arr = np.asarray(stable_log_ks, dtype=float)
    signal_p_values_arr = np.asarray(signal_p_values, dtype=float)

    blocked_child_ids = sorted(blocker_meta.keys())
    tree_undirected = tree.to_undirected(as_view=True)

    @lru_cache(maxsize=None)
    def tree_distance(node_a: str, node_b: str) -> float:
        if node_a == node_b:
            return 0.0
        try:
            return float(nx.shortest_path_length(tree_undirected, node_a, node_b))
        except nx.NetworkXNoPath:
            return _DISTANCE_INF

    scales = _compute_adaptive_scales(
        blocker_meta=blocker_meta,
        blocked_child_ids=blocked_child_ids,
        stable_nodes=stable_nodes,
        signal_nodes=signal_nodes,
        stable_log_ks=stable_log_ks_arr,
        tree_distance=tree_distance,
    )
    tau_b = scales["tau_b"]
    tau_t = scales["tau_t"]
    tau_s = scales["tau_s"]
    h_k = scales["h_k"]

    for record in records:
        if not record.is_gate2_blocked:
            continue

        child_weights: list[float] = []
        child_supports: list[float] = []
        child_ancestor_supports: list[float] = []
        child_stable_shares: list[float] = []

        for child in (record.left, record.right):
            child_id = str(child)

            if child_id not in blocker_meta:
                if child_id in annotations_df.index and np.isfinite(
                    float(annotations_df.at[child_id, "Child_Parent_Divergence_P_Value_BH"])
                ):
                    child_weight = float(
                        annotations_df.at[child_id, "Child_Parent_Divergence_P_Value_BH"]
                    )
                else:
                    child_weight = float(record.edge_weight)

                child_weights.append(child_weight)
                child_supports.append(child_weight)
                child_ancestor_supports.append(child_weight)
                child_stable_shares.append(0.0)
                continue

            blocker_p = float(blocker_meta[child_id]["blocker_p"])
            distance_to_blocker = float(blocker_meta[child_id]["dist_blocker"])
            blocker_kernel = float(np.exp(-max(distance_to_blocker - 1.0, 0.0) / tau_b))
            ancestor_support = blocker_p

            child_log_k = float(
                np.log(
                    max(
                        _edge_structural_dimension(child_id, annotations_df, edge_spectral_dims),
                        1.0,
                    )
                )
            )

            if stable_nodes:
                stable_distances = np.asarray(
                    [tree_distance(child_id, stable_node) for stable_node in stable_nodes],
                    dtype=float,
                )
                tree_kernel = np.exp(-stable_distances / tau_t)
                structural_kernel = _structural_kernel(stable_log_ks_arr, child_log_k, h_k)
                neighbor_weights = tree_kernel * structural_kernel
                stable_mass = float(np.sum(neighbor_weights))
                if stable_mass > 0:
                    stable_support = float(np.average(stable_p_values_arr, weights=neighbor_weights))
                else:
                    stable_support = ancestor_support
            else:
                stable_mass = 0.0
                stable_support = ancestor_support

            support_numerator = blocker_kernel * blocker_p + stable_mass * stable_support
            support_denominator = blocker_kernel + stable_mass
            support = (
                float(support_numerator / support_denominator)
                if support_denominator > 0
                else float(ancestor_support)
            )

            if signal_nodes:
                signal_distances = np.asarray(
                    [tree_distance(child_id, signal_node) for signal_node in signal_nodes],
                    dtype=float,
                )
                signal_terms = (1.0 - signal_p_values_arr) * np.exp(-signal_distances / tau_s)
                signal_penalty = float(np.max(signal_terms)) if len(signal_terms) else 0.0
            else:
                signal_penalty = 0.0

            child_weight = float(np.clip(support * (1.0 - signal_penalty), 0.0, 1.0))
            stable_share = (
                float(stable_mass / (blocker_kernel + stable_mass))
                if (blocker_kernel + stable_mass) > 0
                else 0.0
            )

            child_weights.append(child_weight)
            child_supports.append(stable_support)
            child_ancestor_supports.append(ancestor_support)
            child_stable_shares.append(stable_share)

        record.edge_weight = min(child_weights) if child_weights else record.edge_weight
        record.nearby_stable_support = min(child_supports) if child_supports else None
        record.ancestor_support = min(child_ancestor_supports) if child_ancestor_supports else None
        record.blend_lambda = max(child_stable_shares) if child_stable_shares else None

    logger.debug(
        "Blocked-edge neighborhood enrichment: n_blocked_children=%d, tau_b=%.3f, "
        "tau_t=%.3f, tau_s=%.3f, h_k=%.6f, n_stable=%d, n_signal=%d.",
        len(blocked_child_ids),
        tau_b,
        tau_t,
        tau_s,
        h_k,
        len(stable_nodes),
        len(signal_nodes),
    )
    return records


__all__ = ["enrich_blocked_weights"]
