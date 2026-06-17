"""Adaptive kernel bandwidth helpers for tree-neighborhood weight smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .edge_metadata import StoppingEdgeSummary

_EPS = 1e-12


@dataclass(frozen=True)
class AdaptiveKernelBandwidths:
    """Tree-specific kernel bandwidths for edge-weight smoothing."""

    tau_b: float
    tau_t: float
    tau_s: float
    h_k: float


def safe_positive_bandwidth(values: list[float], default: float) -> float:
    """Return a positive finite bandwidth from a sample list."""
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0]
    if not finite:
        return float(default)
    return max(float(np.median(finite)), _EPS)


def compute_adaptive_kernel_bandwidths(
    *,
    stopping_edge_info: dict[str, StoppingEdgeSummary],
    child_ids: list[str],
    stable_nodes: list[str],
    signal_nodes: list[str],
    stable_log_ks: np.ndarray,
    tree_distance: Callable[[str, str], float],
) -> AdaptiveKernelBandwidths:
    """Infer kernel bandwidths from stopping-edge, stable, and signal neighborhoods."""
    tau_b = safe_positive_bandwidth(
        [meta.distance_to_stopping_edge for meta in stopping_edge_info.values()],
        default=1.0,
    )

    nearest_stable_distances: list[float] = []
    if stable_nodes:
        for child_id in child_ids:
            distances = [tree_distance(child_id, stable_child) for stable_child in stable_nodes]
            nearest_stable_distances.append(min(distances))
    tau_t = safe_positive_bandwidth(nearest_stable_distances, default=1.0)

    nearest_signal_distances: list[float] = []
    if signal_nodes:
        for child_id in child_ids:
            distances = [tree_distance(child_id, signal_child) for signal_child in signal_nodes]
            nearest_signal_distances.append(min(distances))
    tau_s = safe_positive_bandwidth(nearest_signal_distances, default=1.0)

    h_k = float(np.std(stable_log_ks)) if len(stable_log_ks) > 1 else 0.0

    return AdaptiveKernelBandwidths(
        tau_b=tau_b,
        tau_t=tau_t,
        tau_s=tau_s,
        h_k=max(h_k, 0.0),
    )


def structural_kernel(log_k_source: np.ndarray, log_k_target: float, h_k: float) -> np.ndarray:
    """Return Gaussian-in-log-k weights, with exact-match behavior at zero spread."""
    if h_k <= 0:
        return np.where(np.isclose(log_k_source, log_k_target, atol=1e-12), 1.0, 0.0)
    normalized = (log_k_source - log_k_target) / h_k
    return np.exp(-0.5 * normalized**2)


__all__ = [
    "AdaptiveKernelBandwidths",
    "compute_adaptive_kernel_bandwidths",
    "safe_positive_bandwidth",
    "structural_kernel",
]
