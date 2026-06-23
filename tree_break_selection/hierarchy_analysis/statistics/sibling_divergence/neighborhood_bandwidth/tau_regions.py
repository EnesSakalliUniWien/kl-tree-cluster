"""Region-conditioned tau estimation for selected-neighborhood smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TauRegionKey:
    """Topology region used to estimate local neighborhood bandwidths."""

    root_mode: str
    candidate_scope: str
    feature_family: str
    projection_dim_band: str
    topology_state: str
    branch_length_scale_band: str

    def parent(self) -> "TauRegionKey | None":
        """Return the next coarser region used for shrinkage fallback."""

        if self.branch_length_scale_band != "*":
            return TauRegionKey(
                self.root_mode,
                self.candidate_scope,
                self.feature_family,
                self.projection_dim_band,
                self.topology_state,
                "*",
            )
        if self.topology_state != "*":
            return TauRegionKey(
                self.root_mode,
                self.candidate_scope,
                self.feature_family,
                self.projection_dim_band,
                "*",
                "*",
            )
        if self.projection_dim_band != "*":
            return TauRegionKey(
                self.root_mode,
                self.candidate_scope,
                self.feature_family,
                "*",
                "*",
                "*",
            )
        if self.feature_family != "*":
            return TauRegionKey(
                self.root_mode,
                self.candidate_scope,
                "*",
                "*",
                "*",
                "*",
            )
        if self.candidate_scope != "*":
            return TauRegionKey(self.root_mode, "*", "*", "*", "*", "*")
        if self.root_mode != "*":
            return TauRegionKey("*", "*", "*", "*", "*", "*")
        return None


@dataclass(frozen=True)
class TauRegionEstimate:
    """Estimated regional bandwidth set."""

    region: TauRegionKey
    tau_b: float
    tau_t: float
    tau_s: float
    h_k: float
    n_support: int
    n_signal: int
    effective_support: float
    borrowed_from_region: TauRegionKey | None
    tau_status: str

    @property
    def stable_for_promotion(self) -> bool:
        return self.tau_status == "local_stable"


def estimate_tau_region(
    *,
    region: TauRegionKey,
    stopping_edge_distances: Iterable[float],
    stable_neighbor_distances: Iterable[float],
    signal_neighbor_distances: Iterable[float],
    stable_log_ks: Iterable[float],
    n_support: int,
    n_signal: int,
    effective_support: float,
    parent_estimate: TauRegionEstimate | None = None,
    min_effective_support: float = 2.0,
    shrinkage_strength: float = 4.0,
    default_tau: float = 1.0,
) -> TauRegionEstimate:
    """Estimate a local tau set and shrink unstable regions to a parent."""

    local_tau_b = _safe_positive_median(stopping_edge_distances, default_tau)
    local_tau_t = _safe_positive_median(stable_neighbor_distances, default_tau)
    local_tau_s = _safe_positive_median(signal_neighbor_distances, default_tau)
    local_h_k = _safe_nonnegative_std(stable_log_ks)
    n_eff = float(effective_support)
    local_stable = n_eff >= float(min_effective_support) and int(n_support) > 0

    if local_stable or parent_estimate is None:
        status = "local_stable" if local_stable else "borrowed_or_unstable"
        borrowed = None
        return TauRegionEstimate(
            region=region,
            tau_b=local_tau_b,
            tau_t=local_tau_t,
            tau_s=local_tau_s,
            h_k=local_h_k,
            n_support=int(n_support),
            n_signal=int(n_signal),
            effective_support=n_eff,
            borrowed_from_region=borrowed,
            tau_status=status,
        )

    alpha = n_eff / (n_eff + float(shrinkage_strength))
    return TauRegionEstimate(
        region=region,
        tau_b=_shrink(local_tau_b, parent_estimate.tau_b, alpha),
        tau_t=_shrink(local_tau_t, parent_estimate.tau_t, alpha),
        tau_s=_shrink(local_tau_s, parent_estimate.tau_s, alpha),
        h_k=_shrink(local_h_k, parent_estimate.h_k, alpha),
        n_support=int(n_support),
        n_signal=int(n_signal),
        effective_support=n_eff,
        borrowed_from_region=parent_estimate.region,
        tau_status="borrowed_or_unstable",
    )


def _safe_positive_median(values: Iterable[float], default: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value) and float(value) > 0.0]
    if not finite:
        return float(default)
    return max(
        float(np.median(np.asarray(finite, dtype=np.float64))),
        float(np.finfo(np.float64).eps),
    )


def _safe_nonnegative_std(values: Iterable[float]) -> float:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size <= 1:
        return 0.0
    return max(float(np.std(finite)), 0.0)


def _shrink(local_value: float, parent_value: float, alpha: float) -> float:
    return float(alpha) * float(local_value) + (1.0 - float(alpha)) * float(parent_value)


__all__ = [
    "TauRegionEstimate",
    "TauRegionKey",
    "estimate_tau_region",
]
