"""Guarded final decision object for neighborhood support evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoherentSupportDecision:
    """Final guarded support decision for non-direct split recovery candidates."""

    root_usable_or_nonroot: bool
    topology_coherent: bool
    regional_tau_stable: bool
    spectral_flow_supported: bool
    empirical_null_admissible_or_not_required: bool
    hard_negative_control_leak: bool = False
    direct_measurable_split: bool = False
    action_if_supported: str = "promote_guarded_non_direct_split"

    @property
    def promotion_eligible(self) -> bool:
        """Return whether all guards allow the split support action."""

        return self.dominant_blocker == ""

    @property
    def dominant_blocker(self) -> str:
        """Return the first fail-closed blocker in decision order."""

        if self.direct_measurable_split:
            return "direct_measurable_split_not_neighborhood_rescue"
        if self.hard_negative_control_leak:
            return "hard_negative_control_leak"
        if not self.root_usable_or_nonroot:
            return "root_invalid_or_unusable"
        if not self.topology_coherent:
            return "topology_not_coherent"
        if not self.regional_tau_stable:
            return "regional_tau_borrowed_or_unstable"
        if not self.spectral_flow_supported:
            return "spectral_flow_not_supported"
        if not self.empirical_null_admissible_or_not_required:
            return "empirical_null_calibration_not_admissible"
        return ""

    @property
    def method_action(self) -> str:
        """Return the guarded method action for reporting and benchmarks."""

        blocker = self.dominant_blocker
        if not blocker:
            return self.action_if_supported
        if blocker == "root_invalid_or_unusable":
            return "fail_closed_root_invalid"
        if blocker == "hard_negative_control_leak":
            return "fail_closed_hard_negative_control"
        return f"fail_closed_{blocker}"


__all__ = ["CoherentSupportDecision"]
