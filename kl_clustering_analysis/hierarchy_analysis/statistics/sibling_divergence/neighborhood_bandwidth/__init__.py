"""Guarded selected-neighborhood bandwidth primitives."""

from .decision import CoherentSupportDecision
from .kernel_weights import (
    effective_support,
    selected_neighborhood_kernel_weights,
    structural_log_dimension_kernel,
    tree_exponential_kernel,
)
from .support_roles import (
    NULL_CALIBRATION_ROLES,
    SupportRole,
    classify_support_role,
    role_allows_empirical_null_calibration,
)
from .tau_regions import TauRegionEstimate, TauRegionKey, estimate_tau_region
from .tree_distance import BranchLengthDistanceCache, build_branch_length_distance_cache

__all__ = [
    "BranchLengthDistanceCache",
    "CoherentSupportDecision",
    "NULL_CALIBRATION_ROLES",
    "SupportRole",
    "TauRegionEstimate",
    "TauRegionKey",
    "build_branch_length_distance_cache",
    "classify_support_role",
    "effective_support",
    "estimate_tau_region",
    "role_allows_empirical_null_calibration",
    "selected_neighborhood_kernel_weights",
    "structural_log_dimension_kernel",
    "tree_exponential_kernel",
]
