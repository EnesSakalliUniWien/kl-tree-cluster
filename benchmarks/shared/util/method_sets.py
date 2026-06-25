"""Canonical benchmark method sets used across input and execution layers."""

DISTANCE_MATRIX_METHODS = {"leiden", "louvain", "dbscan", "optics", "hdbscan"}

TBS_DISTANCE_TREE_METHODS = {
    "tbs",
    "tbs_continuous_guarded_within_covariance",
    "tbs_complete",
    "tbs_single",
    "tbs_fixed_coordinate_bh",
    "tbs_fixed_coordinate_by",
    "tbs_fixed_coordinate_holm",
    "tbs_fixed_coordinate_bonferroni",
    "tbs_fixed_block_bh",
    "tbs_fixed_block_simes_bh",
    "tbs_conditional_topology_diagnostic",
    "tbs_global_passthrough_refined_diagnostic",
    "tbs_spectral_transport_passthrough_diagnostic",
    "tbs_spectral_transport_passthrough",
    "tbs_legacy_internal_spectral_diagnostic",
    "tbs_legacy_c2ef9a69",
    "tbs_internal_filter_v1",
    "tbs_internal_filter_branch_length_v1",
    "tbs_bandwidth_context_v1",
    "tbs_rescued_legacy_v1",
    "tbs_neighbor_joining",
}

TBS_RUNNER_METHODS = TBS_DISTANCE_TREE_METHODS | {"tbs_iqtree3"}

__all__ = [
    "DISTANCE_MATRIX_METHODS",
    "TBS_DISTANCE_TREE_METHODS",
    "TBS_RUNNER_METHODS",
]
