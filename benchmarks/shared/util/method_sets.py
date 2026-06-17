"""Canonical benchmark method sets used across input and execution layers."""

DISTANCE_MATRIX_METHODS = {"leiden", "louvain", "dbscan", "optics", "hdbscan"}

KL_DISTANCE_TREE_METHODS = {
    "kl",
    "kl_complete",
    "kl_single",
    "kl_conditional_topology_diagnostic",
    "kl_global_passthrough_refined_diagnostic",
    "kl_spectral_transport_passthrough_diagnostic",
    "kl_spectral_transport_passthrough",
    "kl_legacy_internal_spectral_diagnostic",
    "kl_legacy_c2ef9a69",
    "kl_neighbor_joining",
}

KL_RUNNER_METHODS = KL_DISTANCE_TREE_METHODS | {"kl_iqtree3"}

__all__ = [
    "DISTANCE_MATRIX_METHODS",
    "KL_DISTANCE_TREE_METHODS",
    "KL_RUNNER_METHODS",
]
