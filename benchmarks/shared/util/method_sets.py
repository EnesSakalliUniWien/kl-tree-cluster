"""Canonical benchmark method sets used across input and execution layers."""

DISTANCE_MATRIX_METHODS = {"leiden", "louvain", "dbscan", "optics", "hdbscan"}

KL_DISTANCE_TREE_METHODS = {
    "kl",
    "kl_complete",
    "kl_single",
    "kl_conditional_topology_diagnostic",
    "kl_neighbor_joining",
}

KL_RUNNER_METHODS = KL_DISTANCE_TREE_METHODS | {"kl_iqtree3"}

__all__ = [
    "DISTANCE_MATRIX_METHODS",
    "KL_DISTANCE_TREE_METHODS",
    "KL_RUNNER_METHODS",
]
