"""
Benchmarking helpers for running clustering test suites end-to-end.

This package exposes the entry point that lived in ``tests/validation_utils``.
"""

from .pipeline import benchmark_cluster_algorithm
from .utils_decomp import _labels_from_decomposition

__all__ = [
    "benchmark_cluster_algorithm",
    "_labels_from_decomposition",
]
