"""Spectral dimension estimation subpackage.

Provides per-node Marchenko-Pastur spectral dimension computation,
tree traversal helpers, and typed worker payloads.
"""

from .tree_estimator import compute_spectral_decomposition

__all__ = ["compute_spectral_decomposition"]
