"""Shared benchmark configuration constants.

These constants define the canonical default benchmark configuration used by
both the full runner and the shared benchmarking pipeline.
"""

from __future__ import annotations

DEFAULT_METHODS: tuple[str, ...] = (
    "tbs",
    "tbs_diffusion",
    "tbs_diffusion_adaptive_nnls",
    "leiden",
    "louvain",
    "kmeans",
    "spectral",
    "dbscan",
    "optics",
    "hdbscan",
)

__all__ = ["DEFAULT_METHODS"]
