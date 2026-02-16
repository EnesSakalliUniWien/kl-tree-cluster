"""Shared benchmark configuration constants."""

from __future__ import annotations

DEFAULT_METHODS: tuple[str, ...] = (
    "kl",
    "leiden",
    "louvain",
    "kmeans",
    "spectral",
    "dbscan",
    "optics",
    "hdbscan",
)

__all__ = ["DEFAULT_METHODS"]
