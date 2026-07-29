"""Reusable plotting engines for cluster colors, trees, and report panels."""

from .backend import DEFAULT_FILE_BACKEND, configure_matplotlib_backend

configure_matplotlib_backend()

__all__ = [
    "DEFAULT_FILE_BACKEND",
    "configure_matplotlib_backend",
]
