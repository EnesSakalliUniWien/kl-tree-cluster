"""Benchmarking runner package.

Important: do not import optional-dependency runners (igraph/leidenalg/etc.) at
import time.

The benchmarking pipeline loads runners lazily via ``importlib``. Importing
optional runners here makes importing *any* runner (even KL) fail when optional
dependencies aren't installed, because Python executes this package
``__init__.py`` before importing submodules (e.g. ``.kl_runner``).
"""

from __future__ import annotations

__all__ = [
    "_run_dbscan_method",
    "_run_hdbscan_method",
    "_run_kmeans_method",
    "_run_leiden_method",
    "_run_optics_method",
    "_run_spectral_method",
    "_run_louvain_method",
    "_run_kl_method",
]
