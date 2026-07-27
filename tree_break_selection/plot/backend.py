"""Matplotlib backend configuration for noninteractive plotting."""

from __future__ import annotations

import os

DEFAULT_FILE_BACKEND = "Agg"


def configure_matplotlib_backend() -> str:
    """Select a noninteractive backend before importing ``pyplot``.

    Full benchmark plots are written to files from scripts and spawned worker
    processes. On macOS, the default GUI backend can abort the interpreter when
    figures are created from that noninteractive path. Respect an explicit
    ``MPLBACKEND`` override, otherwise use the file-safe Agg backend.
    """
    backend = os.environ.setdefault("MPLBACKEND", DEFAULT_FILE_BACKEND)

    import matplotlib as mpl

    current_backend = str(mpl.get_backend()).lower()
    desired_backend = str(backend).lower()
    if current_backend != desired_backend:
        mpl.use(backend, force=True)
    return backend


__all__ = ["DEFAULT_FILE_BACKEND", "configure_matplotlib_backend"]
