"""Shared plot package defaults."""

from benchmarks.shared.plots.backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib as mpl

# Prefer TrueType embedding (Type 42) for publication-quality PDFs.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
