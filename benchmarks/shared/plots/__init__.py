"""Shared plot package defaults."""

import matplotlib as mpl

# Prefer TrueType embedding (Type 42) for publication-quality PDFs.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
