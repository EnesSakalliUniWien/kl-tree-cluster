"""
KL-divergence based hierarchical clustering analysis.

This package provides tools for hierarchical clustering analysis using
KL-divergence measures with NetworkX and advanced visualization capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core modules
from . import hierarchy_analysis
from . import plot
from . import tree

__all__ = [
    "hierarchy_analysis",
    "plot",
    "tree",
]
