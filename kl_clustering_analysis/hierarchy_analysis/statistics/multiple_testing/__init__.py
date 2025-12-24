"""Multiple testing correction utilities for statistical hypothesis testing.

This package provides methods for controlling false discovery rate (FDR) and
family-wise error rate (FWER) when performing multiple hypothesis tests.

Modules
-------
base
    Core Benjamini-Hochberg FDR correction
flat_correction
    Standard flat BH across all p-values
level_wise_correction
    BH applied separately at each tree level
tree_bh_correction
    Hierarchical TreeBH (Bogomolov et al. 2021)
dispatcher
    Unified interface for selecting correction method
"""

from .base import benjamini_hochberg_correction
from .flat_correction import flat_bh_correction
from .level_wise_correction import level_wise_bh_correction
from .tree_bh_correction import tree_bh_correction, TreeBHResult
from .dispatcher import apply_multiple_testing_correction

__all__ = [
    # Core function
    "benjamini_hochberg_correction",
    # Correction methods
    "flat_bh_correction",
    "level_wise_bh_correction",
    "tree_bh_correction",
    "TreeBHResult",
    # Dispatcher
    "apply_multiple_testing_correction",
]
