"""Tree-construction defaults owned by the tree module.

These values describe the canonical binary-feature linkage route.  They are
constants, not mutable runtime configuration; callers with a different
geometry or topology policy must pass it explicitly.
"""

DEFAULT_BINARY_TREE_DISTANCE_METRIC = "hamming"
DEFAULT_TREE_LINKAGE_METHOD = "average"

__all__ = [
    "DEFAULT_BINARY_TREE_DISTANCE_METRIC",
    "DEFAULT_TREE_LINKAGE_METHOD",
]
