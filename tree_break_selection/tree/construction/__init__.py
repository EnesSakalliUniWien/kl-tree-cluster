"""Tree-construction interface grouped by methodological responsibility."""

from tree_break_selection.tree.construction.build import (
    SUPPORTED_TREE_BUILDERS,
    TreeBuilderName,
    TreeBuildResult,
    build_tree,
)
from tree_break_selection.tree.construction.defaults import (
    DEFAULT_BINARY_TREE_DISTANCE_METRIC,
    DEFAULT_TREE_LINKAGE_METHOD,
)
from tree_break_selection.tree.construction.hierarchical import (
    tree_from_linkage,
    tree_from_linkage_topology,
)

__all__ = [
    "DEFAULT_BINARY_TREE_DISTANCE_METRIC",
    "DEFAULT_TREE_LINKAGE_METHOD",
    "SUPPORTED_TREE_BUILDERS",
    "TreeBuildResult",
    "TreeBuilderName",
    "build_tree",
    "tree_from_linkage",
    "tree_from_linkage_topology",
]
