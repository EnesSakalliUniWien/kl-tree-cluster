"""Tree-construction interface grouped by methodological responsibility."""

from tree_break_selection.tree.construction.build import (
    TreeBuildDiagnostics,
    TreeBuildResult,
    build_tree,
)
from tree_break_selection.tree.construction.defaults import (
    DEFAULT_BINARY_TREE_DISTANCE_METRIC,
    DEFAULT_LINKAGE_TREE_ROOTING,
    DEFAULT_PHYLOGENETIC_TREE_ROOTING,
    DEFAULT_TREE_LINKAGE_METHOD,
    IQTREE3_TREE_BUILDER,
    LINKAGE_TREE_BUILDER,
    NEIGHBOR_JOINING_TREE_BUILDER,
    SUPPORTED_LINKAGE_METHODS,
    SUPPORTED_TREE_BUILDERS,
    TREE_CONSENSUS_PRIORITY_ORDER,
    TREE_CONSENSUS_STRATEGIES,
    TREE_CONSENSUS_STRATEGY_PRIORITY,
    TreeBuilderName,
    TreeLinkageMethod,
)
from tree_break_selection.tree.construction.hierarchical import (
    tree_from_linkage,
)

__all__ = [
    "DEFAULT_BINARY_TREE_DISTANCE_METRIC",
    "DEFAULT_LINKAGE_TREE_ROOTING",
    "DEFAULT_PHYLOGENETIC_TREE_ROOTING",
    "DEFAULT_TREE_LINKAGE_METHOD",
    "IQTREE3_TREE_BUILDER",
    "LINKAGE_TREE_BUILDER",
    "NEIGHBOR_JOINING_TREE_BUILDER",
    "SUPPORTED_LINKAGE_METHODS",
    "SUPPORTED_TREE_BUILDERS",
    "TREE_CONSENSUS_PRIORITY_ORDER",
    "TREE_CONSENSUS_STRATEGIES",
    "TREE_CONSENSUS_STRATEGY_PRIORITY",
    "TreeBuildDiagnostics",
    "TreeBuildResult",
    "TreeBuilderName",
    "TreeLinkageMethod",
    "build_tree",
    "tree_from_linkage",
]
