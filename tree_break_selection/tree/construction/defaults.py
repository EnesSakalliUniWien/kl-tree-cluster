"""Tree-construction method names owned by the tree module.

These values describe the supported construction seams. They are constants,
not mutable runtime configuration; callers with a different geometry or
topology policy must pass it explicitly.
"""

from __future__ import annotations

from typing import Literal

TreeBuilderName = Literal["linkage", "neighbor_joining", "iqtree3"]
TreeLinkageMethod = Literal[
    "average",
    "complete",
    "weighted",
    "single",
    "centroid",
    "median",
    "ward",
]

LINKAGE_TREE_BUILDER = "linkage"
NEIGHBOR_JOINING_TREE_BUILDER = "neighbor_joining"
IQTREE3_TREE_BUILDER = "iqtree3"
DEFAULT_BINARY_TREE_DISTANCE_METRIC = "hamming"
DEFAULT_TREE_LINKAGE_METHOD = "average"
DEFAULT_LINKAGE_TREE_ROOTING = "linkage_root"
DEFAULT_PHYLOGENETIC_TREE_ROOTING = "mad"

SUPPORTED_TREE_BUILDERS: tuple[TreeBuilderName, ...] = (
    LINKAGE_TREE_BUILDER,
    NEIGHBOR_JOINING_TREE_BUILDER,
    IQTREE3_TREE_BUILDER,
)
SUPPORTED_LINKAGE_METHODS: tuple[TreeLinkageMethod, ...] = (
    "average",
    "complete",
    "weighted",
    "single",
    "centroid",
    "median",
    "ward",
)
TREE_CONSENSUS_STRATEGIES: tuple[str, ...] = (
    *SUPPORTED_LINKAGE_METHODS,
    NEIGHBOR_JOINING_TREE_BUILDER,
)
TREE_CONSENSUS_PRIORITY_ORDER: tuple[str, ...] = (
    "weighted",
    "ward",
    "complete",
    "average",
    "centroid",
    "median",
    NEIGHBOR_JOINING_TREE_BUILDER,
    "single",
)
TREE_CONSENSUS_STRATEGY_PRIORITY: dict[str, int] = {
    strategy: index for index, strategy in enumerate(TREE_CONSENSUS_PRIORITY_ORDER)
}

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
    "TreeBuilderName",
    "TreeLinkageMethod",
]
