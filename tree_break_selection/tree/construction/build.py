"""Directory-independent dispatch for supported tree-construction methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from tree_break_selection.tree.construction.hierarchical import (
    tree_from_linkage,
    tree_from_linkage_topology,
)
from tree_break_selection.tree.construction.phylogenetic import (
    MadRootResult,
    iqtree3_tree_from_alignment,
    neighbor_joining_tree_from_distance,
)
from tree_break_selection.tree.poset_tree import PosetTree

TreeBuilderName = Literal["linkage", "neighbor_joining", "iqtree3"]
SUPPORTED_TREE_BUILDERS: tuple[TreeBuilderName, ...] = (
    "linkage",
    "neighbor_joining",
    "iqtree3",
)


@dataclass(frozen=True)
class TreeBuildResult:
    """Tree plus construction evidence needed by runners and reports."""

    tree: PosetTree
    builder: TreeBuilderName
    rooting: str
    linkage_matrix: np.ndarray | None = None
    phylogenetic_rooting: MadRootResult | None = None
    iqtree_metadata: dict[str, object] | None = None
    topology_only_branch_lengths: bool = False
    topology_only_reason: str | None = None


def build_tree(
    data: pd.DataFrame,
    distance_condensed: np.ndarray | None,
    *,
    builder: str,
    rooting: str,
    linkage_method: str,
    allow_topology_only_linkage: bool = False,
    iqtree_executable: str = "iqtree3",
    iqtree_model: str = "JC2",
    iqtree_threads: int = 1,
    iqtree_work_dir: str | None = None,
) -> TreeBuildResult:
    """Build one supported topology through an explicit methodological seam.

    Builder selection is based only on ``builder``. Package layout, filesystem
    ordering, and import discovery never participate in method selection.
    """
    if builder not in SUPPORTED_TREE_BUILDERS:
        supported = ", ".join(SUPPORTED_TREE_BUILDERS)
        raise ValueError(f"Unsupported tree builder {builder!r}; expected one of: {supported}.")

    if builder == "linkage":
        if rooting != "linkage_root":
            raise ValueError("Linkage trees require rooting='linkage_root'.")
        if distance_condensed is None:
            raise ValueError("Linkage tree construction requires distance_condensed.")

        linkage_matrix = linkage(distance_condensed, method=linkage_method)
        try:
            tree = tree_from_linkage(linkage_matrix, leaf_names=data.index.tolist())
        except ValueError as exc:
            if not allow_topology_only_linkage or "nondecreasing" not in str(exc):
                raise
            tree = tree_from_linkage_topology(
                linkage_matrix,
                leaf_names=data.index.tolist(),
                fallback_branch_length=1.0,
            )
            return TreeBuildResult(
                tree=tree,
                builder="linkage",
                rooting=rooting,
                linkage_matrix=linkage_matrix,
                topology_only_branch_lengths=True,
                topology_only_reason=str(exc),
            )
        return TreeBuildResult(
            tree=tree,
            builder="linkage",
            rooting=rooting,
            linkage_matrix=linkage_matrix,
        )

    if rooting != "mad":
        raise ValueError(f"{builder} trees require rooting='mad'.")

    if builder == "neighbor_joining":
        if distance_condensed is None:
            raise ValueError("Neighbor-joining tree construction requires distance_condensed.")
        tree, root = neighbor_joining_tree_from_distance(
            distance_condensed,
            data.index.astype(str).tolist(),
            rooting=rooting,
        )
        return TreeBuildResult(
            tree=tree,
            builder="neighbor_joining",
            rooting=rooting,
            phylogenetic_rooting=root,
        )

    tree, root, metadata = iqtree3_tree_from_alignment(
        data,
        executable=iqtree_executable,
        model=iqtree_model,
        threads=iqtree_threads,
        rooting=rooting,
        work_dir=iqtree_work_dir,
    )
    return TreeBuildResult(
        tree=tree,
        builder="iqtree3",
        rooting=rooting,
        phylogenetic_rooting=root,
        iqtree_metadata=metadata,
    )


__all__ = [
    "SUPPORTED_TREE_BUILDERS",
    "TreeBuildResult",
    "TreeBuilderName",
    "build_tree",
]
