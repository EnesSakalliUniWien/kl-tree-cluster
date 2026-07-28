"""Directory-independent dispatch for supported tree-construction methods."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

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
class TreeBuildDiagnostics:
    """Structural and distance evidence for one constructed tree."""

    leaf_count: int
    internal_node_count: int
    edge_count: int
    root_child_count: int
    zero_branch_length_count: int
    distance_pair_count: int | None
    zero_distance_pair_count: int | None
    repeated_distance_pair_count: int | None
    repeated_distance_value_count: int | None
    repeated_linkage_height_count: int | None


@dataclass(frozen=True)
class TreeBuildResult:
    """Tree plus construction evidence needed by runners and reports."""

    tree: PosetTree
    builder: TreeBuilderName
    rooting: str
    diagnostics: TreeBuildDiagnostics
    linkage_matrix: np.ndarray | None = None
    phylogenetic_rooting: MadRootResult | None = None
    iqtree_metadata: dict[str, object] | None = None
    topology_only_branch_lengths: bool = False
    topology_only_reason: str | None = None


def _validate_condensed_distance(
    distance_condensed: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    """Return a validated condensed distance vector."""
    distances = np.asarray(distance_condensed, dtype=float)
    expected_pairs = sample_count * (sample_count - 1) // 2
    if distances.ndim != 1 or len(distances) != expected_pairs:
        raise ValueError(
            "distance_condensed must contain exactly one value per unordered sample pair; "
            f"expected {expected_pairs}, got shape={distances.shape!r}."
        )
    if not np.isfinite(distances).all():
        raise ValueError("distance_condensed contains non-finite values.")
    if np.any(distances < 0.0):
        raise ValueError("distance_condensed contains negative values.")
    return distances


def _canonical_sample_order(labels: list[object]) -> list[int]:
    """Return a row-order-independent sample order for deterministic tie breaks."""
    if all(isinstance(label, str) for label in labels):
        def natural_string_key(position: int) -> tuple[tuple[tuple[int, object], ...], str]:
            label = labels[position]
            assert isinstance(label, str)
            chunks = tuple(
                (0, int(chunk)) if chunk.isdigit() else (1, chunk)
                for chunk in re.split(r"(\d+)", label)
            )
            return chunks, label

        return sorted(range(len(labels)), key=natural_string_key)

    try:
        return sorted(range(len(labels)), key=labels.__getitem__)
    except TypeError:
        pass

    keys = [
        (
            type(label).__module__,
            type(label).__qualname__,
            repr(label),
        )
        for label in labels
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Mixed-type sample labels must have distinct stable type/repr keys for "
            "deterministic tree construction."
        )
    return sorted(range(len(labels)), key=keys.__getitem__)


def _reorder_condensed_distance(
    distances: np.ndarray,
    order: list[int],
) -> np.ndarray:
    """Reorder a condensed distance vector without changing its values."""
    if order == list(range(len(order))):
        return distances
    square = squareform(distances)
    reordered = square[np.ix_(order, order)]
    return squareform(reordered, checks=False)


def _validate_decomposition_tree(
    tree: PosetTree,
    *,
    expected_leaf_labels: list[object],
) -> None:
    """Enforce the rooted full-binary contract required by TBS traversal."""
    if not nx.is_arborescence(tree):
        raise ValueError("Tree construction must return one directed rooted arborescence.")

    roots = [node for node, degree in tree.in_degree() if degree == 0]
    if len(roots) != 1 or roots[0] != tree.root():
        raise ValueError(
            "Tree construction must return one cached root matching the directed root; "
            f"roots={roots!r}, cached_root={tree.root()!r}."
        )

    invalid_internal_nodes = [
        node for node in tree.nodes if tree.out_degree(node) not in {0, 2}
    ]
    if invalid_internal_nodes:
        raise ValueError(
            "TBS tree construction requires every internal node to have exactly two children; "
            f"invalid_nodes={invalid_internal_nodes[:5]!r}."
        )

    observed_leaf_labels = [
        tree.nodes[node].get("label")
        for node in tree.nodes
        if tree.out_degree(node) == 0
    ]
    if Counter(observed_leaf_labels) != Counter(expected_leaf_labels):
        raise ValueError(
            "Constructed tree leaves do not match the input sample labels exactly; "
            f"expected={expected_leaf_labels!r}, observed={observed_leaf_labels!r}."
        )

    invalid_edges: list[tuple[object, object, object]] = []
    for parent, child, attrs in tree.edges(data=True):
        branch_length = attrs.get("branch_length")
        if branch_length is None:
            invalid_edges.append((parent, child, None))
            continue
        value = float(branch_length)
        if not np.isfinite(value) or value < 0.0:
            invalid_edges.append((parent, child, branch_length))
    if invalid_edges:
        raise ValueError(
            "Constructed tree branch lengths must be finite and non-negative; "
            f"invalid_edges={invalid_edges[:5]!r}."
        )


def _build_diagnostics(
    tree: PosetTree,
    *,
    distances: np.ndarray | None,
    linkage_matrix: np.ndarray | None,
) -> TreeBuildDiagnostics:
    """Summarize tree shape and exact tie burden without another tree fit."""
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    zero_branch_length_count = sum(
        float(attrs["branch_length"]) == 0.0
        for _parent, _child, attrs in tree.edges(data=True)
    )

    distance_pair_count: int | None = None
    zero_distance_pair_count: int | None = None
    repeated_distance_pair_count: int | None = None
    repeated_distance_value_count: int | None = None
    if distances is not None:
        _values, counts = np.unique(distances, return_counts=True)
        repeated = counts > 1
        distance_pair_count = int(len(distances))
        zero_distance_pair_count = int(np.count_nonzero(distances == 0.0))
        repeated_distance_pair_count = int(counts[repeated].sum())
        repeated_distance_value_count = int(np.count_nonzero(repeated))

    repeated_linkage_height_count: int | None = None
    if linkage_matrix is not None:
        _heights, height_counts = np.unique(linkage_matrix[:, 2], return_counts=True)
        repeated_linkage_height_count = int(height_counts[height_counts > 1].sum())

    return TreeBuildDiagnostics(
        leaf_count=len(leaves),
        internal_node_count=tree.number_of_nodes() - len(leaves),
        edge_count=tree.number_of_edges(),
        root_child_count=tree.out_degree(tree.root()),
        zero_branch_length_count=zero_branch_length_count,
        distance_pair_count=distance_pair_count,
        zero_distance_pair_count=zero_distance_pair_count,
        repeated_distance_pair_count=repeated_distance_pair_count,
        repeated_distance_value_count=repeated_distance_value_count,
        repeated_linkage_height_count=repeated_linkage_height_count,
    )


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
    if len(data) < 2:
        raise ValueError("Tree construction requires at least two samples.")
    if not data.index.is_unique:
        raise ValueError("Tree construction requires unique sample labels.")
    if builder not in SUPPORTED_TREE_BUILDERS:
        supported = ", ".join(SUPPORTED_TREE_BUILDERS)
        raise ValueError(f"Unsupported tree builder {builder!r}; expected one of: {supported}.")

    distances: np.ndarray | None = None
    linkage_matrix: np.ndarray | None = None
    phylogenetic_rooting: MadRootResult | None = None
    iqtree_metadata: dict[str, object] | None = None
    topology_only_branch_lengths = False
    topology_only_reason: str | None = None
    expected_leaf_labels: list[object]
    original_labels = data.index.tolist()
    canonical_order = _canonical_sample_order(original_labels)
    canonical_data = data.iloc[canonical_order]

    if builder == "linkage":
        if rooting != "linkage_root":
            raise ValueError("Linkage trees require rooting='linkage_root'.")
        if distance_condensed is None:
            raise ValueError("Linkage tree construction requires distance_condensed.")

        distances = _validate_condensed_distance(
            distance_condensed,
            sample_count=len(data),
        )
        distances = _reorder_condensed_distance(distances, canonical_order)
        linkage_matrix = linkage(distances, method=linkage_method)
        try:
            tree = tree_from_linkage(
                linkage_matrix,
                leaf_names=canonical_data.index.tolist(),
            )
        except ValueError as exc:
            if not allow_topology_only_linkage or "nondecreasing" not in str(exc):
                raise
            tree = tree_from_linkage_topology(
                linkage_matrix,
                leaf_names=canonical_data.index.tolist(),
                fallback_branch_length=1.0,
            )
            topology_only_branch_lengths = True
            topology_only_reason = str(exc)
        expected_leaf_labels = canonical_data.index.tolist()
    else:
        if rooting != "mad":
            raise ValueError(f"{builder} trees require rooting='mad'.")

        expected_leaf_labels = canonical_data.index.astype(str).tolist()
        if len(set(expected_leaf_labels)) != len(expected_leaf_labels):
            raise ValueError(
                "Phylogenetic tree construction requires sample labels that remain unique "
                "after string conversion."
            )
        if builder == "neighbor_joining":
            if distance_condensed is None:
                raise ValueError("Neighbor-joining tree construction requires distance_condensed.")
            distances = _validate_condensed_distance(
                distance_condensed,
                sample_count=len(data),
            )
            distances = _reorder_condensed_distance(distances, canonical_order)
            tree, phylogenetic_rooting = neighbor_joining_tree_from_distance(
                distances,
                expected_leaf_labels,
                rooting=rooting,
            )
        else:
            tree, phylogenetic_rooting, iqtree_metadata = iqtree3_tree_from_alignment(
                canonical_data,
                executable=iqtree_executable,
                model=iqtree_model,
                threads=iqtree_threads,
                rooting=rooting,
                work_dir=iqtree_work_dir,
            )

    _validate_decomposition_tree(
        tree,
        expected_leaf_labels=expected_leaf_labels,
    )
    diagnostics = _build_diagnostics(
        tree,
        distances=distances,
        linkage_matrix=linkage_matrix,
    )
    return TreeBuildResult(
        tree=tree,
        builder=builder,
        rooting=rooting,
        diagnostics=diagnostics,
        linkage_matrix=linkage_matrix,
        phylogenetic_rooting=phylogenetic_rooting,
        iqtree_metadata=iqtree_metadata,
        topology_only_branch_lengths=topology_only_branch_lengths,
        topology_only_reason=topology_only_reason,
    )


__all__ = [
    "SUPPORTED_TREE_BUILDERS",
    "TreeBuildDiagnostics",
    "TreeBuildResult",
    "TreeBuilderName",
    "build_tree",
]
