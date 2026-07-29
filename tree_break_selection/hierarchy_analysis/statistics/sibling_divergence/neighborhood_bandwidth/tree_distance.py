"""Branch-length-aware tree distance cache for selected-neighborhood kernels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable

import networkx as nx
import numpy as np

from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    extract_branch_length_observation,
)


@dataclass(frozen=True)
class BranchLengthDistanceCache:
    """All-pairs undirected tree distances from explicit branch lengths."""

    distances: dict[tuple[Hashable, Hashable], float]
    status: str

    def distance(self, source: Hashable, target: Hashable) -> float:
        """Return the cached distance for a connected pair of nodes."""

        if source == target:
            return 0.0
        key = _distance_key(source, target)
        try:
            return float(self.distances[key])
        except KeyError as exc:
            raise ValueError(
                "Branch-length distance requested for disconnected or missing tree nodes; "
                f"source={source!r}, target={target!r}."
            ) from exc


def build_branch_length_distance_cache(
    tree: nx.DiGraph,
) -> BranchLengthDistanceCache:
    """Build weighted shortest-path distances on the undirected tree skeleton.

    Every edge must provide a finite non-negative ``branch_length``. Missing or
    invalid branch lengths are construction errors because neighborhood
    bandwidth decisions otherwise silently switch geometry.
    """

    graph = nx.Graph()
    observed_edges = 0
    invalid_edges: list[tuple[Hashable, Hashable, object]] = []

    for parent_id, child_id in tree.edges():
        branch_length = extract_branch_length_observation(tree, parent_id, child_id)
        if branch_length is None:
            invalid_edges.append((parent_id, child_id, None))
            continue
        weight = float(branch_length)
        if not np.isfinite(weight) or weight < 0.0:
            invalid_edges.append((parent_id, child_id, branch_length))
            continue
        observed_edges += 1
        graph.add_edge(parent_id, child_id, weight=weight)

    if invalid_edges:
        raise ValueError(
            "Tree-distance cache requires explicit finite non-negative branch_length on "
            f"every edge; invalid_edges={invalid_edges[:5]!r}."
        )
    if tree.number_of_edges() > 0 and observed_edges == 0:
        raise ValueError("Tree-distance cache received a tree with no branch-length edges.")
    if tree.number_of_edges() > 0 and compute_mean_branch_length(tree) is None:
        raise ValueError("Tree-distance cache requires at least one positive branch length.")

    distances: dict[tuple[Hashable, Hashable], float] = {}
    for component in nx.connected_components(graph):
        component_nodes = list(component)
        path_lengths = dict(
            nx.all_pairs_dijkstra_path_length(
                graph.subgraph(component_nodes),
                weight="weight",
            )
        )
        for source, target in combinations(component_nodes, 2):
            distances[_distance_key(source, target)] = float(path_lengths[source][target])

    return BranchLengthDistanceCache(
        distances=distances,
        status="branch_length_observed",
    )


def _distance_key(source: Hashable, target: Hashable) -> tuple[Hashable, Hashable]:
    return tuple(sorted((source, target), key=repr))


__all__ = [
    "BranchLengthDistanceCache",
    "build_branch_length_distance_cache",
]
