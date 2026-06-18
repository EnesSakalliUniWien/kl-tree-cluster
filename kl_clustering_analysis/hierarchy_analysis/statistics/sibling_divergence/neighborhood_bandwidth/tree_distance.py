"""Branch-length-aware tree distance cache for selected-neighborhood kernels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable

import networkx as nx
import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    extract_branch_length_observation,
)


@dataclass(frozen=True)
class BranchLengthDistanceCache:
    """All-pairs undirected tree distances with deterministic branch fallback."""

    distances: dict[tuple[Hashable, Hashable], float]
    fallback_edge_length: float
    status: str

    def distance(self, source: Hashable, target: Hashable) -> float:
        """Return cached distance or infinity for disconnected/missing nodes."""

        if source == target:
            return 0.0
        key = _distance_key(source, target)
        return float(self.distances.get(key, np.inf))


def build_branch_length_distance_cache(
    tree: nx.DiGraph,
) -> BranchLengthDistanceCache:
    """Build weighted shortest-path distances on the undirected tree skeleton.

    Positive ``branch_length`` observations are used directly. Missing edge
    lengths fall back to the mean positive branch length in the tree, and then
    to ``1.0`` when the tree has no positive branch-length observations.
    """

    mean_branch_length = compute_mean_branch_length(tree)
    fallback_edge_length = 1.0 if mean_branch_length is None else float(mean_branch_length)
    graph = nx.Graph()
    observed_edges = 0
    missing_edges = 0

    for parent_id, child_id in tree.edges():
        branch_length = extract_branch_length_observation(tree, parent_id, child_id)
        if branch_length is None:
            weight = fallback_edge_length
            missing_edges += 1
        else:
            weight = float(branch_length)
            observed_edges += 1
        graph.add_edge(parent_id, child_id, weight=max(float(weight), 0.0))

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

    if observed_edges > 0 and missing_edges > 0:
        status = "branch_length_with_mean_fallback"
    elif observed_edges > 0:
        status = "branch_length_observed"
    else:
        status = "unit_fallback"
    return BranchLengthDistanceCache(
        distances=distances,
        fallback_edge_length=float(fallback_edge_length),
        status=status,
    )


def _distance_key(source: Hashable, target: Hashable) -> tuple[Hashable, Hashable]:
    return tuple(sorted((source, target), key=repr))


__all__ = [
    "BranchLengthDistanceCache",
    "build_branch_length_distance_cache",
]
