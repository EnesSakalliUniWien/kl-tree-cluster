"""Pure functions for building cluster assignment structures.

These functions convert final tree boundaries into structured cluster
metadata dictionaries and per-sample DataFrames. They are intentionally
stateless so they can be called from :class:`TreeDecomposition`,
:class:`PosetTree`, or any other consumer without class coupling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClusterBoundary:
    """A final tree boundary whose descendant leaves form one cluster."""

    root_node: object
    leaves: frozenset[str]


def build_cluster_assignments(
    boundaries: Iterable[ClusterBoundary],
) -> dict[int, dict[str, object]]:
    """Build a cluster assignment dictionary from final tree boundaries.

    Parameters
    ----------
    boundaries
        Final tree boundary nodes and their descendant leaf labels.

    Returns
    -------
    dict[int, dict[str, object]]
        Mapping from cluster index to cluster metadata containing
        ``root_node``, ``leaves``, and ``size``.
    """
    cluster_assignments: dict[int, dict[str, object]] = {}
    for cluster_index, boundary in enumerate(boundaries):
        leaves = sorted(boundary.leaves)
        if not leaves:
            continue
        cluster_assignments[cluster_index] = {
            "root_node": boundary.root_node,
            "leaves": leaves,
            "size": len(leaves),
        }
    return cluster_assignments


def build_sample_cluster_assignments(
    decomposition_results: dict[str, object],
) -> pd.DataFrame:
    """Build per-sample cluster assignments from decomposition output.

    Parameters
    ----------
    decomposition_results
        A decomposition result dictionary produced by
        :meth:`TreeDecomposition.decompose_tree` or
        :meth:`PosetTree.decompose`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by ``sample_id`` with columns:

        - ``cluster_id``: integer cluster identifier
        - ``cluster_root``: node identifier that forms the cluster boundary
        - ``cluster_size``: number of samples in the cluster
    """
    raw_cluster_assignments = decomposition_results.get("cluster_assignments", {})
    if not isinstance(raw_cluster_assignments, dict) or not raw_cluster_assignments:
        return pd.DataFrame(columns=["cluster_id", "cluster_root", "cluster_size"])

    rows: dict[str, dict[str, object]] = {}
    for cluster_identifier, cluster_metadata in raw_cluster_assignments.items():
        if not isinstance(cluster_metadata, dict):
            continue
        root = cluster_metadata.get("root_node")
        size = cluster_metadata.get("size", 0)
        for sample_identifier in cluster_metadata.get("leaves", []):
            rows[sample_identifier] = {
                "cluster_id": cluster_identifier,
                "cluster_root": root,
                "cluster_size": size,
            }

    assignments_table = pd.DataFrame.from_dict(rows, orient="index")
    assignments_table.index.name = "sample_id"
    return assignments_table.sort_values("cluster_id")
