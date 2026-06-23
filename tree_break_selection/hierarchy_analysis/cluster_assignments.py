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
        leaf_signature = tuple(leaves)
        cluster_assignments[cluster_index] = {
            "root_node": boundary.root_node,
            "leaves": leaves,
            "leaf_signature": leaf_signature,
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
        - ``cluster_leaf_signature``: sorted, tuple-valued cluster leaf identity
    """
    if "cluster_assignments" not in decomposition_results:
        raise KeyError("Missing required decomposition field 'cluster_assignments'.")

    raw_cluster_assignments = decomposition_results["cluster_assignments"]
    if not isinstance(raw_cluster_assignments, dict):
        raise TypeError("'cluster_assignments' must be a dictionary.")
    if not raw_cluster_assignments:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_root",
                "cluster_size",
                "cluster_leaf_signature",
            ]
        )

    rows: dict[str, dict[str, object]] = {}
    for cluster_identifier, cluster_metadata in raw_cluster_assignments.items():
        if not isinstance(cluster_metadata, dict):
            raise TypeError(
                f"Cluster {cluster_identifier!r} metadata must be a dictionary."
            )
        missing_fields = {"root_node", "leaves", "size"} - set(cluster_metadata)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise KeyError(
                f"Cluster {cluster_identifier!r} metadata is missing required field(s): {missing}."
            )
        leaves = cluster_metadata["leaves"]
        if not isinstance(leaves, list):
            raise TypeError(f"Cluster {cluster_identifier!r} 'leaves' must be a list.")
        root = cluster_metadata["root_node"]
        size = cluster_metadata["size"]
        leaf_signature = tuple(sorted(leaves))
        declared_leaf_signature = cluster_metadata.get("leaf_signature", leaf_signature)
        if tuple(declared_leaf_signature) != leaf_signature:
            raise ValueError(
                f"Cluster {cluster_identifier!r} leaf_signature does not match leaves."
            )
        for sample_identifier in leaves:
            if sample_identifier in rows:
                raise ValueError(
                    f"Sample {sample_identifier!r} appears in multiple clusters."
                )
            rows[sample_identifier] = {
                "cluster_id": cluster_identifier,
                "cluster_root": root,
                "cluster_size": size,
                "cluster_leaf_signature": leaf_signature,
            }

    assignments_table = pd.DataFrame.from_dict(rows, orient="index")
    assignments_table.index.name = "sample_id"
    return assignments_table.sort_values("cluster_id")
