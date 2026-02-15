"""Pure functions for building cluster assignment structures.

These functions convert raw leaf-set partitions into structured cluster
metadata dictionaries and per-sample DataFrames.  They are intentionally
stateless so they can be called from :class:`TreeDecomposition`,
:class:`PosetTree`, or any other consumer without class coupling.
"""

from __future__ import annotations

from typing import Callable, Dict, Set

import pandas as pd


def build_cluster_assignments(
    final_leaf_sets: list[set[str]],
    find_cluster_root: Callable[[Set[str]], str],
) -> dict[int, dict[str, object]]:
    """Build a cluster assignment dictionary from collected leaf sets.

    Parameters
    ----------
    final_leaf_sets
        List of leaf sets, one per cluster.
    find_cluster_root
        Callable that maps a set of leaf labels to the cluster root node
        (typically the lowest common ancestor in the tree).

    Returns
    -------
    dict[int, dict[str, object]]
        Mapping from cluster index to cluster metadata containing
        ``root_node``, ``leaves``, and ``size``.
    """
    cluster_assignments: dict[int, dict[str, object]] = {}
    for cluster_index, leaf_set in enumerate(final_leaf_sets):
        if not leaf_set:
            continue
        cluster_assignments[cluster_index] = {
            "root_node": find_cluster_root(leaf_set),
            "leaves": sorted(leaf_set),
            "size": len(leaf_set),
        }
    return cluster_assignments


def build_sample_cluster_assignments(
    decomposition_results: Dict[str, object],
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
    cluster_assignments = decomposition_results.get("cluster_assignments", {})
    if not cluster_assignments:
        return pd.DataFrame(columns=["cluster_id", "cluster_root", "cluster_size"])

    rows: Dict[str, Dict[str, object]] = {}
    for cluster_identifier, info in cluster_assignments.items():
        root = info.get("root_node")
        size = info.get("size", 0)
        for sample_identifier in info.get("leaves", []):
            rows[sample_identifier] = {
                "cluster_id": cluster_identifier,
                "cluster_root": root,
                "cluster_size": size,
            }

    assignments_table = pd.DataFrame.from_dict(rows, orient="index")
    assignments_table.index.name = "sample_id"
    return assignments_table.sort_values("cluster_id")
