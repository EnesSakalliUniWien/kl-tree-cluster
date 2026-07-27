"""Compare pancreas inner nodes with progenitor-style lineage expectations."""
# ruff: noqa: I001

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


METHOD_KEY = "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001"
LINEAGE_MAP = {
    "alpha": "endocrine",
    "beta": "endocrine",
    "delta": "endocrine",
    "gamma": "endocrine",
    "epsilon": "endocrine",
    "ductal": "exocrine",
    "acinar": "exocrine",
    "activated_stellate": "stromal",
    "quiescent_stellate": "stromal",
    "PSC": "stromal",
    "mesenchymal": "stromal",
    "mesenchyme": "stromal",
    "macrophage": "immune",
    "mast": "immune",
    "endothelial": "endothelial",
}
ENDOCRINE_TYPES = ["alpha", "beta", "delta", "gamma", "epsilon"]
EXOCRINE_TYPES = ["ductal", "acinar"]
STROMAL_TYPES = [
    "activated_stellate",
    "quiescent_stellate",
    "PSC",
    "mesenchymal",
    "mesenchyme",
]
MARKERS = [
    "PDX1",
    "SOX9",
    "ONECUT1",
    "HNF1B",
    "NEUROG3",
    "PAX6",
    "ISL1",
    "NKX6-1",
    "NKX2-2",
    "INS",
    "GCG",
    "SST",
    "PPY",
    "GHRL",
    "KRT19",
    "EPCAM",
    "MUC1",
    "PRSS1",
    "CPA1",
    "REG1A",
    "COL1A1",
    "DCN",
    "PECAM1",
    "PTPRC",
    "LYZ",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def entropy(values: np.ndarray) -> float:
    counts = np.asarray(values, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def top_counts_text(counts: pd.Series, n: int = 10) -> str:
    return "; ".join(f"{key}:{value}" for key, value in counts.head(n).items())


def cluster_ids_text(cluster_ids: list[int] | tuple[int, ...] | set[int] | frozenset[int]) -> str:
    return ",".join(f"C{cluster_id}" for cluster_id in sorted(cluster_ids))


def fractions(counts: pd.Series, names: list[str], total: int) -> dict[str, float]:
    return {name: float(counts.get(name, 0) / total) for name in names}


def grouping_for_node(
    *,
    n_leaves: int,
    dominant_lineage: str,
    dominant_lineage_fraction: float,
    dominant_celltype_fraction: float,
    endocrine_fractions: dict[str, float],
    exocrine_fractions: dict[str, float],
    stromal_fractions: dict[str, float],
) -> tuple[str, str]:
    meaningful_endocrine = sum(value >= 0.02 for value in endocrine_fractions.values())
    meaningful_exocrine = sum(value >= 0.10 for value in exocrine_fractions.values())
    meaningful_stromal = sum(value >= 0.10 for value in stromal_fractions.values())

    if (
        n_leaves >= 50
        and dominant_lineage == "endocrine"
        and dominant_lineage_fraction >= 0.90
        and meaningful_endocrine >= 3
        and dominant_celltype_fraction < 0.95
    ):
        return (
            "endocrine mixed descendant node",
            "lineage-consistent ancestor node, but not validated progenitor",
        )
    if (
        n_leaves >= 50
        and dominant_lineage == "exocrine"
        and dominant_lineage_fraction >= 0.90
        and meaningful_exocrine == 2
        and dominant_celltype_fraction < 0.90
    ):
        return (
            "ductal-acinar mixed descendant node",
            "possible exocrine transition/progenitor-like grouping, requires marker validation",
        )
    if (
        n_leaves >= 50
        and dominant_lineage == "stromal"
        and dominant_lineage_fraction >= 0.90
        and meaningful_stromal >= 2
        and dominant_celltype_fraction < 0.90
    ):
        return (
            "stromal mixed descendant node",
            "stromal state mixture, not pancreatic epithelial progenitor",
        )
    if n_leaves >= 50 and dominant_lineage_fraction < 0.75:
        return ("broad mixed-pancreas node", "too broad; not progenitor-specific")
    if n_leaves >= 50 and dominant_celltype_fraction >= 0.95:
        return ("mature cell-type node", "mature-cell state/subcluster, not progenitor")
    return ("local geometry node", "too small or too local for progenitor interpretation")


def read_marker_frame(output_dir: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    adata = ad.read_h5ad(output_dir / "pancreas_classical_pipeline.h5ad")
    cell_to_obs = {str(cell_id): index for index, cell_id in enumerate(adata.obs_names)}
    subset_obs = np.array([cell_to_obs[cell_id] for cell_id in assignments["cell_id"].astype(str)])
    raw = adata.raw
    if raw is None:
        raise RuntimeError("Expected AnnData raw expression layer to be present.")

    raw_index = {gene: index for index, gene in enumerate(map(str, raw.var_names))}
    markers = [marker for marker in MARKERS if marker in raw_index]
    matrix = raw.X[subset_obs[:, None], [raw_index[marker] for marker in markers]]
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().toarray()
    return pd.DataFrame(np.asarray(matrix), columns=markers)


def main() -> None:
    output_dir = (
        project_root()
        / "raw"
        / "assets"
        / "benchmark-results"
        / "pancreas_scrna_cluster_benchmark_20260623"
    )
    assignments = pd.read_csv(output_dir / "method_assignments.csv")
    edges = pd.read_csv(output_dir / f"{METHOD_KEY}_tree_edges.csv")

    children: dict[str, list[str]] = {}
    parents = set(edges["parent"].astype(str))
    child_nodes = set(edges["child"].astype(str))
    for parent, child in zip(edges["parent"].astype(str), edges["child"].astype(str), strict=True):
        children.setdefault(parent, []).append(child)
    root = next(iter(parents - child_nodes))
    parent_of = {child: parent for parent, child_list in children.items() for child in child_list}

    @lru_cache(maxsize=None)
    def descendant_leaves(node: str) -> tuple[int, ...]:
        if node.startswith("L"):
            return (int(node[1:]),)
        leaves: list[int] = []
        for child in children[node]:
            leaves.extend(descendant_leaves(child))
        return tuple(leaves)

    def depth_from_root(node: str) -> int:
        depth = 0
        while node in parent_of:
            node = parent_of[node]
            depth += 1
        return depth

    celltypes = assignments["celltype"].astype(str).to_numpy()
    clusters = assignments[METHOD_KEY].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for node in sorted(parents, key=lambda value: int(value[1:])):
        leaf_indices = np.array(descendant_leaves(node), dtype=int)
        n_leaves = int(len(leaf_indices))
        celltype_counts = pd.Series(celltypes[leaf_indices]).value_counts()
        lineage_counts = pd.Series(
            [LINEAGE_MAP.get(celltype, "other") for celltype in celltypes[leaf_indices]]
        ).value_counts()
        cluster_counts = pd.Series(clusters[leaf_indices]).value_counts()

        dominant_celltype = str(celltype_counts.index[0])
        dominant_celltype_fraction = float(celltype_counts.iloc[0] / n_leaves)
        dominant_lineage = str(lineage_counts.index[0])
        dominant_lineage_fraction = float(lineage_counts.iloc[0] / n_leaves)
        role = (
            "root"
            if node == root
            else (
                "within_final_cluster"
                if len(cluster_counts) == 1
                else "shared_ancestor_across_final_clusters"
            )
        )
        grouping, progenitor_call = grouping_for_node(
            n_leaves=n_leaves,
            dominant_lineage=dominant_lineage,
            dominant_lineage_fraction=dominant_lineage_fraction,
            dominant_celltype_fraction=dominant_celltype_fraction,
            endocrine_fractions=fractions(celltype_counts, ENDOCRINE_TYPES, n_leaves),
            exocrine_fractions=fractions(celltype_counts, EXOCRINE_TYPES, n_leaves),
            stromal_fractions=fractions(celltype_counts, STROMAL_TYPES, n_leaves),
        )

        child_summary = []
        for child in children[node]:
            child_indices = np.array(descendant_leaves(child), dtype=int)
            child_counts = pd.Series(celltypes[child_indices]).value_counts()
            child_summary.append(
                f"{child} n={len(child_indices)} "
                + ",".join(f"{key}:{value}" for key, value in child_counts.head(4).items())
            )

        rows.append(
            {
                "node": node,
                "n_leaves": n_leaves,
                "depth_from_root": depth_from_root(node),
                "role": role,
                "n_final_clusters": int(len(cluster_counts)),
                "dominant_cluster": str(cluster_counts.index[0]),
                "dominant_cluster_fraction": float(cluster_counts.iloc[0] / n_leaves),
                "n_celltypes": int(len(celltype_counts)),
                "effective_celltypes": float(np.exp(entropy(celltype_counts.to_numpy()))),
                "dominant_celltype": dominant_celltype,
                "dominant_celltype_fraction": dominant_celltype_fraction,
                "top_celltypes": top_counts_text(celltype_counts, 10),
                "dominant_lineage": dominant_lineage,
                "dominant_lineage_fraction": dominant_lineage_fraction,
                "lineage_counts": "; ".join(
                    f"{key}:{value}" for key, value in lineage_counts.items()
                ),
                "grouping": grouping,
                "progenitor_call": progenitor_call,
                "child_summary": " | ".join(child_summary),
            }
        )

    summary = pd.DataFrame(rows)
    marker_frame = read_marker_frame(output_dir, assignments)

    interesting = summary[
        (summary["n_leaves"] >= 50)
        & summary["grouping"].isin(
            [
                "endocrine mixed descendant node",
                "ductal-acinar mixed descendant node",
                "stromal mixed descendant node",
                "broad mixed-pancreas node",
                "mature cell-type node",
            ]
        )
    ].copy()
    keep_nodes = {"N4998"}
    for grouping in [
        "endocrine mixed descendant node",
        "ductal-acinar mixed descendant node",
        "stromal mixed descendant node",
    ]:
        keep_nodes.update(interesting.loc[interesting["grouping"] == grouping, "node"])
    keep_nodes.update(
        interesting[interesting["grouping"] == "mature cell-type node"]
        .sort_values("n_leaves", ascending=False)
        .head(8)["node"]
    )
    review = summary[summary["node"].isin(keep_nodes)].copy()
    for marker in marker_frame.columns:
        marker_values = []
        for node in review["node"]:
            marker_values.append(
                float(marker_frame.iloc[list(descendant_leaves(str(node)))][marker].mean())
            )
        review[marker] = marker_values
    review["marker_evidence"] = ""
    review.loc[
        review["grouping"] == "endocrine mixed descendant node",
        "marker_evidence",
    ] = "NEUROG3 mean 0; mature hormone markers dominate descendant averages"
    review.loc[
        review["grouping"] == "broad mixed-pancreas node",
        "marker_evidence",
    ] = "root/broad mixture; marker averages reflect mixed descendants"
    review.loc[
        review["grouping"] == "mature cell-type node",
        "marker_evidence",
    ] = "composition dominated by one mature annotated cell type"
    terminal_review, terminal_inner_summary = build_terminal_cluster_review(
        assignments=assignments,
        summary=summary,
        marker_frame=marker_frame,
        children=children,
        parent_nodes=parents,
        descendant_leaves=descendant_leaves,
    )
    junction_review, junction_child_summary = build_two_three_cluster_junction_review(
        assignments=assignments,
        marker_frame=marker_frame,
        children=children,
        parent_nodes=parents,
        descendant_leaves=descendant_leaves,
    )
    monophyletic_review, monophyletic_child_summary = (
        build_monophyletic_subtree_meeting_review(
            assignments=assignments,
            marker_frame=marker_frame,
            children=children,
            parent_nodes=parents,
            descendant_leaves=descendant_leaves,
        )
    )

    summary.to_csv(output_dir / "tbs_adaptive_inner_node_lineage_summary.csv", index=False)
    review.sort_values(["grouping", "n_leaves"], ascending=[True, False]).to_csv(
        output_dir / "tbs_adaptive_inner_node_progenitor_review.csv",
        index=False,
    )
    terminal_review.to_csv(
        output_dir / "tbs_adaptive_terminal_cluster_root_review.csv",
        index=False,
    )
    terminal_inner_summary.to_csv(
        output_dir / "tbs_adaptive_terminal_cluster_inner_node_summary.csv",
        index=False,
    )
    junction_review.to_csv(
        output_dir / "tbs_adaptive_two_three_cluster_junction_review.csv",
        index=False,
    )
    junction_child_summary.to_csv(
        output_dir / "tbs_adaptive_two_three_cluster_junction_child_summary.csv",
        index=False,
    )
    monophyletic_review.to_csv(
        output_dir / "tbs_adaptive_monophyletic_subtree_meeting_review.csv",
        index=False,
    )
    monophyletic_child_summary.to_csv(
        output_dir / "tbs_adaptive_monophyletic_subtree_meeting_child_summary.csv",
        index=False,
    )
    write_plot(summary, review, output_dir)
    write_junction_umap_plot(junction_review, assignments, children, descendant_leaves, output_dir)
    write_monophyletic_meeting_umap_plot(
        monophyletic_review,
        assignments,
        children,
        descendant_leaves,
        output_dir,
    )


def build_terminal_cluster_review(
    *,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    marker_frame: pd.DataFrame,
    children: dict[str, list[str]],
    parent_nodes: set[str],
    descendant_leaves: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    celltypes = assignments["celltype"].astype(str).to_numpy()
    clusters = assignments[METHOD_KEY].astype(int).to_numpy()
    cluster_ids = sorted(np.unique(clusters))
    leaf_sets_by_node = {
        node: frozenset(descendant_leaves(node)) for node in parent_nodes  # type: ignore[operator]
    }
    node_by_leaf_set = {leaf_set: node for node, leaf_set in leaf_sets_by_node.items()}
    summary_by_node = summary.set_index("node")

    terminal_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        cluster_indices = frozenset(np.flatnonzero(clusters == cluster_id).astype(int).tolist())
        root_node = node_by_leaf_set.get(cluster_indices)
        root_is_internal = root_node is not None
        if root_node is None:
            if len(cluster_indices) != 1:
                raise RuntimeError(f"Cluster {cluster_id} is not an exact tree clade.")
            root_node = f"L{next(iter(cluster_indices))}"

        internal_nodes = [
            node
            for node, leaves in leaf_sets_by_node.items()
            if leaves.issubset(cluster_indices) and len(leaves) > 1
        ]
        internal_nodes = sorted(internal_nodes, key=lambda node: len(leaf_sets_by_node[node]))
        root_inner = summary_by_node.loc[root_node].to_dict() if root_is_internal else {}
        counts = pd.Series(celltypes[list(cluster_indices)]).value_counts()
        n_leaves = len(cluster_indices)

        marker_values = {
            marker: float(marker_frame.iloc[list(cluster_indices)][marker].mean())
            for marker in marker_frame.columns
        }
        terminal_rows.append(
            {
                "cluster_id": int(cluster_id),
                "terminal_root_node": root_node,
                "terminal_root_is_internal_node": bool(root_is_internal),
                "n_cells": int(n_leaves),
                "n_internal_nodes_inside_terminal_cluster": int(len(internal_nodes)),
                "root_grouping": root_inner.get(
                    "grouping",
                    "singleton leaf cluster",
                ),
                "root_progenitor_call": root_inner.get(
                    "progenitor_call",
                    "singleton observed cell, not an internal progenitor node",
                ),
                "dominant_celltype": str(counts.index[0]),
                "dominant_celltype_fraction": float(counts.iloc[0] / n_leaves),
                "n_celltypes": int(len(counts)),
                "effective_celltypes": float(np.exp(entropy(counts.to_numpy()))),
                "top_celltypes": top_counts_text(counts, 10),
                "root_child_summary": root_inner.get("child_summary", ""),
                **marker_values,
            }
        )

        candidate_inner_nodes = summary[
            summary["node"].isin(internal_nodes)
            & (
                summary["grouping"].isin(
                    [
                        "endocrine mixed descendant node",
                        "ductal-acinar mixed descendant node",
                        "stromal mixed descendant node",
                    ]
                )
                | (summary["n_leaves"] >= 50)
            )
        ].copy()
        for _, row in candidate_inner_nodes.iterrows():
            inner_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "terminal_root_node": root_node,
                    "inner_node": row["node"],
                    "n_leaves": int(row["n_leaves"]),
                    "grouping": row["grouping"],
                    "progenitor_call": row["progenitor_call"],
                    "dominant_celltype": row["dominant_celltype"],
                    "dominant_celltype_fraction": float(row["dominant_celltype_fraction"]),
                    "top_celltypes": row["top_celltypes"],
                    "child_summary": row["child_summary"],
                }
            )

    return pd.DataFrame(terminal_rows), pd.DataFrame(inner_rows)


def build_two_three_cluster_junction_review(
    *,
    assignments: pd.DataFrame,
    marker_frame: pd.DataFrame,
    children: dict[str, list[str]],
    parent_nodes: set[str],
    descendant_leaves: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    celltypes = assignments["celltype"].astype(str).to_numpy()
    clusters = assignments[METHOD_KEY].astype(int).to_numpy()

    @lru_cache(maxsize=None)
    def cluster_set(node: str) -> frozenset[int]:
        return frozenset(int(clusters[index]) for index in descendant_leaves(node))  # type: ignore[operator]

    junction_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    for node in sorted(parent_nodes, key=lambda value: int(value[1:])):
        node_cluster_set = cluster_set(node)
        if len(node_cluster_set) not in {2, 3}:
            continue
        child_cluster_sets = [cluster_set(child) for child in children[node]]
        if any(child_set == node_cluster_set for child_set in child_cluster_sets):
            continue

        leaf_indices = list(descendant_leaves(node))  # type: ignore[operator]
        celltype_counts = pd.Series(celltypes[leaf_indices]).value_counts()
        cluster_counts = pd.Series(clusters[leaf_indices]).value_counts()
        dominant_fraction = float(celltype_counts.iloc[0] / len(leaf_indices))
        effective_celltypes = float(np.exp(entropy(celltype_counts.to_numpy())))
        mixed_flag = bool(dominant_fraction < 0.90 or effective_celltypes > 1.50)
        marker_values = {
            marker: float(marker_frame.iloc[leaf_indices][marker].mean())
            for marker in marker_frame.columns
        }
        junction_rows.append(
            {
                "node": node,
                "n_cells": int(len(leaf_indices)),
                "n_meeting_clusters": int(len(node_cluster_set)),
                "cluster_ids": ",".join(f"C{cluster_id}" for cluster_id in sorted(node_cluster_set)),
                "child_cluster_sets": " | ".join(
                    ",".join(f"C{cluster_id}" for cluster_id in sorted(child_set))
                    for child_set in child_cluster_sets
                ),
                "cluster_counts": "; ".join(
                    f"C{cluster_id}:{count}" for cluster_id, count in cluster_counts.items()
                ),
                "mixed_flag": mixed_flag,
                "n_celltypes": int(len(celltype_counts)),
                "effective_celltypes": effective_celltypes,
                "dominant_celltype": str(celltype_counts.index[0]),
                "dominant_celltype_fraction": dominant_fraction,
                "top_celltypes": top_counts_text(celltype_counts, 8),
                "progenitor_call": (
                    "mixed junction but NEUROG3 is absent; not a validated progenitor"
                    if mixed_flag
                    else "same-celltype cluster junction; not progenitor"
                ),
                **marker_values,
            }
        )
        for child, child_set in zip(children[node], child_cluster_sets, strict=True):
            child_indices = list(descendant_leaves(child))  # type: ignore[operator]
            child_celltype_counts = pd.Series(celltypes[child_indices]).value_counts()
            child_cluster_counts = pd.Series(clusters[child_indices]).value_counts()
            child_marker_values = {
                marker: float(marker_frame.iloc[child_indices][marker].mean())
                for marker in marker_frame.columns
            }
            child_rows.append(
                {
                    "junction_node": node,
                    "child_node": child,
                    "child_n_cells": int(len(child_indices)),
                    "child_cluster_ids": ",".join(
                        f"C{cluster_id}" for cluster_id in sorted(child_set)
                    ),
                    "child_cluster_counts": "; ".join(
                        f"C{cluster_id}:{count}"
                        for cluster_id, count in child_cluster_counts.items()
                    ),
                    "child_top_celltypes": top_counts_text(child_celltype_counts, 8),
                    **child_marker_values,
                }
            )

    review = pd.DataFrame(junction_rows)
    if not review.empty:
        review = review.sort_values(["mixed_flag", "n_cells"], ascending=[False, False])
    return review, pd.DataFrame(child_rows)


def build_monophyletic_subtree_meeting_review(
    *,
    assignments: pd.DataFrame,
    marker_frame: pd.DataFrame,
    children: dict[str, list[str]],
    parent_nodes: set[str],
    descendant_leaves: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    celltypes = assignments["celltype"].astype(str).to_numpy()
    clusters = assignments[METHOD_KEY].astype(int).to_numpy()
    cluster_leaf_sets = {
        int(cluster_id): frozenset(np.flatnonzero(clusters == cluster_id).astype(int).tolist())
        for cluster_id in np.unique(clusters)
    }

    @lru_cache(maxsize=None)
    def leaf_set(node: str) -> frozenset[int]:
        return frozenset(descendant_leaves(node))  # type: ignore[operator]

    @lru_cache(maxsize=None)
    def cluster_set(node: str) -> frozenset[int]:
        return frozenset(int(clusters[index]) for index in descendant_leaves(node))  # type: ignore[operator]

    def full_cluster_leaf_union(cluster_ids: frozenset[int]) -> frozenset[int]:
        leaves: set[int] = set()
        for cluster_id in cluster_ids:
            leaves.update(cluster_leaf_sets[cluster_id])
        return frozenset(leaves)

    def is_complete_cluster_union(node: str) -> bool:
        node_cluster_set = cluster_set(node)
        return leaf_set(node) == full_cluster_leaf_union(node_cluster_set)

    def lineage_counts_for_indices(indices: list[int]) -> pd.Series:
        return pd.Series(
            [LINEAGE_MAP.get(celltype, "other") for celltype in celltypes[indices]]
        ).value_counts()

    meeting_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    for node in sorted(parent_nodes, key=lambda value: int(value[1:])):
        node_cluster_set = cluster_set(node)
        if len(node_cluster_set) < 2:
            continue

        child_cluster_sets = [cluster_set(child) for child in children[node]]
        if any(child_set == node_cluster_set for child_set in child_cluster_sets):
            continue
        if not all(is_complete_cluster_union(child) for child in children[node]):
            continue

        leaf_indices = list(descendant_leaves(node))  # type: ignore[operator]
        celltype_counts = pd.Series(celltypes[leaf_indices]).value_counts()
        lineage_counts = lineage_counts_for_indices(leaf_indices)
        cluster_counts = pd.Series(clusters[leaf_indices]).value_counts()
        dominant_celltype_fraction = float(celltype_counts.iloc[0] / len(leaf_indices))
        dominant_lineage_fraction = float(lineage_counts.iloc[0] / len(leaf_indices))
        effective_celltypes = float(np.exp(entropy(celltype_counts.to_numpy())))
        child_n_clusters = [len(child_set) for child_set in child_cluster_sets]
        child_n_cells = [len(leaf_set(child)) for child in children[node]]
        child_dominant_celltypes: list[str] = []
        child_dominant_lineages: list[str] = []

        marker_values = {
            marker: float(marker_frame.iloc[leaf_indices][marker].mean())
            for marker in marker_frame.columns
        }
        for child, child_set in zip(children[node], child_cluster_sets, strict=True):
            child_indices = list(descendant_leaves(child))  # type: ignore[operator]
            child_celltype_counts = pd.Series(celltypes[child_indices]).value_counts()
            child_lineage_counts = lineage_counts_for_indices(child_indices)
            child_cluster_counts = pd.Series(clusters[child_indices]).value_counts()
            child_marker_values = {
                marker: float(marker_frame.iloc[child_indices][marker].mean())
                for marker in marker_frame.columns
            }
            child_dominant_celltypes.append(str(child_celltype_counts.index[0]))
            child_dominant_lineages.append(str(child_lineage_counts.index[0]))
            child_rows.append(
                {
                    "meeting_node": node,
                    "child_node": child,
                    "child_n_cells": int(len(child_indices)),
                    "child_n_final_clusters": int(len(child_set)),
                    "child_cluster_ids": cluster_ids_text(child_set),
                    "child_cluster_counts": "; ".join(
                        f"C{cluster_id}:{count}"
                        for cluster_id, count in child_cluster_counts.items()
                    ),
                    "child_dominant_celltype": str(child_celltype_counts.index[0]),
                    "child_dominant_celltype_fraction": float(
                        child_celltype_counts.iloc[0] / len(child_indices)
                    ),
                    "child_top_celltypes": top_counts_text(child_celltype_counts, 8),
                    "child_dominant_lineage": str(child_lineage_counts.index[0]),
                    "child_lineage_counts": "; ".join(
                        f"{key}:{value}" for key, value in child_lineage_counts.items()
                    ),
                    **child_marker_values,
                }
            )

        meaningful_endocrine = sum(
            celltype_counts.get(celltype, 0) / len(leaf_indices) >= 0.02
            for celltype in ENDOCRINE_TYPES
        )
        min_child_cluster_fraction = min(child_n_clusters) / len(node_cluster_set)
        min_child_cell_fraction = min(child_n_cells) / len(leaf_indices)
        ladder_like_split = bool(
            len(node_cluster_set) > 3
            and (min_child_cluster_fraction < 0.20 or min_child_cell_fraction < 0.08)
        )
        mixed_flag = bool(
            dominant_celltype_fraction < 0.90
            or effective_celltypes > 1.50
            or len(set(child_dominant_celltypes)) > 1
        )

        if (
            len(leaf_indices) >= 50
            and str(lineage_counts.index[0]) == "endocrine"
            and dominant_lineage_fraction >= 0.90
            and meaningful_endocrine >= 3
        ):
            meeting_type = "endocrine mixed monophyletic-subtree meeting"
            progenitor_call = (
                "lineage-coherent endocrine hierarchy ancestor; NEUROG3 absent, "
                "so not a validated progenitor state"
            )
        elif len(leaf_indices) >= 50 and dominant_lineage_fraction < 0.75:
            meeting_type = "broad mixed-pancreas monophyletic-subtree meeting"
            progenitor_call = "too broad across pancreatic lineages; not progenitor-specific"
        elif dominant_celltype_fraction >= 0.95:
            meeting_type = "same mature-celltype monophyletic-subtree meeting"
            progenitor_call = "same annotated cell type split into monophyletic subclusters"
        elif len(leaf_indices) < 50:
            meeting_type = "small local monophyletic-subtree junction"
            progenitor_call = (
                "small local boundary; not a supported progenitor without marker evidence"
            )
        else:
            meeting_type = "mixed monophyletic-subtree meeting"
            progenitor_call = "mixed descendant clade; requires marker evidence before ancestry claim"

        display_focus = bool(
            mixed_flag
            and (
                not ladder_like_split
                or len(node_cluster_set) <= 3
                or meeting_type == "endocrine mixed monophyletic-subtree meeting"
            )
        )
        meeting_rows.append(
            {
                "node": node,
                "n_cells": int(len(leaf_indices)),
                "n_meeting_clusters": int(len(node_cluster_set)),
                "cluster_ids": cluster_ids_text(node_cluster_set),
                "child_cluster_sets": " | ".join(
                    cluster_ids_text(child_set) for child_set in child_cluster_sets
                ),
                "child_n_final_clusters": " | ".join(
                    str(value) for value in child_n_clusters
                ),
                "child_n_cells": " | ".join(str(value) for value in child_n_cells),
                "min_child_cluster_fraction": float(min_child_cluster_fraction),
                "min_child_cell_fraction": float(min_child_cell_fraction),
                "ladder_like_split": ladder_like_split,
                "display_focus": display_focus,
                "mixed_flag": mixed_flag,
                "meeting_type": meeting_type,
                "cluster_counts": "; ".join(
                    f"C{cluster_id}:{count}" for cluster_id, count in cluster_counts.items()
                ),
                "n_celltypes": int(len(celltype_counts)),
                "effective_celltypes": effective_celltypes,
                "dominant_celltype": str(celltype_counts.index[0]),
                "dominant_celltype_fraction": dominant_celltype_fraction,
                "top_celltypes": top_counts_text(celltype_counts, 10),
                "dominant_lineage": str(lineage_counts.index[0]),
                "dominant_lineage_fraction": dominant_lineage_fraction,
                "lineage_counts": "; ".join(
                    f"{key}:{value}" for key, value in lineage_counts.items()
                ),
                "child_dominant_celltypes": " | ".join(child_dominant_celltypes),
                "child_dominant_lineages": " | ".join(child_dominant_lineages),
                "progenitor_call": progenitor_call,
                **marker_values,
            }
        )

    review = pd.DataFrame(meeting_rows)
    if not review.empty:
        review = review.sort_values(
            ["display_focus", "n_cells", "n_meeting_clusters"],
            ascending=[False, False, False],
        )
    return review, pd.DataFrame(child_rows)


def write_plot(summary: pd.DataFrame, review: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary[summary["n_leaves"] >= 50].copy()
    colors = {
        "broad mixed-pancreas node": "#64748b",
        "endocrine mixed descendant node": "#2563eb",
        "ductal-acinar mixed descendant node": "#f97316",
        "stromal mixed descendant node": "#16a34a",
        "mature cell-type node": "#7c3aed",
        "local geometry node": "#d1d5db",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.8),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    ax = axes[0]
    for grouping, group in plot_df.groupby("grouping"):
        ax.scatter(
            np.log10(group["n_leaves"]),
            group["effective_celltypes"],
            s=np.clip(group["n_final_clusters"] * 12, 18, 180),
            c=colors.get(str(grouping), "#111827"),
            label=str(grouping),
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
    for node in ["N4963", "N4962", "N4881", "N4789", "N4998"]:
        match = plot_df[plot_df["node"] == node]
        if match.empty:
            continue
        row = match.iloc[0]
        ax.annotate(
            node,
            (np.log10(float(row["n_leaves"])), float(row["effective_celltypes"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("log10(descendant leaves)")
    ax.set_ylabel("Effective annotated cell types")
    ax.set_title("TBS adaptive internal nodes >= 50 leaves")
    ax.legend(fontsize=7, loc="upper left")

    selected = review.sort_values("n_leaves", ascending=False).head(12).copy()
    heat_markers = [
        marker
        for marker in [
            "NEUROG3",
            "PDX1",
            "SOX9",
            "HNF1B",
            "PAX6",
            "ISL1",
            "NKX6-1",
            "NKX2-2",
            "INS",
            "GCG",
            "SST",
            "PPY",
            "GHRL",
            "KRT19",
            "PRSS1",
            "CPA1",
            "REG1A",
            "COL1A1",
            "PECAM1",
            "LYZ",
        ]
        if marker in selected.columns
    ]
    matrix = selected[heat_markers].to_numpy(dtype=float)
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    denom = np.where(col_max - col_min > 0, col_max - col_min, 1)
    scaled = (matrix - col_min) / denom

    ax = axes[1]
    image = ax.imshow(scaled, aspect="auto", cmap="magma", vmin=0, vmax=1)
    labels = [
        f"{row.node} n={int(row.n_leaves)}\n{str(row.grouping).replace(' descendant node', '')}"
        for row in selected.itertuples()
    ]
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(len(heat_markers)))
    ax.set_xticklabels(heat_markers, rotation=60, ha="right", fontsize=8)
    ax.set_title("Marker means in selected internal nodes\n(column-scaled for display)")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="relative mean")
    fig.suptitle(
        "Inner nodes compared to progenitor-style lineage expectations",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "tbs_adaptive_inner_node_progenitor_comparison.png", dpi=240)
    fig.savefig(output_dir / "tbs_adaptive_inner_node_progenitor_comparison.pdf")
    plt.close(fig)


def write_junction_umap_plot(
    junction_review: pd.DataFrame,
    assignments: pd.DataFrame,
    children: dict[str, list[str]],
    descendant_leaves: object,
    output_dir: Path,
) -> None:
    mixed = junction_review[junction_review["mixed_flag"].astype(bool)].copy()
    if mixed.empty:
        return
    ncols = 2
    nrows = int(np.ceil(len(mixed) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.7 * nrows), squeeze=False)
    palette = plt.get_cmap("tab10")
    for ax, row in zip(axes.ravel(), mixed.itertuples(index=False), strict=False):
        leaf_indices = list(descendant_leaves(str(row.node)))  # type: ignore[operator]
        frame = assignments.iloc[leaf_indices].copy()
        ax.scatter(
            assignments["umap1"],
            assignments["umap2"],
            s=3,
            c="#e5e7eb",
            alpha=0.45,
            linewidths=0,
        )
        for color_index, (cluster_id, group) in enumerate(frame.groupby(METHOD_KEY, sort=True)):
            ax.scatter(
                group["umap1"],
                group["umap2"],
                s=24,
                c=[palette(color_index % 10)],
                label=f"C{int(cluster_id)} n={len(group)}",
                alpha=0.92,
                linewidths=0.2,
                edgecolors="white",
            )
        ax.set_title(
            f"{row.node}: {row.cluster_ids}\n{row.top_celltypes}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(mixed) :]:
        ax.axis("off")
    fig.suptitle(
        "Mixed direct junctions where two or three final TBS clusters meet",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "tbs_adaptive_two_three_cluster_junction_mixed_umap.png", dpi=240)
    fig.savefig(output_dir / "tbs_adaptive_two_three_cluster_junction_mixed_umap.pdf")
    plt.close(fig)


def write_monophyletic_meeting_umap_plot(
    meeting_review: pd.DataFrame,
    assignments: pd.DataFrame,
    children: dict[str, list[str]],
    descendant_leaves: object,
    output_dir: Path,
) -> None:
    selected = meeting_review[meeting_review["display_focus"].astype(bool)].copy()
    if selected.empty:
        return
    selected = selected.sort_values("n_cells", ascending=False).head(8)
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.8 * ncols, 6.4 * nrows), squeeze=False)
    palette = plt.get_cmap("tab20")
    for ax, row in zip(axes.ravel(), selected.itertuples(index=False), strict=False):
        leaf_indices = list(descendant_leaves(str(row.node)))  # type: ignore[operator]
        frame = assignments.iloc[leaf_indices].copy()
        ax.scatter(
            assignments["umap1"],
            assignments["umap2"],
            s=4,
            c="#d1d5db",
            alpha=0.22,
            linewidths=0,
        )
        point_size = 18 if len(frame) >= 200 else 34
        for color_index, (cluster_id, group) in enumerate(frame.groupby(METHOD_KEY, sort=True)):
            color = palette(color_index % 20)
            ax.scatter(
                group["umap1"],
                group["umap2"],
                s=point_size,
                c=[color],
                label=f"C{int(cluster_id)} n={len(group)}",
                alpha=0.90,
                linewidths=0.25,
                edgecolors="white",
            )
            if len(group) >= 8:
                ax.text(
                    float(group["umap1"].median()),
                    float(group["umap2"].median()),
                    f"C{int(cluster_id)}",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.82,
                    },
                )

        x_min, x_max = float(frame["umap1"].min()), float(frame["umap1"].max())
        y_min, y_max = float(frame["umap2"].min()), float(frame["umap2"].max())
        x_pad = max(0.7, (x_max - x_min) * 0.12)
        y_pad = max(0.7, (y_max - y_min) * 0.12)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(
            f"{row.node}: {row.cluster_ids}\n{row.top_celltypes}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(fontsize=7, loc="best", frameon=True)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle(
        "Mixed junctions where complete monophyletic final-cluster subtrees meet",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "tbs_adaptive_monophyletic_subtree_meeting_mixed_umap.png", dpi=240)
    fig.savefig(output_dir / "tbs_adaptive_monophyletic_subtree_meeting_mixed_umap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
