"""Audit mass-weighted distributional movement in scRNA TBS benchmark trees."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.distributional_action import (
    edge_distributional_action_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "raw/assets/benchmark-results"
OUTPUT_ROOT = BENCHMARK_ROOT / "scrna_distributional_action_audit_20260624"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    output_dir: Path


@dataclass(frozen=True)
class MethodSpec:
    geometry: str
    variant: str
    method_label: str
    assignment_key: str
    edge_csv: str


DATASETS = (
    DatasetSpec(
        key="adult_pancreas",
        label="Adult pancreas",
        output_dir=BENCHMARK_ROOT / "pancreas_scrna_cluster_benchmark_20260623",
    ),
    DatasetSpec(
        key="goncalves_fetal",
        label="Goncalves fetal pancreas",
        output_dir=BENCHMARK_ROOT / "goncalves_fetal_pancreas_progenitor_benchmark_20260624",
    ),
)

METHODS = (
    MethodSpec(
        geometry="pca_linkage",
        variant="topology_only",
        method_label="TBS topology projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv="tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    ),
    MethodSpec(
        geometry="pca_linkage",
        variant="branch_time_nnls",
        method_label="TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv=(
            "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_"
            "tree_edges.csv"
        ),
    ),
    MethodSpec(
        geometry="pca_linkage",
        variant="branch_time_raw_linkage",
        method_label=(
            "TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 "
            "edge=0.001"
        ),
        assignment_key="tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv=(
            "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_"
            "edge0p001_tree_edges.csv"
        ),
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="topology_only",
        method_label="TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv=(
            "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_"
            "tree_edges.csv"
        ),
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="branch_time_nnls",
        method_label=(
            "TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 "
            "alpha=0.01 edge=0.001"
        ),
        assignment_key=(
            "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_"
            "alpha0p01_edge0p001"
        ),
        edge_csv=(
            "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_"
            "alpha0p01_edge0p001_tree_edges.csv"
        ),
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="branch_time_raw_linkage",
        method_label=(
            "TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 "
            "alpha=0.01 edge=0.001"
        ),
        assignment_key=(
            "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_"
            "k90_alpha0p01_edge0p001"
        ),
        edge_csv=(
            "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_"
            "k90_alpha0p01_edge0p001_tree_edges.csv"
        ),
    ),
)

EDGE_STAT_COLUMN = "Child_Parent_Divergence_Test_Statistic"
EDGE_P_COLUMN = "Child_Parent_Divergence_P_Value"
EDGE_SIGNIFICANT_COLUMN = "Child_Parent_Divergence_Significant"
EDGE_TESTED_COLUMN = "Child_Parent_Divergence_Tested"
EDGE_ANCESTOR_BLOCKED_COLUMN = "Child_Parent_Divergence_Ancestor_Blocked"


def _node_index(node_id: object) -> int:
    text = str(node_id)
    if len(text) < 2 or text[0] not in {"L", "N"}:
        raise ValueError(f"Unexpected tree node id {node_id!r}.")
    return int(text[1:])


def _node_is_leaf(node_id: object) -> bool:
    return str(node_id).startswith("L")


def _node_is_internal(node_id: object) -> bool:
    return str(node_id).startswith("N")


def _read_pca(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = pd.read_csv(output_dir / "benchmark_subset_pca.csv", index_col=0)
    pca = pca.astype(float)
    scale = pca.std(axis=0, ddof=1)
    scale = scale.mask(~np.isfinite(scale) | (scale <= 0.0), 1.0)
    standardized = (pca - pca.mean(axis=0)) / scale
    return pca, standardized


def _leaf_node_vectors(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    values = frame.to_numpy(dtype=np.float64, copy=False)
    return {f"L{index}": values[index] for index in range(values.shape[0])}


def _reconstruct_node_vectors(
    edges: pd.DataFrame,
    leaf_vectors: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    node_vectors = dict(leaf_vectors)
    leaf_counts = {node_id: 1 for node_id in leaf_vectors}
    children_by_parent = {
        parent: group["child"].astype(str).tolist()
        for parent, group in edges.groupby("parent", sort=False)
    }

    for parent in sorted(children_by_parent, key=_node_index):
        children = children_by_parent[parent]
        missing = [child for child in children if child not in node_vectors]
        if missing:
            raise ValueError(f"Cannot reconstruct {parent!r}; missing child vectors {missing!r}.")
        counts = np.asarray([leaf_counts[child] for child in children], dtype=np.float64)
        values = np.vstack([node_vectors[child] for child in children])
        total_count = int(np.sum(counts))
        if total_count <= 0:
            raise ValueError(f"Node {parent!r} has non-positive reconstructed leaf count.")
        node_vectors[parent] = np.average(values, axis=0, weights=counts)
        leaf_counts[parent] = total_count

    return node_vectors, leaf_counts


def _rank_within(values: pd.Series, *, ascending: bool = False) -> pd.Series:
    return values.rank(method="min", ascending=ascending).astype(int)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    data = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    if len(data) < 3 or data["left"].nunique() < 2 or data["right"].nunique() < 2:
        return float("nan")
    return float(data["left"].rank().corr(data["right"].rank()))


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_goncalves_progenitor_tables(output_dir: Path) -> dict[str, dict[str, object]]:
    node_info: dict[str, dict[str, object]] = {}
    inner_path = output_dir / "goncalves_tbs_inner_node_progenitor_signature_scores.csv"
    if inner_path.exists():
        inner = pd.read_csv(inner_path)
        for _, row in inner.iterrows():
            node = str(row["node"])
            node_info[node] = {
                "progenitor_group_type": row.get("group_type"),
                "progenitor_interpretation": row.get("progenitor_interpretation"),
                "progenitor_population_fraction": row.get("progenitor_population_fraction"),
                "dominant_population": row.get("dominant_population"),
                "top_populations": row.get("top_populations"),
                "is_monophyletic_meeting_node": False,
            }

    meeting_path = output_dir / "goncalves_tbs_monophyletic_meeting_progenitor_signature_scores.csv"
    if meeting_path.exists():
        meeting = pd.read_csv(meeting_path)
        for _, row in meeting.iterrows():
            node = str(row["node"])
            info = node_info.setdefault(node, {})
            info.update(
                {
                    "progenitor_group_type": row.get("group_type"),
                    "progenitor_interpretation": row.get("progenitor_interpretation"),
                    "progenitor_population_fraction": row.get("progenitor_population_fraction"),
                    "dominant_population": row.get("dominant_population"),
                    "top_populations": row.get("top_populations"),
                    "meeting_clusters": row.get("meeting_clusters"),
                    "is_monophyletic_meeting_node": True,
                }
            )
    return node_info


def _edge_records_for_method(
    dataset: DatasetSpec,
    method: MethodSpec,
    pca: pd.DataFrame,
    standardized_pca: pd.DataFrame,
    progenitor_info: dict[str, dict[str, object]],
) -> pd.DataFrame:
    edge_path = dataset.output_dir / method.edge_csv
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)
    edges = pd.read_csv(edge_path)
    edges["parent"] = edges["parent"].astype(str)
    edges["child"] = edges["child"].astype(str)

    raw_vectors, raw_counts = _reconstruct_node_vectors(edges, _leaf_node_vectors(pca))
    std_vectors, std_counts = _reconstruct_node_vectors(edges, _leaf_node_vectors(standardized_pca))
    if raw_counts != std_counts:
        raise ValueError(f"Raw and standardized reconstructed counts differ for {edge_path}.")

    records: list[dict[str, object]] = []
    n_leaves = int(pca.shape[0])
    for _, edge in edges.iterrows():
        parent = str(edge["parent"])
        child = str(edge["child"])
        parent_count = int(edge["parent_leaf_count"])
        child_count = int(edge["child_leaf_count"])
        reconstructed_parent_count = raw_counts[parent]
        reconstructed_child_count = raw_counts[child]
        if parent_count != reconstructed_parent_count or child_count != reconstructed_child_count:
            raise ValueError(
                f"Leaf-count mismatch in {edge_path.name} for {parent}->{child}: "
                f"csv=({parent_count}, {child_count}), reconstructed="
                f"({reconstructed_parent_count}, {reconstructed_child_count})."
            )

        raw_delta = raw_vectors[child] - raw_vectors[parent]
        std_delta = std_vectors[child] - std_vectors[parent]
        raw_delta_sq_sum = float(np.sum(raw_delta * raw_delta))
        standardized_delta_sq_sum = float(np.sum(std_delta * std_delta))
        standardized_delta_sq_mean = float(standardized_delta_sq_sum / len(std_delta))
        child_leaf_fraction = float(child_count / n_leaves)
        standardized_action = edge_distributional_action_summary(
            std_vectors[parent],
            std_vectors[child],
            parent_count,
            child_count,
        )
        raw_action = edge_distributional_action_summary(
            raw_vectors[parent],
            raw_vectors[child],
            parent_count,
            child_count,
        )
        child_parent_fraction = standardized_action.child_parent_mass_fraction
        subtree_distributional_action = float(
            standardized_action.action / len(std_delta)
        )
        parent_fraction_distributional_action = float(
            child_parent_fraction * standardized_delta_sq_mean
        )
        raw_subtree_distributional_action = raw_action.action
        branch_length = float(edge["branch_length"])
        child_info = progenitor_info.get(child, {})
        parent_info = progenitor_info.get(parent, {})

        records.append(
            {
                "dataset": dataset.key,
                "dataset_label": dataset.label,
                "geometry": method.geometry,
                "variant": method.variant,
                "method": method.method_label,
                "assignment_key": method.assignment_key,
                "edge_csv": method.edge_csv,
                "parent": parent,
                "child": child,
                "child_is_leaf": _node_is_leaf(child),
                "child_is_internal": _node_is_internal(child),
                "parent_leaf_count": parent_count,
                "child_leaf_count": child_count,
                "child_leaf_fraction": child_leaf_fraction,
                "child_parent_fraction": child_parent_fraction,
                "branch_length": branch_length,
                "raw_delta_norm": float(np.sqrt(raw_delta_sq_sum)),
                "standardized_delta_norm": float(np.sqrt(standardized_delta_sq_sum)),
                "standardized_delta_sq_mean": standardized_delta_sq_mean,
                "subtree_distributional_action": subtree_distributional_action,
                "parent_fraction_distributional_action": (
                    parent_fraction_distributional_action
                ),
                "raw_subtree_distributional_action": raw_subtree_distributional_action,
                "action_per_branch_length": float(
                    subtree_distributional_action / max(branch_length, 1e-12)
                ),
                "edge_test_statistic": float(edge.get(EDGE_STAT_COLUMN, np.nan)),
                "edge_p_value": float(edge.get(EDGE_P_COLUMN, np.nan)),
                "edge_significant": _safe_bool(edge.get(EDGE_SIGNIFICANT_COLUMN, False)),
                "edge_tested": _safe_bool(edge.get(EDGE_TESTED_COLUMN, False)),
                "edge_ancestor_blocked": _safe_bool(
                    edge.get(EDGE_ANCESTOR_BLOCKED_COLUMN, False)
                ),
                "split_filter_policy": edge.get(
                    "Distributional_Split_Action_Filter_Policy"
                ),
                "split_filter_quantile": float(
                    edge.get("Distributional_Split_Action_Filter_Quantile", np.nan)
                ),
                "split_filter_threshold": float(
                    edge.get("Distributional_Split_Action_Filter_Threshold", np.nan)
                ),
                "split_filter_passes": _safe_bool(
                    edge.get("Distributional_Split_Action_Filter_Passes", True)
                ),
                "split_filter_filtered": _safe_bool(
                    edge.get("Distributional_Split_Action_Filtered", False)
                ),
                "child_progenitor_interpretation": child_info.get(
                    "progenitor_interpretation"
                ),
                "child_progenitor_population_fraction": child_info.get(
                    "progenitor_population_fraction"
                ),
                "child_dominant_population": child_info.get("dominant_population"),
                "child_top_populations": child_info.get("top_populations"),
                "child_is_monophyletic_meeting_node": bool(
                    child_info.get("is_monophyletic_meeting_node", False)
                ),
                "child_meeting_clusters": child_info.get("meeting_clusters"),
                "parent_progenitor_interpretation": parent_info.get(
                    "progenitor_interpretation"
                ),
                "parent_progenitor_population_fraction": parent_info.get(
                    "progenitor_population_fraction"
                ),
                "parent_dominant_population": parent_info.get("dominant_population"),
                "parent_top_populations": parent_info.get("top_populations"),
                "parent_is_monophyletic_meeting_node": bool(
                    parent_info.get("is_monophyletic_meeting_node", False)
                ),
                "parent_meeting_clusters": parent_info.get("meeting_clusters"),
            }
        )

    result = pd.DataFrame.from_records(records)
    result["action_rank"] = _rank_within(result["subtree_distributional_action"])
    result["leaf_count_rank"] = _rank_within(result["child_leaf_count"])
    result["branch_length_rank"] = _rank_within(result["branch_length"])
    result["edge_statistic_rank"] = _rank_within(result["edge_test_statistic"])
    result["action_minus_leaf_count_rank"] = result["leaf_count_rank"] - result["action_rank"]
    result["action_minus_branch_length_rank"] = (
        result["branch_length_rank"] - result["action_rank"]
    )
    return result


def _summarize_method(group: pd.DataFrame) -> dict[str, object]:
    internal = group[group["child_is_internal"]].copy()
    top_action = group.sort_values("subtree_distributional_action", ascending=False).head(10)
    top_leaf = group.sort_values("child_leaf_count", ascending=False).head(10)
    top_branch = group.sort_values("branch_length", ascending=False).head(10)
    top_action_children = set(top_action["child"])

    tested = group[group["edge_tested"]].copy()
    return {
        "dataset": group["dataset"].iloc[0],
        "dataset_label": group["dataset_label"].iloc[0],
        "geometry": group["geometry"].iloc[0],
        "variant": group["variant"].iloc[0],
        "method": group["method"].iloc[0],
        "n_edges": int(len(group)),
        "n_internal_child_edges": int(len(internal)),
        "edge_open_count": int(group["edge_significant"].sum()),
        "split_filter_filtered_count": int(group["split_filter_filtered"].sum()),
        "split_filter_pass_count": int(group["split_filter_passes"].sum()),
        "median_child_leaf_count": float(group["child_leaf_count"].median()),
        "median_standardized_delta_norm": float(group["standardized_delta_norm"].median()),
        "median_subtree_distributional_action": float(
            group["subtree_distributional_action"].median()
        ),
        "max_subtree_distributional_action": float(
            group["subtree_distributional_action"].max()
        ),
        "spearman_action_vs_leaf_count": _spearman(
            group["subtree_distributional_action"],
            group["child_leaf_count"],
        ),
        "spearman_action_vs_branch_length": _spearman(
            group["subtree_distributional_action"],
            group["branch_length"],
        ),
        "spearman_action_vs_edge_statistic_tested": _spearman(
            tested["subtree_distributional_action"],
            tested["edge_test_statistic"],
        ),
        "spearman_internal_action_vs_leaf_count": _spearman(
            internal["subtree_distributional_action"],
            internal["child_leaf_count"],
        ),
        "spearman_internal_action_vs_branch_length": _spearman(
            internal["subtree_distributional_action"],
            internal["branch_length"],
        ),
        "top10_action_leaf_count_overlap": int(
            len(top_action_children & set(top_leaf["child"]))
        ),
        "top10_action_branch_length_overlap": int(
            len(top_action_children & set(top_branch["child"]))
        ),
        "top_action_edge": str(
            top_action.iloc[0]["parent"] + "->" + top_action.iloc[0]["child"]
        ),
        "top_action_child_leaf_count": int(top_action.iloc[0]["child_leaf_count"]),
        "top_action_standardized_delta_norm": float(
            top_action.iloc[0]["standardized_delta_norm"]
        ),
        "top_action_value": float(top_action.iloc[0]["subtree_distributional_action"]),
        "top_action_branch_length": float(top_action.iloc[0]["branch_length"]),
    }


def _plot_action_vs_branch_length(edges: pd.DataFrame) -> None:
    plot_data = edges[
        (edges["variant"] == "topology_only")
        & (edges["geometry"].isin(["pca_linkage", "adaptive_diffusion"]))
    ].copy()
    panels = list(plot_data.groupby(["dataset_label", "geometry"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, ((dataset_label, geometry), group) in zip(axes_flat, panels, strict=False):
        colors = np.where(group["edge_significant"], "#b23a48", "#52796f")
        sizes = 8.0 + 15.0 * np.log1p(group["child_leaf_count"].to_numpy(dtype=float))
        ax.scatter(
            group["branch_length"].to_numpy(dtype=float) + 1e-6,
            group["subtree_distributional_action"].to_numpy(dtype=float) + 1e-6,
            s=sizes,
            c=colors,
            alpha=0.55,
            linewidths=0,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{dataset_label}\n{geometry.replace('_', ' ')}")
        ax.set_xlabel("branch length + 1e-6")
        ax.set_ylabel("child_count * standardized delta^2")
        ax.grid(alpha=0.2, linewidth=0.6)
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        "Distributional action is not the same object as branch length",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_ROOT / "distributional_action_vs_branch_length.png", dpi=220)
    plt.close(fig)


def _plot_top_internal_edges(edges: pd.DataFrame) -> None:
    focus = edges[
        (edges["geometry"] == "adaptive_diffusion")
        & (edges["variant"] == "topology_only")
        & (edges["child_is_internal"])
    ].copy()
    panels = [
        (label, group.sort_values("subtree_distributional_action", ascending=False).head(12))
        for label, group in focus.groupby("dataset_label", sort=False)
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 7), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (dataset_label, group) in zip(axes, panels, strict=True):
        group = group.iloc[::-1].copy()
        labels = [f"{row.parent}->{row.child}" for row in group.itertuples()]
        ax.barh(labels, group["subtree_distributional_action"], color="#4c78a8")
        for index, row in enumerate(group.itertuples()):
            ax.text(
                row.subtree_distributional_action,
                index,
                f" n={int(row.child_leaf_count)}, d={row.standardized_delta_norm:.2f}",
                va="center",
                ha="left",
                fontsize=7,
            )
        ax.set_title(dataset_label)
        ax.set_xlabel("distributional action")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Top adaptive-diffusion internal edges by mass-weighted distributional movement",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "top_internal_distributional_action_edges.png", dpi=220)
    plt.close(fig)


def _plot_action_vs_edge_statistic(edges: pd.DataFrame) -> None:
    focus = edges[
        (edges["geometry"] == "adaptive_diffusion")
        & (edges["variant"] == "topology_only")
        & (edges["edge_tested"])
    ].copy()
    panels = list(focus.groupby("dataset_label", sort=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5), sharex=False, sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (dataset_label, group) in zip(axes, panels, strict=True):
        colors = np.where(group["edge_significant"], "#b23a48", "#52796f")
        ax.scatter(
            group["subtree_distributional_action"].to_numpy(dtype=float) + 1e-6,
            group["edge_test_statistic"].to_numpy(dtype=float) + 1e-6,
            s=22,
            c=colors,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dataset_label)
        ax.set_xlabel("distributional action + 1e-6")
        ax.set_ylabel("edge projected-Wald statistic + 1e-6")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Mass-weighted movement tracks, but does not duplicate, edge-test evidence",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_ROOT / "distributional_action_vs_edge_statistic.png", dpi=220)
    plt.close(fig)


def _format_top_table(edges: pd.DataFrame, dataset: str, geometry: str) -> str:
    rows = edges[
        (edges["dataset_label"] == dataset)
        & (edges["geometry"] == geometry)
        & (edges["variant"] == "topology_only")
        & (edges["child_is_internal"])
    ].copy()
    rows = rows.sort_values("subtree_distributional_action", ascending=False).head(10)
    columns = [
        "parent",
        "child",
        "child_leaf_count",
        "standardized_delta_norm",
        "subtree_distributional_action",
        "branch_length",
        "edge_test_statistic",
        "edge_significant",
        "child_progenitor_interpretation",
        "child_top_populations",
    ]
    available = [column for column in columns if column in rows.columns]
    return rows[available].to_markdown(index=False, floatfmt=".4f")


def _artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        record["rows"] = int(len(frame))
        record["columns"] = int(len(frame.columns))
    return record


def _write_manifest(generated_at: str) -> None:
    artifact_paths = [
        OUTPUT_ROOT / "scrna_distributional_action_edges.csv",
        OUTPUT_ROOT / "scrna_distributional_action_method_summary.csv",
        OUTPUT_ROOT / "distributional_action_vs_branch_length.png",
        OUTPUT_ROOT / "top_internal_distributional_action_edges.png",
        OUTPUT_ROOT / "distributional_action_vs_edge_statistic.png",
        OUTPUT_ROOT / "scrna_distributional_action_audit.md",
    ]
    manifest = {
        "manifest_schema_version": "static_artifact_provenance/v1",
        "bundle": "scrna_distributional_action_audit_20260624",
        "generated_at": generated_at,
        "source_script": "scripts/audit_scrna_distributional_action.py",
        "source_inputs": [
            str(dataset.output_dir.relative_to(PROJECT_ROOT)) for dataset in DATASETS
        ],
        "artifacts": [_artifact_record(path) for path in artifact_paths],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_report(
    edges: pd.DataFrame,
    summary: pd.DataFrame,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    adaptive_summary = summary[
        (summary["geometry"] == "adaptive_diffusion")
        & (summary["variant"] == "topology_only")
    ].copy()
    pca_summary = summary[
        (summary["geometry"] == "pca_linkage")
        & (summary["variant"] == "topology_only")
    ].copy()

    report = f"""# scRNA Distributional Action Audit

Generated at: {generated_at}

## Summary

The check confirms the concern: descendant leaf count is only sample mass. It is
not enough to describe how much a tree edge or internal node changes the
distribution. The audit adds a mass-weighted movement score

`subtree_distributional_action = child_leaf_count * mean_standardized_PCA_delta^2`

where the delta is the child barycenter minus the parent barycenter in the saved
benchmark PCA space. This is the between-subtree contribution that was missing
from the branch-length-only discussion.

Branch length and distributional action are related but not interchangeable.
Top-action edges do not perfectly overlap the largest subtrees or the longest
branches, so the report/plots should include distributional action when we
interpret which internal nodes matter.

## Method

1. Read `benchmark_subset_pca.csv` and each canonical TBS `*_tree_edges.csv`.
2. Reconstruct every node barycenter bottom-up from `L0`, `L1`, ... leaf rows.
3. Standardize each PCA coordinate over the benchmark subset.
4. For every edge, compute child-parent displacement, child mass, branch length,
   edge-test metadata, and action scores.
5. Join Goncalves progenitor node annotations when the edge child or parent is a
   labeled progenitor analysis node.

The audit does not rerun clustering and does not change TBS. It is a diagnostic
over current benchmark artifacts.

## Adaptive-Diffusion Topology Summary

{adaptive_summary.to_markdown(index=False, floatfmt=".4f")}

## PCA-Linkage Topology Summary

{pca_summary.to_markdown(index=False, floatfmt=".4f")}

## Top Adult Adaptive-Diffusion Internal Edges By Action

{_format_top_table(edges, "Adult pancreas", "adaptive_diffusion")}

## Top Goncalves Adaptive-Diffusion Internal Edges By Action

{_format_top_table(edges, "Goncalves fetal pancreas", "adaptive_diffusion")}

## Interpretation

- `leaf_count` is already present and useful, but it only measures how many
  cells descend from a node.
- `standardized_delta_norm` measures how far the child distribution moves away
  from the parent distribution per cell.
- `subtree_distributional_action` combines both. A node can be important by
  being large, by moving strongly, or by doing both.
- NNLS branch lengths still fit additive path distances. They do not currently
  optimize or display this mass-weighted distributional-action quantity.
- Edge projected-Wald statistics include sample-size effects and covariance
  normalization, so they are closer to the desired notion than branch length,
  but they are threshold/test statistics rather than a direct visual
  contribution score.

## Output Files

- `scrna_distributional_action_edges.csv`
- `scrna_distributional_action_method_summary.csv`
- `distributional_action_vs_branch_length.png`
- `top_internal_distributional_action_edges.png`
- `distributional_action_vs_edge_statistic.png`
"""
    (OUTPUT_ROOT / "scrna_distributional_action_audit.md").write_text(report)
    return generated_at


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    edge_tables = []
    for dataset in DATASETS:
        pca, standardized_pca = _read_pca(dataset.output_dir)
        progenitor_info = (
            _load_goncalves_progenitor_tables(dataset.output_dir)
            if dataset.key == "goncalves_fetal"
            else {}
        )
        for method in METHODS:
            edge_tables.append(
                _edge_records_for_method(
                    dataset,
                    method,
                    pca,
                    standardized_pca,
                    progenitor_info,
                )
            )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The behavior of DataFrame concatenation with empty or all-NA "
                "entries is deprecated.*"
            ),
            category=FutureWarning,
        )
        edges = pd.concat(edge_tables, ignore_index=True)
    summary = pd.DataFrame.from_records(
        [
            _summarize_method(group)
            for _, group in edges.groupby(["dataset", "geometry", "variant"], sort=False)
        ]
    )

    edges.to_csv(OUTPUT_ROOT / "scrna_distributional_action_edges.csv", index=False)
    summary.to_csv(
        OUTPUT_ROOT / "scrna_distributional_action_method_summary.csv",
        index=False,
    )
    _plot_action_vs_branch_length(edges)
    _plot_top_internal_edges(edges)
    _plot_action_vs_edge_statistic(edges)
    _write_report(edges, summary, generated_at)
    _write_manifest(generated_at)
    print(f"Wrote distributional-action audit outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
