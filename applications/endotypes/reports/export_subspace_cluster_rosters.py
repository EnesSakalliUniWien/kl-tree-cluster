#!/usr/bin/env python3
"""Export organized per-subspace cluster rosters, annotations, and radial trees."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import cophenet, cut_tree, leaves_list, to_tree
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from applications.endotypes.reports.build_subspace_gene_annotation_pdf import (  # noqa: E402
    gene_annotation_blurbs,
    load_binary_matrix,
    load_cache,
    local_top_terms_for_cluster,
    protein_label,
    representative_genes_for_cluster,
)

GO_ID_RE = re.compile(r"(GO:\d{7})")
COPY_ARTIFACTS = {
    "cluster_assignments": "cluster_assignments.csv",
    "cluster_coherence": "cluster_coherence.csv",
    "tfidf_cluster_quality": "tfidf_cluster_quality.csv",
    "go_ic_quality_summary": "go_ic_quality_summary.csv",
    "axis_top_terms": "axis_top_terms.csv",
    "axis_term_loadings_all": "axis_term_loadings_all.csv",
    "linkage_matrix": "linkage_matrix.csv",
    "tree_dendrogram_png": "source_rectangular_tree_dendrogram.png",
    "tree_subtree_clusters_png": "source_rectangular_tree_subtree_clusters.png",
    "subspace_embedding_png": "subspace_embedding_clusters.png",
    "adaptive_diffusion_embedding_png": "adaptive_diffusion_embedding_clusters.png",
    "axis_terms_combined_png": "axis_terms_combined.png",
    "subspace_embedding_annotated_terms_png": "subspace_embedding_clusters_annotated_terms.png",
    "adaptive_diffusion_embedding_annotated_terms_png": "adaptive_diffusion_embedding_clusters_annotated_terms.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--genes-per-cluster", type=int, default=10)
    parser.add_argument("--terms-per-cluster", type=int, default=4)
    parser.add_argument("--markdown-gene-preview", type=int, default=18)
    parser.add_argument("--skip-radial-trees", action="store_true")
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")


def parse_go_term(value: object) -> tuple[str, str]:
    text = "" if pd.isna(value) else str(value)
    match = GO_ID_RE.search(text)
    go_id = match.group(1) if match else ""
    term = text.replace(f"({go_id})", "").strip() if go_id else text.strip()
    return term, go_id


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def display_number(value: object, digits: int = 3) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return ""
    if abs(number) < 0.001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def resolve_path(value: object, experiment_dir: Path) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    candidate = experiment_dir / path
    return candidate if candidate.exists() else path


def path_text(path: Path | None) -> str:
    return "" if path is None else str(path)


def subspace_dir_name(row: dict[str, object] | pd.Series) -> str:
    rank = safe_int(row.get("_rank", row.get("specificity_aware_rank", row.get("display_rank"))))
    status = str(row.get("status", ""))
    if rank:
        prefix = f"rank{rank:02d}"
    elif status and status != "ok":
        block_id = safe_int(row.get("block_id"))
        prefix = f"failed{block_id:02d}" if block_id or block_id == 0 else "failed"
    else:
        prefix = "rankNA"
    return f"{prefix}_{safe_name(row.get('weighting', ''))}_{safe_name(row.get('block_name', ''))}"


def load_artifact_index(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "artifact_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    rank_col = (
        "specificity_aware_rank" if "specificity_aware_rank" in frame.columns else "display_rank"
    )
    frame["_rank"] = pd.to_numeric(frame.get(rank_col), errors="coerce")
    if "run_id" not in frame.columns:
        if "method_run_id" in frame.columns:
            frame["run_id"] = frame["method_run_id"].astype(str)
        else:
            frame["run_id"] = (
                "current__adaptive_diffusion_cosine_subspace__"
                + frame["weighting"].astype(str)
                + "__"
                + frame["block_name"].astype(str)
            )
    return frame.sort_values(["_rank", "weighting", "block_name"], na_position="last").reset_index(
        drop=True
    )


def load_optional_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.is_dir():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_linkage_matrix(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    cols = ["left", "right", "distance", "count"]
    if all(col in frame.columns for col in cols):
        return frame[cols].to_numpy(dtype=float)
    return frame.iloc[:, :4].to_numpy(dtype=float)


def cluster_lookup(frame: pd.DataFrame) -> dict[int, dict[str, object]]:
    if frame.empty or "cluster_id" not in frame.columns:
        return {}
    rows: dict[int, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        rows[safe_int(row.get("cluster_id"))] = row
    return rows


def preview_genes(genes: list[str], limit: int) -> str:
    if len(genes) <= limit:
        return "; ".join(genes)
    return "; ".join(genes[:limit]) + f"; ... (+{len(genes) - limit})"


def compute_full_space_embedding(data: pd.DataFrame) -> pd.DataFrame:
    coords = PCA(n_components=2, random_state=1729).fit_transform(data.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "gene": data.index.astype(str),
            "full_space_axis_1": coords[:, 0],
            "full_space_axis_2": coords[:, 1],
        }
    )


def plot_full_space_embedding(
    coordinates: pd.DataFrame,
    assignments: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> pd.DataFrame:
    frame = coordinates.merge(assignments[["gene", "cluster_id"]], on="gene", how="inner")
    frame["cluster_id"] = frame["cluster_id"].map(safe_int)
    palette = cluster_colors(frame["cluster_id"].tolist())
    colors = [palette[int(cluster_id)] for cluster_id in frame["cluster_id"]]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.scatter(
        frame["full_space_axis_1"],
        frame["full_space_axis_2"],
        c=colors,
        s=22,
        alpha=0.88,
        linewidths=0,
    )
    ax.set_title(f"{title}\nfull feature-space PCA by cluster")
    ax.set_xlabel("full-space PC1")
    ax.set_ylabel("full-space PC2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return frame


def plot_subspace_coordinate_embedding(
    coordinates_path: Path | None,
    assignments: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> pd.DataFrame:
    if coordinates_path is None or not coordinates_path.exists():
        return pd.DataFrame()
    coords = pd.read_csv(coordinates_path)
    if "gene" not in coords.columns:
        return pd.DataFrame()
    mode_columns = [col for col in coords.columns if col != "gene"]
    if not mode_columns:
        return pd.DataFrame()
    matrix = coords[mode_columns].to_numpy(dtype=float)
    if matrix.shape[1] == 1:
        embedding = np.column_stack([matrix[:, 0], np.zeros(matrix.shape[0])])
        method = mode_columns[0]
    else:
        embedding = PCA(n_components=2, random_state=1729).fit_transform(matrix)
        method = "subspace PCA"
    embedded = pd.DataFrame(
        {
            "gene": coords["gene"].astype(str),
            "subspace_axis_1": embedding[:, 0],
            "subspace_axis_2": embedding[:, 1],
        }
    )
    frame = embedded.merge(assignments[["gene", "cluster_id"]], on="gene", how="inner")
    frame["cluster_id"] = frame["cluster_id"].map(safe_int)
    palette = cluster_colors(frame["cluster_id"].tolist())
    colors = [palette[int(cluster_id)] for cluster_id in frame["cluster_id"]]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.scatter(
        frame["subspace_axis_1"],
        frame["subspace_axis_2"],
        c=colors,
        s=22,
        alpha=0.88,
        linewidths=0,
    )
    ax.set_title(f"{title}\nsubspace coordinates by cluster ({method})")
    ax.set_xlabel("subspace axis 1")
    ax.set_ylabel("subspace axis 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return frame


def plot_tree_distance_embedding(
    linkage_matrix: np.ndarray,
    assignments: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> pd.DataFrame:
    n_leaves = linkage_matrix.shape[0] + 1
    cophenetic_distances = squareform(cophenet(linkage_matrix))
    embedding = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=1729,
        normalized_stress="auto",
    ).fit_transform(cophenetic_distances)
    frame = pd.DataFrame(
        {
            "gene": assignments["gene"].astype(str).tolist()[:n_leaves],
            "tree_distance_axis_1": embedding[:, 0],
            "tree_distance_axis_2": embedding[:, 1],
            "cluster_id": assignments["cluster_id"].map(safe_int).to_numpy(dtype=int)[:n_leaves],
        }
    )
    palette = cluster_colors(frame["cluster_id"].tolist())
    colors = [palette[int(cluster_id)] for cluster_id in frame["cluster_id"]]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.scatter(
        frame["tree_distance_axis_1"],
        frame["tree_distance_axis_2"],
        c=colors,
        s=22,
        alpha=0.88,
        linewidths=0,
    )
    ax.set_title(f"{title}\ntree-distance MDS by cluster")
    ax.set_xlabel("tree-distance MDS axis 1")
    ax.set_ylabel("tree-distance MDS axis 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return frame


def cluster_colors(cluster_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    unique_ids = sorted(set(int(cluster_id) for cluster_id in cluster_ids))
    if not unique_ids:
        return {}
    colors: dict[int, tuple[float, float, float, float]] = {}
    for idx, cluster_id in enumerate(unique_ids):
        hue = idx / max(len(unique_ids), 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.58)
        colors[cluster_id] = (red, green, blue, 0.94)
    return colors


def diagnostic_cluster_count(artifact_index: pd.DataFrame, n_leaves: int) -> int:
    completed = pd.to_numeric(
        artifact_index.loc[artifact_index["status"].astype(str).eq("ok"), "n_clusters"],
        errors="coerce",
    ).dropna()
    if len(completed):
        count = int(round(float(completed.median())))
    else:
        count = int(round(math.sqrt(n_leaves)))
    return max(2, min(count, n_leaves, 80))


def diagnostic_linkage_assignments(
    linkage_matrix: np.ndarray,
    genes: list[str],
    *,
    n_clusters: int,
) -> pd.DataFrame:
    labels = cut_tree(linkage_matrix, n_clusters=[n_clusters]).reshape(-1).astype(int)
    frame = pd.DataFrame({"gene": genes, "cluster_id": labels})
    sizes = frame.groupby("cluster_id").size().rename("cluster_size")
    return frame.merge(sizes, on="cluster_id", how="left")


def plot_radial_tree(
    linkage_matrix: np.ndarray,
    assignments: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    include_legend: bool = True,
    include_caption: bool = True,
) -> None:
    n_leaves = linkage_matrix.shape[0] + 1
    if len(assignments) != n_leaves:
        raise ValueError(
            f"linkage has {n_leaves} leaves but assignments has {len(assignments)} rows"
        )

    assignments = assignments.reset_index(drop=True).copy()
    assignments["cluster_id"] = assignments["cluster_id"].map(safe_int)
    leaf_cluster = assignments["cluster_id"].to_numpy(dtype=int)
    leaf_order = leaves_list(linkage_matrix).astype(int).tolist()
    theta_by_leaf = {
        leaf_id: (2.0 * math.pi * idx / max(n_leaves, 1)) for idx, leaf_id in enumerate(leaf_order)
    }
    root = to_tree(linkage_matrix, rd=False)
    max_distance = max(float(np.nanmax(linkage_matrix[:, 2])), float(root.dist), 1e-12)
    palette = cluster_colors(leaf_cluster.tolist())
    mixed_color = (0.58, 0.58, 0.58, 0.42)

    leaf_sets: dict[int, list[int]] = {}
    node_positions: dict[int, tuple[float, float]] = {}
    pure_clusters: dict[int, int | None] = {}

    def visit(node: object) -> list[int]:
        if node.is_leaf():
            leaves = [int(node.id)]
            pure = int(leaf_cluster[int(node.id)])
        else:
            leaves = visit(node.left) + visit(node.right)
            child_pure = {pure_clusters[int(node.left.id)], pure_clusters[int(node.right.id)]}
            pure = child_pure.pop() if len(child_pure) == 1 else None
        leaf_sets[int(node.id)] = leaves
        theta = float(np.mean([theta_by_leaf[leaf] for leaf in leaves]))
        radius = 0.08 + 0.90 * (1.0 - float(getattr(node, "dist", 0.0)) / max_distance)
        node_positions[int(node.id)] = (theta, radius)
        pure_clusters[int(node.id)] = pure
        return leaves

    visit(root)

    fig = plt.figure(figsize=(11, 11) if include_legend else (8.5, 8.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_axis_off()
    ax.set_ylim(0, 1.05)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(math.pi / 2)

    def draw_arc(
        theta_a: float, theta_b: float, radius: float, color: tuple[float, float, float, float]
    ) -> None:
        if theta_b < theta_a:
            theta_a, theta_b = theta_b, theta_a
        theta = np.linspace(theta_a, theta_b, max(3, int(abs(theta_b - theta_a) * 80)))
        ax.plot(theta, np.full_like(theta, radius), color=color, lw=0.55, solid_capstyle="round")

    def draw_node(node: object) -> None:
        if node.is_leaf():
            return
        parent_theta, parent_radius = node_positions[int(node.id)]
        for child in (node.left, node.right):
            child_theta, child_radius = node_positions[int(child.id)]
            pure = pure_clusters[int(child.id)]
            color = palette[pure] if pure is not None else mixed_color
            ax.plot([child_theta, child_theta], [parent_radius, child_radius], color=color, lw=0.55)
            draw_arc(child_theta, parent_theta, parent_radius, color)
            draw_node(child)

    draw_node(root)

    leaf_theta = [theta_by_leaf[idx] for idx in range(n_leaves)]
    leaf_color = [palette[int(cluster_id)] for cluster_id in leaf_cluster]
    ax.scatter(leaf_theta, np.full(n_leaves, 1.0), c=leaf_color, s=8, linewidths=0, alpha=0.95)

    cluster_sizes = assignments.groupby("cluster_id").size().sort_values(ascending=False)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[int(cluster_id)],
            markersize=6,
            label=f"C{int(cluster_id)} n={int(size)}",
        )
        for cluster_id, size in cluster_sizes.head(18).items()
    ]
    if len(cluster_sizes) > 18:
        legend_items.append(
            Line2D([0], [0], color=mixed_color, lw=2, label=f"+{len(cluster_sizes) - 18} clusters")
        )
    if include_legend:
        ax.legend(
            handles=legend_items,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(title, fontsize=13, y=0.98)
    if include_caption:
        fig.text(
            0.5,
            0.025,
            "Radial hierarchy from the saved linkage matrix. Leaf points and pure subtrees are colored by final TBS cluster id; mixed ancestral branches are gray.",
            ha="center",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_cluster(
    data: pd.DataFrame,
    genes: list[str],
    *,
    terms_per_cluster: int,
    genes_per_cluster: int,
    cache: dict[str, object],
) -> dict[str, object]:
    available_genes = [gene for gene in genes if gene in data.index]
    if not available_genes:
        return {
            "local_top_terms": "",
            "local_top_go_ids": "",
            "representative_genes": "",
            "representative_gene_proteins": "",
            "representative_gene_annotations": "",
        }
    local_terms = local_top_terms_for_cluster(data, available_genes, top_n=terms_per_cluster)
    top_columns = local_terms["column"].tolist()
    representatives = representative_genes_for_cluster(
        data,
        available_genes,
        top_columns,
        top_n=genes_per_cluster,
    )
    proteins = [protein_label(gene, cache) for gene in representatives]
    annotations = gene_annotation_blurbs(data, representatives, top_columns)
    return {
        "local_top_terms": ";".join(local_terms["go_term"].astype(str).tolist()),
        "local_top_go_ids": ";".join(local_terms["go_id"].astype(str).tolist()),
        "representative_genes": ";".join(representatives),
        "representative_gene_proteins": ";".join(proteins),
        "representative_gene_annotations": " | ".join(annotations),
    }


def build_rosters(
    artifact_index: pd.DataFrame,
    *,
    experiment_dir: Path,
    output_dir: Path,
    data: pd.DataFrame,
    full_space_coordinates: pd.DataFrame,
    cache: dict[str, object],
    terms_per_cluster: int,
    genes_per_cluster: int,
    render_radial_trees: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    default_diagnostic_clusters = diagnostic_cluster_count(artifact_index, len(data.index))

    for row in artifact_index.to_dict(orient="records"):
        status = str(row.get("status", ""))
        run_id = str(row.get("run_id", ""))
        rank = safe_int(row.get("_rank"))
        weighting = str(row.get("weighting", ""))
        block_name = str(row.get("block_name", ""))
        organized_subspace_dir = output_dir / "subspaces" / subspace_dir_name(row)
        organized_subspace_dir.mkdir(parents=True, exist_ok=True)
        source_artifact_dir = organized_subspace_dir / "source_artifacts"
        source_artifact_dir.mkdir(parents=True, exist_ok=True)

        resolved_paths = {key: resolve_path(row.get(key), experiment_dir) for key in COPY_ARTIFACTS}
        assignments_path = resolved_paths["cluster_assignments"]
        coherence_path = resolved_paths["cluster_coherence"]
        tfidf_path = resolved_paths["tfidf_cluster_quality"]
        linkage_path = resolved_paths["linkage_matrix"]
        linkage_matrix = load_linkage_matrix(linkage_path)
        source_subspace_dir = resolve_path(row.get("subspace_dir"), experiment_dir)
        subspace_coordinates_path = None
        if (
            source_subspace_dir is not None
            and source_subspace_dir.exists()
            and source_subspace_dir.is_dir()
        ):
            coordinate_candidates = sorted(source_subspace_dir.glob("*__subspace_coordinates.csv"))
            subspace_coordinates_path = coordinate_candidates[0] if coordinate_candidates else None

        assignments = load_optional_table(assignments_path)
        coherence = cluster_lookup(load_optional_table(coherence_path))
        tfidf_quality = cluster_lookup(load_optional_table(tfidf_path))
        assignment_source = "accepted_tbs" if status == "ok" and not assignments.empty else ""
        if assignments.empty and linkage_matrix is not None:
            assignments = diagnostic_linkage_assignments(
                linkage_matrix,
                data.index.astype(str).tolist(),
                n_clusters=default_diagnostic_clusters,
            )
            assignment_source = "diagnostic_linkage_cut"
        if not assignments.empty:
            assignments["gene"] = assignments["gene"].astype(str)
            assignments["cluster_id"] = assignments["cluster_id"].map(safe_int)
            sizes = assignments.groupby("cluster_id").size().rename("cluster_size")
            assignments = assignments.drop(columns=["cluster_size"], errors="ignore").merge(
                sizes,
                on="cluster_id",
                how="left",
            )
        has_assignments = not assignments.empty

        copied_rows = []
        for key, target_name in COPY_ARTIFACTS.items():
            source = resolved_paths[key]
            copied = ""
            exists = bool(source is not None and source.exists() and source.is_file())
            if exists:
                target = source_artifact_dir / target_name
                shutil.copy2(source, target)
                copied = str(target)
            copied_rows.append(
                {
                    "artifact": key,
                    "source_path": path_text(source),
                    "copied_path": copied,
                    "exists": exists,
                }
            )
        if subspace_coordinates_path is not None and subspace_coordinates_path.exists():
            target = source_artifact_dir / "subspace_coordinates.csv"
            shutil.copy2(subspace_coordinates_path, target)
            copied_rows.append(
                {
                    "artifact": "subspace_coordinates",
                    "source_path": str(subspace_coordinates_path),
                    "copied_path": str(target),
                    "exists": True,
                }
            )
        else:
            copied_rows.append(
                {
                    "artifact": "subspace_coordinates",
                    "source_path": path_text(subspace_coordinates_path),
                    "copied_path": "",
                    "exists": False,
                }
            )
        pd.DataFrame(copied_rows).to_csv(
            organized_subspace_dir / "source_artifacts_manifest.csv", index=False
        )

        radial_path = organized_subspace_dir / "radial_tree_clusters.png"
        radial_compact_path = organized_subspace_dir / "radial_tree_clusters_compact.png"
        full_space_path = organized_subspace_dir / "full_space_embedding_clusters.png"
        full_space_coordinates_path = (
            organized_subspace_dir / "full_space_embedding_cluster_coordinates.csv"
        )
        subspace_embedding_path = organized_subspace_dir / "subspace_embedding_clusters.png"
        subspace_embedding_coordinates_path = (
            organized_subspace_dir / "subspace_embedding_cluster_coordinates.csv"
        )
        tree_distance_embedding_path = (
            organized_subspace_dir / "tree_distance_embedding_clusters.png"
        )
        tree_distance_embedding_coordinates_path = (
            organized_subspace_dir / "tree_distance_embedding_cluster_coordinates.csv"
        )
        diagnostic_assignment_path = (
            organized_subspace_dir / "diagnostic_linkage_cluster_assignments.csv"
        )
        accepted_assignment_path = organized_subspace_dir / "accepted_tbs_cluster_assignments.csv"
        radial_rendered = False
        full_space_rendered = False
        subspace_embedding_rendered = False
        tree_distance_embedding_rendered = False
        if has_assignments and assignment_source == "diagnostic_linkage_cut":
            assignments.to_csv(diagnostic_assignment_path, index=False)
        if has_assignments and assignment_source == "accepted_tbs":
            assignments.to_csv(accepted_assignment_path, index=False)
        if has_assignments and render_radial_trees:
            if linkage_matrix is not None:
                plot_radial_tree(
                    linkage_matrix,
                    assignments,
                    radial_path,
                    title=f"{assignment_source}: {weighting} / {block_name}",
                )
                plot_radial_tree(
                    linkage_matrix,
                    assignments,
                    radial_compact_path,
                    title=f"{assignment_source}: {weighting} / {block_name}",
                    include_legend=False,
                    include_caption=False,
                )
                radial_rendered = True
        if has_assignments:
            full_space_frame = plot_full_space_embedding(
                full_space_coordinates,
                assignments,
                full_space_path,
                title=f"{assignment_source}: {weighting} / {block_name}",
            )
            full_space_frame.to_csv(full_space_coordinates_path, index=False)
            full_space_rendered = True
            subspace_frame = plot_subspace_coordinate_embedding(
                subspace_coordinates_path,
                assignments,
                subspace_embedding_path,
                title=f"{assignment_source}: {weighting} / {block_name}",
            )
            if not subspace_frame.empty:
                subspace_frame.to_csv(subspace_embedding_coordinates_path, index=False)
                subspace_embedding_rendered = True
            if linkage_matrix is not None:
                tree_distance_frame = plot_tree_distance_embedding(
                    linkage_matrix,
                    assignments,
                    tree_distance_embedding_path,
                    title=f"{assignment_source}: {weighting} / {block_name}",
                )
                tree_distance_frame.to_csv(tree_distance_embedding_coordinates_path, index=False)
                tree_distance_embedding_rendered = True

        status_record = {
            "run_id": run_id,
            "specificity_aware_rank": rank if rank else math.nan,
            "display_rank": safe_int(row.get("display_rank")) or math.nan,
            "weighting": weighting,
            "block_name": block_name,
            "status": status,
            "assignment_source": assignment_source,
            "has_cluster_assignments": bool(has_assignments),
            "n_genes": len(assignments) if has_assignments else 0,
            "n_unique_genes": assignments["gene"].nunique() if has_assignments else 0,
            "linkage_leaves": int(linkage_matrix.shape[0] + 1) if linkage_matrix is not None else 0,
            "n_clusters": assignments["cluster_id"].nunique()
            if has_assignments
            else safe_int(row.get("n_clusters")),
            "coherent_cluster_count": safe_int(row.get("coherent_cluster_count")),
            "organized_subspace_dir": str(organized_subspace_dir),
            "radial_tree_clusters_png": str(radial_path) if radial_rendered else "",
            "radial_tree_clusters_compact_png": str(radial_compact_path) if radial_rendered else "",
            "full_space_embedding_clusters_png": str(full_space_path)
            if full_space_rendered
            else "",
            "full_space_embedding_cluster_coordinates": str(full_space_coordinates_path)
            if full_space_rendered
            else "",
            "subspace_embedding_clusters_png": str(subspace_embedding_path)
            if subspace_embedding_rendered
            else "",
            "subspace_embedding_cluster_coordinates": str(subspace_embedding_coordinates_path)
            if subspace_embedding_rendered
            else "",
            "tree_distance_embedding_clusters_png": str(tree_distance_embedding_path)
            if tree_distance_embedding_rendered
            else "",
            "tree_distance_embedding_cluster_coordinates": str(
                tree_distance_embedding_coordinates_path
            )
            if tree_distance_embedding_rendered
            else "",
            "diagnostic_linkage_cluster_assignments": str(diagnostic_assignment_path)
            if assignment_source == "diagnostic_linkage_cut"
            else "",
            "accepted_tbs_cluster_assignments": str(accepted_assignment_path)
            if assignment_source == "accepted_tbs"
            else "",
            "failure_status": row.get("failure_status", ""),
            **{f"{key}_path": path_text(value) for key, value in resolved_paths.items()},
            "subspace_coordinates_path": path_text(subspace_coordinates_path),
        }
        status_rows.append(status_record)
        pd.DataFrame([status_record]).to_csv(
            organized_subspace_dir / "subspace_status.csv", index=False
        )

        if not has_assignments:
            write_subspace_readme(
                organized_subspace_dir, status_record, cluster_count=0, gene_memberships=0
            )
            continue

        grouped = assignments.groupby("cluster_id", sort=True)
        subspace_cluster_rows: list[dict[str, object]] = []
        subspace_gene_rows: list[dict[str, object]] = []
        for cluster_id, cluster_frame in grouped:
            genes = sorted(cluster_frame["gene"].astype(str).tolist())
            coherence_row = coherence.get(safe_int(cluster_id), {})
            tfidf_row = tfidf_quality.get(safe_int(cluster_id), {})
            top_term, top_go_id = parse_go_term(coherence_row.get("top_term", ""))
            cluster_size = len(genes)
            recorded_size = safe_int(
                coherence_row.get("cluster_size"), safe_int(cluster_frame["cluster_size"].iloc[0])
            )
            annotation = annotate_cluster(
                data,
                genes,
                terms_per_cluster=terms_per_cluster,
                genes_per_cluster=genes_per_cluster,
                cache=cache,
            )
            if not top_term and annotation["local_top_terms"]:
                top_terms = str(annotation["local_top_terms"]).split(";")
                top_ids = str(annotation["local_top_go_ids"]).split(";")
                top_term = top_terms[0] if top_terms else ""
                top_go_id = top_ids[0] if top_ids else ""
            base = {
                "run_id": run_id,
                "specificity_aware_rank": rank,
                "display_rank": safe_int(row.get("display_rank")),
                "weighting": weighting,
                "block_name": block_name,
                "status": status,
                "assignment_source": assignment_source,
                "organized_subspace_dir": str(organized_subspace_dir),
                "radial_tree_clusters_png": str(radial_path) if radial_rendered else "",
                "radial_tree_clusters_compact_png": str(radial_compact_path)
                if radial_rendered
                else "",
                "full_space_embedding_clusters_png": str(full_space_path)
                if full_space_rendered
                else "",
                "full_space_embedding_cluster_coordinates": str(full_space_coordinates_path)
                if full_space_rendered
                else "",
                "subspace_embedding_clusters_png": str(subspace_embedding_path)
                if subspace_embedding_rendered
                else "",
                "subspace_embedding_cluster_coordinates": str(subspace_embedding_coordinates_path)
                if subspace_embedding_rendered
                else "",
                "tree_distance_embedding_clusters_png": str(tree_distance_embedding_path)
                if tree_distance_embedding_rendered
                else "",
                "tree_distance_embedding_cluster_coordinates": str(
                    tree_distance_embedding_coordinates_path
                )
                if tree_distance_embedding_rendered
                else "",
                "cluster_id": safe_int(cluster_id),
                "cluster_size": cluster_size,
                "recorded_cluster_size": recorded_size,
                "coherent_by_rule": coherence_row.get("coherent_by_rule", ""),
                "n_significant_terms_q05": safe_int(coherence_row.get("n_significant_terms_q05")),
                "min_q_value": safe_float(coherence_row.get("min_q_value")),
                "top_term": top_term,
                "top_go_id": top_go_id,
                "top_term_prevalence_delta": safe_float(
                    coherence_row.get("top_term_prevalence_delta")
                ),
                "mean_within_tfidf_cosine": safe_float(tfidf_row.get("mean_within_tfidf_cosine")),
                **annotation,
            }
            cluster_record = {**base, "member_genes": ";".join(genes)}
            subspace_cluster_rows.append(cluster_record)
            cluster_rows.append(cluster_record)
            for gene in genes:
                gene_record = {**base, "gene": gene}
                subspace_gene_rows.append(gene_record)
                gene_rows.append(gene_record)

        subspace_clusters = pd.DataFrame(subspace_cluster_rows)
        subspace_genes = pd.DataFrame(subspace_gene_rows)
        subspace_clusters.to_csv(organized_subspace_dir / "cluster_roster.csv", index=False)
        subspace_clusters.drop(columns=["member_genes"]).to_csv(
            organized_subspace_dir / "cluster_annotations.csv", index=False
        )
        subspace_genes.to_csv(organized_subspace_dir / "gene_membership.csv", index=False)
        write_subspace_readme(
            organized_subspace_dir,
            status_record,
            cluster_count=len(subspace_clusters),
            gene_memberships=len(subspace_genes),
        )

    return pd.DataFrame(cluster_rows), pd.DataFrame(gene_rows), pd.DataFrame(status_rows)


def write_subspace_readme(
    path: Path,
    status: dict[str, object],
    *,
    cluster_count: int,
    gene_memberships: int,
) -> None:
    lines = [
        f"# {status.get('weighting')} / {status.get('block_name')}",
        "",
        f"Run ID: `{status.get('run_id')}`",
        f"Status: `{status.get('status')}`",
        f"Assignment source: `{status.get('assignment_source')}`",
        f"Specificity-aware rank: `{status.get('specificity_aware_rank')}`",
        f"Linkage leaves: `{status.get('linkage_leaves')}`",
        f"Assigned genes: `{status.get('n_unique_genes')}`",
        f"Clusters exported: `{cluster_count}`",
        f"Gene memberships exported: `{gene_memberships}`",
        "",
        "Files:",
        "- `cluster_roster.csv`: one row per cluster with complete member-gene lists and local GO annotations.",
        "- `cluster_annotations.csv`: cluster-level annotation fields without the long member-gene cell.",
        "- `gene_membership.csv`: one row per gene in this subspace with its cluster annotation context.",
        "- `accepted_tbs_cluster_assignments.csv`: accepted final TBS assignments when the TBS gate completed.",
        "- `diagnostic_linkage_cluster_assignments.csv`: diagnostic linkage-cut assignments when the TBS gate failed.",
        "- `radial_tree_clusters.png`: radial hierarchy colored by the recorded cluster id when linkage and assignments are available.",
        "- `radial_tree_clusters_compact.png`: compact radial hierarchy used inside the annotation PDF.",
        "- `full_space_embedding_clusters.png`: full feature-matrix PCA coordinates colored by this subspace's cluster ids.",
        "- `full_space_embedding_cluster_coordinates.csv`: full-space PCA coordinates joined to this subspace's cluster ids.",
        "- `subspace_embedding_clusters.png`: regenerated subspace-coordinate embedding colored by this subspace's cluster ids.",
        "- `subspace_embedding_cluster_coordinates.csv`: subspace coordinates joined to this subspace's cluster ids.",
        "- `tree_distance_embedding_clusters.png`: tree-distance MDS embedding from the saved linkage matrix colored by cluster id.",
        "- `tree_distance_embedding_cluster_coordinates.csv`: tree-distance MDS coordinates joined to cluster ids.",
        "- `source_artifacts/`: copied source tables and plots from the canonical subspace experiment.",
        "- `source_artifacts_manifest.csv`: copied-source inventory.",
        "- `subspace_status.csv`: one-row status record.",
    ]
    failure = str(status.get("failure_status", "") or "")
    if failure:
        lines.extend(["", "Failure status:", "", f"`{failure}`"])
    path.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_markdown(
    path: Path,
    *,
    dataset_label: str,
    experiment_dir: Path,
    status: pd.DataFrame,
    clusters: pd.DataFrame,
    gene_preview: int,
) -> None:
    lines: list[str] = [
        f"# {dataset_label} Subspace Cluster Roster",
        "",
        f"Experiment: `{experiment_dir}`",
        "",
        "This file lists every exported cluster for each accepted TBS or diagnostic linkage-cut subspace. The companion",
        "`subspace_cluster_roster.csv` keeps the complete semicolon-delimited gene list for",
        "each cluster; per-subspace folders under `subspaces/` hold the same data split by",
        "subspace plus radial tree plots, embedding plots, and copied source artifacts.",
        "",
        "Rows with `assignment_source=accepted_tbs` are final accepted TBS assignments. Rows with",
        "`assignment_source=diagnostic_linkage_cut` are diagnostic cuts from saved linkage trees",
        "for failed gates, not accepted TBS output.",
        "",
    ]
    assigned = status[status["has_cluster_assignments"].eq(True)]
    unassigned = status[~status["has_cluster_assignments"].eq(True)]
    lines.extend(
        [
            f"Subspaces with cluster rosters: {len(assigned)}",
            f"Subspaces without cluster rosters: {len(unassigned)}",
            f"Total cluster rows: {len(clusters)}",
            "",
        ]
    )
    for subspace in (
        assigned.sort_values(["weighting", "block_name"])
        .sort_values("specificity_aware_rank", na_position="last")
        .to_dict(orient="records")
    ):
        mask = clusters["run_id"].eq(subspace["run_id"])
        sub_clusters = clusters.loc[mask].sort_values(["cluster_id"])
        lines.extend(
            [
                f"## Rank {subspace['specificity_aware_rank']}: {subspace['weighting']} / {subspace['block_name']}",
                "",
                f"Directory: `{subspace['organized_subspace_dir']}`",
                f"Assignment source: `{subspace['assignment_source']}`",
                f"Leaves/genes: `{subspace['linkage_leaves']}` leaves, `{subspace['n_unique_genes']}` assigned genes",
                f"Radial tree: `{subspace['radial_tree_clusters_png']}`",
                f"Full-space embedding: `{subspace['full_space_embedding_clusters_png']}`",
                f"Tree-distance embedding: `{subspace['tree_distance_embedding_clusters_png']}`",
                "",
                f"Clusters: {len(sub_clusters)}; coherent clusters: {subspace['coherent_cluster_count']}",
                "",
                "| cluster | size | coherent | top GO term | local GO terms | representative genes | members |",
                "|---:|---:|:---:|---|---|---|---|",
            ]
        )
        for cluster in sub_clusters.to_dict(orient="records"):
            genes = (
                str(cluster.get("member_genes", "")).split(";")
                if cluster.get("member_genes")
                else []
            )
            top_term = str(cluster.get("top_term", ""))
            top_go_id = str(cluster.get("top_go_id", ""))
            if top_go_id:
                top_term = f"{top_term} ({top_go_id})"
            coherent = cluster.get("coherent_by_rule")
            coherent_text = (
                "yes"
                if str(coherent).lower() == "true"
                else "no"
                if str(coherent).lower() == "false"
                else ""
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(cluster.get("cluster_id", "")),
                        str(cluster.get("cluster_size", "")),
                        coherent_text,
                        top_term.replace("|", "/"),
                        str(cluster.get("local_top_terms", "")).replace("|", "/"),
                        str(cluster.get("representative_genes", "")).replace("|", "/"),
                        preview_genes(genes, gene_preview).replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")

    if not unassigned.empty:
        lines.extend(
            [
                "## Subspaces Without Cluster Rosters",
                "",
                "| rank | weighting | block | status | failure |",
                "|---:|---|---|---|---|",
            ]
        )
        for subspace in (
            unassigned.sort_values(["weighting", "block_name"])
            .sort_values("specificity_aware_rank", na_position="last")
            .to_dict(orient="records")
        ):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(subspace.get("specificity_aware_rank", "")),
                        str(subspace.get("weighting", "")),
                        str(subspace.get("block_name", "")),
                        str(subspace.get("status", "")),
                        str(subspace.get("failure_status", "")).replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_radial_pdf(path: Path, status: pd.DataFrame) -> None:
    rows = (
        status[status["radial_tree_clusters_png"].astype(str).ne("")]
        .sort_values(["weighting", "block_name"])
        .sort_values("specificity_aware_rank", na_position="last")
    )
    if rows.empty:
        return
    with PdfPages(path) as pdf:
        for row in rows.to_dict(orient="records"):
            image_path = Path(str(row["radial_tree_clusters_png"]))
            if not image_path.exists():
                continue
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.axis("off")
            ax.imshow(plt.imread(str(image_path)))
            ax.set_title(
                f"{row['assignment_source']}: {row['weighting']} / {row['block_name']}",
                loc="left",
                fontsize=12,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    output_dir = args.output_dir or experiment_dir / "systematic_subspace_gene_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "subspaces").exists():
        shutil.rmtree(output_dir / "subspaces")
    dataset_label = args.dataset_label or experiment_dir.name

    artifact_index = load_artifact_index(experiment_dir)
    data = load_binary_matrix(args.feature_matrix)
    full_space_coordinates = compute_full_space_embedding(data)
    full_space_coordinates.to_csv(output_dir / "full_space_embedding_coordinates.csv", index=False)
    cache = load_cache(output_dir / "life_science_lookup_cache.json")
    clusters, genes, status = build_rosters(
        artifact_index,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        data=data,
        full_space_coordinates=full_space_coordinates,
        cache=cache,
        terms_per_cluster=args.terms_per_cluster,
        genes_per_cluster=args.genes_per_cluster,
        render_radial_trees=not args.skip_radial_trees,
    )

    clusters.to_csv(output_dir / "subspace_cluster_roster.csv", index=False)
    genes.to_csv(output_dir / "subspace_gene_membership_long.csv", index=False)
    status.to_csv(output_dir / "subspace_cluster_status.csv", index=False)
    write_root_markdown(
        output_dir / "subspace_cluster_roster.md",
        dataset_label=dataset_label,
        experiment_dir=experiment_dir,
        status=status,
        clusters=clusters,
        gene_preview=args.markdown_gene_preview,
    )
    write_radial_pdf(
        output_dir / f"{safe_name(dataset_label).lower()}_radial_tree_clusters.pdf", status
    )
    structure = {
        "dataset_label": dataset_label,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "root_files": [
            "subspace_cluster_roster.csv",
            "subspace_gene_membership_long.csv",
            "subspace_cluster_status.csv",
            "subspace_cluster_roster.md",
            "full_space_embedding_coordinates.csv",
            f"{safe_name(dataset_label).lower()}_radial_tree_clusters.pdf",
        ],
        "subspace_directory_patterns": [
            "subspaces/rank##_weighting_block_name/",
            "subspaces/failed##_weighting_block_name/",
        ],
        "assignment_sources": {
            "accepted_tbs": "final accepted TBS assignments",
            "diagnostic_linkage_cut": "diagnostic linkage-tree cuts for failed gates",
        },
    }
    (output_dir / "directory_structure.json").write_text(
        json.dumps(structure, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(clusters)} clusters and {len(genes)} gene memberships to {output_dir}")


if __name__ == "__main__":
    main()
