#!/usr/bin/env python3
"""Rank and plot allGO tree assignments by GO-annotation information criteria."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import hypergeom
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import MDS
from statsmodels.stats.multitest import multipletests
from tree_break_selection.space_separation import (
    SpectralBlock,
    adaptive_diffusion_distance,
    block_adaptive_diffusion_distance,
    block_diffusion_distance,
    coordinates_for_block,
    cosine_eigendecomposition,
    weight_feature_matrix,
)

from applications.endotypes._shared import safe_name

RESULT_PREFIX = "allgo_new_quality_aware_go_ic"


@dataclass
class Candidate:
    family: str
    run_id: str
    weighting: str
    block_name: str
    block_start: int | None
    block_end: int | None
    assignments_path: Path
    summary: dict[str, object]


@dataclass
class CandidateArtifact:
    candidate: Candidate
    assignments: pd.DataFrame
    labels: np.ndarray
    linkage_matrix: np.ndarray | None
    subspace_embedding: np.ndarray | None
    ranking: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-kak-dir", type=Path)
    parser.add_argument("--adaptive-diffusion-kak-dir", type=Path)
    parser.add_argument("--whole-adaptive-dir", type=Path)
    parser.add_argument(
        "--method-matrix-summary",
        type=Path,
        help=(
            "Generic method/tree-geometry summary CSV produced by "
            "run_allgo_method_version_tree_matrix.py."
        ),
    )
    parser.add_argument("--global-embedding", type=Path)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--plot-all", action="store_true", default=True)
    parser.add_argument("--top-summary-count", type=int, default=12)
    return parser.parse_args()


def reader_family(family: str) -> str:
    return {
        "adaptive_diffusion_kak": "adaptive_diffusion_cosine_subspace",
        "raw_kak": "raw_cosine_subspace",
    }.get(family, family)


def reader_run_id(run_id: str) -> str:
    return run_id.replace(
        "adaptive_diffusion_kak__", "adaptive_diffusion_cosine_subspace__"
    ).replace("raw_kak__", "raw_cosine_subspace__")


def display_label(run_id: str) -> str:
    replacements = {
        "whole_adaptive_diffusion": "whole adaptive diffusion",
    }
    if run_id in replacements:
        return replacements[run_id]
    label = run_id
    label = label.replace("adaptive_diffusion_kak__", "adaptive diffusion cosine subspace, ")
    label = label.replace(
        "adaptive_diffusion_cosine_subspace__", "adaptive diffusion cosine subspace, "
    )
    label = label.replace("raw_kak__", "raw cosine subspace, ")
    label = label.replace("raw_cosine_subspace__", "raw cosine subspace, ")
    label = label.replace("current__", "current, ")
    label = label.replace("whole_adaptive_diffusion__", "whole adaptive diffusion, ")
    label = label.replace("__", ", ")
    label = label.replace("adaptive_modes_", "modes ")
    label = label.replace("adaptive_common_mode_01", "common mode 01")
    label = label.replace("_", " ")
    return label


def wrapped_label(run_id: str, width: int = 48) -> str:
    return "\n".join(textwrap.wrap(display_label(run_id), width=width))


def tier_color(label: str) -> str:
    return {
        "quality_plausible": "#4c78a8",
        "broad_coherent": "#72b7b2",
        "quality_mixed": "#f58518",
        "degenerate": "#b279a2",
    }.get(label, "#777777")


def load_binary_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data = data.apply(pd.to_numeric, errors="raise")
    values = data.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{path} contains values outside {{0,1}}.")
    zero_columns = data.columns[(data.sum(axis=0) == 0).to_numpy()]
    if len(zero_columns):
        data = data.drop(columns=zero_columns)
    zero_rows = data.index[(data.sum(axis=1) == 0).to_numpy()]
    if len(zero_rows):
        raise ValueError(f"Rows with no active GO terms: {list(zero_rows[:10])!r}")
    return data.astype(int)


def load_global_embedding(path: Path | None, data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if path is not None and path.exists():
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        frame = pd.read_csv(path, sep=sep)
        first = frame.columns[0]
        if first != "gene":
            frame = frame.rename(columns={first: "gene"})
        frame["gene"] = frame["gene"].astype(str)
        numeric = [
            col
            for col in frame.columns
            if col != "gene" and pd.api.types.is_numeric_dtype(frame[col])
        ]
        numeric = [col for col in numeric if col.lower() not in {"cluster_id", "cluster"}]
        if len(numeric) >= 2:
            return frame[["gene", numeric[0], numeric[1]]].copy(), numeric[0], numeric[1]

    values = data.to_numpy(dtype=float)
    n_components = min(50, values.shape[0] - 1, values.shape[1])
    reduced = TruncatedSVD(n_components=n_components, random_state=1729).fit_transform(values)
    try:
        import umap

        embedding = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=1729,
        ).fit_transform(reduced)
        x_col, y_col = "UMAP-1", "UMAP-2"
    except Exception:
        embedding = PCA(n_components=2, random_state=1729).fit_transform(reduced)
        x_col, y_col = "PCA-1", "PCA-2"
    return (
        pd.DataFrame(
            {"gene": data.index.astype(str), x_col: embedding[:, 0], y_col: embedding[:, 1]}
        ),
        x_col,
        y_col,
    )


def load_assignment(path: Path, data_index: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        if "sample_id" in frame.columns:
            frame = frame.rename(columns={"sample_id": "gene"})
        elif "Unnamed: 0" in frame.columns:
            frame = frame.rename(columns={"Unnamed: 0": "gene"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "gene"})
    if "cluster_id" not in frame.columns:
        raise ValueError(f"{path} has no cluster_id column.")
    frame["gene"] = frame["gene"].astype(str)
    frame = frame.set_index("gene").reindex(data_index.astype(str))
    if frame["cluster_id"].isna().any():
        missing = frame.index[frame["cluster_id"].isna()].tolist()[:10]
        raise ValueError(f"{path} is missing assignment rows, e.g. {missing!r}.")
    labels = pd.factorize(frame["cluster_id"].astype(int), sort=True)[0].astype(int)
    sizes = pd.Series(labels, index=frame.index).value_counts()
    out = pd.DataFrame({"gene": frame.index, "cluster_id": labels})
    out["cluster_size"] = out["cluster_id"].map(sizes).astype(int)
    return out


def add_candidates_from_raw_kak(path: Path | None) -> list[Candidate]:
    if path is None:
        return []
    summary_path = path / "matrix_kak_probe_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    rows: list[Candidate] = []
    for _, row in summary[summary["status"].eq("ok")].iterrows():
        run_id = f"raw_kak__{row['weighting']}__{row['block_name']}"
        rows.append(
            Candidate(
                family="raw_kak",
                run_id=run_id,
                weighting=str(row["weighting"]),
                block_name=str(row["block_name"]),
                block_start=int(row["block_start"]),
                block_end=int(row["block_end"]),
                assignments_path=Path(str(row["assignments_path"])),
                summary=row.to_dict(),
            )
        )
    return rows


def add_candidates_from_diffusion_kak(path: Path | None) -> list[Candidate]:
    if path is None:
        return []
    summary_path = path / "matrix_kak_diffusion_probe_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    rows: list[Candidate] = []
    for _, row in summary[summary["status"].eq("ok")].iterrows():
        run_id = f"adaptive_diffusion_kak__{row['weighting']}__{row['block_name']}"
        rows.append(
            Candidate(
                family="adaptive_diffusion_kak",
                run_id=run_id,
                weighting=str(row["weighting"]),
                block_name=str(row["block_name"]),
                block_start=int(row["block_start"]),
                block_end=int(row["block_end"]),
                assignments_path=Path(str(row["assignments_path"])),
                summary=row.to_dict(),
            )
        )
    return rows


def add_whole_adaptive_candidate(path: Path | None) -> list[Candidate]:
    if path is None:
        return []
    assignments = path / "cluster_assignments.csv"
    summary = path / "summary.json"
    if not assignments.exists():
        return []
    meta: dict[str, object] = {}
    if summary.exists():
        meta = json.loads(summary.read_text(encoding="utf-8"))
    return [
        Candidate(
            family="whole_adaptive_diffusion",
            run_id="whole_adaptive_diffusion",
            weighting="whole",
            block_name="whole_matrix",
            block_start=None,
            block_end=None,
            assignments_path=assignments,
            summary=meta,
        )
    ]


def add_candidates_from_method_matrix(summary_path: Path | None) -> list[Candidate]:
    if summary_path is None:
        return []
    if summary_path.is_dir():
        summary_path = summary_path / "method_tree_matrix_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    if summary.empty or "status" not in summary.columns:
        return []
    rows: list[Candidate] = []
    for _, row in summary[summary["status"].eq("ok")].iterrows():
        tree_geometry = str(row.get("tree_geometry", ""))
        method_version = str(row.get("method_version", ""))
        family = str(row.get("method_family", f"{method_version}__{tree_geometry}"))
        run_id = str(row.get("run_id", ""))
        weighting = str(row.get("weighting", "whole"))
        block_name = str(row.get("block_name", "whole_matrix"))

        def optional_int(value: object) -> int | None:
            if pd.isna(value):
                return None
            return int(float(value))

        rows.append(
            Candidate(
                family=family,
                run_id=reader_run_id(run_id),
                weighting=weighting,
                block_name=block_name,
                block_start=optional_int(row.get("block_start")),
                block_end=optional_int(row.get("block_end")),
                assignments_path=Path(str(row["assignments_path"])),
                summary={**row.to_dict(), "tree_geometry": tree_geometry},
            )
        )
    return rows


def eigensystems_for_data(
    data: pd.DataFrame, max_rank: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    systems: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for weighting in ("binary", "tfidf"):
        values = weight_feature_matrix(data, weighting)
        systems[weighting] = cosine_eigendecomposition(values, max_rank)
    return systems


def candidate_subspace_and_tree(
    candidate: Candidate,
    data: pd.DataFrame,
    eigensystems: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    tree_geometry = str(candidate.summary.get("tree_geometry", ""))
    if tree_geometry == "":
        if candidate.family == "whole_adaptive_diffusion":
            tree_geometry = "whole_adaptive_diffusion"
        elif candidate.family == "raw_kak":
            tree_geometry = "raw_cosine_subspace"
        elif candidate.family == "adaptive_diffusion_kak":
            tree_geometry = "adaptive_diffusion_cosine_subspace"

    if tree_geometry == "whole_adaptive_diffusion":
        distances, _metadata = adaptive_diffusion_distance(
            data,
            k_neighbors=15,
            diffusion_time=3,
            n_components=30,
            metric="hamming",
            bandwidth_type="-1/(d+2)",
            epsilon="median",
            return_metadata=True,
        )
        matrix = squareform(distances)
        embedding = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=1729,
            normalized_stress="auto",
        ).fit_transform(matrix)
        return embedding, distances

    if candidate.weighting not in eigensystems:
        return None, None
    eigvals, eigvecs = eigensystems[candidate.weighting]
    if candidate.block_start is None or candidate.block_end is None:
        return None, None
    block = SpectralBlock(
        block_id=0,
        block_name=candidate.block_name,
        block_start=int(candidate.block_start),
        block_end=int(candidate.block_end),
        block_type=str(candidate.summary.get("block_type", "")),
    )
    coords = coordinates_for_block(eigvals, eigvecs, block)
    if tree_geometry == "raw_cosine_subspace":
        return coords, pdist(coords, metric="euclidean")
    if tree_geometry == "adaptive_diffusion_cosine_subspace":
        mode = str(candidate.summary.get("diffusion_mode", "adaptive"))
        if mode == "fixed":
            distances, _metadata = block_diffusion_distance(
                coords,
                k_neighbors=int(candidate.summary.get("diffusion_k_neighbors", 15)),
                diffusion_time=int(candidate.summary.get("diffusion_time", 3)),
                n_components=int(candidate.summary.get("diffusion_components", 30)),
            )
        else:
            distances, _metadata = block_adaptive_diffusion_distance(
                coords,
                k_neighbors=int(candidate.summary.get("diffusion_k_neighbors", 15)),
                diffusion_time=int(candidate.summary.get("diffusion_time", 3)),
                n_components=int(candidate.summary.get("diffusion_components", 30)),
                metric=str(candidate.summary.get("adaptive_metric", "euclidean")),
                bandwidth_type=candidate.summary.get("adaptive_bandwidth_type", "-1/(d+2)"),
                epsilon=candidate.summary.get("adaptive_epsilon", "median"),
            )
        return coords, distances
    return None, None


def go_information_criterion(data: pd.DataFrame, labels: np.ndarray) -> dict[str, object]:
    values = data.to_numpy(dtype=float)
    n, p = values.shape
    total_observations = n * p
    ll = 0.0
    active_params = 0
    full_params = 0
    for cluster_id in sorted(set(labels.astype(int))):
        idx = labels == cluster_id
        cluster = values[idx]
        nk = int(cluster.shape[0])
        counts = cluster.sum(axis=0)
        full_params += p
        active_params += int(((counts > 0) & (counts < nk)).sum())
        probs = counts / max(nk, 1)
        positive = counts > 0
        negative = counts < nk
        ll += float(np.sum(counts[positive] * np.log(probs[positive])))
        ll += float(np.sum((nk - counts[negative]) * np.log1p(-probs[negative])))
    return {
        "go_log_likelihood": ll,
        "go_bic_active": float(active_params * math.log(total_observations) - 2.0 * ll),
        "go_aic_active": float(2.0 * active_params - 2.0 * ll),
        "go_bic_full": float(full_params * math.log(total_observations) - 2.0 * ll),
        "go_active_parameters": int(active_params),
        "go_full_parameters": int(full_params),
        "go_bic_active_per_gene": float(
            (active_params * math.log(total_observations) - 2.0 * ll) / n
        ),
    }


def cluster_coherence(data: pd.DataFrame, labels: np.ndarray) -> dict[str, object]:
    presence = data.to_numpy(dtype=int)
    n, _p = presence.shape
    totals = presence.sum(axis=0)
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(set(labels.astype(int))):
        idx = labels == cluster_id
        size = int(idx.sum())
        rest_size = int(n - size)
        cluster_counts = presence[idx].sum(axis=0)
        rest_counts = totals - cluster_counts
        p_values = hypergeom.sf(cluster_counts - 1, n, totals, size)
        q_values = multipletests(p_values, method="fdr_bh")[1]
        cluster_prev = cluster_counts / max(size, 1)
        rest_prev = rest_counts / max(rest_size, 1)
        delta = cluster_prev - rest_prev
        best = int(np.nanargmin(q_values))
        significant = int((q_values < 0.05).sum())
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": size,
                "n_significant_terms_q05": significant,
                "min_q_value": float(q_values[best]),
                "top_term": str(data.columns[best]),
                "top_term_prevalence_delta": float(delta[best]),
                "coherent_by_rule": bool(
                    size >= 3
                    and significant >= 3
                    and float(q_values[best]) < 0.05
                    and float(delta[best]) >= 0.25
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    return {
        "coherent_cluster_count": int(frame["coherent_by_rule"].sum()),
        "coherent_cluster_fraction": float(frame["coherent_by_rule"].mean()),
        "median_significant_terms_q05": float(frame["n_significant_terms_q05"].median()),
        "min_cluster_q_value": float(frame["min_q_value"].min()),
        "coherence_table": frame,
    }


def tfidf_within_cosine_quality(data: pd.DataFrame, labels: np.ndarray) -> dict[str, object]:
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(
        data.to_numpy(dtype=float)
    )
    rows: list[dict[str, object]] = []
    values = labels.astype(int)
    for cluster_id in sorted(set(values)):
        idx = np.where(values == cluster_id)[0]
        if len(idx) < 2:
            mean_sim = math.nan
        else:
            gram = (tfidf[idx] @ tfidf[idx].T).toarray()
            mean_sim = float((gram.sum() - np.trace(gram)) / (len(idx) * (len(idx) - 1)))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(idx)),
                "mean_within_tfidf_cosine": mean_sim,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    valid = frame["mean_within_tfidf_cosine"].dropna()
    weighted = frame.dropna(subset=["mean_within_tfidf_cosine"])
    if weighted.empty:
        weighted_mean = math.nan
    else:
        weighted_mean = float(
            np.average(
                weighted["mean_within_tfidf_cosine"].to_numpy(),
                weights=weighted["cluster_size"].to_numpy(),
            )
        )
    return {
        "median_within_tfidf_cosine": float(valid.median()) if not valid.empty else math.nan,
        "weighted_mean_within_tfidf_cosine": weighted_mean,
        "tfidf_quality_table": frame,
    }


def cluster_size_metrics(labels: np.ndarray) -> dict[str, object]:
    sizes = pd.Series(labels.astype(int)).value_counts()
    return {
        "n_clusters": int(len(sizes)),
        "largest_cluster_size": int(sizes.max()),
        "largest_cluster_fraction": float(sizes.max() / len(labels)),
        "singleton_clusters": int((sizes == 1).sum()),
        "singleton_fraction": float((sizes == 1).sum() / max(len(sizes), 1)),
        "singleton_gene_fraction": float(sizes[sizes == 1].sum() / len(labels)),
        "median_cluster_size": float(sizes.median()),
    }


def quality_tier(row: pd.Series) -> tuple[int, str]:
    """Coarse quality tier used to prevent GO-IC singleton pathologies."""
    n_clusters = int(row["n_clusters"])
    largest = float(row["largest_cluster_fraction"])
    singleton_gene = float(row["singleton_gene_fraction"])
    coherent = float(row["coherent_cluster_fraction"])
    if 5 <= n_clusters <= 150 and largest <= 0.75 and singleton_gene <= 0.05 and coherent >= 0.25:
        return 0, "quality_plausible"
    if 2 <= n_clusters < 5 and largest <= 0.75 and coherent >= 0.25:
        return 1, "broad_coherent"
    if n_clusters == 1 or largest >= 0.90 or singleton_gene >= 0.50:
        return 3, "degenerate"
    return 2, "quality_mixed"


def cluster_colors(labels: np.ndarray) -> np.ndarray:
    sizes = pd.Series(labels).value_counts()
    rank = {int(cluster_id): i for i, cluster_id in enumerate(sizes.index)}
    return np.asarray([rank[int(label)] for label in labels], dtype=int)


def cluster_cmap(n_colors: int) -> ListedColormap:
    if n_colors <= 20:
        base = plt.get_cmap("tab20")
        return ListedColormap([base(i % base.N) for i in range(max(n_colors, 1))])
    base = plt.get_cmap("turbo")
    values = np.linspace(0.02, 0.98, max(n_colors, 1))
    return ListedColormap([base(value) for value in values])


def embedding_2d(values: np.ndarray | None, random_state: int) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return None
    if values.shape[1] == 1:
        return np.column_stack([values[:, 0], np.zeros(values.shape[0])])
    try:
        import umap

        return umap.UMAP(
            n_components=2,
            n_neighbors=min(18, max(2, values.shape[0] - 1)),
            min_dist=0.05,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(values)
    except Exception:
        return PCA(n_components=2, random_state=random_state).fit_transform(values)


def draw_tree_strip(
    ax_tree: plt.Axes, ax_strip: plt.Axes, linkage_matrix: np.ndarray | None, colors: np.ndarray
) -> None:
    if linkage_matrix is None:
        ax_tree.axis("off")
        ax_strip.axis("off")
        ax_tree.text(0.5, 0.5, "Tree not available", ha="center", va="center")
        return
    result = dendrogram(
        linkage_matrix,
        ax=ax_tree,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
        link_color_func=lambda _: "#333333",
    )
    for collection in ax_tree.collections:
        collection.set_linewidth(0.45)
        collection.set_alpha(0.82)
    leaves = np.asarray(result["leaves"], dtype=int)
    strip_values = colors[leaves]
    n_colors = max(int(strip_values.max()) + 1, 1)
    cmap = cluster_cmap(n_colors)
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), cmap.N)
    ax_strip.imshow(strip_values[np.newaxis, :], aspect="auto", cmap=cmap, norm=norm)
    ax_tree.set_ylabel("distance", fontsize=8)
    ax_tree.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_tree.tick_params(axis="y", labelsize=7)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])


def draw_candidate_page(
    artifact: CandidateArtifact,
    global_embedding: pd.DataFrame,
    global_x: str,
    global_y: str,
    output_path: Path,
) -> plt.Figure:
    colors = cluster_colors(artifact.labels)
    color_frame = artifact.assignments.copy()
    color_frame["cluster_color_rank"] = colors
    n_colors = max(int(colors.max()) + 1, 1)
    cmap = cluster_cmap(n_colors)
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), cmap.N)
    merged = global_embedding.merge(color_frame, on="gene", how="inner")
    sub = embedding_2d(
        artifact.subspace_embedding, random_state=abs(hash(artifact.candidate.run_id)) % 100000
    )
    fig = plt.figure(figsize=(20.0, 11.2), layout="constrained")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.05, 1.05, 1.45, 1.05],
        height_ratios=[1.0, 1.0],
        hspace=0.12,
        wspace=0.16,
    )
    ax_global = fig.add_subplot(grid[0, 0])
    ax_sub = fig.add_subplot(grid[0, 1])
    tree_grid = grid[:, 2].subgridspec(2, 1, height_ratios=[0.86, 0.14], hspace=0.03)
    ax_tree = fig.add_subplot(tree_grid[0, 0])
    ax_strip = fig.add_subplot(tree_grid[1, 0])
    ax_size = fig.add_subplot(grid[1, 0])
    ax_quality = fig.add_subplot(grid[1, 1])
    ax_text = fig.add_subplot(grid[:, 3])

    ax_global.scatter(
        merged[global_x],
        merged[global_y],
        c=merged["cluster_color_rank"],
        cmap=cmap,
        norm=norm,
        s=16,
        alpha=0.86,
        linewidths=0,
    )
    ax_global.set_title("Full allGO UMAP")
    ax_global.set_xlabel(global_x)
    ax_global.set_ylabel(global_y)
    ax_global.tick_params(labelsize=7)

    if sub is None:
        ax_sub.axis("off")
        ax_sub.text(0.5, 0.5, "No subspace embedding", ha="center", va="center")
    else:
        ax_sub.scatter(
            sub[:, 0], sub[:, 1], c=colors, cmap=cmap, norm=norm, s=16, alpha=0.86, linewidths=0
        )
        ax_sub.set_title("Tree subspace embedding")
        ax_sub.set_xlabel("axis 1")
        ax_sub.set_ylabel("axis 2")
        ax_sub.tick_params(labelsize=7)

    draw_tree_strip(ax_tree, ax_strip, artifact.linkage_matrix, colors)
    ax_tree.set_title("Tree leaf order")

    sizes = pd.Series(artifact.labels).value_counts().sort_values(ascending=False)
    top_sizes = sizes.head(18).iloc[::-1]
    ax_size.barh([f"C{int(i)}" for i in top_sizes.index], top_sizes.to_numpy(), color="#4c78a8")
    ax_size.set_title("Largest clusters")
    ax_size.set_xlabel("genes")
    ax_size.tick_params(labelsize=7)
    ax_size.grid(axis="x", alpha=0.18)

    rank = artifact.ranking
    quality_items = [
        ("coherent", "coherent_cluster_fraction", "#54a24b"),
        ("within TF-IDF", "weighted_mean_within_tfidf_cosine", "#f58518"),
        ("largest", "largest_cluster_fraction", "#72b7b2"),
        ("singletons", "singleton_gene_fraction", "#e45756"),
    ]
    quality_values = [float(rank.get(item[1], math.nan)) for item in quality_items]
    quality_colors = [item[2] for item in quality_items]
    quality_labels = [item[0] for item in quality_items]
    y_positions = np.arange(len(quality_items))
    bars = ax_quality.barh(y_positions, quality_values[::-1], color=quality_colors[::-1])
    ax_quality.set_yticks([])
    ax_quality.set_xlim(
        0.0, max(1.0, max([v for v in quality_values if math.isfinite(v)] or [1.0]) * 1.05)
    )
    ax_quality.set_title("Cluster quality")
    ax_quality.tick_params(labelsize=8)
    ax_quality.grid(axis="x", alpha=0.18)
    for bar, label, value in zip(bars, quality_labels[::-1], quality_values[::-1], strict=False):
        y_center = bar.get_y() + bar.get_height() / 2
        ax_quality.text(
            0.02,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
        if math.isfinite(value):
            ax_quality.text(
                ax_quality.get_xlim()[1] * 0.98,
                y_center,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=8,
            )

    text_lines = [
        reader_run_id(artifact.candidate.run_id),
        "",
        f"display rank: {int(rank.get('display_rank', -1))}",
        f"raw GO-IC rank: {int(rank.get('raw_go_ic_rank', -1))}",
        f"quality tier: {rank.get('quality_tier_label', '')}",
        "",
        f"method family: {reader_family(artifact.candidate.family)}",
        f"weighting: {artifact.candidate.weighting}",
        f"block: {artifact.candidate.block_name}",
        f"modes: {artifact.candidate.block_start}-{artifact.candidate.block_end}",
        "",
        f"GO-BIC active: {rank['go_bic_active']:.2f}",
        f"GO-BIC active/gene: {rank['go_bic_active_per_gene']:.2f}",
        f"GO active params: {int(rank['go_active_parameters'])}",
        f"GO logLik: {rank['go_log_likelihood']:.2f}",
        "",
        f"clusters: {int(rank['n_clusters'])}",
        f"largest cluster: {int(rank['largest_cluster_size'])} ({rank['largest_cluster_fraction']:.3f})",
        f"singletons: {int(rank['singleton_clusters'])} ({rank['singleton_fraction']:.3f} of clusters)",
        f"median cluster size: {rank['median_cluster_size']:.1f}",
        "",
        f"coherent clusters: {int(rank['coherent_cluster_count'])} ({rank['coherent_cluster_fraction']:.3f})",
        f"median significant GO terms: {rank['median_significant_terms_q05']:.1f}",
        f"min cluster q: {rank['min_cluster_q_value']:.2e}",
        f"median within TF-IDF cosine: {rank['median_within_tfidf_cosine']:.4f}",
        f"weighted within TF-IDF cosine: {rank['weighted_mean_within_tfidf_cosine']:.4f}",
    ]
    for key in ("block_energy_fraction", "segmentation_bic"):
        value = artifact.candidate.summary.get(key)
        if value is not None and pd.notna(value):
            text_lines.append(f"{key}: {float(value):.4f}")
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        linespacing=1.18,
    )
    fig.suptitle(
        f"Rank {int(rank.get('display_rank', -1))}: {display_label(artifact.candidate.run_id)}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    return fig


def draw_ranking_plots(ranking: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    top = ranking.head(top_n).iloc[::-1]
    labels = [
        f"{int(row.display_rank):02d}  {wrapped_label(str(row.run_id), 42)}"
        for row in top.itertuples(index=False)
    ]
    colors = [tier_color(str(label)) for label in top["quality_tier_label"]]
    fig, ax = plt.subplots(figsize=(13.8, max(6.2, 0.62 * len(top) + 1.8)))
    bars = ax.barh(labels, top["go_bic_active_per_gene"], color=colors)
    ax.set_xlabel("GO-BIC active per gene (lower is better)")
    ax.set_ylabel("tree")
    ax.set_title("Top trees by quality tier, then GO annotation information criterion")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(0, float(top["go_bic_active_per_gene"].max()) + 420.0)
    for bar, row in zip(bars, top.itertuples(index=False), strict=False):
        ax.text(
            bar.get_width() + 22,
            bar.get_y() + bar.get_height() / 2,
            f"K={int(row.n_clusters)}, coherent={int(row.coherent_cluster_count)}/{int(row.n_clusters)}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / f"{RESULT_PREFIX}_top_trees.png", dpi=180)
    plt.close(fig)

    def scatter_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10.2, 6.8))
        scatter = ax.scatter(
            frame["go_bic_active_per_gene"],
            frame["coherent_cluster_fraction"],
            c=frame["weighted_mean_within_tfidf_cosine"],
            s=48 + 150 * (1.0 - frame["singleton_gene_fraction"].clip(0, 1)),
            cmap="viridis",
            alpha=0.82,
            linewidths=0.4,
            edgecolors="#333333",
        )
        for tier, group in frame.groupby("quality_tier_label"):
            ax.scatter(
                group["go_bic_active_per_gene"],
                group["coherent_cluster_fraction"],
                facecolors="none",
                edgecolors=tier_color(str(tier)),
                linewidths=1.7,
                s=70 + 170 * (1.0 - group["singleton_gene_fraction"].clip(0, 1)),
                label=str(tier),
            )
        ax.set_xlabel("GO-BIC active per gene (lower is better)")
        ax.set_ylabel("coherent cluster fraction")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("weighted within-cluster TF-IDF cosine")
        for _, row in frame.head(min(12, len(frame))).iterrows():
            ax.annotate(
                str(row["display_rank"]),
                (row["go_bic_active_per_gene"], row["coherent_cluster_fraction"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    scatter_plot(
        ranking,
        output_dir / f"{RESULT_PREFIX}_quality_scatter_all_trees.png",
        "All trees: GO information criterion vs cluster quality",
    )
    plausible = ranking[ranking["quality_tier_label"].eq("quality_plausible")]
    if not plausible.empty:
        scatter_plot(
            plausible,
            output_dir / f"{RESULT_PREFIX}_quality_scatter_plausible_trees.png",
            "Plausible trees: GO information criterion vs cluster quality",
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.output_dir / "tree_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in pages_dir.glob("*.png"):
        stale_page.unlink()

    data = load_binary_matrix(args.input)
    global_embedding, global_x, global_y = load_global_embedding(args.global_embedding, data)
    global_embedding.to_csv(args.output_dir / "global_embedding_used.csv", index=False)
    eigensystems = eigensystems_for_data(data, args.max_rank)

    candidates = []
    candidates.extend(add_candidates_from_method_matrix(args.method_matrix_summary))
    candidates.extend(add_candidates_from_raw_kak(args.raw_kak_dir))
    candidates.extend(add_candidates_from_diffusion_kak(args.adaptive_diffusion_kak_dir))
    candidates.extend(add_whole_adaptive_candidate(args.whole_adaptive_dir))
    if not candidates:
        raise ValueError("No candidate tree assignments were found.")

    artifacts: list[CandidateArtifact] = []
    ranking_rows: list[dict[str, object]] = []
    coherence_frames: list[pd.DataFrame] = []
    tfidf_frames: list[pd.DataFrame] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] score {candidate.run_id}", flush=True)
        assignments = load_assignment(candidate.assignments_path, data.index)
        labels = assignments["cluster_id"].to_numpy(dtype=int)
        subspace, distances = candidate_subspace_and_tree(candidate, data, eigensystems)
        linkage_matrix = None
        if (
            distances is not None
            and np.isfinite(distances).all()
            and not np.allclose(distances, 0.0)
        ):
            linkage_matrix = linkage(distances, method="average")
        info = go_information_criterion(data, labels)
        coherence = cluster_coherence(data, labels)
        tfidf_quality = tfidf_within_cosine_quality(data, labels)
        sizes = cluster_size_metrics(labels)
        rank = {
            "run_id": candidate.run_id,
            "method_run_id": reader_run_id(candidate.run_id),
            "family": candidate.family,
            "method_family": reader_family(candidate.family),
            "weighting": candidate.weighting,
            "block_name": candidate.block_name,
            "block_start": candidate.block_start,
            "block_end": candidate.block_end,
            **info,
            **sizes,
            "coherent_cluster_count": coherence["coherent_cluster_count"],
            "coherent_cluster_fraction": coherence["coherent_cluster_fraction"],
            "median_significant_terms_q05": coherence["median_significant_terms_q05"],
            "min_cluster_q_value": coherence["min_cluster_q_value"],
            "median_within_tfidf_cosine": tfidf_quality["median_within_tfidf_cosine"],
            "weighted_mean_within_tfidf_cosine": tfidf_quality["weighted_mean_within_tfidf_cosine"],
            "assignments_path": str(candidate.assignments_path),
            "block_energy_fraction": candidate.summary.get("block_energy_fraction", math.nan),
            "segmentation_bic": candidate.summary.get("segmentation_bic", math.nan),
        }
        coherence_frame = coherence["coherence_table"].copy()
        coherence_frame.insert(0, "run_id", candidate.run_id)
        coherence_frames.append(coherence_frame)
        tfidf_frame = tfidf_quality["tfidf_quality_table"].copy()
        tfidf_frame.insert(0, "run_id", candidate.run_id)
        tfidf_frames.append(tfidf_frame)
        artifact = CandidateArtifact(
            candidate=candidate,
            assignments=assignments,
            labels=labels,
            linkage_matrix=linkage_matrix,
            subspace_embedding=subspace,
            ranking=rank,
        )
        artifacts.append(artifact)
        ranking_rows.append(rank)

    ranking = pd.DataFrame.from_records(ranking_rows)
    raw_order = ranking.sort_values(
        [
            "go_bic_active",
            "coherent_cluster_fraction",
            "weighted_mean_within_tfidf_cosine",
            "singleton_fraction",
        ],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    raw_rank = dict(zip(raw_order["run_id"], np.arange(1, len(raw_order) + 1), strict=False))
    tiers = ranking.apply(quality_tier, axis=1, result_type="expand")
    ranking["quality_tier"] = tiers[0].astype(int)
    ranking["quality_tier_label"] = tiers[1].astype(str)
    ranking["raw_go_ic_rank"] = ranking["run_id"].map(raw_rank).astype(int)
    ranking = ranking.sort_values(
        [
            "quality_tier",
            "go_bic_active",
            "coherent_cluster_fraction",
            "weighted_mean_within_tfidf_cosine",
            "singleton_gene_fraction",
        ],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "display_rank", np.arange(1, len(ranking) + 1))
    ranking_path = args.output_dir / f"{RESULT_PREFIX}_tree_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    pd.concat(coherence_frames, ignore_index=True).to_csv(
        args.output_dir / f"{RESULT_PREFIX}_cluster_coherence_long.csv",
        index=False,
    )
    pd.concat(tfidf_frames, ignore_index=True).to_csv(
        args.output_dir / f"{RESULT_PREFIX}_tfidf_quality_long.csv",
        index=False,
    )

    rank_by_run = dict(zip(ranking["run_id"], ranking["display_rank"], strict=False))
    raw_rank_by_run = dict(zip(ranking["run_id"], ranking["raw_go_ic_rank"], strict=False))
    tier_by_run = dict(zip(ranking["run_id"], ranking["quality_tier_label"], strict=False))
    tier_number_by_run = dict(zip(ranking["run_id"], ranking["quality_tier"], strict=False))
    artifact_by_run = {artifact.candidate.run_id: artifact for artifact in artifacts}
    ordered_artifacts = [artifact_by_run[run_id] for run_id in ranking["run_id"]]
    for artifact in ordered_artifacts:
        run_id = artifact.candidate.run_id
        artifact.ranking["display_rank"] = int(rank_by_run[run_id])
        artifact.ranking["raw_go_ic_rank"] = int(raw_rank_by_run[run_id])
        artifact.ranking["quality_tier"] = int(tier_number_by_run[run_id])
        artifact.ranking["quality_tier_label"] = str(tier_by_run[run_id])

    draw_ranking_plots(ranking, args.output_dir, args.top_summary_count)
    pdf_path = args.output_dir / f"{RESULT_PREFIX}_all_tree_pages.pdf"
    with PdfPages(pdf_path) as pdf:
        for artifact in ordered_artifacts:
            path = (
                pages_dir
                / f"{int(artifact.ranking['display_rank']):02d}_{safe_name(reader_run_id(artifact.candidate.run_id))}.png"
            )
            fig = draw_candidate_page(artifact, global_embedding, global_x, global_y, path)
            pdf.savefig(fig)
            plt.close(fig)

    readme = [
        "# allGO New Quality-Aware GO-IC Tree Summary",
        "",
        f"Input: `{args.input}`",
        f"Trees scored: `{len(ranking)}`",
        "",
        "Primary display ordering: quality tier, then GO annotation Bernoulli BIC with active within-cluster GO parameters.",
        "The raw GO-IC-only order is preserved in `raw_go_ic_rank` because near-singleton trees can overfit GO labels.",
        "Quality tiers: `quality_plausible` requires 5-150 clusters, largest cluster <= 75%, singleton-gene fraction <= 5%, and coherent-cluster fraction >= 25%; `broad_coherent` is a small-cluster coherent fallback; `degenerate` includes one-cluster, giant-cluster, or singleton-dominated trees.",
        "Lower `go_bic_active` is better within a quality tier.",
        "",
        "## Top Trees",
        "",
        ranking[
            [
                "display_rank",
                "raw_go_ic_rank",
                "method_run_id",
                "method_family",
                "quality_tier_label",
                "go_bic_active_per_gene",
                "n_clusters",
                "coherent_cluster_count",
                "coherent_cluster_fraction",
                "weighted_mean_within_tfidf_cosine",
                "singleton_fraction",
                "largest_cluster_fraction",
            ]
        ]
        .head(args.top_summary_count)
        .to_string(index=False),
        "",
        "## Outputs",
        "",
        f"- `{RESULT_PREFIX}_tree_ranking.csv`",
        f"- `{RESULT_PREFIX}_top_trees.png`",
        f"- `{RESULT_PREFIX}_quality_scatter_all_trees.png`",
        f"- `{RESULT_PREFIX}_quality_scatter_plausible_trees.png`",
        f"- `{RESULT_PREFIX}_all_tree_pages.pdf`",
        "- `tree_pages/*.png`",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Wrote GO-IC tree plots: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
