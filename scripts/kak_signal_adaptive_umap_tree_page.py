#!/usr/bin/env python3
"""Create KAK/cosine geometry diagnostic pages from matrix-probe outputs.

This restores the historical whole-UMAP -> KAK sub-UMAP -> tree page as a
diagnostic-only visualization layer. It reads current
``adaptive_cosine_kak_matrix_probe`` outputs and recomputes the KAK/cosine
coordinates needed to inspect:

* global embedding position,
* within-block embedding position,
* radius relative to the common/invariant cosine axis,
* angle to the leading block axis,
* independent/orthogonal radius inside the selected block,
* tree leaf order and TreeDecomposition cluster labels.

It does not run clustering and does not make production calibration claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    SpectralBlock,
    coords_for_block,
    cosine_eigendecomposition,
    weighted_matrix,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_matrix_probe import (
    load_matrix,
    safe_name,
)

EmbeddingMethod = Literal["umap", "svd"]


@dataclass
class BlockArtifact:
    run_id: str
    weighting: str
    block_name: str
    summary_row: pd.Series
    assignments: pd.DataFrame
    colors: np.ndarray
    coords: np.ndarray
    embedding_2d: np.ndarray
    embedding_3d: np.ndarray
    linkage_matrix: np.ndarray
    sample_geometry: pd.DataFrame
    internal_geometry: pd.DataFrame
    summary_geometry: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build whole-embedding -> KAK sub-embedding -> geometry -> tree pages "
            "from adaptive cosine/KAK matrix-probe outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
        help="Feature matrix TSV used by the matrix KAK probe.",
    )
    parser.add_argument(
        "--kak-results-dir",
        type=Path,
        required=True,
        help="Directory containing matrix_kak_probe_summary.csv and assignments/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for restored diagnostic pages and geometry tables.",
    )
    parser.add_argument(
        "--global-embedding",
        type=Path,
        default=None,
        help=(
            "Optional existing 2D global embedding CSV/TSV. If omitted, the script "
            "computes one from the input matrix."
        ),
    )
    parser.add_argument(
        "--embedding-method",
        choices=["umap", "svd"],
        default="umap",
        help="Method for computed global and block embeddings.",
    )
    parser.add_argument(
        "--global-weighting",
        choices=["binary", "tfidf"],
        default="binary",
        help="Feature weighting used when computing a global embedding.",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=["binary", "tfidf"],
        default=None,
        help="Optional weighting filter.",
    )
    parser.add_argument(
        "--block-limit",
        type=int,
        default=0,
        help="Limit number of ok blocks plotted; 0 means all ok blocks.",
    )
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=1729)
    parser.add_argument(
        "--reference-endotypes",
        type=Path,
        default=None,
        help=(
            "Optional Julia reference endotype table. When provided, block pages "
            "are ordered by descending NMI against matched reference labels."
        ),
    )
    parser.add_argument(
        "--order-by",
        choices=["input", "reference_nmi"],
        default="reference_nmi",
        help="Ordering for displayed block/tree pages.",
    )
    return parser.parse_args()


def load_probe_summary(results_dir: Path, weightings: list[str] | None) -> pd.DataFrame:
    path = results_dir / "matrix_kak_probe_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pd.read_csv(path)
    required = {
        "weighting",
        "block_name",
        "block_start",
        "block_end",
        "status",
        "assignments_path",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Probe summary missing required columns: {sorted(missing)!r}")
    summary = summary[summary["status"].eq("ok")].copy()
    if weightings is not None:
        summary = summary[summary["weighting"].isin(weightings)].copy()
    if summary.empty:
        raise ValueError("No ok KAK matrix-probe blocks are available to visualize.")
    return summary.reset_index(drop=True)


def resolve_assignment_path(results_dir: Path, value: object) -> Path:
    if value is None or pd.isna(value):
        raise ValueError("Ok block is missing assignments_path.")
    path = Path(str(value))
    candidates = [
        path,
        results_dir / path,
        results_dir / "assignments" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve assignment path {path!s}")


def load_assignment(path: Path, data_index: pd.Index) -> pd.DataFrame:
    assignment = pd.read_csv(path)
    if "gene" not in assignment.columns:
        first_column = assignment.columns[0]
        assignment = assignment.rename(columns={first_column: "gene"})
    required = {"gene", "cluster_id", "cluster_root", "cluster_size"}
    missing = required - set(assignment.columns)
    if missing:
        raise ValueError(f"Assignment file {path} missing columns: {sorted(missing)!r}")
    assignment = assignment.set_index("gene", drop=False)
    missing_genes = data_index.difference(assignment.index)
    if len(missing_genes):
        raise ValueError(
            f"Assignment file {path} is missing {len(missing_genes)} gene(s), "
            f"for example {missing_genes[:5].tolist()!r}."
        )
    return assignment.loc[data_index].reset_index(drop=True)


def parse_reference_endotypes(path: Path) -> dict[str, dict[str, object]]:
    entrez_to_endotype: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            first_cell = row[0].strip()
            if first_cell.startswith("#") or first_cell in {"SUM", "AVERAGE"}:
                continue
            if len(row) < 10:
                continue
            cluster_id_text = row[4].strip()
            if not cluster_id_text.isdigit():
                continue
            endotype = {
                "reference_cluster_id": int(cluster_id_text),
                "reference_cluster_color": row[5].strip() or "#808080",
                "reference_cluster_rank": row[6].strip(),
                "reference_cluster_name": row[7].strip() or f"Cluster {cluster_id_text}",
                "reference_cluster_type": row[8].strip() or "cluster",
            }
            for gene_id in (cell.strip() for cell in row[9:] if cell.strip()):
                entrez_to_endotype[str(gene_id)] = endotype
    return entrez_to_endotype


def map_symbols_to_entrez(symbols: list[str], *, batch_size: int = 200) -> dict[str, str | None]:
    mapped: dict[str, str | None] = {}
    endpoint = "https://mygene.info/v3/query"
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        body = urlencode(
            {
                "q": ",".join(batch),
                "scopes": "symbol",
                "fields": "symbol,entrezgene,taxid",
                "species": "human",
            }
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Failed to query mygene.info for gene symbol -> Entrez mapping."
            ) from exc

        if isinstance(payload, dict):
            payload = [payload]
        for record in payload:
            query = record.get("query")
            if not query:
                continue
            entrez = record.get("entrezgene") or record.get("_id")
            mapped[str(query)] = None if entrez is None else str(entrez)
        for symbol in batch:
            mapped.setdefault(symbol, None)
    return mapped


def build_reference_labels(data_index: pd.Index, reference_endotypes_path: Path) -> pd.DataFrame:
    symbol_to_entrez = map_symbols_to_entrez([str(value) for value in data_index.tolist()])
    entrez_to_endotype = parse_reference_endotypes(reference_endotypes_path)
    rows: list[dict[str, object]] = []
    for gene in data_index:
        gene_text = str(gene)
        entrez_id = symbol_to_entrez.get(gene_text)
        endotype = entrez_to_endotype.get(str(entrez_id)) if entrez_id is not None else None
        row = {
            "gene": gene_text,
            "entrez_id": entrez_id,
            "matched_reference_label": isinstance(endotype, dict),
            "reference_cluster_id": pd.NA,
            "reference_cluster_name": "Unassigned",
            "reference_cluster_type": "unassigned",
            "reference_cluster_color": "#bdbdbd",
            "reference_cluster_rank": pd.NA,
        }
        if isinstance(endotype, dict):
            row.update(endotype)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def reference_metrics_for_assignment(assignment: pd.DataFrame) -> dict[str, object]:
    if "reference_cluster_id" not in assignment.columns:
        return {
            "reference_matched_genes": 0,
            "reference_unmatched_genes": int(len(assignment)),
            "reference_cluster_count_observed": 0,
            "reference_ari": math.nan,
            "reference_nmi": math.nan,
        }
    matched = assignment[assignment["reference_cluster_id"].notna()].copy()
    if matched.empty:
        return {
            "reference_matched_genes": 0,
            "reference_unmatched_genes": int(len(assignment)),
            "reference_cluster_count_observed": 0,
            "reference_ari": math.nan,
            "reference_nmi": math.nan,
        }
    y_ref = matched["reference_cluster_id"].astype(int).to_numpy()
    y_pred = matched["cluster_id"].astype(int).to_numpy()
    return {
        "reference_matched_genes": int(matched.shape[0]),
        "reference_unmatched_genes": int(len(assignment) - matched.shape[0]),
        "reference_cluster_count_observed": int(matched["reference_cluster_id"].nunique()),
        "reference_ari": float(adjusted_rand_score(y_ref, y_pred)),
        "reference_nmi": float(normalized_mutual_info_score(y_ref, y_pred)),
    }


def attach_reference_labels(
    artifacts: list[BlockArtifact],
    reference_labels: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        labels = reference_labels.drop(columns=["matched_reference_label"], errors="ignore")
        artifact.assignments = artifact.assignments.merge(labels, on="gene", how="left")
        artifact.sample_geometry = artifact.sample_geometry.merge(
            reference_labels,
            on="gene",
            how="left",
        )
        metrics = reference_metrics_for_assignment(artifact.assignments)
        artifact.summary_geometry.update(metrics)
        rows.append(
            {
                "run_id": artifact.run_id,
                "weighting": artifact.weighting,
                "block_name": artifact.block_name,
                "n_clusters": int(artifact.summary_row["n_clusters"]),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def order_artifacts(
    artifacts: list[BlockArtifact],
    *,
    order_by: str,
) -> list[BlockArtifact]:
    if order_by != "reference_nmi":
        for rank, artifact in enumerate(artifacts, start=1):
            artifact.summary_geometry["display_order_rank"] = rank
        return artifacts

    def key(artifact: BlockArtifact) -> tuple[float, int, str]:
        value = artifact.summary_geometry.get("reference_nmi", math.nan)
        nmi = float(value) if value is not None and pd.notna(value) else -math.inf
        return (-nmi, int(artifact.summary_row["n_clusters"]), artifact.run_id)

    ordered = sorted(artifacts, key=key)
    for rank, artifact in enumerate(ordered, start=1):
        artifact.summary_geometry["display_order_rank"] = rank
    return ordered


def cluster_color_values(labels: pd.Series) -> tuple[np.ndarray, dict[int, int]]:
    labels_int = labels.astype(int)
    sizes = labels_int.value_counts()
    rank_by_cluster = {
        int(cluster_id): int(rank) for rank, cluster_id in enumerate(sizes.index)
    }
    return labels_int.map(rank_by_cluster).to_numpy(dtype=int), rank_by_cluster


def compute_embedding(
    values: np.ndarray,
    *,
    n_components: int,
    method: EmbeddingMethod,
    random_state: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Embedding input must be 2-D, got shape {values.shape}.")
    if values.shape[0] < 2:
        raise ValueError("Embedding requires at least two rows.")

    if values.shape[1] == 0:
        raise ValueError("Embedding input has zero columns.")
    if values.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        values = np.column_stack(
            [values[:, 0], rng.normal(0.0, 1e-9, size=values.shape[0])]
        )

    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(18, max(2, values.shape[0] - 1)),
            min_dist=0.05,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(values)

    max_components = min(n_components, values.shape[0] - 1, values.shape[1])
    if max_components <= 0:
        raise ValueError("SVD embedding has no available components.")
    if values.shape[1] > 256:
        model = TruncatedSVD(n_components=max_components, random_state=random_state)
        embedding = model.fit_transform(values)
    else:
        model = PCA(n_components=max_components, random_state=random_state)
        embedding = model.fit_transform(values)
    if embedding.shape[1] < n_components:
        padding = np.zeros((embedding.shape[0], n_components - embedding.shape[1]))
        embedding = np.column_stack([embedding, padding])
    return embedding


def load_or_compute_global_embedding(
    *,
    data: pd.DataFrame,
    global_embedding_path: Path | None,
    weighting: str,
    method: EmbeddingMethod,
    random_state: int,
) -> tuple[pd.DataFrame, str, str]:
    if global_embedding_path is not None:
        sep = "\t" if global_embedding_path.suffix.lower() in {".tsv", ".tab"} else ","
        global_embedding = pd.read_csv(global_embedding_path, sep=sep)
        if "gene" not in global_embedding.columns:
            global_embedding = global_embedding.rename(columns={global_embedding.columns[0]: "gene"})
        numeric_columns = [
            column
            for column in global_embedding.columns
            if column != "gene" and pd.api.types.is_numeric_dtype(global_embedding[column])
        ]
        if len(numeric_columns) < 2:
            raise ValueError(
                f"Global embedding {global_embedding_path} must contain gene plus two numeric columns."
            )
        return (
            global_embedding[["gene", numeric_columns[0], numeric_columns[1]]].copy(),
            numeric_columns[0],
            numeric_columns[1],
        )

    values = weighted_matrix(data, weighting)
    if method == "umap" and values.shape[1] > 80:
        svd_components = min(50, values.shape[0] - 1, values.shape[1])
        values = TruncatedSVD(
            n_components=svd_components,
            random_state=random_state,
        ).fit_transform(values)
    embedding = compute_embedding(
        values,
        n_components=2,
        method=method,
        random_state=random_state,
    )
    x_col = f"Global {method.upper()} axis 1"
    y_col = f"Global {method.upper()} axis 2"
    return (
        pd.DataFrame({"gene": data.index.to_numpy(), x_col: embedding[:, 0], y_col: embedding[:, 1]}),
        x_col,
        y_col,
    )


def spectral_block_from_row(row: pd.Series) -> SpectralBlock:
    return SpectralBlock(
        block_id=int(row.get("block_id", 0)),
        block_name=str(row["block_name"]),
        block_start=int(row["block_start"]),
        block_end=int(row["block_end"]),
        block_type=str(row.get("block_type", "adaptive_decay_regime")),
    )


def corr_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return math.nan
    if float(np.std(left[valid])) <= 1e-12 or float(np.std(right[valid])) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def pairwise_direction_cosine_quantiles(coords: np.ndarray) -> dict[str, float]:
    radius = np.linalg.norm(coords, axis=1)
    valid = radius > 1e-12
    if int(valid.sum()) < 3:
        return {
            "pairwise_direction_cosine_q10": math.nan,
            "pairwise_direction_cosine_q50": math.nan,
            "pairwise_direction_cosine_q90": math.nan,
        }
    unit = coords[valid] / radius[valid, np.newaxis]
    cosine = unit @ unit.T
    tri = cosine[np.triu_indices_from(cosine, k=1)]
    return {
        "pairwise_direction_cosine_q10": quantile_or_nan(tri, 0.10),
        "pairwise_direction_cosine_q50": quantile_or_nan(tri, 0.50),
        "pairwise_direction_cosine_q90": quantile_or_nan(tri, 0.90),
    }


def vector_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float)))


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return math.nan
    return float(np.dot(left, right) / (left_norm * right_norm))


def angle_between_vectors_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = vector_cosine(left, right)
    if not math.isfinite(cosine):
        return math.nan
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def unoriented_angle_to_leading_axis_deg(vector: np.ndarray) -> float:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return math.nan
    cosine = abs(float(vector[0]) / norm)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def independent_fraction(vector: np.ndarray) -> float:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return math.nan
    if vector.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(vector[1:]) / norm)


def cluster_concentration(cluster_ids: np.ndarray) -> tuple[int, float]:
    if cluster_ids.size == 0:
        return 0, math.nan
    values, counts = np.unique(cluster_ids.astype(int), return_counts=True)
    return int(values.size), float(counts.max() / cluster_ids.size)


def compute_internal_tree_geometry(
    *,
    row: pd.Series,
    linkage_matrix: np.ndarray,
    coords: np.ndarray,
    common_axis_score: np.ndarray,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per internal linkage node in KAK block coordinates."""
    n_leaves = coords.shape[0]
    leaves_by_node: dict[int, np.ndarray] = {
        leaf: np.array([leaf], dtype=int) for leaf in range(n_leaves)
    }
    centroid_by_node: dict[int, np.ndarray] = {
        leaf: coords[leaf].astype(float) for leaf in range(n_leaves)
    }
    cluster_ids = assignment["cluster_id"].astype(int).to_numpy()
    rows: list[dict[str, object]] = []

    for merge_index, merge in enumerate(linkage_matrix):
        left_id = int(merge[0])
        right_id = int(merge[1])
        parent_id = n_leaves + merge_index
        left_leaves = leaves_by_node[left_id]
        right_leaves = leaves_by_node[right_id]
        parent_leaves = np.concatenate([left_leaves, right_leaves])

        left_centroid = np.mean(coords[left_leaves], axis=0)
        right_centroid = np.mean(coords[right_leaves], axis=0)
        parent_centroid = np.mean(coords[parent_leaves], axis=0)
        leaves_by_node[parent_id] = parent_leaves
        centroid_by_node[parent_id] = parent_centroid

        left_branch = left_centroid - parent_centroid
        right_branch = right_centroid - parent_centroid
        sibling_vector = left_centroid - right_centroid
        parent_centered = coords[parent_leaves] - parent_centroid
        left_centered = coords[left_leaves] - left_centroid
        right_centered = coords[right_leaves] - right_centroid

        parent_radius = np.linalg.norm(parent_centered, axis=1)
        left_radius = np.linalg.norm(left_centered, axis=1)
        right_radius = np.linalg.norm(right_centered, axis=1)
        parent_cluster_count, parent_dominant_cluster_fraction = cluster_concentration(
            cluster_ids[parent_leaves]
        )
        left_cluster_count, left_dominant_cluster_fraction = cluster_concentration(
            cluster_ids[left_leaves]
        )
        right_cluster_count, right_dominant_cluster_fraction = cluster_concentration(
            cluster_ids[right_leaves]
        )

        left_size = int(left_leaves.size)
        right_size = int(right_leaves.size)
        parent_size = int(parent_leaves.size)
        sibling_centroid_distance = vector_norm(sibling_vector)
        parent_radius_q50 = quantile_or_nan(parent_radius, 0.50)
        parent_radius_scale = (
            max(parent_radius_q50, 1e-12)
            if math.isfinite(parent_radius_q50)
            else math.nan
        )
        child_radius_scale = max(
            quantile_or_nan(left_radius, 0.50) + quantile_or_nan(right_radius, 0.50),
            1e-12,
        )
        common_axis_mean_delta = float(
            np.mean(common_axis_score[left_leaves])
            - np.mean(common_axis_score[right_leaves])
        )

        rows.append(
            {
                "weighting": row["weighting"],
                "block_name": row["block_name"],
                "parent_linkage_id": int(parent_id),
                "left_child_linkage_id": int(left_id),
                "right_child_linkage_id": int(right_id),
                "linkage_merge_distance": float(merge[2]),
                "parent_size": parent_size,
                "left_size": left_size,
                "right_size": right_size,
                "balance_fraction": float(min(left_size, right_size) / parent_size),
                "parent_centroid_norm": vector_norm(parent_centroid),
                "left_centroid_norm": vector_norm(left_centroid),
                "right_centroid_norm": vector_norm(right_centroid),
                "parent_within_radius_q50": parent_radius_q50,
                "parent_within_radius_q90": quantile_or_nan(parent_radius, 0.90),
                "left_within_radius_q50": quantile_or_nan(left_radius, 0.50),
                "right_within_radius_q50": quantile_or_nan(right_radius, 0.50),
                "left_branch_norm": vector_norm(left_branch),
                "right_branch_norm": vector_norm(right_branch),
                "sibling_centroid_distance": sibling_centroid_distance,
                "sibling_centroid_cosine_from_origin": vector_cosine(
                    left_centroid, right_centroid
                ),
                "sibling_centroid_angle_from_origin_deg": angle_between_vectors_deg(
                    left_centroid, right_centroid
                ),
                "sibling_branch_cosine_from_parent": vector_cosine(
                    left_branch, right_branch
                ),
                "sibling_branch_angle_from_parent_deg": angle_between_vectors_deg(
                    left_branch, right_branch
                ),
                "sibling_separation_to_parent_radius_ratio": (
                    float(sibling_centroid_distance / parent_radius_scale)
                    if math.isfinite(parent_radius_scale)
                    else math.nan
                ),
                "sibling_separation_to_child_radius_ratio": float(
                    sibling_centroid_distance / child_radius_scale
                ),
                "parent_centroid_angle_to_leading_axis_deg": (
                    unoriented_angle_to_leading_axis_deg(parent_centroid)
                ),
                "left_branch_angle_to_leading_axis_deg": (
                    unoriented_angle_to_leading_axis_deg(left_branch)
                ),
                "right_branch_angle_to_leading_axis_deg": (
                    unoriented_angle_to_leading_axis_deg(right_branch)
                ),
                "parent_centroid_independent_fraction": independent_fraction(
                    parent_centroid
                ),
                "left_branch_independent_fraction": independent_fraction(left_branch),
                "right_branch_independent_fraction": independent_fraction(right_branch),
                "parent_common_axis_mean": float(np.mean(common_axis_score[parent_leaves])),
                "left_common_axis_mean": float(np.mean(common_axis_score[left_leaves])),
                "right_common_axis_mean": float(np.mean(common_axis_score[right_leaves])),
                "sibling_common_axis_mean_delta": common_axis_mean_delta,
                "abs_sibling_common_axis_mean_delta": abs(common_axis_mean_delta),
                "parent_cluster_count": parent_cluster_count,
                "parent_dominant_cluster_fraction": parent_dominant_cluster_fraction,
                "left_cluster_count": left_cluster_count,
                "right_cluster_count": right_cluster_count,
                "left_dominant_cluster_fraction": left_dominant_cluster_fraction,
                "right_dominant_cluster_fraction": right_dominant_cluster_fraction,
            }
        )

    return pd.DataFrame.from_records(rows)


def compute_block_geometry(
    *,
    data: pd.DataFrame,
    row: pd.Series,
    coords: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    common_axis_score: np.ndarray,
    assignment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    radius = np.linalg.norm(coords, axis=1)
    leading_coord = coords[:, 0] if coords.shape[1] else np.zeros(len(data))
    independent_radius = (
        np.linalg.norm(coords[:, 1:], axis=1)
        if coords.shape[1] > 1
        else np.zeros(len(data))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_cos_to_axis = np.divide(
            leading_coord,
            radius,
            out=np.zeros_like(leading_coord, dtype=float),
            where=radius > 1e-12,
        )
        independent_radius_fraction = np.divide(
            independent_radius,
            radius,
            out=np.zeros_like(independent_radius, dtype=float),
            where=radius > 1e-12,
        )
    angle_to_axis_deg = np.degrees(
        np.arccos(np.clip(np.abs(signed_cos_to_axis), 0.0, 1.0))
    )

    sample_geometry = pd.DataFrame(
        {
            "gene": data.index.to_numpy(),
            "weighting": row["weighting"],
            "block_name": row["block_name"],
            "cluster_id": assignment["cluster_id"].astype(int).to_numpy(),
            "cluster_size": assignment["cluster_size"].astype(int).to_numpy(),
            "common_invariant_axis_score": common_axis_score,
            "block_radius": radius,
            "leading_axis_coordinate": leading_coord,
            "signed_cosine_to_leading_axis": signed_cos_to_axis,
            "angle_to_leading_axis_deg": angle_to_axis_deg,
            "independent_radius": independent_radius,
            "independent_radius_fraction": independent_radius_fraction,
        }
    )

    start = int(row["block_start"]) - 1
    end = int(row["block_end"])
    block_eigvals = eigvals[start:end]
    block_energy = float(np.sum(block_eigvals)) if block_eigvals.size else math.nan
    leading_axis_energy_fraction = (
        float(block_eigvals[0] / block_energy)
        if block_eigvals.size and block_energy > 0
        else math.nan
    )

    pairwise = pairwise_direction_cosine_quantiles(coords)
    summary = {
        "weighting": row["weighting"],
        "block_name": row["block_name"],
        "block_start": int(row["block_start"]),
        "block_end": int(row["block_end"]),
        "block_type": row.get("block_type", ""),
        "subspace_dimensions": int(row["subspace_dimensions"]),
        "block_energy_fraction": float(row["block_energy_fraction"]),
        "leading_axis_energy_fraction_in_block": leading_axis_energy_fraction,
        "n_clusters": int(row["n_clusters"]),
        "largest_cluster_fraction": float(row["largest_cluster_fraction"]),
        "singleton_fraction": float(row["singleton_fraction"]),
        "radius_mean": float(np.mean(radius)),
        "radius_std": float(np.std(radius)),
        "radius_q10": quantile_or_nan(radius, 0.10),
        "radius_q50": quantile_or_nan(radius, 0.50),
        "radius_q90": quantile_or_nan(radius, 0.90),
        "radius_max": float(np.max(radius)),
        "radius_common_axis_correlation": corr_or_nan(radius, common_axis_score),
        "leading_coord_common_axis_correlation": corr_or_nan(
            leading_coord, common_axis_score
        ),
        "angle_to_leading_axis_deg_q10": quantile_or_nan(angle_to_axis_deg, 0.10),
        "angle_to_leading_axis_deg_q50": quantile_or_nan(angle_to_axis_deg, 0.50),
        "angle_to_leading_axis_deg_q90": quantile_or_nan(angle_to_axis_deg, 0.90),
        "independent_radius_fraction_q50": quantile_or_nan(
            independent_radius_fraction, 0.50
        ),
        "independent_radius_fraction_q90": quantile_or_nan(
            independent_radius_fraction, 0.90
        ),
        **pairwise,
    }
    return sample_geometry, summary


def summarize_internal_geometry(internal_geometry: pd.DataFrame) -> dict[str, float]:
    if internal_geometry.empty:
        return {
            "internal_balance_q50": math.nan,
            "internal_sibling_separation_ratio_q50": math.nan,
            "internal_sibling_center_cosine_q50": math.nan,
            "internal_parent_radius_q50": math.nan,
            "internal_parent_cluster_count_q50": math.nan,
        }
    return {
        "internal_balance_q50": quantile_or_nan(
            internal_geometry["balance_fraction"].to_numpy(), 0.50
        ),
        "internal_sibling_separation_ratio_q50": quantile_or_nan(
            internal_geometry["sibling_separation_to_child_radius_ratio"].to_numpy(),
            0.50,
        ),
        "internal_sibling_center_cosine_q50": quantile_or_nan(
            internal_geometry["sibling_centroid_cosine_from_origin"].to_numpy(), 0.50
        ),
        "internal_sibling_angle_from_origin_deg_q50": quantile_or_nan(
            internal_geometry["sibling_centroid_angle_from_origin_deg"].to_numpy(), 0.50
        ),
        "internal_sibling_branch_angle_from_parent_deg_q50": quantile_or_nan(
            internal_geometry["sibling_branch_angle_from_parent_deg"].to_numpy(), 0.50
        ),
        "internal_sibling_separation_parent_ratio_q50": quantile_or_nan(
            internal_geometry["sibling_separation_to_parent_radius_ratio"].to_numpy(),
            0.50,
        ),
        "internal_parent_radius_q50": quantile_or_nan(
            internal_geometry["parent_within_radius_q50"].to_numpy(), 0.50
        ),
        "internal_parent_cluster_count_q50": quantile_or_nan(
            internal_geometry["parent_cluster_count"].to_numpy(), 0.50
        ),
        "internal_parent_dominant_cluster_fraction_q50": quantile_or_nan(
            internal_geometry["parent_dominant_cluster_fraction"].to_numpy(), 0.50
        ),
        "internal_abs_common_axis_gap_q50": quantile_or_nan(
            internal_geometry["abs_sibling_common_axis_mean_delta"].to_numpy(), 0.50
        ),
    }


def build_artifacts(
    *,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: Path,
    embedding_method: EmbeddingMethod,
    random_state: int,
    max_rank: int,
) -> tuple[list[BlockArtifact], pd.DataFrame]:
    artifacts: list[BlockArtifact] = []
    spectrum_rows: list[dict[str, object]] = []
    eigensystems: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for weighting in sorted(summary["weighting"].unique()):
        values = weighted_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, max_rank)
        common_axis_score = eigvecs[:, 0] * math.sqrt(max(float(eigvals[0]), 0.0))
        eigensystems[weighting] = (eigvals, eigvecs, common_axis_score)
        total_energy = float(np.sum(eigvals))
        for component, eigval in enumerate(eigvals, start=1):
            spectrum_rows.append(
                {
                    "weighting": weighting,
                    "component": component,
                    "eigenvalue": float(eigval),
                    "fraction_of_kept_operator_energy": (
                        float(eigval / total_energy) if total_energy > 0 else math.nan
                    ),
                    "is_common_invariant_axis": component == 1,
                }
            )

    for row_index, row in summary.iterrows():
        weighting = str(row["weighting"])
        eigvals, eigvecs, common_axis_score = eigensystems[weighting]
        block = spectral_block_from_row(row)
        coords = coords_for_block(eigvals, eigvecs, block)
        distances = pdist(coords, metric="euclidean")
        if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
            raise ValueError(f"Degenerate block distances for {weighting}/{block.block_name}")
        linkage_matrix = linkage(distances, method="average")
        assignment = load_assignment(resolve_assignment_path(results_dir, row["assignments_path"]), data.index)
        colors, _ = cluster_color_values(assignment["cluster_id"])
        emb2 = compute_embedding(
            coords,
            n_components=2,
            method=embedding_method,
            random_state=random_state + 101 + int(row_index),
        )
        emb3 = compute_embedding(
            coords,
            n_components=3,
            method=embedding_method,
            random_state=random_state + 201 + int(row_index),
        )
        sample_geometry, summary_geometry = compute_block_geometry(
            data=data,
            row=row,
            coords=coords,
            eigvals=eigvals,
            eigvecs=eigvecs,
            common_axis_score=common_axis_score,
            assignment=assignment,
        )
        internal_geometry = compute_internal_tree_geometry(
            row=row,
            linkage_matrix=linkage_matrix,
            coords=coords,
            common_axis_score=common_axis_score,
            assignment=assignment,
        )
        summary_geometry.update(summarize_internal_geometry(internal_geometry))
        artifacts.append(
            BlockArtifact(
                run_id=f"{weighting}__{block.block_name}",
                weighting=weighting,
                block_name=block.block_name,
                summary_row=row,
                assignments=assignment,
                colors=colors,
                coords=coords,
                embedding_2d=emb2,
                embedding_3d=emb3,
                linkage_matrix=linkage_matrix,
                sample_geometry=sample_geometry,
                internal_geometry=internal_geometry,
                summary_geometry=summary_geometry,
            )
        )

    return artifacts, pd.DataFrame.from_records(spectrum_rows)


def top_cluster_sizes(labels: pd.Series, *, limit: int = 6) -> str:
    sizes = labels.astype(int).value_counts().head(limit)
    return ", ".join(f"C{int(cluster_id)}={int(size)}" for cluster_id, size in sizes.items())


def reference_metric_text(summary_geometry: dict[str, object]) -> list[str]:
    if "reference_nmi" not in summary_geometry:
        return []
    nmi = summary_geometry.get("reference_nmi", math.nan)
    ari = summary_geometry.get("reference_ari", math.nan)
    matched = summary_geometry.get("reference_matched_genes", 0)
    observed = summary_geometry.get("reference_cluster_count_observed", 0)
    if nmi is None or pd.isna(nmi):
        return [f"ref matched {int(matched)}"]
    return [
        f"ref NMI {float(nmi):.3f}",
        f"ref ARI {float(ari):.3f}" if ari is not None and pd.notna(ari) else "ref ARI NA",
        f"ref genes {int(matched)}",
        f"ref classes {int(observed)}",
    ]


def draw_reference_nmi_ranking(metrics: pd.DataFrame, output_png: Path) -> None:
    if metrics.empty or "reference_nmi" not in metrics.columns:
        return
    frame = metrics.copy()
    frame["reference_nmi"] = pd.to_numeric(frame["reference_nmi"], errors="coerce")
    frame = frame.dropna(subset=["reference_nmi"]).sort_values("reference_nmi", ascending=True)
    if frame.empty:
        return

    labels = frame["run_id"].astype(str).to_list()
    fig_height = max(4.8, 0.42 * len(frame) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    bars = ax.barh(labels, frame["reference_nmi"], color="#386cb0", alpha=0.86)
    ax.set_xlabel("NMI against Julia reference endotypes")
    ax.set_ylabel("KAK/cosine block")
    ax.set_title("KAK/cosine blocks ordered by reference NMI")
    ax.set_xlim(0.0, max(1.0, float(frame["reference_nmi"].max()) * 1.08))
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)
    for bar, nmi, clusters in zip(bars, frame["reference_nmi"], frame["n_clusters"], strict=False):
        ax.text(
            float(nmi) + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{float(nmi):.3f} / {int(clusters)} clusters",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def draw_tree_with_strip(
    *,
    ax_tree: plt.Axes,
    ax_strip: plt.Axes,
    artifact: BlockArtifact,
) -> None:
    result = dendrogram(
        artifact.linkage_matrix,
        ax=ax_tree,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
        link_color_func=lambda _: "#333333",
    )
    ax_tree.set_title("Tree leaf order", fontsize=9)
    ax_tree.set_ylabel("KAK distance", fontsize=8)
    ax_tree.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_tree.tick_params(axis="y", labelsize=7)

    leaf_order = np.asarray(result["leaves"], dtype=int)
    strip_values = artifact.colors[leaf_order]
    n_colors = max(int(strip_values.max()) + 1, 1)
    base_cmap = plt.get_cmap("tab20")
    strip_cmap = ListedColormap([base_cmap(i % base_cmap.N) for i in range(n_colors)])
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), strip_cmap.N)
    ax_strip.imshow(
        strip_values[np.newaxis, :],
        aspect="auto",
        cmap=strip_cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax_strip.set_yticks([])
    ax_strip.set_xticks([])
    ax_strip.set_xlabel("Tree leaves; color = cluster", fontsize=7)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)


def draw_internal_geometry_plot(*, ax: plt.Axes, artifact: BlockArtifact) -> None:
    internal = artifact.internal_geometry
    if internal.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No internal tree geometry", ha="center", va="center")
        return

    x = internal["balance_fraction"].to_numpy(dtype=float)
    y = internal["sibling_separation_to_parent_radius_ratio"].to_numpy(dtype=float)
    color = internal["parent_dominant_cluster_fraction"].to_numpy(dtype=float)
    size_source = internal["parent_size"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(color) & (y > 0)
    if int(valid.sum()) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Internal ratios are not finite", ha="center", va="center")
        return

    max_size = max(float(np.nanmax(size_source[valid])), 1.0)
    sizes = 12.0 + 54.0 * np.sqrt(size_source[valid] / max_size)
    scatter = ax.scatter(
        x[valid],
        y[valid],
        c=color[valid],
        s=sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title("Internal tree geometry", fontsize=10)
    ax.set_xlabel("Split balance", fontsize=9)
    ax.set_ylabel("Sibling separation / parent radius", fontsize=9)
    ax.set_xlim(-0.02, 0.52)
    ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Dominant cluster fraction", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def draw_consolidated_page(
    *,
    artifacts: list[BlockArtifact],
    global_embedding: pd.DataFrame,
    global_x: str,
    global_y: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    n_rows = len(artifacts)
    height = max(8.6, 3.35 * n_rows + 1.45)
    fig = plt.figure(figsize=(22.0, height))
    grid = fig.add_gridspec(
        nrows=n_rows,
        ncols=5,
        width_ratios=[1.0, 1.0, 1.0, 1.28, 0.96],
        hspace=0.54,
        wspace=0.30,
    )
    cmap = plt.get_cmap("tab20")

    for i, artifact in enumerate(artifacts):
        merged = global_embedding.merge(artifact.assignments, on="gene", how="inner")
        if merged.shape[0] != artifact.assignments.shape[0]:
            raise ValueError(f"Global embedding does not cover all genes for {artifact.run_id}.")
        colors = artifact.colors

        ax_global = fig.add_subplot(grid[i, 0])
        ax_sub = fig.add_subplot(grid[i, 1])
        ax_geom = fig.add_subplot(grid[i, 2])
        tree_grid = grid[i, 3].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[0.80, 0.20],
            hspace=0.03,
        )
        ax_tree = fig.add_subplot(tree_grid[0, 0])
        ax_strip = fig.add_subplot(tree_grid[1, 0])
        ax_text = fig.add_subplot(grid[i, 4])

        ax_global.scatter(
            merged[global_x],
            merged[global_y],
            c=colors,
            cmap=cmap,
            s=12,
            alpha=0.82,
            linewidths=0,
        )
        ax_global.set_title("Whole matrix embedding", fontsize=9)
        ax_global.set_xlabel(global_x, fontsize=8)
        ax_global.set_ylabel(global_y, fontsize=8)
        ax_global.tick_params(labelsize=7)

        ax_sub.scatter(
            artifact.embedding_2d[:, 0],
            artifact.embedding_2d[:, 1],
            c=colors,
            cmap=cmap,
            s=12,
            alpha=0.82,
            linewidths=0,
        )
        ax_sub.set_title("KAK block embedding", fontsize=9)
        ax_sub.set_xlabel("Block axis 1", fontsize=8)
        ax_sub.set_ylabel("Block axis 2", fontsize=8)
        ax_sub.tick_params(labelsize=7)

        geom = artifact.sample_geometry
        ax_geom.scatter(
            geom["common_invariant_axis_score"],
            geom["block_radius"],
            c=colors,
            cmap=cmap,
            s=12,
            alpha=0.78,
            linewidths=0,
        )
        ax_geom.set_title("Invariant axis vs radius", fontsize=9)
        ax_geom.set_xlabel("Common/invariant axis score", fontsize=8)
        ax_geom.set_ylabel("Block radius", fontsize=8)
        ax_geom.tick_params(labelsize=7)

        draw_tree_with_strip(ax_tree=ax_tree, ax_strip=ax_strip, artifact=artifact)

        row = artifact.summary_row
        g = artifact.summary_geometry
        text = "\n".join(
            [
                artifact.run_id,
                "",
                f"display rank {int(g.get('display_order_rank', i + 1))}",
                f"modes {int(row['block_start'])}-{int(row['block_end'])}",
                f"dim {int(row['subspace_dimensions'])}",
                f"energy {float(row['block_energy_fraction']):.3f}",
                "",
                *reference_metric_text(g),
                "",
                f"clusters {int(row['n_clusters'])}",
                f"largest {100.0 * float(row['largest_cluster_fraction']):.1f}%",
                f"singletons {100.0 * float(row['singleton_fraction']):.1f}%",
                "",
                f"radius q50 {g['radius_q50']:.4f}",
                f"radius q90 {g['radius_q90']:.4f}",
                f"angle q50 {g['angle_to_leading_axis_deg_q50']:.1f} deg",
                f"indep frac q50 {g['independent_radius_fraction_q50']:.3f}",
                f"corr(r,inv) {g['radius_common_axis_correlation']:.3f}",
                "",
                f"tree balance q50 {g['internal_balance_q50']:.3f}",
                f"tree sep/r q50 {g['internal_sibling_separation_parent_ratio_q50']:.2f}",
                f"tree angle q50 {g['internal_sibling_angle_from_origin_deg_q50']:.1f} deg",
                f"tree dom frac q50 {g['internal_parent_dominant_cluster_fraction_q50']:.3f}",
                "",
                "largest groups:",
                top_cluster_sizes(artifact.assignments["cluster_id"]),
            ]
        )
        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            linespacing=1.18,
        )

    fig.suptitle(
        "Diagnostic adaptive cosine/KAK geometry: global view, block view, radius law, tree",
        fontsize=14,
        y=0.990,
    )
    fig.subplots_adjust(top=0.935, bottom=0.030, left=0.045, right=0.990)
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def draw_per_block_pages(
    *,
    artifacts: list[BlockArtifact],
    global_embedding: pd.DataFrame,
    global_x: str,
    global_y: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    paths: list[Path] = []
    for artifact in artifacts:
        merged = global_embedding.merge(artifact.assignments, on="gene", how="inner")
        colors = artifact.colors
        geom = artifact.sample_geometry
        fig = plt.figure(figsize=(22.0, 10.2))
        grid = fig.add_gridspec(
            2,
            4,
            width_ratios=[1, 1, 1.15, 1.08],
            height_ratios=[1, 1],
            wspace=0.30,
            hspace=0.34,
        )
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
        tree_grid = grid[0, 2].subgridspec(2, 1, height_ratios=[0.82, 0.18], hspace=0.03)
        ax_tree = fig.add_subplot(tree_grid[0, 0])
        ax_strip = fig.add_subplot(tree_grid[1, 0])
        ax_angle = fig.add_subplot(grid[1, 1])
        ax_internal = fig.add_subplot(grid[1, 2])
        ax_text = fig.add_subplot(grid[:, 3])

        axes[0].scatter(merged[global_x], merged[global_y], c=colors, cmap=cmap, s=16, alpha=0.84)
        axes[0].set_title("Whole matrix embedding")
        axes[0].set_xlabel(global_x)
        axes[0].set_ylabel(global_y)

        axes[1].scatter(
            artifact.embedding_2d[:, 0],
            artifact.embedding_2d[:, 1],
            c=colors,
            cmap=cmap,
            s=16,
            alpha=0.84,
        )
        axes[1].set_title("KAK block embedding")
        axes[1].set_xlabel("Block embedding axis 1")
        axes[1].set_ylabel("Block embedding axis 2")

        axes[2].scatter(
            geom["common_invariant_axis_score"],
            geom["block_radius"],
            c=colors,
            cmap=cmap,
            s=16,
            alpha=0.80,
        )
        axes[2].set_title("Radius against common/invariant axis")
        axes[2].set_xlabel("Common/invariant axis score")
        axes[2].set_ylabel("Block radius")

        ax_angle.scatter(
            geom["angle_to_leading_axis_deg"],
            geom["independent_radius_fraction"],
            c=colors,
            cmap=cmap,
            s=16,
            alpha=0.80,
        )
        ax_angle.set_title("Angle and independent radius")
        ax_angle.set_xlabel("Angle to leading block axis (deg)")
        ax_angle.set_ylabel("Independent radius fraction")

        draw_tree_with_strip(ax_tree=ax_tree, ax_strip=ax_strip, artifact=artifact)
        draw_internal_geometry_plot(ax=ax_internal, artifact=artifact)

        g = artifact.summary_geometry
        preferred = [
            "display_order_rank",
            "reference_nmi",
            "reference_ari",
            "reference_matched_genes",
            "reference_cluster_count_observed",
        ]
        ordered_items = [(key, g[key]) for key in preferred if key in g]
        ordered_items.extend((key, value) for key, value in g.items() if key not in set(preferred))
        text = "\n".join(f"{key}: {value}" for key, value in ordered_items)
        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            linespacing=1.18,
        )

        fig.suptitle(artifact.run_id, fontsize=13, y=0.975)
        fig.subplots_adjust(
            top=0.915,
            bottom=0.070,
            left=0.055,
            right=0.985,
            wspace=0.30,
            hspace=0.34,
        )
        path = output_dir / f"{safe_name(artifact.run_id)}_geometry_page.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def write_interactive_html(
    *,
    artifacts: list[BlockArtifact],
    output_path: Path,
) -> None:
    fig = go.Figure()
    trace_groups: dict[str, list[int]] = {}
    for artifact_index, artifact in enumerate(artifacts):
        trace_groups[artifact.run_id] = []
        coords = artifact.embedding_3d
        geom = artifact.sample_geometry
        for cluster_id, group in artifact.assignments.groupby("cluster_id", sort=True):
            row_indices = group.index.to_numpy()
            visible = artifact_index == 0
            trace_index = len(fig.data)
            trace_groups[artifact.run_id].append(trace_index)
            customdata = np.stack(
                [
                    group["cluster_id"].to_numpy(),
                    group["cluster_root"].astype(str).to_numpy(),
                    group["cluster_size"].to_numpy(),
                    geom.loc[row_indices, "block_radius"].to_numpy(),
                    geom.loc[row_indices, "angle_to_leading_axis_deg"].to_numpy(),
                    geom.loc[row_indices, "independent_radius_fraction"].to_numpy(),
                    geom.loc[row_indices, "common_invariant_axis_score"].to_numpy(),
                ],
                axis=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=coords[row_indices, 0],
                    y=coords[row_indices, 1],
                    z=coords[row_indices, 2],
                    mode="markers",
                    visible=visible,
                    name=f"{artifact.run_id}: C{int(cluster_id)}",
                    text=group["gene"],
                    customdata=customdata,
                    marker={"size": 4, "opacity": 0.82},
                    hovertemplate=(
                        "gene=%{text}<br>"
                        "cluster=%{customdata[0]}<br>"
                        "root=%{customdata[1]}<br>"
                        "cluster size=%{customdata[2]}<br>"
                        "radius=%{customdata[3]:.4f}<br>"
                        "angle=%{customdata[4]:.2f} deg<br>"
                        "indep radius frac=%{customdata[5]:.3f}<br>"
                        "invariant score=%{customdata[6]:.4f}<br>"
                        "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

    buttons = []
    n_traces = len(fig.data)
    for run_id, trace_indices in trace_groups.items():
        visible = [False] * n_traces
        for trace_index in trace_indices:
            visible[trace_index] = True
        buttons.append(
            {
                "label": run_id,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"KAK/cosine block geometry: {run_id}"},
                ],
            }
        )

    initial = artifacts[0].run_id if artifacts else ""
    fig.update_layout(
        title=f"KAK/cosine block geometry: {initial}",
        scene={
            "xaxis_title": "Block embedding axis 1",
            "yaxis_title": "Block embedding axis 2",
            "zaxis_title": "Block embedding axis 3",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"l": 0, "r": 0, "t": 85, "b": 0},
        height=820,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_block_png").mkdir(exist_ok=True)

    data = load_matrix(args.input)
    summary = load_probe_summary(args.kak_results_dir, args.weightings)
    if args.block_limit > 0:
        summary = summary.head(args.block_limit).copy()

    global_embedding, global_x, global_y = load_or_compute_global_embedding(
        data=data,
        global_embedding_path=args.global_embedding,
        weighting=args.global_weighting,
        method=args.embedding_method,
        random_state=args.random_state,
    )
    global_embedding.to_csv(out / "kak_signal_adaptive_global_embedding.csv", index=False)

    artifacts, spectrum = build_artifacts(
        data=data,
        summary=summary,
        results_dir=args.kak_results_dir,
        embedding_method=args.embedding_method,
        random_state=args.random_state,
        max_rank=args.max_rank,
    )
    reference_metrics = pd.DataFrame()
    if args.reference_endotypes is not None:
        if not args.reference_endotypes.exists():
            raise FileNotFoundError(args.reference_endotypes)
        reference_labels = build_reference_labels(data.index, args.reference_endotypes)
        reference_labels.to_csv(out / "kak_signal_adaptive_reference_labels.csv", index=False)
        reference_metrics = attach_reference_labels(artifacts, reference_labels)
        artifacts = order_artifacts(artifacts, order_by=args.order_by)
        reference_metrics = pd.DataFrame.from_records(
            [
                {
                    "display_order_rank": artifact.summary_geometry.get("display_order_rank"),
                    "run_id": artifact.run_id,
                    "weighting": artifact.weighting,
                    "block_name": artifact.block_name,
                    "n_clusters": int(artifact.summary_row["n_clusters"]),
                    "reference_matched_genes": artifact.summary_geometry.get(
                        "reference_matched_genes"
                    ),
                    "reference_unmatched_genes": artifact.summary_geometry.get(
                        "reference_unmatched_genes"
                    ),
                    "reference_cluster_count_observed": artifact.summary_geometry.get(
                        "reference_cluster_count_observed"
                    ),
                    "reference_ari": artifact.summary_geometry.get("reference_ari"),
                    "reference_nmi": artifact.summary_geometry.get("reference_nmi"),
                }
                for artifact in artifacts
            ]
        )
        reference_metrics.to_csv(
            out / "kak_signal_adaptive_reference_nmi_metrics.csv",
            index=False,
        )
        draw_reference_nmi_ranking(
            reference_metrics,
            out / "kak_signal_adaptive_reference_nmi_ranking.png",
        )
    else:
        artifacts = order_artifacts(artifacts, order_by="input")
    spectrum.to_csv(out / "kak_signal_adaptive_spectrum_for_plot.csv", index=False)

    sample_geometry = pd.concat(
        [artifact.sample_geometry for artifact in artifacts],
        ignore_index=True,
    )
    sample_geometry.to_csv(
        out / "kak_signal_adaptive_sample_geometry_long.csv",
        index=False,
    )
    geometry_summary = pd.DataFrame.from_records(
        [artifact.summary_geometry for artifact in artifacts]
    )
    geometry_summary.to_csv(
        out / "kak_signal_adaptive_block_geometry_summary.csv",
        index=False,
    )
    internal_geometry = pd.concat(
        [artifact.internal_geometry for artifact in artifacts],
        ignore_index=True,
    )
    internal_geometry.to_csv(
        out / "kak_signal_adaptive_internal_tree_geometry.csv",
        index=False,
    )

    coordinate_frames = []
    for artifact in artifacts:
        frame = pd.DataFrame(
            {
                "run_id": artifact.run_id,
                "gene": data.index.to_numpy(),
                "block_embedding_2d_axis_1": artifact.embedding_2d[:, 0],
                "block_embedding_2d_axis_2": artifact.embedding_2d[:, 1],
                "block_embedding_3d_axis_1": artifact.embedding_3d[:, 0],
                "block_embedding_3d_axis_2": artifact.embedding_3d[:, 1],
                "block_embedding_3d_axis_3": artifact.embedding_3d[:, 2],
            }
        ).merge(artifact.assignments, on="gene", how="left")
        coordinate_frames.append(frame)
    pd.concat(coordinate_frames, ignore_index=True).to_csv(
        out / "kak_signal_adaptive_subembedding_coordinates_long.csv",
        index=False,
    )

    per_block_paths = draw_per_block_pages(
        artifacts=artifacts,
        global_embedding=global_embedding,
        global_x=global_x,
        global_y=global_y,
        output_dir=out / "per_block_png",
    )
    with PdfPages(out / "kak_signal_adaptive_per_block_pages.pdf") as pdf:
        for path in per_block_paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(17.5, 10.0))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    draw_consolidated_page(
        artifacts=artifacts,
        global_embedding=global_embedding,
        global_x=global_x,
        global_y=global_y,
        output_png=out / "kak_signal_adaptive_umap_tree_page.png",
        output_pdf=out / "kak_signal_adaptive_umap_tree_page.pdf",
    )
    write_interactive_html(
        artifacts=artifacts,
        output_path=out / "kak_signal_adaptive_subembedding_3d.html",
    )

    readme = [
        "# KAK Signal-Adaptive UMAP/Tree Diagnostic Page",
        "",
        f"Input matrix: `{args.input}`",
        f"KAK results: `{args.kak_results_dir}`",
        f"Blocks plotted: `{len(artifacts)}`",
        f"Embedding method: `{args.embedding_method}`",
        "",
        "## Outputs",
        "",
        "- `kak_signal_adaptive_umap_tree_page.png/pdf`: consolidated visual page.",
        "- `kak_signal_adaptive_per_block_pages.pdf`: one detailed page per block.",
        "- `kak_signal_adaptive_subembedding_3d.html`: interactive block embedding.",
        "- `kak_signal_adaptive_block_geometry_summary.csv`: radius/angle/invariant-axis diagnostics.",
        "- `kak_signal_adaptive_sample_geometry_long.csv`: per-gene geometry diagnostics.",
        "- `kak_signal_adaptive_internal_tree_geometry.csv`: per-merge internal tree geometry.",
        "- `kak_signal_adaptive_reference_nmi_metrics.csv`: optional reference NMI ordering metrics.",
        "- `kak_signal_adaptive_reference_nmi_ranking.png`: optional reference NMI ranking plot.",
        "",
        f"Reference endotypes: `{args.reference_endotypes}`",
        f"Block display order: `{args.order_by if args.reference_endotypes is not None else 'input'}`",
        "",
        "Diagnostic-only: selected KAK/cosine bases are not production-calibrated tests.",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote KAK signal-adaptive diagnostic pages: {out}", flush=True)


if __name__ == "__main__":
    main()
