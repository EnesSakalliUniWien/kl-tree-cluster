"""Pancreas scRNA classical-pipeline clustering benchmark.

This script downloads the public Scanpy pancreas AnnData object, recomputes a
standard Scanpy-style PCA/neighborhood/UMAP/Leiden workflow, and benchmarks TBS
against common clustering baselines on a deterministic stratified subset.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.util.decomposition import labels_and_report_from_decomposition
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.annotation_bundle import (
    GateAnnotationBundle,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.column_contracts import (
    EDGE_GATE_COLUMNS,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_tree_bh import (
    apply_child_parent_divergence_tree_bh_correction,
)
from tree_break_selection.hierarchy_analysis.statistics.distributional_action import (
    edge_distributional_action_summary,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.space_separation import adaptive_diffusion_geometry
from tree_break_selection.tree.continuous_distance import (
    CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC,
)
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns

DATA_URL = "https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1"
AMBIGUOUS_LABELS = {
    "not applicable",
    "dropped",
    "co-expression",
    "unclear",
    "unclassified",
    "unclassified endocrine",
    "MHC class II",
}


def _scanpy_module() -> Any:
    import scanpy as sc

    return sc


def _scanpy_version() -> str:
    try:
        return version("scanpy")
    except PackageNotFoundError as exc:
        return f"unavailable: {exc}"


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    label: str
    params: dict[str, object]
    needs_distance_matrix: bool = False
    needs_distance_condensed: bool = False
    distance_source: str = "euclidean"
    significance_level: float | None = None
    edge_alpha: float | None = None

    @property
    def assignment_key(self) -> str:
        return (
            self.label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("=", "")
            .replace(".", "p")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _prepare_adata(input_h5ad: Path, output_dir: Path, n_pcs: int) -> Any:
    sc = _scanpy_module()
    adata = sc.read(input_h5ad, backup_url=DATA_URL)
    adata.var_names_make_unique()

    if "n_genes" in adata.obs:
        adata.obs["qc_n_genes_by_counts"] = pd.to_numeric(adata.obs["n_genes"])
    if "n_counts" in adata.obs:
        adata.obs["qc_total_counts"] = pd.to_numeric(adata.obs["n_counts"])

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    if bool(adata.var["mt"].any()):
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    else:
        adata.obs["pct_counts_mt"] = np.nan

    # The Scanpy pancreas object is already intersected and preprocessed; keep a
    # raw snapshot, then recompute the standard latent-space workflow.
    if adata.raw is None:
        adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs, random_state=0)
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="classical_leiden")

    adata.write_h5ad(output_dir / "pancreas_classical_pipeline.h5ad")
    return adata


def _write_qc_outputs(adata: Any, output_dir: Path) -> dict[str, object]:
    qc = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "celltype": adata.obs["celltype"].astype(str).to_numpy(),
            "batch": adata.obs["batch"].astype(str).to_numpy()
            if "batch" in adata.obs
            else "unknown",
            "n_genes": pd.to_numeric(
                adata.obs.get("qc_n_genes_by_counts", adata.obs.get("n_genes")),
                errors="coerce",
            ),
            "total_counts": pd.to_numeric(
                adata.obs.get("qc_total_counts", adata.obs.get("n_counts")),
                errors="coerce",
            ),
            "percent_mt": pd.to_numeric(adata.obs["pct_counts_mt"], errors="coerce"),
        }
    )
    qc.to_csv(output_dir / "qc_cell_metrics.csv", index=False)

    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "batches": int(adata.obs["batch"].nunique()) if "batch" in adata.obs else 0,
        "celltype_count": int(adata.obs["celltype"].nunique()),
        "raw_layer_present": bool(adata.raw is not None),
        "mitochondrial_gene_count": int(adata.var["mt"].sum()),
        "scdblfinder_status": (
            "not_run_prepackaged_post_count_object; R/Bioconductor scDblFinder "
            "was not available in this Python-only benchmark run"
        ),
        "ambient_rna_status": ("not_run_no_empty_droplet_channel_inputs_in_prepackaged_h5ad"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    axes[0].hist(qc["n_genes"].dropna(), bins=60, color="#3b82f6")
    axes[0].set_title("Detected genes")
    axes[1].hist(qc["total_counts"].dropna(), bins=60, color="#059669")
    axes[1].set_title("Total counts")
    if qc["percent_mt"].notna().any():
        axes[2].hist(qc["percent_mt"].dropna(), bins=60, color="#dc2626")
    axes[2].set_title("Mitochondrial %")
    for ax in axes:
        ax.set_ylabel("Cells")
    fig.tight_layout()
    fig.savefig(output_dir / "qc_metric_distributions.png", dpi=180)
    plt.close(fig)
    return summary


def _benchmark_subset(
    adata: Any,
    *,
    max_cells: int,
    min_cells_per_label: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = adata.obs["celltype"].astype(str)
    eligible = labels[~labels.isin(AMBIGUOUS_LABELS)]
    counts = eligible.value_counts()
    keep_labels = counts[counts >= min_cells_per_label].index.tolist()
    eligible_indices = np.flatnonzero(labels.isin(keep_labels).to_numpy())

    if len(eligible_indices) <= max_cells:
        return eligible_indices

    selected: list[int] = []
    eligible_labels = labels.iloc[eligible_indices]
    quotas = (
        eligible_labels.value_counts(normalize=True)
        .mul(max_cells)
        .round()
        .astype(int)
        .clip(lower=1)
    )
    while int(quotas.sum()) > max_cells:
        largest = quotas.idxmax()
        quotas.loc[largest] -= 1
    while int(quotas.sum()) < max_cells:
        largest = eligible_labels.value_counts().idxmax()
        quotas.loc[largest] += 1

    for label, quota in quotas.items():
        label_indices = eligible_indices[eligible_labels.to_numpy() == label]
        chosen = rng.choice(label_indices, size=min(int(quota), len(label_indices)), replace=False)
        selected.extend(int(i) for i in chosen)

    return np.array(sorted(selected), dtype=int)


def _base_tbs_params() -> dict[str, object]:
    return {
        "tree_distance_metric": CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC,
        "tree_linkage_method": "average",
        "spectral_minimum_dimension": 2,
        "adaptive_projection_dimension_energy_fraction": 0.90,
        "continuous_covariance_policy": "guarded_within_child",
        "continuous_covariance_min_child_leaf_count": 8,
        "sibling_gate_method": "projected_wald_inflation",
        "edge_branch_length_variance_policy": "none",
        "passthrough": True,
    }


def _method_configs(true_k: int) -> list[MethodConfig]:
    tbs_topology_params = _base_tbs_params()
    tbs_raw_linkage_branch_time_params = {
        **_base_tbs_params(),
        "edge_branch_length_variance_policy": "normalized_branch_length",
        "allow_linkage_ultrametric_branch_time": True,
    }
    tbs_branch_time_params = {
        **_base_tbs_params(),
        "edge_branch_length_variance_policy": "normalized_branch_length",
        "branch_length_optimization_method": "fixed_topology_nnls",
        "branch_length_optimization_target_metric": "squared_standardized_euclidean",
        "branch_length_optimization_pair_sample_size": 50_000,
        "branch_length_optimization_random_state": 0,
        "branch_length_optimization_solver_tolerance": 1e-5,
        "branch_length_optimization_max_iterations": 1000,
    }
    tbs_adaptive_topology_params = {
        **_base_tbs_params(),
        "tree_distance_metric": "adaptive_diffusion_euclidean",
    }
    tbs_adaptive_raw_linkage_branch_time_params = {
        **tbs_adaptive_topology_params,
        "edge_branch_length_variance_policy": "normalized_branch_length",
        "allow_linkage_ultrametric_branch_time": True,
    }
    tbs_adaptive_branch_time_params = {
        **tbs_adaptive_topology_params,
        "edge_branch_length_variance_policy": "normalized_branch_length",
        "branch_length_optimization_method": "fixed_topology_nnls",
        "branch_length_optimization_target_metric": "squared_standardized_euclidean",
        "branch_length_optimization_pair_sample_size": 50_000,
        "branch_length_optimization_random_state": 0,
        "branch_length_optimization_solver_tolerance": 1e-5,
        "branch_length_optimization_max_iterations": 1000,
    }
    return [
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            "TBS topology projected adaptive-k90 alpha=0.01 edge=0.001",
            tbs_topology_params,
            needs_distance_condensed=True,
            distance_source="euclidean",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            "TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001",
            tbs_branch_time_params,
            needs_distance_condensed=True,
            distance_source="euclidean",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            ("TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001"),
            tbs_raw_linkage_branch_time_params,
            needs_distance_condensed=True,
            distance_source="euclidean",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            "TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001",
            tbs_adaptive_topology_params,
            needs_distance_condensed=True,
            distance_source="adaptive_diffusion",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            (
                "TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 "
                "alpha=0.01 edge=0.001"
            ),
            tbs_adaptive_branch_time_params,
            needs_distance_condensed=True,
            distance_source="adaptive_diffusion",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "tbs_continuous_guarded_within_covariance",
            (
                "TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 "
                "alpha=0.01 edge=0.001"
            ),
            tbs_adaptive_raw_linkage_branch_time_params,
            needs_distance_condensed=True,
            distance_source="adaptive_diffusion",
            significance_level=0.01,
            edge_alpha=0.001,
        ),
        MethodConfig(
            "leiden",
            "Leiden",
            {"n_neighbors": 15, "resolution": 1.0},
            needs_distance_matrix=True,
        ),
        MethodConfig(
            "louvain",
            "Louvain",
            {"n_neighbors": 15, "resolution": 1.0},
            needs_distance_matrix=True,
        ),
        MethodConfig("kmeans", "K-means true K", {"n_clusters": true_k, "n_init": 20}),
        MethodConfig(
            "spectral",
            "Spectral true K",
            {
                "n_clusters": true_k,
                "affinity": "nearest_neighbors",
                "assign_labels": "cluster_qr",
                "n_neighbors": 15,
            },
        ),
        MethodConfig(
            "hdbscan",
            "HDBSCAN",
            {
                "min_cluster_size": 25,
                "min_samples": 10,
                "cluster_selection_epsilon": 0.0,
            },
            needs_distance_matrix=True,
        ),
    ]


def _score_labels(
    *,
    method_label: str,
    method_id: str,
    params: dict[str, object],
    status: str,
    skip_reason: str | None,
    elapsed_sec: float,
    y_true: np.ndarray,
    y_pred: np.ndarray | None,
    X: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method_label,
        "method_id": method_id,
        "params": json.dumps(params, sort_keys=True, default=_json_default),
        "significance_level": params.get("_significance_level"),
        "edge_alpha": params.get("_edge_alpha"),
        "status": status,
        "skip_reason": skip_reason,
        "elapsed_sec": elapsed_sec,
    }
    if y_pred is None:
        row.update(
            {
                "n_clusters": 0,
                "ari": math.nan,
                "nmi": math.nan,
                "ami": math.nan,
                "homogeneity": math.nan,
                "completeness": math.nan,
                "v_measure": math.nan,
                "silhouette": math.nan,
            }
        )
        return row

    unique = np.unique(y_pred)
    row.update(
        {
            "n_clusters": int(unique.size),
            "ari": adjusted_rand_score(y_true, y_pred),
            "nmi": normalized_mutual_info_score(y_true, y_pred),
            "ami": adjusted_mutual_info_score(y_true, y_pred),
            "homogeneity": homogeneity_score(y_true, y_pred),
            "completeness": completeness_score(y_true, y_pred),
            "v_measure": v_measure_score(y_true, y_pred),
        }
    )
    if 1 < unique.size < len(y_pred):
        row["silhouette"] = silhouette_score(X, y_pred, metric="euclidean")
    else:
        row["silhouette"] = math.nan
    row.update(_split_merge_summary_metrics(y_true, y_pred))
    return row


def _entropy(values: np.ndarray) -> float:
    weights = np.asarray(values, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    probabilities = weights[weights > 0.0] / total
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def _contingency_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.crosstab(
        pd.Series(y_pred, name="cluster"),
        pd.Series(y_true, name="celltype"),
        dropna=False,
    )


def _split_merge_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    table = _contingency_table(y_true, y_pred)
    n_cells = float(table.to_numpy(dtype=float).sum())
    if n_cells <= 0.0:
        return {}

    cluster_sizes = table.sum(axis=1).astype(float)
    label_sizes = table.sum(axis=0).astype(float)
    cluster_count = int(table.shape[0])
    label_count = int(table.shape[1])

    purity = float(table.max(axis=1).sum() / n_cells)
    dominant_cluster_recall = float(table.max(axis=0).sum() / n_cells)

    cluster_entropies = []
    effective_labels = []
    mixed_cluster_cell_count = 0.0
    for cluster_id, row in table.iterrows():
        counts = row.to_numpy(dtype=float)
        size = float(cluster_sizes.loc[cluster_id])
        entropy = _entropy(counts)
        cluster_entropies.append((size, entropy))
        effective_labels.append((size, float(np.exp(entropy))))
        if size > 0.0 and float(np.max(counts)) / size < 0.8:
            mixed_cluster_cell_count += size

    label_entropies = []
    effective_clusters = []
    fragmented_label_cell_count = 0.0
    clusters_per_label_ge_5pct = []
    clusters_per_label_ge_10pct = []
    for label, column in table.items():
        counts = column.to_numpy(dtype=float)
        size = float(label_sizes.loc[label])
        entropy = _entropy(counts)
        label_entropies.append((size, entropy))
        effective_clusters.append((size, float(np.exp(entropy))))
        if size > 0.0:
            dominant_recall = float(np.max(counts)) / size
            if dominant_recall < 0.8:
                fragmented_label_cell_count += size
            clusters_per_label_ge_5pct.append(int(np.sum(counts >= 0.05 * size)))
            clusters_per_label_ge_10pct.append(int(np.sum(counts >= 0.10 * size)))

    weighted_cluster_entropy = sum(w * e for w, e in cluster_entropies) / n_cells
    weighted_label_entropy = sum(w * e for w, e in label_entropies) / n_cells
    weighted_effective_labels = sum(w * e for w, e in effective_labels) / n_cells
    weighted_effective_clusters = sum(w * e for w, e in effective_clusters) / n_cells

    return {
        "n_true_labels": float(label_count),
        "overcluster_ratio": float(cluster_count / label_count) if label_count else math.nan,
        "weighted_cluster_purity": purity,
        "merge_error_rate": 1.0 - purity,
        "weighted_label_dominant_cluster_recall": dominant_cluster_recall,
        "split_error_rate": 1.0 - dominant_cluster_recall,
        "weighted_cluster_label_entropy": weighted_cluster_entropy,
        "weighted_label_cluster_entropy": weighted_label_entropy,
        "weighted_effective_labels_per_cluster": weighted_effective_labels,
        "weighted_effective_clusters_per_label": weighted_effective_clusters,
        "mixed_cluster_cell_fraction": mixed_cluster_cell_count / n_cells,
        "fragmented_label_cell_fraction": fragmented_label_cell_count / n_cells,
        "mean_clusters_per_label_ge_5pct": float(np.mean(clusters_per_label_ge_5pct)),
        "mean_clusters_per_label_ge_10pct": float(np.mean(clusters_per_label_ge_10pct)),
    }


def _split_merge_detail_tables(
    *,
    method_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table = _contingency_table(y_true, y_pred)
    label_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    for label, column in table.items():
        counts = column.astype(float)
        size = float(counts.sum())
        if size <= 0.0:
            continue
        dominant_cluster = counts.idxmax()
        label_rows.append(
            {
                "method": method_label,
                "celltype": label,
                "n_cells": int(size),
                "n_predicted_clusters": int((counts > 0).sum()),
                "effective_cluster_count": float(np.exp(_entropy(counts.to_numpy()))),
                "dominant_cluster": dominant_cluster,
                "dominant_cluster_cells": int(counts.loc[dominant_cluster]),
                "dominant_cluster_recall": float(counts.loc[dominant_cluster] / size),
                "clusters_ge_5pct": int((counts >= 0.05 * size).sum()),
                "clusters_ge_10pct": int((counts >= 0.10 * size).sum()),
            }
        )

    for cluster_id, row in table.iterrows():
        counts = row.astype(float)
        size = float(counts.sum())
        if size <= 0.0:
            continue
        dominant_label = counts.idxmax()
        cluster_rows.append(
            {
                "method": method_label,
                "cluster": cluster_id,
                "n_cells": int(size),
                "n_celltypes": int((counts > 0).sum()),
                "effective_celltype_count": float(np.exp(_entropy(counts.to_numpy()))),
                "dominant_celltype": dominant_label,
                "dominant_celltype_cells": int(counts.loc[dominant_label]),
                "cluster_purity": float(counts.loc[dominant_label] / size),
                "celltypes_ge_10pct": int((counts >= 0.10 * size).sum()),
            }
        )

    return label_rows, cluster_rows


def _score_sensitivity_labels(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    unique = np.unique(y_pred)
    row: dict[str, object] = {
        "n_clusters": int(unique.size),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ami": adjusted_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
    }
    row.update(_split_merge_summary_metrics(y_true, y_pred))
    return row


def _branch_time_model_grid() -> list[dict[str, object]]:
    return [
        {"length_model": "none", "scale": 0.0, "gamma": 1.0, "clip_quantile": None},
        {
            "length_model": "linear_scale_0p25",
            "scale": 0.25,
            "gamma": 1.0,
            "clip_quantile": None,
        },
        {"length_model": "linear_scale_0p5", "scale": 0.5, "gamma": 1.0, "clip_quantile": None},
        {"length_model": "linear_scale_1", "scale": 1.0, "gamma": 1.0, "clip_quantile": None},
        {"length_model": "linear_scale_2", "scale": 2.0, "gamma": 1.0, "clip_quantile": None},
        {"length_model": "linear_scale_4", "scale": 4.0, "gamma": 1.0, "clip_quantile": None},
        {"length_model": "sqrt_scale_1", "scale": 1.0, "gamma": 0.5, "clip_quantile": None},
        {"length_model": "quadratic_scale_1", "scale": 1.0, "gamma": 2.0, "clip_quantile": None},
        {
            "length_model": "linear_scale_1_clip_p95",
            "scale": 1.0,
            "gamma": 1.0,
            "clip_quantile": 0.95,
        },
    ]


def _branch_time_multipliers(
    branch_lengths: np.ndarray,
    *,
    scale: float,
    gamma: float,
    clip_quantile: float | None,
) -> np.ndarray:
    lengths = np.asarray(branch_lengths, dtype=float)
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if float(scale) <= 0.0 or positive.size == 0:
        return np.ones_like(lengths, dtype=float)

    active_lengths = lengths.copy()
    if clip_quantile is not None:
        cap = float(np.quantile(positive, float(clip_quantile)))
        active_lengths = np.minimum(active_lengths, cap)

    normalizer = float(np.mean(positive))
    normalized = np.clip(active_lengths / normalizer, a_min=0.0, a_max=None)
    return 1.0 + float(scale) * np.power(normalized, float(gamma))


def _serialize_trace_value(value: object) -> object:
    if isinstance(value, tuple | list):
        return json.dumps(list(value), default=_json_default)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return value


def _write_traversal_trace_csv(
    trace: object,
    *,
    method: str,
    path: Path,
) -> None:
    trace_df = pd.DataFrame(trace if isinstance(trace, list) else [])
    trace_df.insert(0, "method", method)
    for column in trace_df.columns:
        if trace_df[column].map(lambda value: isinstance(value, tuple | list | dict)).any():
            trace_df[column] = trace_df[column].map(_serialize_trace_value)
    trace_df.to_csv(path, index=False)


def _decompose_with_hypothetical_edge_annotations(
    *,
    tree: object,
    base_bundle: GateAnnotationBundle,
    annotations_df: pd.DataFrame,
    config: MethodConfig,
    sample_ids: np.ndarray,
) -> np.ndarray:
    edge_result = replace(base_bundle.edge_gate_result, annotated_df=annotations_df)
    bundle = replace(
        base_bundle,
        annotated_df=annotations_df,
        edge_gate_result=edge_result,
        stage_timings={},
    )
    decomposer = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=bundle,
        passthrough=bool(config.params.get("passthrough", True)),
    )
    labels, _report = labels_and_report_from_decomposition(
        decomposer.decompose_tree(),
        sample_ids,
    )
    return labels


def _write_branch_time_sensitivity_plot(
    sensitivity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if sensitivity_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, group in sensitivity_df.groupby("method", sort=False):
        ax.plot(
            group["weighted_cluster_purity"],
            group["weighted_label_dominant_cluster_recall"],
            marker="o",
            linewidth=1.2,
            label=str(method),
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(row["length_model"]),
                (
                    float(row["weighted_cluster_purity"]),
                    float(row["weighted_label_dominant_cluster_recall"]),
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Weighted cluster purity (merge control)")
    ax.set_ylabel("Weighted dominant-cluster recall (split control)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "tbs_branch_time_sensitivity.png", dpi=200)
    plt.close(fig)


def _write_tbs_tree_diagnostics(
    *,
    config: MethodConfig,
    result_extra: dict[str, object],
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
    length_sensitivity_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    tree = result_extra.get("tree")
    if tree is None or not hasattr(tree, "edges"):
        return None

    key = config.assignment_key
    annotations = result_extra.get("annotations")
    annotations_df = annotations if isinstance(annotations, pd.DataFrame) else None

    rows: list[dict[str, object]] = []
    for parent, child, attrs in tree.edges(data=True):
        parent_distribution = np.asarray(tree.nodes[parent]["distribution"], dtype=float)
        child_distribution = np.asarray(tree.nodes[child]["distribution"], dtype=float)
        parent_leaf_count = tree.nodes[parent].get("leaf_count", math.nan)
        child_leaf_count = tree.nodes[child].get("leaf_count", math.nan)
        action_summary = edge_distributional_action_summary(
            parent_distribution,
            child_distribution,
            parent_leaf_count,
            child_leaf_count,
        )
        branch_length = attrs.get("branch_length", math.nan)
        row: dict[str, object] = {
            "method": config.label,
            "parent": parent,
            "child": child,
            "branch_length": branch_length,
            "parent_leaf_count": parent_leaf_count,
            "child_leaf_count": child_leaf_count,
            "distributional_action_parent_mass": action_summary.parent_mass,
            "distributional_action_child_mass": action_summary.child_mass,
            "distributional_action_child_parent_mass_fraction": (
                action_summary.child_parent_mass_fraction
            ),
            "distributional_action_squared_displacement": (action_summary.squared_displacement),
            "distributional_action": action_summary.action,
            "distributional_action_per_branch_length": (
                action_summary.action / max(float(branch_length), 1e-12)
                if np.isfinite(float(branch_length))
                else math.nan
            ),
        }
        if annotations_df is not None and child in annotations_df.index:
            for column in EDGE_GATE_COLUMNS:
                if column in annotations_df.columns:
                    row[column] = annotations_df.loc[child, column]
            for column in annotations_df.columns:
                if (
                    column == "Distributional_Action"
                    or column.startswith("Distributional_Action_")
                    or column == "Distributional_Split_Action"
                    or column.startswith("Distributional_Split_Action_")
                ):
                    row[column] = annotations_df.loc[child, column]
        rows.append(row)

    edge_df = pd.DataFrame(rows)
    edge_df.to_csv(output_dir / f"{key}_tree_edges.csv", index=False)
    _write_traversal_trace_csv(
        result_extra.get("traversal_trace", []),
        method=config.label,
        path=output_dir / f"{key}_traversal_trace.csv",
    )
    _write_traversal_trace_csv(
        result_extra.get("full_edge_traversal_trace", []),
        method=config.label,
        path=output_dir / f"{key}_full_edge_traversal_trace.csv",
    )

    branch_lengths = pd.to_numeric(edge_df["branch_length"], errors="coerce").dropna()
    summary: dict[str, object] = {
        "method": config.label,
        "edge_count": int(len(edge_df)),
        "branch_length_count": int(branch_lengths.size),
        "branch_length_mean": float(branch_lengths.mean())
        if not branch_lengths.empty
        else math.nan,
        "branch_length_median": float(branch_lengths.median())
        if not branch_lengths.empty
        else math.nan,
        "branch_length_min": float(branch_lengths.min()) if not branch_lengths.empty else math.nan,
        "branch_length_max": float(branch_lengths.max()) if not branch_lengths.empty else math.nan,
        "edge_branch_length_variance_policy": result_extra.get(
            "edge_branch_length_variance_policy",
            config.params.get("edge_branch_length_variance_policy"),
        ),
        "spectral_minimum_dimension": result_extra.get("spectral_minimum_dimension"),
        "adaptive_projection_dimension_energy_fraction": result_extra.get(
            "adaptive_projection_dimension_energy_fraction"
        ),
        "tree_builder": result_extra.get("tree_builder"),
        "branch_length_optimization_method": result_extra.get(
            "branch_length_optimization_method",
            config.params.get("branch_length_optimization_method", "linkage_ultrametric"),
        ),
        "branch_length_optimization_residual_rmse": result_extra.get(
            "branch_length_optimization_residual_rmse"
        ),
        "branch_length_optimization_residual_mae": result_extra.get(
            "branch_length_optimization_residual_mae"
        ),
        "branch_length_optimization_n_pairs_used": result_extra.get(
            "branch_length_optimization_n_pairs_used"
        ),
        "branch_length_optimization_elapsed_sec": result_extra.get(
            "branch_length_optimization_elapsed_sec"
        ),
    }

    if not branch_lengths.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(branch_lengths, bins=60, color="#64748b")
        ax.set_title(config.label)
        ax.set_xlabel("Branch length")
        ax.set_ylabel("Edges")
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}_branch_lengths.png", dpi=200)
        plt.close(fig)

    linkage_matrix = result_extra.get("linkage_matrix")
    if linkage_matrix is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(
            np.asarray(linkage_matrix, dtype=float),
            truncate_mode="lastp",
            p=60,
            no_labels=True,
            ax=ax,
        )
        ax.set_title(f"{config.label} truncated dendrogram")
        ax.set_xlabel("Merged subtrees")
        ax.set_ylabel("Linkage distance")
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}_tree_dendrogram.png", dpi=200)
        plt.close(fig)

    base_bundle = result_extra.get("gate_bundle")
    if (
        isinstance(base_bundle, GateAnnotationBundle)
        and annotations_df is not None
        and str(summary["edge_branch_length_variance_policy"]) == "none"
        and all(column in annotations_df.columns for column in EDGE_GATE_COLUMNS)
    ):
        tree_edges = list(tree.edges())
        child_ids = [child for _parent, child in tree_edges]
        base_statistics = (
            annotations_df.loc[child_ids, "Child_Parent_Divergence_Test_Statistic"]
            .astype(float)
            .to_numpy()
        )
        degrees = (
            annotations_df.loc[child_ids, "Child_Parent_Divergence_df"].astype(float).to_numpy()
        )
        edge_lengths = np.array(
            [
                float(tree.edges[parent, child].get("branch_length", math.nan))
                for parent, child in tree_edges
            ],
            dtype=float,
        )
        finite_inputs = (
            np.isfinite(base_statistics)
            & np.isfinite(degrees)
            & (degrees > 0.0)
            & np.isfinite(edge_lengths)
            & (edge_lengths >= 0.0)
        )
        if finite_inputs.all():
            for model in _branch_time_model_grid():
                multipliers = _branch_time_multipliers(
                    edge_lengths,
                    scale=float(model["scale"]),
                    gamma=float(model["gamma"]),
                    clip_quantile=model["clip_quantile"],
                )
                transformed_statistics = base_statistics / multipliers
                transformed_p_values = chi2.sf(transformed_statistics, degrees)
                (
                    reject_null,
                    p_values_corrected,
                    tested_edge_flags,
                    ancestor_blocked_flags,
                ) = apply_child_parent_divergence_tree_bh_correction(
                    tree=tree,
                    p_values_for_correction=transformed_p_values,
                    child_ids=child_ids,
                    edge_alpha=float(config.edge_alpha or 0.001),
                )
                transformed_annotations = annotations_df.copy()
                transformed_annotations.loc[child_ids, "Child_Parent_Divergence_Test_Statistic"] = (
                    transformed_statistics
                )
                transformed_annotations.loc[child_ids, "Child_Parent_Divergence_P_Value"] = (
                    transformed_p_values
                )
                transformed_annotations.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = (
                    p_values_corrected
                )
                transformed_annotations.loc[child_ids, "Child_Parent_Divergence_Significant"] = (
                    reject_null
                )
                transformed_annotations.loc[child_ids, "Child_Parent_Divergence_Tested"] = (
                    tested_edge_flags
                )
                transformed_annotations.loc[
                    child_ids, "Child_Parent_Divergence_Ancestor_Blocked"
                ] = ancestor_blocked_flags
                labels = _decompose_with_hypothetical_edge_annotations(
                    tree=tree,
                    base_bundle=base_bundle,
                    annotations_df=transformed_annotations,
                    config=config,
                    sample_ids=sample_ids,
                )
                row = {
                    "method": config.label,
                    "length_model": model["length_model"],
                    "scale": model["scale"],
                    "gamma": model["gamma"],
                    "clip_quantile": model["clip_quantile"],
                    "edge_open_count": int(np.sum(reject_null)),
                    "tested_edge_count": int(np.sum(tested_edge_flags)),
                    "ancestor_blocked_edge_count": int(np.sum(ancestor_blocked_flags)),
                    "median_variance_multiplier": float(np.median(multipliers)),
                    "max_variance_multiplier": float(np.max(multipliers)),
                }
                row.update(_score_sensitivity_labels(y_true, labels))
                length_sensitivity_rows.append(row)

    return summary


def _write_distance_time_model_analysis(
    *,
    output_dir: Path,
    tree_summaries: list[dict[str, object]],
    adaptive_metadata: dict[str, object] | None,
    length_sensitivity_rows: list[dict[str, object]],
    generated_at: str,
) -> None:
    tree_summary_df = pd.DataFrame(tree_summaries)
    if not tree_summary_df.empty:
        tree_summary_df.to_csv(output_dir / "tbs_tree_branch_length_summary.csv", index=False)
        tree_table = tree_summary_df.to_markdown(index=False, floatfmt=".4f")
    else:
        tree_table = "_No TBS tree diagnostics were produced._"

    adaptive_text = (
        json.dumps(adaptive_metadata, indent=2, sort_keys=True, default=_json_default)
        if adaptive_metadata is not None
        else "adaptive diffusion distance was unavailable in this run"
    )
    length_sensitivity_df = pd.DataFrame(length_sensitivity_rows)
    if not length_sensitivity_df.empty:
        best_rows = (
            length_sensitivity_df.sort_values(
                [
                    "method",
                    "v_measure",
                    "weighted_label_dominant_cluster_recall",
                    "weighted_cluster_purity",
                ],
                ascending=[True, False, False, False],
            )
            .groupby("method", as_index=False)
            .head(3)
        )
        length_sensitivity_table = best_rows[
            [
                "method",
                "length_model",
                "edge_open_count",
                "n_clusters",
                "weighted_cluster_purity",
                "weighted_label_dominant_cluster_recall",
                "merge_error_rate",
                "split_error_rate",
                "v_measure",
                "ari",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    else:
        length_sensitivity_table = "_No branch-time sensitivity rows were produced._"

    analysis = f"""# Edge gate distance/time model analysis

Generated at: {generated_at}

## Interpretation

- Linkage and adaptive diffusion distances are used here as topology models:
  they choose which leaves/subtrees merge and assign dendrogram branch lengths.
- The default edge-gate null is sampling covariance only. It records branch
  lengths for diagnostics but does not treat them as elapsed stochastic time.
- The `normalized_branch_length` policy is the explicit branch-time variance
  model: it multiplies child-parent contrast variance by
  `1 + branch_length / mean_branch_length`.
- Production-facing linkage branch-time rows recompute all edge lengths on the
  fixed topology with native NNLS before applying that policy. Raw linkage
  ultrametric heights are kept only in explicitly labeled diagnostic rows.
- The TBS rows use projected-Wald sibling gates with local adaptive projected
  dimensions. For each edge or sibling contrast, the test starts from the local
  PCA basis and keeps the shortest prefix explaining 90% of that contrast's
  projected energy; the MP/floor dimension remains diagnostic metadata.
- The branch-time sensitivity table below is supervised failure analysis. It
  reuses a fixed topology and sibling gate, rescales edge Wald statistics by
  candidate time multipliers, reruns Tree-BH and traversal, then scores the
  resulting labels. These rows should not be interpreted as an unsupervised
  production rule.

## Adaptive diffusion topology

```json
{adaptive_text}
```

## Branch length summaries

{tree_table}

## Branch-time sensitivity, top rows per topology

{length_sensitivity_table}
"""
    (output_dir / "edge_gate_distance_time_model_analysis.md").write_text(analysis)


def _run_benchmarks(
    adata: Any,
    output_dir: Path,
    *,
    max_cells: int,
    n_pcs: int,
    seed: int,
    generated_at: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    dict[str, object] | None,
    list[dict[str, object]],
]:
    subset_idx = _benchmark_subset(
        adata,
        max_cells=max_cells,
        min_cells_per_label=20,
        seed=seed,
    )
    subset = adata[subset_idx].copy()
    X = np.asarray(subset.obsm["X_pca"][:, :n_pcs], dtype=float)
    sample_ids = subset.obs_names.astype(str).to_numpy()
    X_df = pd.DataFrame(X, index=sample_ids, columns=[f"PC{i + 1}" for i in range(X.shape[1])])
    y_true = subset.obs["celltype"].astype(str).to_numpy()
    true_k = int(pd.Series(y_true).nunique())

    feature_space = continuous_feature_space_from_columns(X_df.columns)
    distance_condensed = pdist(X, metric="euclidean")
    distance_matrix = squareform(distance_condensed)
    adaptive_distance_condensed: np.ndarray | None = None
    adaptive_metadata: dict[str, object] | None = None
    adaptive_error: str | None = None
    try:
        geometry = adaptive_diffusion_geometry(
            X_df,
            k_neighbors=15,
            diffusion_time=3,
            n_components=n_pcs,
            metric="euclidean",
            bandwidth_type="-1/(d+2)",
            epsilon="median",
        )
        adaptive_distance_condensed = np.asarray(geometry.distance_condensed, dtype=float)
        adaptive_metadata = dict(geometry.metadata)
        adaptive_metadata.update(
            {
                "diffusion_time": 3,
                "n_components": int(n_pcs),
                "k_neighbors_requested": 15,
            }
        )
    except Exception as exc:
        adaptive_error = str(exc)

    subset_obs = subset.obs[["celltype", "batch", "sample"]].copy()
    subset_obs.insert(0, "cell_id", sample_ids)
    subset_obs.to_csv(output_dir / "benchmark_subset_cells.csv", index=False)
    pd.DataFrame(X, index=sample_ids, columns=X_df.columns).to_csv(
        output_dir / "benchmark_subset_pca.csv"
    )

    rows = []
    subset_umap = np.asarray(subset.obsm["X_umap"], dtype=float)
    assignments = pd.DataFrame(
        {
            "cell_id": sample_ids,
            "celltype": y_true,
            "umap1": subset_umap[:, 0],
            "umap2": subset_umap[:, 1],
        }
    )
    tree_summaries: list[dict[str, object]] = []
    length_sensitivity_rows: list[dict[str, object]] = []
    celltype_fragmentation_rows: list[dict[str, object]] = []
    cluster_composition_rows: list[dict[str, object]] = []
    for config in _method_configs(true_k):
        start = time.perf_counter()
        active_distance_condensed = distance_condensed
        if config.distance_source == "adaptive_diffusion":
            active_distance_condensed = adaptive_distance_condensed
        if config.needs_distance_condensed and active_distance_condensed is None:
            elapsed = time.perf_counter() - start
            assignments[config.assignment_key] = pd.NA
            scored_params = dict(config.params)
            if config.significance_level is not None:
                scored_params["_significance_level"] = config.significance_level
            if config.edge_alpha is not None:
                scored_params["_edge_alpha"] = config.edge_alpha
            rows.append(
                _score_labels(
                    method_label=config.label,
                    method_id=config.method_id,
                    params=scored_params,
                    status="skip",
                    skip_reason=adaptive_error or "distance source unavailable",
                    elapsed_sec=elapsed,
                    y_true=y_true,
                    y_pred=None,
                    X=X,
                )
            )
            continue
        result = run_clustering_result(
            X_df,
            config.method_id,
            config.params,
            seed=seed,
            significance_level=config.significance_level,
            edge_alpha=config.edge_alpha,
            distance_matrix=distance_matrix if config.needs_distance_matrix else None,
            distance_condensed=active_distance_condensed
            if config.needs_distance_condensed
            else None,
            feature_space=feature_space,
        )
        elapsed = time.perf_counter() - start
        labels = None if result.labels is None else np.asarray(result.labels, dtype=int)
        assignments[config.assignment_key] = labels if labels is not None else pd.NA
        scored_params = dict(config.params)
        if config.significance_level is not None:
            scored_params["_significance_level"] = config.significance_level
        if config.edge_alpha is not None:
            scored_params["_edge_alpha"] = config.edge_alpha
        scored_params["_distance_source"] = config.distance_source
        rows.append(
            _score_labels(
                method_label=config.label,
                method_id=config.method_id,
                params=scored_params,
                status=result.status,
                skip_reason=result.skip_reason,
                elapsed_sec=elapsed,
                y_true=y_true,
                y_pred=labels,
                X=X,
            )
        )
        if result.status == "ok" and config.method_id.startswith("tbs"):
            summary = _write_tbs_tree_diagnostics(
                config=config,
                result_extra=result.extra,
                sample_ids=sample_ids,
                y_true=y_true,
                output_dir=output_dir,
                length_sensitivity_rows=length_sensitivity_rows,
            )
            if summary is not None:
                tree_summaries.append(summary)
        if labels is not None:
            label_rows, cluster_rows = _split_merge_detail_tables(
                method_label=config.label,
                y_true=y_true,
                y_pred=labels,
            )
            celltype_fragmentation_rows.extend(label_rows)
            cluster_composition_rows.extend(cluster_rows)

    results = pd.DataFrame(rows).sort_values(
        [
            "v_measure",
            "weighted_label_dominant_cluster_recall",
            "weighted_cluster_purity",
        ],
        ascending=False,
    )
    results.to_csv(output_dir / "method_metrics.csv", index=False)
    assignments.to_csv(output_dir / "method_assignments.csv", index=False)
    pd.DataFrame(celltype_fragmentation_rows).to_csv(
        output_dir / "celltype_fragmentation_by_method.csv",
        index=False,
    )
    pd.DataFrame(cluster_composition_rows).to_csv(
        output_dir / "cluster_composition_by_method.csv",
        index=False,
    )
    length_sensitivity_df = pd.DataFrame(length_sensitivity_rows)
    if not length_sensitivity_df.empty:
        length_sensitivity_df = length_sensitivity_df.sort_values(
            [
                "method",
                "v_measure",
                "weighted_label_dominant_cluster_recall",
                "weighted_cluster_purity",
            ],
            ascending=[True, False, False, False],
        )
        length_sensitivity_rows = length_sensitivity_df.to_dict("records")
        length_sensitivity_df.to_csv(
            output_dir / "tbs_branch_time_sensitivity.csv",
            index=False,
        )
        _write_branch_time_sensitivity_plot(length_sensitivity_df, output_dir)
    _write_distance_time_model_analysis(
        output_dir=output_dir,
        tree_summaries=tree_summaries,
        adaptive_metadata=adaptive_metadata,
        length_sensitivity_rows=length_sensitivity_rows,
        generated_at=(
            generated_at
            if generated_at is not None
            else datetime.now().astimezone().isoformat(timespec="seconds")
        ),
    )
    return results, assignments, tree_summaries, adaptive_metadata, length_sensitivity_rows


def _write_plots(
    adata: Any,
    results: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    emb = np.asarray(adata.obsm["X_umap"])
    for ax, color_key, title in [
        (axes[0], "batch", "UMAP by batch"),
        (axes[1], "celltype", "UMAP by cell type"),
    ]:
        values = adata.obs[color_key].astype("category")
        codes = values.cat.codes.to_numpy()
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=codes, s=2, cmap="tab20", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        del scatter
    fig.tight_layout()
    fig.savefig(output_dir / "classical_umap_overview.png", dpi=200)
    plt.close(fig)

    metric_frame = results[results["status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(metric_frame["method"], metric_frame["ari"], color="#2563eb")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(0, max(1.0, float(metric_frame["ari"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "method_ari_barplot.png", dpi=200)
    plt.close(fig)

    diagnostic_frame = metric_frame.dropna(
        subset=[
            "weighted_cluster_purity",
            "weighted_label_dominant_cluster_recall",
            "n_clusters",
        ]
    ).copy()
    if not diagnostic_frame.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = 40.0 + 3.0 * diagnostic_frame["n_clusters"].astype(float)
        ax.scatter(
            diagnostic_frame["weighted_cluster_purity"],
            diagnostic_frame["weighted_label_dominant_cluster_recall"],
            s=sizes,
            c=diagnostic_frame["overcluster_ratio"],
            cmap="viridis",
            alpha=0.75,
            edgecolors="#111827",
            linewidths=0.6,
        )
        for _, row in diagnostic_frame.iterrows():
            ax.annotate(
                str(row["method"]),
                (
                    float(row["weighted_cluster_purity"]),
                    float(row["weighted_label_dominant_cluster_recall"]),
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Weighted cluster purity (merge control)")
        ax.set_ylabel("Weighted dominant-cluster recall (split control)")
        colorbar = fig.colorbar(ax.collections[0], ax=ax)
        colorbar.set_label("Overcluster ratio")
        fig.tight_layout()
        fig.savefig(output_dir / "method_split_merge_diagnostic.png", dpi=200)
        plt.close(fig)

    method_columns = [
        column
        for column in assignments.columns
        if column not in {"cell_id", "celltype", "umap1", "umap2"}
    ]
    panel_count = len(method_columns) + 1
    ncols = 3
    nrows = int(math.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows))
    flat_axes = np.asarray(axes).reshape(-1)

    true_codes = assignments["celltype"].astype("category").cat.codes.to_numpy()
    flat_axes[0].scatter(
        assignments["umap1"],
        assignments["umap2"],
        c=true_codes,
        s=5,
        cmap="tab20",
        linewidths=0,
    )
    flat_axes[0].set_title("Curated cell type")

    for ax, column in zip(flat_axes[1:], method_columns, strict=False):
        values = pd.to_numeric(assignments[column], errors="coerce")
        if values.notna().any():
            colors = values.fillna(-1).astype(int).to_numpy()
        else:
            colors = np.zeros(len(assignments), dtype=int)
        ax.scatter(
            assignments["umap1"],
            assignments["umap2"],
            c=colors,
            s=5,
            cmap="tab20",
            linewidths=0,
        )
        ax.set_title(column.replace("_", " "))

    for ax in flat_axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    for ax in flat_axes[panel_count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "method_umap_clusters.png", dpi=200)
    plt.close(fig)


def _write_report(
    *,
    output_dir: Path,
    input_h5ad: Path,
    adata: Any,
    qc_summary: dict[str, object],
    results: pd.DataFrame,
    tree_summaries: list[dict[str, object]],
    adaptive_metadata: dict[str, object] | None,
    length_sensitivity_rows: list[dict[str, object]],
    max_cells: int,
    n_pcs: int,
    generated_at: str,
) -> None:
    label_counts = adata.obs["celltype"].astype(str).value_counts()
    top_counts = "\n".join(f"- {label}: {count}" for label, count in label_counts.head(12).items())
    results_table = results[
        [
            "method",
            "status",
            "significance_level",
            "edge_alpha",
            "n_clusters",
            "overcluster_ratio",
            "weighted_cluster_purity",
            "weighted_label_dominant_cluster_recall",
            "merge_error_rate",
            "split_error_rate",
            "weighted_effective_clusters_per_label",
            "homogeneity",
            "completeness",
            "v_measure",
            "nmi",
            "ari",
            "silhouette",
            "elapsed_sec",
            "skip_reason",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    tree_summary_text = (
        pd.DataFrame(tree_summaries).to_markdown(index=False, floatfmt=".4f")
        if tree_summaries
        else "_No TBS tree summaries were produced._"
    )
    adaptive_summary = (
        json.dumps(adaptive_metadata, sort_keys=True, default=_json_default)
        if adaptive_metadata is not None
        else "adaptive diffusion distance unavailable"
    )
    length_sensitivity_df = pd.DataFrame(length_sensitivity_rows)
    if not length_sensitivity_df.empty:
        length_sensitivity_text = (
            length_sensitivity_df[
                [
                    "method",
                    "length_model",
                    "edge_open_count",
                    "n_clusters",
                    "weighted_cluster_purity",
                    "weighted_label_dominant_cluster_recall",
                    "merge_error_rate",
                    "split_error_rate",
                    "v_measure",
                    "ari",
                ]
            ]
            .groupby("method", as_index=False)
            .head(3)
            .to_markdown(
                index=False,
                floatfmt=".4f",
            )
        )
    else:
        length_sensitivity_text = "_No branch-time sensitivity rows were produced._"

    report = f"""# Pancreas scRNA clustering benchmark

Generated at: {generated_at}

## Data

- Source: Scanpy pancreas AnnData tutorial object downloaded from `{DATA_URL}`.
- Local raw object: `{input_h5ad}`.
- Shape: {adata.n_obs} cells x {adata.n_vars} genes.
- Batches: {qc_summary["batches"]}.
- Cell-type labels: {qc_summary["celltype_count"]}.

Top labels:

{top_counts}

## Classical pipeline

The run recomputed a conventional Scanpy latent-space workflow on the downloaded
AnnData object: scaling, PCA ({n_pcs} components), 15-neighbor graph, UMAP, and
Leiden clustering at resolution 1.0. The input object is a prepackaged post-count
AnnData object, so FASTQ-to-count generation was not performed.

QC notes:

- Raw snapshot present: {qc_summary["raw_layer_present"]}.
- Mitochondrial genes detected by `MT-` prefix: {qc_summary["mitochondrial_gene_count"]}.
- Doublet status: {qc_summary["scdblfinder_status"]}.
- Ambient RNA status: {qc_summary["ambient_rna_status"]}.

## Benchmark design

The benchmark uses the first {n_pcs} recomputed PCs and a deterministic
stratified subset capped at {max_cells} cells, excluding ambiguous labels:
{", ".join(sorted(AMBIGUOUS_LABELS))}. Metrics compare cluster assignments to
the curated `celltype` labels.

TBS is run at sibling alpha `0.01` and edge alpha `0.001`. Both production gates
use projected-Wald tests with a repaired local adaptive dimension rule. The
stored MP/floor dimension remains `2`, but each edge or sibling contrast can use
the shortest local PCA prefix explaining `90%` of that contrast's projected
energy.

The TBS variants separate topology from time:

- topology-only average linkage on standardized PCA distance;
- recomputed fixed-topology native NNLS branch lengths fitted to squared
  standardized continuous distances, used with normalized branch-time variance;
- explicitly labeled raw-linkage branch-time diagnostics, retained only as a
  negative-control sensitivity check;
- adaptive diffusion topology using a variable-bandwidth diffusion distance;
- adaptive diffusion with the same recomputed-NNLS versus raw-linkage
  diagnostic branch-time split.

Adaptive diffusion metadata: `{adaptive_summary}`.

## Results

{results_table}

## TBS tree and branch lengths

{tree_summary_text}

## Branch-time sensitivity

The length-transform search is a supervised diagnostic on fixed TBS topologies:
it rescales edge Wald statistics, reapplies Tree-BH, traverses the same
hierarchy, and scores split/merge behavior against curated cell types. It is not
used as a production clustering rule.

{length_sensitivity_text}

## Files

- `method_metrics.csv`: benchmark metrics.
- `method_assignments.csv`: per-cell benchmark labels.
- `celltype_fragmentation_by_method.csv`: per-cell-type split diagnostics.
- `cluster_composition_by_method.csv`: per-cluster merge/purity diagnostics.
- `tbs_branch_time_sensitivity.csv`: fixed-topology branch-time transform
  sensitivity scored with split/merge metrics.
- `tbs_branch_time_sensitivity.png`: branch-time sensitivity plot.
- `edge_gate_distance_time_model_analysis.md`: distance-vs-time model notes,
  adaptive diffusion metadata, and branch-length summary.
- `tbs_tree_branch_length_summary.csv`: per-TBS tree branch-length summaries.
- `*_tree_edges.csv`: per-edge TBS tree diagnostics with edge-gate p-values.
- `*_traversal_trace.csv`: live TBS traversal diagnostics with sibling-gate
  and final-boundary decisions.
- `*_full_edge_traversal_trace.csv`: edge-reachable traversal diagnostics,
  independent of sibling-gate stops.
- `*_tree_dendrogram.png`: truncated TBS dendrogram plots.
- `*_branch_lengths.png`: TBS branch-length histograms.
- `benchmark_subset_cells.csv`: subset metadata.
- `benchmark_subset_pca.csv`: PCA features used for benchmarking.
- `pancreas_classical_pipeline.h5ad`: processed AnnData checkpoint.
- `classical_umap_overview.png`: UMAP colored by batch and cell type.
- `method_umap_clusters.png`: subset UMAP colored by curated labels and method
  cluster assignments.
- `method_ari_barplot.png`: ARI comparison.
- `method_split_merge_diagnostic.png`: purity-vs-fragmentation diagnostic.
- `manifest.json`: parameters, package versions, and provenance.
"""
    (output_dir / "summary.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=2500)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark artifacts.",
    )
    args = parser.parse_args()

    root = _project_root()
    input_h5ad = root / "raw" / "inbox" / "pancreas.h5ad"
    output_dir = args.output_dir or (
        root / "raw" / "assets" / "benchmark-results" / "pancreas_scrna_cluster_benchmark_20260623"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.perf_counter()
    adata = _prepare_adata(input_h5ad, output_dir, args.n_pcs)
    qc_summary = _write_qc_outputs(adata, output_dir)
    (
        results,
        _assignments,
        tree_summaries,
        adaptive_metadata,
        length_sensitivity_rows,
    ) = _run_benchmarks(
        adata,
        output_dir,
        max_cells=args.max_cells,
        n_pcs=args.n_pcs,
        seed=args.seed,
        generated_at=generated_at,
    )
    _write_plots(adata, results, _assignments, output_dir)

    manifest = {
        "generated_at": generated_at,
        "data_url": DATA_URL,
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "max_cells": args.max_cells,
        "n_pcs": args.n_pcs,
        "seed": args.seed,
        "elapsed_sec": time.perf_counter() - start,
        "python": platform.python_version(),
        "scanpy": _scanpy_version(),
        "qc_summary": qc_summary,
        "adaptive_diffusion_metadata": adaptive_metadata,
        "tbs_tree_summaries": tree_summaries,
        "tbs_branch_time_sensitivity_rows": length_sensitivity_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
    )
    _write_report(
        output_dir=output_dir,
        input_h5ad=input_h5ad,
        adata=adata,
        qc_summary=qc_summary,
        results=results,
        tree_summaries=tree_summaries,
        adaptive_metadata=adaptive_metadata,
        length_sensitivity_rows=length_sensitivity_rows,
        max_cells=args.max_cells,
        n_pcs=args.n_pcs,
        generated_at=generated_at,
    )
    print(results.to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
