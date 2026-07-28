#!/usr/bin/env python3
"""Generate clustering diagnostics for one or more assignment files.

The panel is intentionally method-agnostic: it accepts assignment CSVs and
optional UMAP coordinates, reference labels, and a feature matrix.  It reports
fragmentation, reference recovery, spatial compactness, and feature coherence
without treating any one metric as sufficient evidence of clustering quality.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)


@dataclass(frozen=True)
class AssignmentSpec:
    method: str
    path: Path


def parse_assignment_spec(value: str) -> AssignmentSpec:
    if "=" not in value:
        msg = "Assignment inputs must use METHOD=PATH syntax."
        raise argparse.ArgumentTypeError(msg)
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        msg = "Assignment method name cannot be empty."
        raise argparse.ArgumentTypeError(msg)
    return AssignmentSpec(method=method, path=Path(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clustering diagnostics from assignment CSVs."
    )
    parser.add_argument(
        "--assignments",
        type=parse_assignment_spec,
        action="append",
        required=True,
        help="Assignment file as METHOD=PATH. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for diagnostic CSVs, manifest, and plot panel.",
    )
    parser.add_argument(
        "--umap-coordinates",
        type=Path,
        default=None,
        help="Optional CSV with gene_symbol, UMAP-1, and UMAP-2 columns.",
    )
    parser.add_argument(
        "--reference-labels",
        type=Path,
        default=None,
        help="Optional reference-label CSV for ARI/NMI and fragmentation diagnostics.",
    )
    parser.add_argument(
        "--reference-gene-column",
        default=None,
        help="Gene identifier column in --reference-labels. Inferred when omitted.",
    )
    parser.add_argument(
        "--reference-label-column",
        default=None,
        help="Reference cluster label column. Inferred when omitted.",
    )
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=None,
        help="Optional TSV feature matrix with genes as rows for feature-coherence diagnostics.",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=15,
        help="Number of largest non-singleton clusters to retain in top-cluster outputs.",
    )
    return parser.parse_args()


def load_assignments(spec: AssignmentSpec) -> pd.DataFrame:
    data = pd.read_csv(spec.path)
    if "gene_symbol" not in data.columns:
        first = data.columns[0]
        if first.startswith("Unnamed") or first in {"gene", "sample_id", "index"}:
            data = data.rename(columns={first: "gene_symbol"})
        else:
            data = data.reset_index().rename(columns={"index": "gene_symbol"})
    if "cluster_id" not in data.columns:
        msg = f"{spec.path} does not contain a cluster_id column."
        raise ValueError(msg)
    data = data[["gene_symbol", "cluster_id"]].copy()
    data["gene_symbol"] = data["gene_symbol"].astype(str)
    data["method"] = spec.method
    sizes = data["cluster_id"].value_counts()
    data["cluster_size"] = data["cluster_id"].map(sizes).astype(int)
    return data


def cluster_size_bin(size: int) -> str:
    if size <= 1:
        return "singleton"
    if size == 2:
        return "n=2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    if size <= 20:
        return "11-20"
    return ">20"


def effective_cluster_count(sizes: pd.Series) -> float:
    probs = sizes.to_numpy(dtype=float) / float(sizes.sum())
    probs = probs[probs > 0.0]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def gini(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    if array.size == 0 or np.isclose(array.sum(), 0.0):
        return float("nan")
    index = np.arange(1, array.size + 1)
    return float(((2.0 * index - array.size - 1.0) * array).sum() / (array.size * array.sum()))


def size_diagnostics(assignments: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bin_rows = []
    for method, data in assignments.items():
        sizes = data.groupby("cluster_id").size().sort_values(ascending=False)
        n = int(sizes.sum())
        bins = sizes.map(cluster_size_bin).value_counts()
        genes_by_bin = (
            sizes.to_frame("cluster_size")
            .assign(size_bin=lambda frame: frame["cluster_size"].map(cluster_size_bin))
            .groupby("size_bin")["cluster_size"]
            .sum()
        )
        for bin_name in ["singleton", "n=2", "3-5", "6-10", "11-20", ">20"]:
            bin_rows.append(
                {
                    "method": method,
                    "size_bin": bin_name,
                    "clusters": int(bins.get(bin_name, 0)),
                    "genes": int(genes_by_bin.get(bin_name, 0)),
                    "gene_fraction": float(genes_by_bin.get(bin_name, 0) / n),
                }
            )
        summary_rows.append(
            {
                "method": method,
                "n_genes": n,
                "n_clusters": int(sizes.size),
                "singletons": int((sizes == 1).sum()),
                "singleton_gene_fraction": float((sizes == 1).sum() / n),
                "non_singleton_clusters": int((sizes > 1).sum()),
                "genes_in_non_singletons": int(sizes[sizes > 1].sum()),
                "median_cluster_size": float(sizes.median()),
                "mean_cluster_size": float(sizes.mean()),
                "p90_cluster_size": float(sizes.quantile(0.90)),
                "max_cluster_size": int(sizes.max()),
                "clusters_ge5": int((sizes >= 5).sum()),
                "genes_in_clusters_ge5": int(sizes[sizes >= 5].sum()),
                "clusters_ge10": int((sizes >= 10).sum()),
                "genes_in_clusters_ge10": int(sizes[sizes >= 10].sum()),
                "effective_cluster_count": effective_cluster_count(sizes),
                "cluster_size_gini": gini(sizes),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def infer_reference_columns(
    reference: pd.DataFrame,
    gene_column: str | None,
    label_column: str | None,
) -> tuple[str, str]:
    if gene_column is None:
        for candidate in ("gene_symbol", "gene", "sample_id", "id"):
            if candidate in reference.columns:
                gene_column = candidate
                break
    if label_column is None:
        for candidate in (
            "reference_cluster_id",
            "reference_label",
            "cluster_id",
            "label",
            "reference_cluster_name",
        ):
            if candidate in reference.columns:
                label_column = candidate
                break
    if gene_column is None or label_column is None:
        msg = "Could not infer reference gene/label columns."
        raise ValueError(msg)
    return gene_column, label_column


def normalize_label(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_reference(path: Path, gene_column: str | None, label_column: str | None) -> pd.DataFrame:
    reference = pd.read_csv(path)
    gene_column, label_column = infer_reference_columns(reference, gene_column, label_column)
    reference = reference[[gene_column, label_column]].rename(
        columns={gene_column: "gene_symbol", label_column: "reference_label"}
    )
    reference = reference.dropna(subset=["gene_symbol", "reference_label"]).copy()
    reference["gene_symbol"] = reference["gene_symbol"].astype(str)
    reference["reference_label"] = reference["reference_label"].map(normalize_label)
    return reference


def reference_diagnostics(
    assignments: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    fragmentation_rows = []
    purity_rows = []
    for method, data in assignments.items():
        merged = data.merge(reference, on="gene_symbol", how="inner")
        y_true = merged["reference_label"].astype(str)
        y_pred = merged["cluster_id"].astype(str)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
        summary_rows.append(
            {
                "method": method,
                "matched_reference_genes": int(len(merged)),
                "reference_labels": int(merged["reference_label"].nunique()),
                "method_clusters_on_ref_genes": int(merged["cluster_id"].nunique()),
                "adjusted_rand_index": adjusted_rand_score(y_true, y_pred),
                "normalized_mutual_info": normalized_mutual_info_score(y_true, y_pred),
                "homogeneity": homogeneity,
                "completeness": completeness,
                "v_measure": v_measure,
            }
        )
        for reference_label, group in merged.groupby("reference_label"):
            counts = group["cluster_id"].value_counts()
            fragmentation_rows.append(
                {
                    "method": method,
                    "reference_label": reference_label,
                    "reference_label_size": int(len(group)),
                    "method_clusters": int(counts.size),
                    "largest_method_cluster_id": counts.index[0],
                    "largest_method_cluster_size": int(counts.iloc[0]),
                    "largest_method_cluster_fraction": float(counts.iloc[0] / len(group)),
                }
            )
        for cluster_id, group in merged.groupby("cluster_id"):
            counts = group["reference_label"].value_counts()
            purity_rows.append(
                {
                    "method": method,
                    "cluster_id": cluster_id,
                    "matched_cluster_size": int(len(group)),
                    "top_reference_label": counts.index[0],
                    "top_reference_count": int(counts.iloc[0]),
                    "top_reference_fraction": float(counts.iloc[0] / len(group)),
                    "reference_label_count": int(counts.size),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fragmentation_rows),
        pd.DataFrame(purity_rows),
    )


def umap_diagnostics(
    assignments: dict[str, pd.DataFrame],
    umap_coordinates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinates = umap_coordinates[["gene_symbol", "UMAP-1", "UMAP-2"]].copy()
    coordinates["gene_symbol"] = coordinates["gene_symbol"].astype(str)
    summary_rows = []
    cluster_rows = []
    for method, data in assignments.items():
        merged = data.merge(coordinates, on="gene_symbol", how="inner")
        encoded = pd.factorize(merged["cluster_id"])[0]
        silhouette = np.nan
        if 1 < np.unique(encoded).size < len(encoded):
            try:
                silhouette = float(
                    silhouette_score(merged[["UMAP-1", "UMAP-2"]].to_numpy(), encoded)
                )
            except ValueError:
                silhouette = np.nan
        for cluster_id, group in merged.groupby("cluster_id"):
            if len(group) < 2:
                continue
            xy = group[["UMAP-1", "UMAP-2"]].to_numpy()
            centroid = xy.mean(axis=0)
            distances = np.linalg.norm(xy - centroid, axis=1)
            cluster_rows.append(
                {
                    "method": method,
                    "cluster_id": cluster_id,
                    "cluster_size": int(len(group)),
                    "umap_centroid_1": float(centroid[0]),
                    "umap_centroid_2": float(centroid[1]),
                    "mean_umap_radius": float(distances.mean()),
                    "max_umap_radius": float(distances.max()),
                }
            )
        clusters = pd.DataFrame([row for row in cluster_rows if row["method"] == method])
        if clusters.empty:
            summary_rows.append({"method": method, "umap_silhouette_all_labels": silhouette})
            continue
        summary_rows.append(
            {
                "method": method,
                "non_singleton_clusters_on_umap": int(len(clusters)),
                "weighted_mean_umap_radius": float(
                    np.average(clusters["mean_umap_radius"], weights=clusters["cluster_size"])
                ),
                "median_cluster_mean_umap_radius": float(clusters["mean_umap_radius"].median()),
                "p90_cluster_mean_umap_radius": float(clusters["mean_umap_radius"].quantile(0.90)),
                "max_cluster_mean_umap_radius": float(clusters["mean_umap_radius"].max()),
                "umap_silhouette_all_labels": silhouette,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(cluster_rows)


def feature_coherence_diagnostics(
    assignments: dict[str, pd.DataFrame],
    feature_matrix_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(feature_matrix_path, sep="\t", index_col=0)
    features.index = features.index.astype(str)
    cluster_rows = []
    for method, data in assignments.items():
        for cluster_id, group in data.groupby("cluster_id"):
            genes = [gene for gene in group["gene_symbol"].astype(str) if gene in features.index]
            if len(genes) < 2:
                continue
            sub = features.loc[genes]
            values = sub.to_numpy(dtype=float)
            if values.shape[0] > 1:
                mean_pairwise_hamming_similarity = float(
                    1.0 - pdist(values, metric="hamming").mean()
                )
                mean_pairwise_jaccard_similarity = float(
                    1.0 - pdist(values, metric="jaccard").mean()
                )
            else:
                mean_pairwise_hamming_similarity = np.nan
                mean_pairwise_jaccard_similarity = np.nan
            feature_prevalence = sub.mean(axis=0)
            cluster_rows.append(
                {
                    "method": method,
                    "cluster_id": cluster_id,
                    "cluster_size": int(len(genes)),
                    "mean_active_features_per_gene": float(sub.sum(axis=1).mean()),
                    "active_feature_union": int((feature_prevalence > 0.0).sum()),
                    "features_shared_by_all": int((feature_prevalence >= 1.0).sum()),
                    "features_shared_by_half": int((feature_prevalence >= 0.5).sum()),
                    "mean_feature_prevalence_over_union": float(
                        feature_prevalence[feature_prevalence > 0.0].mean()
                    ),
                    "mean_pairwise_hamming_similarity": mean_pairwise_hamming_similarity,
                    "mean_pairwise_jaccard_similarity": mean_pairwise_jaccard_similarity,
                }
            )
    clusters = pd.DataFrame(cluster_rows)
    if clusters.empty:
        return pd.DataFrame(), clusters
    summary = (
        clusters.groupby("method")
        .agg(
            non_singleton_clusters=("cluster_id", "count"),
            mean_pairwise_hamming_similarity=(
                "mean_pairwise_hamming_similarity",
                lambda values: float(np.nanmean(values)),
            ),
            median_pairwise_hamming_similarity=("mean_pairwise_hamming_similarity", "median"),
            mean_pairwise_jaccard_similarity=("mean_pairwise_jaccard_similarity", "mean"),
            median_pairwise_jaccard_similarity=("mean_pairwise_jaccard_similarity", "median"),
            median_features_shared_by_all=("features_shared_by_all", "median"),
            median_features_shared_by_half=("features_shared_by_half", "median"),
            median_active_feature_union=("active_feature_union", "median"),
        )
        .reset_index()
    )
    return summary, clusters


def top_cluster_table(assignments: dict[str, pd.DataFrame], limit: int) -> pd.DataFrame:
    rows = []
    for method, data in assignments.items():
        sizes = data.groupby("cluster_id").size().sort_values(ascending=False)
        for rank, (cluster_id, size) in enumerate(sizes[sizes > 1].head(limit).items(), start=1):
            genes = data.loc[data["cluster_id"] == cluster_id, "gene_symbol"].tolist()
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "cluster_id": cluster_id,
                    "cluster_size": int(size),
                    "genes": ";".join(genes),
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def plot_review_panel(
    output_path: Path,
    size_summary: pd.DataFrame,
    size_bins: pd.DataFrame,
    reference_summary: pd.DataFrame | None,
    umap_summary: pd.DataFrame | None,
    feature_summary: pd.DataFrame | None,
) -> None:
    methods = size_summary["method"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Clustering diagnostic review panel", fontsize=16)

    ax = axes[0, 0]
    ax.bar(methods, size_summary["singleton_gene_fraction"], color="#9ecae1", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of genes")
    ax.set_title("Singleton fragmentation")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    pivot = size_bins.pivot(index="method", columns="size_bin", values="gene_fraction").reindex(
        methods
    )
    bottom = np.zeros(len(methods))
    colors = {
        "singleton": "#d9d9d9",
        "n=2": "#9ecae1",
        "3-5": "#3182bd",
        "6-10": "#fdae6b",
        "11-20": "#de2d26",
        ">20": "#54278f",
    }
    for bin_name in ["singleton", "n=2", "3-5", "6-10", "11-20", ">20"]:
        values = pivot.get(bin_name, pd.Series(0.0, index=methods)).fillna(0.0).to_numpy()
        ax.bar(
            methods, values, bottom=bottom, label=bin_name, color=colors[bin_name], edgecolor="none"
        )
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_title("Genes by cluster-size bin")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    ax.bar(methods, size_summary["max_cluster_size"], color="#74c476", edgecolor="black")
    ax.set_title("Largest cluster")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    if reference_summary is not None and not reference_summary.empty:
        width = 0.28
        x = np.arange(len(methods))
        ref = reference_summary.set_index("method").reindex(methods)
        ax.bar(x - width, ref["adjusted_rand_index"], width, label="ARI", color="#756bb1")
        ax.bar(x, ref["normalized_mutual_info"], width, label="NMI", color="#31a354")
        ax.bar(x + width, ref["completeness"], width, label="completeness", color="#fd8d3c")
        ax.set_xticks(x, methods, rotation=25)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reference recovery")

    ax = axes[1, 1]
    if umap_summary is not None and not umap_summary.empty:
        umap = umap_summary.set_index("method").reindex(methods)
        ax.bar(methods, umap["weighted_mean_umap_radius"], color="#6baed6", edgecolor="black")
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Weighted mean UMAP radius")

    ax = axes[1, 2]
    if feature_summary is not None and not feature_summary.empty:
        feature = feature_summary.set_index("method").reindex(methods)
        ax.bar(
            methods,
            feature["median_pairwise_jaccard_similarity"],
            color="#bdbdbd",
            edgecolor="black",
        )
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Median active-feature Jaccard")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assignments = {spec.method: load_assignments(spec) for spec in args.assignments}
    size_summary, size_bins = size_diagnostics(assignments)
    top_clusters = top_cluster_table(assignments, args.top_clusters)

    write_csv(size_summary, args.output_dir / "cluster_size_summary.csv")
    write_csv(size_bins, args.output_dir / "cluster_size_bins.csv")
    write_csv(top_clusters, args.output_dir / "top_non_singleton_clusters.csv")

    reference_summary = None
    reference_fragmentation = None
    method_cluster_reference_purity = None
    if args.reference_labels is not None:
        reference = load_reference(
            args.reference_labels,
            args.reference_gene_column,
            args.reference_label_column,
        )
        (
            reference_summary,
            reference_fragmentation,
            method_cluster_reference_purity,
        ) = reference_diagnostics(assignments, reference)
        write_csv(reference_summary, args.output_dir / "reference_recovery_summary.csv")
        write_csv(reference_fragmentation, args.output_dir / "reference_fragmentation_by_label.csv")
        write_csv(
            method_cluster_reference_purity,
            args.output_dir / "method_cluster_reference_purity.csv",
        )

    umap_summary = None
    umap_clusters = None
    if args.umap_coordinates is not None:
        umap_coordinates = pd.read_csv(args.umap_coordinates)
        umap_summary, umap_clusters = umap_diagnostics(assignments, umap_coordinates)
        write_csv(umap_summary, args.output_dir / "umap_compactness_summary.csv")
        write_csv(umap_clusters, args.output_dir / "umap_cluster_compactness.csv")

    feature_summary = None
    feature_clusters = None
    if args.feature_matrix is not None:
        feature_summary, feature_clusters = feature_coherence_diagnostics(
            assignments,
            args.feature_matrix,
        )
        write_csv(feature_summary, args.output_dir / "feature_coherence_summary.csv")
        write_csv(feature_clusters, args.output_dir / "feature_coherence_by_cluster.csv")

    plot_review_panel(
        args.output_dir / "clustering_diagnostic_review_panel.png",
        size_summary=size_summary,
        size_bins=size_bins,
        reference_summary=reference_summary,
        umap_summary=umap_summary,
        feature_summary=feature_summary,
    )

    manifest = {
        "assignments": {spec.method: str(spec.path) for spec in args.assignments},
        "feature_matrix": str(args.feature_matrix) if args.feature_matrix else None,
        "reference_labels": str(args.reference_labels) if args.reference_labels else None,
        "umap_coordinates": str(args.umap_coordinates) if args.umap_coordinates else None,
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote clustering diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
