#!/usr/bin/env python3
"""Build cosine eigen-subspace trees and apply the KL tree split method."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For a blob feature matrix, decompose the gene-gene cosine operator into "
            "independent eigen-bands, build one hierarchical tree per band, and run "
            "the project KL tree split method on each tree."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Blob TSV matrix.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--baseline-assignments",
        type=Path,
        default=None,
        help="Optional assignment CSV to compare subspace splits against.",
    )
    parser.add_argument("--alpha-local", type=float, default=float(config.EDGE_ALPHA))
    parser.add_argument("--sibling-alpha", type=float, default=float(config.SIBLING_ALPHA))
    parser.add_argument(
        "--max-rank",
        type=int,
        default=80,
        help="Maximum positive eigenvectors to keep for band construction.",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
        help="Cosine operator inputs to test.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data = data.apply(pd.to_numeric, errors="raise").astype(float)
    data = data.loc[:, data.sum(axis=0) > 0]
    if data.empty:
        raise ValueError(f"No non-empty features in {path}")
    if (data.sum(axis=1) == 0).any():
        missing = data.index[data.sum(axis=1) == 0].tolist()[:10]
        raise ValueError(f"Rows with no active features cannot enter cosine operator: {missing}")
    return data


def weighted_matrix(data: pd.DataFrame, weighting: str) -> np.ndarray:
    values = data.to_numpy(dtype=float)
    if weighting == "binary":
        return values
    if weighting == "tfidf":
        return TfidfTransformer(norm=None, use_idf=True, smooth_idf=True).fit_transform(values).toarray()
    raise ValueError(f"Unknown weighting: {weighting}")


def cosine_eigendecomposition(values: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    row_normed = normalize(values, norm="l2", axis=1)
    operator = row_normed @ row_normed.T
    eigvals, eigvecs = np.linalg.eigh(operator)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-10
    eigvals = eigvals[keep][:max_rank]
    eigvecs = eigvecs[:, keep][:, :max_rank]
    return eigvals, eigvecs


def make_bands(rank: int) -> list[tuple[str, int, int]]:
    candidates = [
        ("common_mode_01", 1, 1),
        ("variation_02_05", 2, 5),
        ("variation_06_15", 6, 15),
        ("variation_16_35", 16, 35),
        ("variation_36_80", 36, 80),
        ("broad_variation_02_35", 2, 35),
        ("broad_variation_02_80", 2, 80),
        ("all_modes_01_80", 1, 80),
    ]
    bands: list[tuple[str, int, int]] = []
    for name, start, end in candidates:
        if start <= rank:
            bands.append((name, start, min(end, rank)))
    return bands


def coords_for_band(eigvals: np.ndarray, eigvecs: np.ndarray, start_1: int, end_1: int) -> np.ndarray:
    start = start_1 - 1
    end = end_1
    coords = eigvecs[:, start:end] * np.sqrt(np.maximum(eigvals[start:end], 0.0))
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def load_baseline(path: Path | None, index: pd.Index) -> pd.Series | None:
    if path is None or not path.exists():
        return None
    baseline = pd.read_csv(path)
    if "Unnamed: 0" in baseline.columns:
        baseline = baseline.rename(columns={"Unnamed: 0": "gene"})
    if "gene" not in baseline.columns:
        baseline = baseline.rename(columns={baseline.columns[0]: "gene"})
    cluster_col = "cluster_id" if "cluster_id" in baseline.columns else None
    if cluster_col is None:
        candidates = [c for c in baseline.columns if "cluster" in c.lower()]
        if not candidates:
            return None
        cluster_col = candidates[0]
    baseline["gene"] = baseline["gene"].astype(str)
    return baseline.set_index("gene").loc[index, cluster_col].astype(int)


def cluster_sizes(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "n_genes"})
        .sort_values(["n_genes", "cluster_id"], ascending=[False, True])
    )


def top_terms_for_assignments(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    max_terms_per_cluster: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        cluster_size = int(mask.sum())
        rest_size = int((~mask).sum())
        if cluster_size == 0 or rest_size == 0:
            continue
        cluster_prev = data.loc[mask].mean(axis=0)
        rest_prev = data.loc[~mask].mean(axis=0)
        delta = (cluster_prev - rest_prev).sort_values(ascending=False)
        for term, value in delta.head(max_terms_per_cluster).items():
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": cluster_size,
                    "term": term,
                    "cluster_prevalence": float(cluster_prev[term]),
                    "rest_prevalence": float(rest_prev[term]),
                    "prevalence_delta": float(value),
                }
            )
    return pd.DataFrame(rows)


def plot_band_assignments(
    coords: np.ndarray,
    assignments: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    labels = assignments["cluster_id"].astype(str).to_numpy()
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else np.zeros(coords.shape[0])
    plot_df = pd.DataFrame(
        {
            "Cosine subspace direction 1": x,
            "Cosine subspace direction 2": y,
            "cluster": labels,
        }
    )
    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="Cosine subspace direction 1",
        y="Cosine subspace direction 2",
        hue="cluster",
        palette="tab20",
        s=38,
        alpha=0.85,
        linewidth=0,
    )
    plt.title(title)
    plt.legend(title="Our split", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "assignments").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "top_terms").mkdir(exist_ok=True)

    data = load_matrix(args.input)
    baseline = load_baseline(args.baseline_assignments, data.index)

    summary_rows: list[dict[str, object]] = []
    size_frames: list[pd.DataFrame] = []
    all_assignment_frames: list[pd.DataFrame] = []
    spectrum_frames: list[pd.DataFrame] = []

    for weighting in args.weightings:
        values = weighted_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
        total = float(np.sum(eigvals))
        rank = len(eigvals)
        spectrum = pd.DataFrame(
            {
                "weighting": weighting,
                "component": np.arange(1, rank + 1),
                "eigenvalue": eigvals,
                "fraction_of_kept_operator_energy": eigvals / total if total > 0 else np.nan,
                "cumulative_fraction": np.cumsum(eigvals) / total if total > 0 else np.nan,
            }
        )
        spectrum_frames.append(spectrum)

        for band_name, band_start, band_end in make_bands(rank):
            run_id = f"{weighting}__{band_name}"
            coords = coords_for_band(eigvals, eigvecs, band_start, band_end)
            if coords.shape[1] == 0 or not np.isfinite(coords).all():
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "band": band_name,
                        "band_start": band_start,
                        "band_end": band_end,
                        "status": "failed_invalid_coordinates",
                    }
                )
                continue

            distances = pdist(coords, metric="euclidean")
            if not np.isfinite(distances).all() or np.allclose(distances, 0):
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "band": band_name,
                        "band_start": band_start,
                        "band_end": band_end,
                        "status": "failed_degenerate_distances",
                    }
                )
                continue

            try:
                linkage_matrix = linkage(distances, method="average")
                tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data.index.tolist())
                decomposition = tree.decompose(
                    leaf_data=data,
                    alpha_local=args.alpha_local,
                    sibling_alpha=args.sibling_alpha,
                )
                assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
                assignments.index.name = "gene"
                assignments.to_csv(out / "assignments" / f"{run_id}_cluster_assignments.csv")

                sizes = cluster_sizes(assignments)
                sizes.insert(0, "run_id", run_id)
                sizes.insert(1, "weighting", weighting)
                sizes.insert(2, "band", band_name)
                size_frames.append(sizes)

                labels = assignments["cluster_id"].astype(int)
                assignment_long = assignments[["cluster_id", "cluster_root", "cluster_size"]].copy()
                assignment_long.insert(0, "run_id", run_id)
                assignment_long.insert(1, "weighting", weighting)
                assignment_long.insert(2, "band", band_name)
                all_assignment_frames.append(assignment_long.reset_index())

                top_terms = top_terms_for_assignments(data, labels)
                if not top_terms.empty:
                    top_terms.insert(0, "run_id", run_id)
                    top_terms.insert(1, "weighting", weighting)
                    top_terms.insert(2, "band", band_name)
                    top_terms.to_csv(out / "top_terms" / f"{run_id}_top_terms.csv", index=False)

                n_clusters = int(decomposition.get("num_clusters", labels.nunique()))
                largest = int(sizes["n_genes"].max())
                singleton_clusters = int((sizes["n_genes"] == 1).sum())
                try:
                    sil = (
                        float(silhouette_score(coords, labels.to_numpy()))
                        if n_clusters > 1 and n_clusters < len(labels)
                        else math.nan
                    )
                except Exception:
                    sil = math.nan
                ari = math.nan
                nmi = math.nan
                if baseline is not None:
                    ari = float(adjusted_rand_score(baseline.to_numpy(), labels.to_numpy()))
                    nmi = float(normalized_mutual_info_score(baseline.to_numpy(), labels.to_numpy()))

                plot_band_assignments(
                    coords,
                    assignments,
                    f"{weighting} cosine tree: {band_name}",
                    out / "plots" / f"{run_id}_subspace_tree_split.png",
                )

                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "band": band_name,
                        "band_start": band_start,
                        "band_end": band_end,
                        "status": "ok",
                        "n_genes": int(data.shape[0]),
                        "n_features_tested": int(data.shape[1]),
                        "subspace_dimensions": int(coords.shape[1]),
                        "band_energy_fraction": float(np.sum(eigvals[band_start - 1 : band_end]) / total),
                        "n_clusters": n_clusters,
                        "largest_cluster": largest,
                        "largest_cluster_fraction": float(largest / len(labels)),
                        "singleton_clusters": singleton_clusters,
                        "silhouette_in_subspace": sil,
                        "ari_vs_baseline": ari,
                        "nmi_vs_baseline": nmi,
                    }
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "band": band_name,
                        "band_start": band_start,
                        "band_end": band_end,
                        "status": "failed_exception",
                        "error": repr(exc),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "cosine_subspace_tree_sweep_summary.csv", index=False)

    if spectrum_frames:
        spectrum_df = pd.concat(spectrum_frames, ignore_index=True)
        spectrum_df.to_csv(out / "cosine_operator_eigen_spectrum.csv", index=False)
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=spectrum_df[spectrum_df["component"] <= 80],
            x="component",
            y="fraction_of_kept_operator_energy",
            hue="weighting",
            marker="o",
        )
        plt.xlabel("Cosine operator component")
        plt.ylabel("Fraction of kept operator energy")
        plt.title("Cosine operator eigen-spectrum")
        plt.tight_layout()
        plt.savefig(out / "plots" / "cosine_operator_eigen_spectrum.png", dpi=180)
        plt.close()

    if size_frames:
        sizes_df = pd.concat(size_frames, ignore_index=True)
        sizes_df.to_csv(out / "cosine_subspace_cluster_sizes.csv", index=False)

    if all_assignment_frames:
        assignments_df = pd.concat(all_assignment_frames, ignore_index=True)
        assignments_df.to_csv(out / "cosine_subspace_cluster_assignments_long.csv", index=False)

    ok = summary[summary["status"].eq("ok")].copy()
    if not ok.empty:
        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=ok, x="run_id", y="largest_cluster_fraction", hue="weighting", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Cosine subspace tree")
        plt.ylabel("Largest cluster fraction after our split")
        plt.title("How our method reacts to each cosine subspace tree")
        plt.tight_layout()
        plt.savefig(out / "plots" / "largest_core_reaction_by_subspace.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        sns.scatterplot(
            data=ok,
            x="n_clusters",
            y="largest_cluster_fraction",
            hue="weighting",
            style="band",
            s=90,
        )
        plt.xlabel("Clusters found by our method")
        plt.ylabel("Largest remaining core fraction")
        plt.title("Subspace tree split reaction map")
        plt.tight_layout()
        plt.savefig(out / "plots" / "subspace_tree_reaction_map.png", dpi=180)
        plt.close()

    manifest_rows = []
    for file_path in sorted(out.rglob("*")):
        if file_path.is_file():
            manifest_rows.append(
                f"{file_path.relative_to(out).as_posix()}\t{file_path.stat().st_size} bytes"
            )
    (out / "MANIFEST.txt").write_text("\n".join(manifest_rows) + "\n")

    if ok.empty:
        best_core = pd.DataFrame()
        stable = pd.DataFrame()
    else:
        best_core = ok.sort_values(
            ["largest_cluster_fraction", "n_clusters"], ascending=[True, False]
        ).head(8)
        stable = ok.sort_values(["silhouette_in_subspace"], ascending=False, na_position="last").head(8)
    readme = [
        "# Cosine Eigen-Subspace Tree Sweep",
        "",
        f"Input: `{args.input}`",
        f"Genes: `{data.shape[0]}`",
        f"Deduplicated non-empty GO features used for split tests: `{data.shape[1]}`",
        "",
        "Method: build gene-gene cosine operators, decompose each operator into eigen-bands,",
        "construct one hierarchical tree per eigen-band, then apply the existing project",
        "`PosetTree.decompose(...)` split method on each subspace tree.",
        "",
        "The tree topology comes from the cosine subspace. The split tests use the original",
        "deduplicated GO features for the blob, so this is a test of whether each independent",
        "subspace topology is supported by the GO annotation data.",
        "",
        "## Best Reduction Of The Largest Core",
        "",
        best_core.to_string(index=False) if not best_core.empty else "No successful runs.",
        "",
        "## Highest Subspace Silhouette",
        "",
        stable.to_string(index=False) if not stable.empty else "No successful runs.",
        "",
        f"Runtime seconds: `{time.time() - start_time:.2f}`",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme))


if __name__ == "__main__":
    main()
