#!/usr/bin/env python3
"""Adaptive cosine spectral-block trees from the eigenspectrum."""

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
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cosine KAK/spectral blocks adaptively from the log eigenspectrum, "
            "then apply the project tree split method to each block."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="TSV feature matrix.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--baseline-assignments",
        type=Path,
        default=None,
        help="Optional assignment CSV for ARI/NMI comparison.",
    )
    parser.add_argument("--alpha-local", type=float, default=float(config.EDGE_ALPHA))
    parser.add_argument("--sibling-alpha", type=float, default=float(config.SIBLING_ALPHA))
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=10)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
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
    return eigvals[keep][:max_rank], eigvecs[:, keep][:, :max_rank]


def interval_linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    return float(np.sum(residual * residual))


def should_isolate_common_mode(log_eigvals: np.ndarray) -> bool:
    if len(log_eigvals) < 8:
        return True
    gaps = log_eigvals[:-1] - log_eigvals[1:]
    tail = gaps[5:] if len(gaps) > 5 else gaps
    median = float(np.median(tail))
    mad = float(np.median(np.abs(tail - median)) * 1.4826)
    return bool(gaps[0] > median + 3.0 * max(mad, 1e-12))


def adaptive_spectral_blocks(
    eigvals: np.ndarray,
    *,
    min_segment_length: int,
    max_segments: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Segment log eigenvalue decay by BIC-selected piecewise-linear blocks."""
    rank = len(eigvals)
    if rank == 0:
        return [], {"selected_segments": 0, "bic": math.nan, "segmentation_sse": math.nan}
    if rank <= min_segment_length:
        return [
            {
                "block_id": 0,
                "block_name": f"adaptive_modes_01_{rank:02d}",
                "block_start": 1,
                "block_end": rank,
                "block_type": "all_available",
            }
        ], {"selected_segments": 1, "bic": math.nan, "segmentation_sse": 0.0}

    log_eigvals = np.log(np.maximum(eigvals, 1e-300))
    offset = 0
    blocks: list[dict[str, object]] = []
    if should_isolate_common_mode(log_eigvals):
        blocks.append(
            {
                "block_id": 0,
                "block_name": "adaptive_common_mode_01",
                "block_start": 1,
                "block_end": 1,
                "block_type": "common_mode",
            }
        )
        offset = 1

    y = log_eigvals[offset:]
    n = len(y)
    x = np.arange(offset + 1, rank + 1, dtype=float)
    if n < min_segment_length:
        if n:
            blocks.append(
                {
                    "block_id": len(blocks),
                    "block_name": f"adaptive_modes_{offset + 1:02d}_{rank:02d}",
                    "block_start": offset + 1,
                    "block_end": rank,
                    "block_type": "tail",
                }
            )
        return blocks, {"selected_segments": len(blocks), "bic": math.nan, "segmentation_sse": 0.0}

    max_k = min(max_segments, max(1, n // min_segment_length))
    sse = np.full((n, n), np.inf)
    for start in range(n):
        for end in range(start + min_segment_length - 1, n):
            sse[start, end] = interval_linear_sse(x[start : end + 1], y[start : end + 1])

    dp = np.full((max_k + 1, n), np.inf)
    prev = np.full((max_k + 1, n), -1, dtype=int)
    for end in range(min_segment_length - 1, n):
        dp[1, end] = sse[0, end]
    for k in range(2, max_k + 1):
        first_valid_end = k * min_segment_length - 1
        for end in range(first_valid_end, n):
            for cut in range((k - 1) * min_segment_length - 1, end - min_segment_length + 1):
                value = dp[k - 1, cut] + sse[cut + 1, end]
                if value < dp[k, end]:
                    dp[k, end] = value
                    prev[k, end] = cut

    selected: tuple[float, int, float] | None = None
    for k in range(1, max_k + 1):
        total_sse = float(dp[k, n - 1])
        if not np.isfinite(total_sse):
            continue
        total_sse = max(total_sse, 1e-12)
        parameter_count = 3 * k
        bic = n * math.log(total_sse / n) + parameter_count * math.log(n)
        if selected is None or bic < selected[0]:
            selected = (bic, k, total_sse)
    if selected is None:
        raise RuntimeError("Could not select adaptive spectral segmentation.")

    _, k, total_sse = selected
    segments: list[tuple[int, int]] = []
    end = n - 1
    while k >= 1:
        cut = prev[k, end]
        start = 0 if k == 1 else cut + 1
        segments.append((start + offset + 1, end + offset + 1))
        end = cut
        k -= 1

    for start, end in reversed(segments):
        blocks.append(
            {
                "block_id": len(blocks),
                "block_name": f"adaptive_modes_{start:02d}_{end:02d}",
                "block_start": int(start),
                "block_end": int(end),
                "block_type": "adaptive_decay_regime",
            }
        )

    return blocks, {
        "selected_segments": len(segments),
        "bic": float(selected[0]),
        "segmentation_sse": float(total_sse),
        "common_mode_isolated": bool(offset == 1),
        "min_segment_length": int(min_segment_length),
        "max_segments": int(max_segments),
    }


def coords_for_block(eigvals: np.ndarray, eigvecs: np.ndarray, start_1: int, end_1: int) -> np.ndarray:
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


def top_terms_for_assignments(data: pd.DataFrame, labels: pd.Series, max_terms_per_cluster: int = 12) -> pd.DataFrame:
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


def plot_block_assignments(coords: np.ndarray, assignments: pd.DataFrame, title: str, path: Path) -> None:
    labels = assignments["cluster_id"].astype(str).to_numpy()
    plot_df = pd.DataFrame(
        {
            "Cosine spectral-block direction 1": coords[:, 0],
            "Cosine spectral-block direction 2": coords[:, 1] if coords.shape[1] > 1 else np.zeros(coords.shape[0]),
            "cluster": labels,
        }
    )
    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="Cosine spectral-block direction 1",
        y="Cosine spectral-block direction 2",
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


def plot_spectrum_with_blocks(spectrum: pd.DataFrame, blocks: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5.8))
    ax = sns.lineplot(
        data=spectrum,
        x="component",
        y="log_eigenvalue",
        hue="weighting",
        marker="o",
    )
    for _, block in blocks.iterrows():
        if block["block_start"] == 1 and block["block_end"] == 1:
            continue
        ax.axvspan(
            float(block["block_start"]) - 0.5,
            float(block["block_end"]) + 0.5,
            alpha=0.08,
            color="grey",
        )
    ax.set_xlabel("Cosine operator component")
    ax.set_ylabel("log eigenvalue: strength of independent direction")
    ax.set_title("Adaptive spectral blocks from the cosine KAK/eigen decomposition")
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
    block_rows: list[dict[str, object]] = []
    spectrum_frames: list[pd.DataFrame] = []
    size_frames: list[pd.DataFrame] = []
    all_assignment_frames: list[pd.DataFrame] = []
    segmentation_rows: list[dict[str, object]] = []

    for weighting in args.weightings:
        values = weighted_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
        total = float(np.sum(eigvals))
        spectrum = pd.DataFrame(
            {
                "weighting": weighting,
                "component": np.arange(1, len(eigvals) + 1),
                "eigenvalue": eigvals,
                "log_eigenvalue": np.log(np.maximum(eigvals, 1e-300)),
                "fraction_of_kept_operator_energy": eigvals / total if total > 0 else np.nan,
                "cumulative_fraction": np.cumsum(eigvals) / total if total > 0 else np.nan,
            }
        )
        spectrum_frames.append(spectrum)

        blocks, diagnostics = adaptive_spectral_blocks(
            eigvals,
            min_segment_length=args.min_segment_length,
            max_segments=args.max_segments,
        )
        segmentation_rows.append({"weighting": weighting, **diagnostics})

        for block in blocks:
            start_1 = int(block["block_start"])
            end_1 = int(block["block_end"])
            energy = float(np.sum(eigvals[start_1 - 1 : end_1]) / total)
            block_rows.append(
                {
                    "weighting": weighting,
                    **block,
                    "subspace_dimensions": end_1 - start_1 + 1,
                    "block_energy_fraction": energy,
                    "block_first_eigenvalue": float(eigvals[start_1 - 1]),
                    "block_last_eigenvalue": float(eigvals[end_1 - 1]),
                }
            )

            run_id = f"{weighting}__{block['block_name']}"
            coords = coords_for_block(eigvals, eigvecs, start_1, end_1)
            distances = pdist(coords, metric="euclidean")
            if not np.isfinite(distances).all() or np.allclose(distances, 0):
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "block_name": block["block_name"],
                        "block_start": start_1,
                        "block_end": end_1,
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
                sizes.insert(2, "block_name", block["block_name"])
                size_frames.append(sizes)

                labels = assignments["cluster_id"].astype(int)
                assignment_long = assignments[["cluster_id", "cluster_root", "cluster_size"]].copy()
                assignment_long.insert(0, "run_id", run_id)
                assignment_long.insert(1, "weighting", weighting)
                assignment_long.insert(2, "block_name", block["block_name"])
                assignment_long.insert(3, "block_start", start_1)
                assignment_long.insert(4, "block_end", end_1)
                all_assignment_frames.append(assignment_long.reset_index())

                top_terms = top_terms_for_assignments(data, labels)
                if not top_terms.empty:
                    top_terms.insert(0, "run_id", run_id)
                    top_terms.insert(1, "weighting", weighting)
                    top_terms.insert(2, "block_name", block["block_name"])
                    top_terms.to_csv(out / "top_terms" / f"{run_id}_top_terms.csv", index=False)

                n_clusters = int(decomposition.get("num_clusters", labels.nunique()))
                largest = int(sizes["n_genes"].max())
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

                plot_block_assignments(
                    coords,
                    assignments,
                    f"{weighting} adaptive spectral block: modes {start_1}-{end_1}",
                    out / "plots" / f"{run_id}_spectral_block_split.png",
                )

                summary_rows.append(
                    {
                        "run_id": run_id,
                        "weighting": weighting,
                        "block_name": block["block_name"],
                        "block_type": block["block_type"],
                        "block_start": start_1,
                        "block_end": end_1,
                        "status": "ok",
                        "n_genes": int(data.shape[0]),
                        "n_features_tested": int(data.shape[1]),
                        "subspace_dimensions": int(coords.shape[1]),
                        "block_energy_fraction": energy,
                        "n_clusters": n_clusters,
                        "largest_cluster": largest,
                        "largest_cluster_fraction": float(largest / len(labels)),
                        "singleton_clusters": int((sizes["n_genes"] == 1).sum()),
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
                        "block_name": block["block_name"],
                        "block_type": block["block_type"],
                        "block_start": start_1,
                        "block_end": end_1,
                        "status": "failed_exception",
                        "error": repr(exc),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "adaptive_spectral_block_tree_summary.csv", index=False)
    block_df = pd.DataFrame(block_rows)
    block_df.to_csv(out / "adaptive_spectral_blocks.csv", index=False)
    pd.DataFrame(segmentation_rows).to_csv(out / "adaptive_spectral_segmentation_diagnostics.csv", index=False)

    if spectrum_frames:
        spectrum_df = pd.concat(spectrum_frames, ignore_index=True)
        spectrum_df.to_csv(out / "cosine_kak_eigenspectrum.csv", index=False)
        plot_spectrum_with_blocks(spectrum_df, block_df, out / "plots" / "adaptive_spectral_blocks_on_eigenspectrum.png")

    if size_frames:
        pd.concat(size_frames, ignore_index=True).to_csv(out / "adaptive_spectral_block_cluster_sizes.csv", index=False)
    if all_assignment_frames:
        pd.concat(all_assignment_frames, ignore_index=True).to_csv(
            out / "adaptive_spectral_block_cluster_assignments_long.csv",
            index=False,
        )

    ok = summary[summary["status"].eq("ok")].copy()
    if not ok.empty:
        plt.figure(figsize=(11, 5.5))
        sns.barplot(data=ok, x="run_id", y="largest_cluster_fraction", hue="weighting", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Adaptive spectral block tree")
        plt.ylabel("Largest cluster fraction after split")
        plt.title("Adaptive KAK/eigenspectrum blocks: split reaction")
        plt.tight_layout()
        plt.savefig(out / "plots" / "adaptive_block_largest_core_reaction.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9.5, 5.5))
        sns.scatterplot(
            data=ok,
            x="n_clusters",
            y="largest_cluster_fraction",
            hue="weighting",
            size="block_energy_fraction",
            sizes=(60, 260),
        )
        plt.xlabel("Clusters found by our method")
        plt.ylabel("Largest remaining core fraction")
        plt.title("Adaptive spectral-block reaction map")
        plt.tight_layout()
        plt.savefig(out / "plots" / "adaptive_block_reaction_map.png", dpi=180)
        plt.close()

    manifest_rows = []
    for file_path in sorted(out.rglob("*")):
        if file_path.is_file():
            manifest_rows.append(f"{file_path.relative_to(out).as_posix()}\t{file_path.stat().st_size} bytes")
    (out / "MANIFEST.txt").write_text("\n".join(manifest_rows) + "\n")

    best_core = ok.sort_values(["largest_cluster_fraction", "n_clusters"], ascending=[True, False]).head(10)
    stable = ok.sort_values(["silhouette_in_subspace"], ascending=False, na_position="last").head(10)
    readme = [
        "# Adaptive Cosine KAK / Spectral-Block Trees",
        "",
        f"Input: `{args.input}`",
        f"Genes: `{data.shape[0]}`",
        f"Deduplicated non-empty GO features: `{data.shape[1]}`",
        "",
        "Method: build the parameter-free gene-gene cosine operator, decompose it as",
        "`S = Q Lambda Q^T`, segment the log eigenspectrum with BIC-selected",
        "piecewise-linear decay regimes, then build one tree per adaptive spectral block.",
        "",
        f"Split thresholds: `EDGE_ALPHA={args.alpha_local}`, `SIBLING_ALPHA={args.sibling_alpha}`",
        f"Segmentation controls: `min_segment_length={args.min_segment_length}`, `max_segments={args.max_segments}`",
        "",
        "## Adaptive Blocks",
        "",
        block_df.to_string(index=False) if not block_df.empty else "No blocks.",
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
