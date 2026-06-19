#!/usr/bin/env python3
"""Quality analysis for a binary GO feature matrix."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

GO_ID_RE = re.compile(r"(GO:\d{7})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--svd-components", type=int, default=80)
    parser.add_argument("--top-pairs", type=int, default=200)
    return parser.parse_args()


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{q}": math.nan for q in ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")}
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p01": float(np.quantile(values, 0.01)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p25": float(np.quantile(values, 0.25)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p75": float(np.quantile(values, 0.75)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(np.max(values)),
    }


def binary_entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    entropy = np.zeros_like(probabilities, dtype=float)
    valid = (probabilities > 0.0) & (probabilities < 1.0)
    p = probabilities[valid]
    entropy[valid] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy


def shannon_entropy_bits(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    probabilities = weights[weights > 0.0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)
    return data.apply(pd.to_numeric, errors="raise")


def parse_go_term(column: str) -> tuple[str, str | None]:
    match = GO_ID_RE.search(column)
    go_id = match.group(1) if match else None
    if go_id is None:
        return column, None
    term = column.replace(f"({go_id})", "").strip()
    return term, go_id


def packed_pattern_groups(binary: np.ndarray, labels: pd.Index, *, axis: int) -> pd.DataFrame:
    """Find exact duplicate binary patterns over rows or columns."""
    values = binary if axis == 0 else binary.T
    packed = np.packbits(values.astype(np.uint8), axis=1)
    groups: dict[bytes, list[str]] = {}
    for idx, row in enumerate(packed):
        groups.setdefault(row.tobytes(), []).append(str(labels[idx]))
    rows = []
    for group_id, members in enumerate(groups.values(), start=1):
        if len(members) <= 1:
            continue
        rows.append(
            {
                "duplicate_group_id": group_id,
                "n_members": len(members),
                "members": ";".join(members),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_members", "duplicate_group_id"], ascending=[False, True])


def top_similarity_pairs(
    matrix: np.ndarray,
    labels: pd.Index,
    *,
    metric_name: str,
    top_n: int,
) -> pd.DataFrame:
    similarity = cosine_similarity(matrix.astype(float))
    np.fill_diagonal(similarity, -np.inf)
    tri_i, tri_j = np.triu_indices_from(similarity, k=1)
    values = similarity[tri_i, tri_j]
    if values.size == 0:
        return pd.DataFrame(columns=["left", "right", metric_name])
    keep = np.argsort(values)[::-1][:top_n]
    return pd.DataFrame(
        {
            "left": labels[tri_i[keep]].astype(str),
            "right": labels[tri_j[keep]].astype(str),
            metric_name: values[keep],
        }
    )


def feature_filter_grid(data: pd.DataFrame) -> pd.DataFrame:
    supports = data.sum(axis=0).to_numpy(dtype=int)
    rows = []
    for min_support in (1, 2, 3, 5, 10, 20, 50):
        for max_prevalence in (1.0, 0.75, 0.50, 0.25):
            max_support = int(math.floor(max_prevalence * len(data)))
            keep = (supports >= min_support) & (supports <= max_support)
            filtered = data.iloc[:, keep]
            row_sums = filtered.sum(axis=1).to_numpy(dtype=int) if filtered.shape[1] else np.zeros(len(data), dtype=int)
            rows.append(
                {
                    "min_support": min_support,
                    "max_prevalence": max_prevalence,
                    "max_support": max_support,
                    "kept_terms": int(keep.sum()),
                    "removed_terms": int((~keep).sum()),
                    "kept_term_fraction": float(keep.mean()),
                    "row_zero_count_after_filter": int((row_sums == 0).sum()),
                    "row_active_terms_median_after_filter": float(np.median(row_sums)),
                    "row_active_terms_p05_after_filter": float(np.quantile(row_sums, 0.05)),
                    "row_active_terms_p95_after_filter": float(np.quantile(row_sums, 0.95)),
                    "density_after_filter": float(row_sums.sum() / max(filtered.shape[0] * filtered.shape[1], 1)),
                }
            )
    return pd.DataFrame(rows)


def entropy_bin_table(values: np.ndarray, *, value_name: str) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    bins = [
        (0.0, 0.05, "<=0.05"),
        (0.05, 0.10, "0.05-0.10"),
        (0.10, 0.20, "0.10-0.20"),
        (0.20, 0.50, "0.20-0.50"),
        (0.50, 0.80, "0.50-0.80"),
        (0.80, 1.00, "0.80-1.00"),
    ]
    rows = []
    for lower, upper, label in bins:
        if lower == 0.0:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values > lower) & (values <= upper)
        rows.append(
            {
                f"{value_name}_bin": label,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()) if values.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    *,
    data: pd.DataFrame,
    gene_stats: pd.DataFrame,
    term_stats: pd.DataFrame,
    pairwise_stats: dict[str, float],
    svd_df: pd.DataFrame,
    embedding: pd.DataFrame,
    filter_grid: pd.DataFrame,
    duplicate_gene_groups: pd.DataFrame,
    duplicate_term_groups: pd.DataFrame,
    entropy_summary: pd.DataFrame,
    term_entropy_bins: pd.DataFrame,
    gene_entropy_bins: pd.DataFrame,
) -> None:
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        summary_lines = [
            "feature_matrix_julia_allGO_new.tsv quality analysis",
            "",
            f"Genes: {data.shape[0]:,}",
            f"GO terms: {data.shape[1]:,}",
            f"Density: {data.to_numpy().mean():.4f}",
            f"Median active terms/gene: {gene_stats['active_terms'].median():.1f}",
            f"Median genes/term: {term_stats['support'].median():.1f}",
            f"Singleton/rare terms support <= 2: {int((term_stats['support'] <= 2).sum()):,}",
            f"Terms support <= 5: {int((term_stats['support'] <= 5).sum()):,}",
            f"Terms present in >= 50% genes: {int((term_stats['prevalence'] >= 0.5).sum()):,}",
            f"Duplicate gene patterns: {len(duplicate_gene_groups):,} groups",
            f"Duplicate GO-term patterns: {len(duplicate_term_groups):,} groups",
            "",
            "Interpretation: sparse rare terms increase GO-IC overfitting risk, while broad terms",
            "can dominate distance geometry. Use this report to decide filtering/sensitivity grids.",
        ]
        ax.text(0.05, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=13, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].hist(gene_stats["active_terms"], bins=40, color="#4c78a8", alpha=0.86)
        axes[0, 0].set_title("Annotation burden per gene")
        axes[0, 0].set_xlabel("active GO terms")
        axes[0, 0].set_ylabel("genes")
        axes[0, 1].hist(term_stats["support"], bins=60, color="#f58518", alpha=0.86)
        axes[0, 1].set_title("GO term support")
        axes[0, 1].set_xlabel("genes annotated")
        axes[0, 1].set_ylabel("GO terms")
        axes[1, 0].hist(np.log10(term_stats["support"].clip(lower=1)), bins=40, color="#54a24b", alpha=0.86)
        axes[1, 0].set_title("GO term support, log10")
        axes[1, 0].set_xlabel("log10 support")
        axes[1, 1].bar(
            ["<=1", "2-5", "6-10", "11-50", "51-200", ">200"],
            [
                int((term_stats["support"] <= 1).sum()),
                int(((term_stats["support"] >= 2) & (term_stats["support"] <= 5)).sum()),
                int(((term_stats["support"] >= 6) & (term_stats["support"] <= 10)).sum()),
                int(((term_stats["support"] >= 11) & (term_stats["support"] <= 50)).sum()),
                int(((term_stats["support"] >= 51) & (term_stats["support"] <= 200)).sum()),
                int((term_stats["support"] > 200).sum()),
            ],
            color="#72b7b2",
        )
        axes[1, 1].set_title("GO term support bins")
        axes[1, 1].set_ylabel("GO terms")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        axes[0, 0].hist(term_stats["entropy_bits"], bins=50, color="#4c78a8", alpha=0.88)
        axes[0, 0].set_title("GO-term Bernoulli entropy")
        axes[0, 0].set_xlabel("entropy bits")
        axes[0, 0].set_ylabel("GO terms")
        axes[0, 1].scatter(
            term_stats["support"],
            term_stats["entropy_bits"],
            s=8,
            alpha=0.45,
            color="#f58518",
            edgecolors="none",
        )
        axes[0, 1].set_title("GO support versus entropy")
        axes[0, 1].set_xlabel("genes annotated")
        axes[0, 1].set_ylabel("entropy bits")
        axes[0, 1].set_xscale("log")
        axes[1, 0].hist(gene_stats["row_entropy_bits"], bins=40, color="#54a24b", alpha=0.88)
        axes[1, 0].set_title("Gene row entropy")
        axes[1, 0].set_xlabel("entropy bits from active-term fraction")
        axes[1, 0].set_ylabel("genes")
        summary_text = ["Entropy summary"]
        for row in entropy_summary.itertuples(index=False):
            value = row.value
            if isinstance(value, float):
                summary_text.append(f"{row.metric}: {value:.6g}")
            else:
                summary_text.append(f"{row.metric}: {value}")
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.0,
            1.0,
            "\n".join(summary_text),
            va="top",
            ha="left",
            fontsize=8.8,
            family="monospace",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(svd_df["component"], svd_df["explained_variance_ratio"], marker="o", linewidth=1)
        axes[0].set_title("SVD explained variance ratio")
        axes[0].set_xlabel("component")
        axes[0].set_ylabel("variance ratio")
        axes[0].grid(alpha=0.25)
        axes[1].plot(svd_df["component"], svd_df["cumulative_explained_variance_ratio"], marker="o", linewidth=1)
        axes[1].set_title("Cumulative SVD variance")
        axes[1].set_xlabel("component")
        axes[1].set_ylabel("cumulative ratio")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            embedding["axis_1"],
            embedding["axis_2"],
            c=embedding["active_terms"],
            cmap="viridis",
            s=28,
            alpha=0.88,
            linewidth=0.2,
            edgecolor="#333333",
        )
        ax.set_title("Gene embedding colored by annotation burden")
        ax.set_xlabel("SVD axis 1")
        ax.set_ylabel("SVD axis 2")
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("active GO terms")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].hist(pairwise_stats["gene_cosine_values"], bins=50, color="#4c78a8", alpha=0.86)
        axes[0].set_title("Pairwise gene cosine similarity")
        axes[0].set_xlabel("cosine")
        axes[0].set_ylabel("gene pairs")
        axes[1].hist(pairwise_stats["gene_jaccard_similarity_values"], bins=50, color="#e45756", alpha=0.86)
        axes[1].set_title("Pairwise gene Jaccard similarity")
        axes[1].set_xlabel("Jaccard similarity")
        axes[1].set_ylabel("gene pairs")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        pivot = filter_grid.pivot(index="min_support", columns="max_prevalence", values="kept_terms")
        image = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        ax.set_xlabel("max prevalence")
        ax.set_ylabel("min support")
        ax.set_title("GO terms retained under support/prevalence filters")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{int(pivot.iloc[i, j])}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="kept GO terms")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(args.input)
    values = data.to_numpy()
    binary_ok = bool(np.isin(values[~pd.isna(values)], (0, 1)).all())
    if not binary_ok:
        raise ValueError("Feature matrix is not binary.")
    binary = values.astype(np.uint8)
    n_genes, n_terms = binary.shape

    row_sums = binary.sum(axis=1)
    col_sums = binary.sum(axis=0)
    gene_stats = pd.DataFrame(
        {
            "gene": data.index,
            "active_terms": row_sums,
            "active_term_fraction": row_sums / max(n_terms, 1),
        }
    ).sort_values("active_terms", ascending=False)
    gene_stats["row_entropy_bits"] = binary_entropy_bits(gene_stats["active_term_fraction"].to_numpy())
    q1, q3 = np.quantile(row_sums, [0.25, 0.75])
    iqr = q3 - q1
    gene_stats["burden_outlier_iqr"] = (gene_stats["active_terms"] < q1 - 1.5 * iqr) | (
        gene_stats["active_terms"] > q3 + 1.5 * iqr
    )

    parsed_terms = [parse_go_term(col) for col in data.columns]
    term_stats = pd.DataFrame(
        {
            "go_term": [term for term, _go_id in parsed_terms],
            "go_id": [go_id for _term, go_id in parsed_terms],
            "column": data.columns,
            "support": col_sums,
            "prevalence": col_sums / max(n_genes, 1),
        }
    ).sort_values(["support", "go_id"], ascending=[False, True])
    term_stats["entropy_bits"] = binary_entropy_bits(term_stats["prevalence"].to_numpy())
    term_entropy_bins = entropy_bin_table(term_stats["entropy_bits"].to_numpy(), value_name="go_term_entropy_bits")
    gene_entropy_bins = entropy_bin_table(gene_stats["row_entropy_bits"].to_numpy(), value_name="gene_row_entropy_bits")

    total_go_marginal_entropy_bits = float(term_stats["entropy_bits"].sum())
    matrix_density_entropy_bits_per_cell = float(binary_entropy_bits(np.array([binary.mean()]))[0])
    gene_entry_entropy_bits = shannon_entropy_bits(row_sums)
    term_entry_entropy_bits = shannon_entropy_bits(col_sums)
    entropy_summary_rows = [
        ("go_term_entropy_bits_mean", float(term_stats["entropy_bits"].mean())),
        ("go_term_entropy_bits_median", float(term_stats["entropy_bits"].median())),
        ("go_term_entropy_bits_p25", float(term_stats["entropy_bits"].quantile(0.25))),
        ("go_term_entropy_bits_p75", float(term_stats["entropy_bits"].quantile(0.75))),
        ("go_term_entropy_bits_p95", float(term_stats["entropy_bits"].quantile(0.95))),
        ("go_term_entropy_bits_max", float(term_stats["entropy_bits"].max())),
        ("go_terms_entropy_le_0_05", int((term_stats["entropy_bits"] <= 0.05).sum())),
        ("go_terms_entropy_le_0_10", int((term_stats["entropy_bits"] <= 0.10).sum())),
        ("go_terms_entropy_le_0_20", int((term_stats["entropy_bits"] <= 0.20).sum())),
        ("go_terms_entropy_ge_0_50", int((term_stats["entropy_bits"] >= 0.50).sum())),
        ("go_terms_entropy_ge_0_80", int((term_stats["entropy_bits"] >= 0.80).sum())),
        ("total_go_marginal_entropy_bits", total_go_marginal_entropy_bits),
        ("mean_go_marginal_entropy_bits_per_term", total_go_marginal_entropy_bits / max(n_terms, 1)),
        ("matrix_density_entropy_bits_per_cell", matrix_density_entropy_bits_per_cell),
        ("gene_row_entropy_bits_mean", float(gene_stats["row_entropy_bits"].mean())),
        ("gene_row_entropy_bits_median", float(gene_stats["row_entropy_bits"].median())),
        ("gene_row_entropy_bits_p95", float(gene_stats["row_entropy_bits"].quantile(0.95))),
        ("gene_row_entropy_bits_max", float(gene_stats["row_entropy_bits"].max())),
        ("annotation_entry_entropy_across_genes_bits", gene_entry_entropy_bits),
        ("annotation_entry_effective_genes", float(2.0**gene_entry_entropy_bits)),
        ("annotation_entry_entropy_across_go_terms_bits", term_entry_entropy_bits),
        ("annotation_entry_effective_go_terms", float(2.0**term_entry_entropy_bits)),
    ]
    entropy_summary = pd.DataFrame(entropy_summary_rows, columns=["metric", "value"])

    duplicate_gene_groups = packed_pattern_groups(binary, data.index, axis=0)
    duplicate_term_groups = packed_pattern_groups(binary, data.columns, axis=1)

    cosine = cosine_similarity(binary.astype(float))
    tri = np.triu_indices_from(cosine, k=1)
    gene_cosine_values = cosine[tri]
    jaccard_dist = pdist(binary.astype(bool), metric="jaccard")
    gene_jaccard_similarity_values = 1.0 - jaccard_dist
    pairwise_summary = {
        **quantiles(gene_cosine_values, "gene_cosine"),
        **quantiles(gene_jaccard_similarity_values, "gene_jaccard_similarity"),
    }
    top_gene_pairs = top_similarity_pairs(binary, data.index, metric_name="cosine_similarity", top_n=args.top_pairs)
    top_term_pairs = top_similarity_pairs(binary.T, data.columns, metric_name="cosine_similarity", top_n=args.top_pairs)

    n_components = min(args.svd_components, n_genes - 1, n_terms - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=1729)
    coords = svd.fit_transform(binary.astype(float))
    svd_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "singular_value": svd.singular_values_,
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(svd.explained_variance_ratio_),
        }
    )
    embedding = pd.DataFrame(
        {
            "gene": data.index,
            "axis_1": coords[:, 0],
            "axis_2": coords[:, 1] if coords.shape[1] > 1 else np.zeros(n_genes),
            "active_terms": row_sums,
        }
    )
    filter_grid = feature_filter_grid(data)

    summary_rows = [
        ("input_path", str(args.input)),
        ("n_genes", n_genes),
        ("n_go_terms", n_terms),
        ("binary_values_only", binary_ok),
        ("missing_values", int(pd.isna(values).sum())),
        ("density", float(binary.mean())),
        ("total_active_entries", int(binary.sum())),
        ("zero_gene_rows", int((row_sums == 0).sum())),
        ("zero_go_term_columns", int((col_sums == 0).sum())),
        ("duplicate_gene_pattern_groups", int(len(duplicate_gene_groups))),
        ("duplicate_go_term_pattern_groups", int(len(duplicate_term_groups))),
        ("terms_support_eq_1", int((col_sums == 1).sum())),
        ("terms_support_le_2", int((col_sums <= 2).sum())),
        ("terms_support_le_5", int((col_sums <= 5).sum())),
        ("terms_support_ge_50pct", int((col_sums >= 0.5 * n_genes).sum())),
        ("terms_support_ge_75pct", int((col_sums >= 0.75 * n_genes).sum())),
        ("gene_active_terms_mean", float(np.mean(row_sums))),
        ("gene_active_terms_median", float(np.median(row_sums))),
        ("go_term_support_mean", float(np.mean(col_sums))),
        ("go_term_support_median", float(np.median(col_sums))),
        ("go_term_entropy_bits_mean", float(term_stats["entropy_bits"].mean())),
        ("go_term_entropy_bits_median", float(term_stats["entropy_bits"].median())),
        ("go_terms_entropy_le_0_10", int((term_stats["entropy_bits"] <= 0.10).sum())),
        ("go_terms_entropy_ge_0_80", int((term_stats["entropy_bits"] >= 0.80).sum())),
        ("matrix_density_entropy_bits_per_cell", matrix_density_entropy_bits_per_cell),
        ("total_go_marginal_entropy_bits", total_go_marginal_entropy_bits),
        ("svd_components_for_50pct_variance", int((svd_df["cumulative_explained_variance_ratio"] < 0.50).sum() + 1)),
        ("svd_components_for_80pct_variance", int((svd_df["cumulative_explained_variance_ratio"] < 0.80).sum() + 1)),
        *pairwise_summary.items(),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    summary.to_csv(args.output_dir / "matrix_quality_summary.csv", index=False)
    gene_stats.to_csv(args.output_dir / "gene_annotation_burden.csv", index=False)
    term_stats.to_csv(args.output_dir / "go_term_supports.csv", index=False)
    duplicate_gene_groups.to_csv(args.output_dir / "duplicate_gene_patterns.csv", index=False)
    duplicate_term_groups.to_csv(args.output_dir / "duplicate_go_term_patterns.csv", index=False)
    entropy_summary.to_csv(args.output_dir / "entropy_summary.csv", index=False)
    term_entropy_bins.to_csv(args.output_dir / "go_term_entropy_bins.csv", index=False)
    gene_entropy_bins.to_csv(args.output_dir / "gene_row_entropy_bins.csv", index=False)
    pd.DataFrame([pairwise_summary]).to_csv(args.output_dir / "pairwise_gene_similarity_summary.csv", index=False)
    top_gene_pairs.to_csv(args.output_dir / "top_gene_cosine_pairs.csv", index=False)
    top_term_pairs.to_csv(args.output_dir / "top_go_term_cosine_pairs.csv", index=False)
    svd_df.to_csv(args.output_dir / "svd_spectrum.csv", index=False)
    embedding.to_csv(args.output_dir / "gene_svd_embedding.csv", index=False)
    filter_grid.to_csv(args.output_dir / "feature_filter_sensitivity.csv", index=False)

    write_report(
        args.output_dir / "feature_matrix_quality_report.pdf",
        data=data,
        gene_stats=gene_stats,
        term_stats=term_stats,
        pairwise_stats={
            "gene_cosine_values": gene_cosine_values,
            "gene_jaccard_similarity_values": gene_jaccard_similarity_values,
        },
        svd_df=svd_df,
        embedding=embedding,
        filter_grid=filter_grid,
        duplicate_gene_groups=duplicate_gene_groups,
        duplicate_term_groups=duplicate_term_groups,
        entropy_summary=entropy_summary,
        term_entropy_bins=term_entropy_bins,
        gene_entropy_bins=gene_entropy_bins,
    )

    readme = [
        "# Feature Matrix Quality Analysis",
        "",
        f"Input: `{args.input}`",
        "",
        "Primary outputs:",
        "- `feature_matrix_quality_report.pdf`",
        "- `matrix_quality_summary.csv`",
        "- `gene_annotation_burden.csv`",
        "- `go_term_supports.csv`",
        "- `entropy_summary.csv`",
        "- `go_term_entropy_bins.csv`",
        "- `gene_row_entropy_bins.csv`",
        "- `feature_filter_sensitivity.csv`",
        "- `pairwise_gene_similarity_summary.csv`",
        "- `top_gene_cosine_pairs.csv`",
        "- `top_go_term_cosine_pairs.csv`",
        "",
        "Short interpretation:",
        f"- Matrix has `{n_genes}` genes and `{n_terms}` GO terms with density `{binary.mean():.4f}`.",
        f"- `{int((col_sums <= 2).sum())}` GO terms have support <= 2 genes; these are overfitting-prone for GO-IC.",
        f"- Median GO-term Bernoulli entropy is `{float(term_stats['entropy_bits'].median()):.4f}` bits; `{int((term_stats['entropy_bits'] <= 0.10).sum())}` terms are <= 0.10 bits.",
        f"- Total marginal GO-term entropy is `{total_go_marginal_entropy_bits:.1f}` bits across `{n_terms}` terms.",
        f"- `{int((col_sums >= 0.5 * n_genes).sum())}` GO terms occur in at least half of genes; these broad terms can dominate global geometry.",
        f"- Median gene has `{float(np.median(row_sums)):.1f}` active GO terms.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote feature matrix quality analysis: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
