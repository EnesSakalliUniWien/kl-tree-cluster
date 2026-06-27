#!/usr/bin/env python3
"""Split allGO GO-IC plot outputs into method-specific result folders."""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

METHOD_ORDER = [
    "whole_adaptive_diffusion",
    "adaptive_diffusion_kak",
    "raw_kak",
    "current__whole_adaptive_diffusion",
    "current__raw_cosine_subspace",
    "current__adaptive_diffusion_cosine_subspace",
]

METHOD_SLUGS = {
    "whole_adaptive_diffusion": "whole_adaptive_diffusion",
    "adaptive_diffusion_kak": "adaptive_diffusion_cosine_subspace",
    "raw_kak": "raw_cosine_subspace",
    "current__whole_adaptive_diffusion": "current__whole_adaptive_diffusion",
    "current__raw_cosine_subspace": "current__raw_cosine_subspace",
    "current__adaptive_diffusion_cosine_subspace": "current__adaptive_diffusion_cosine_subspace",
}


METHOD_LABELS = {
    "whole_adaptive_diffusion": "whole adaptive diffusion",
    "adaptive_diffusion_kak": "adaptive diffusion cosine subspace",
    "raw_kak": "raw cosine subspace",
    "current__whole_adaptive_diffusion": "current + whole adaptive diffusion",
    "current__raw_cosine_subspace": "current + raw cosine subspace",
    "current__adaptive_diffusion_cosine_subspace": "current + adaptive diffusion cosine subspace",
}


METHOD_DESCRIPTIONS = {
    "whole_adaptive_diffusion": (
        "Adaptive diffusion tree built from the full allGO feature matrix, "
        "using the full matrix rather than a decomposed cosine subspace."
    ),
    "adaptive_diffusion_kak": (
        "Cosine eigenspace component-block tree after adaptive diffusion "
        "smoothing within each selected subspace."
    ),
    "raw_kak": (
        "Raw cosine eigenspace component-block tree: the PosetTree gate is "
        "run directly on cosine eigenspace blocks without adaptive diffusion smoothing."
    ),
    "current__whole_adaptive_diffusion": (
        "Current TBS gates applied to a full-matrix adaptive diffusion tree."
    ),
    "current__raw_cosine_subspace": (
        "Current TBS gates applied to a raw cosine eigenspace block tree."
    ),
    "current__adaptive_diffusion_cosine_subspace": (
        "Current TBS gates applied to adaptive diffusion trees built inside cosine eigenspace blocks."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="allgo_new_quality_aware_go_ic",
        help="Prefix used for method-specific output file names.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def get_method_slug(method: str) -> str:
    return METHOD_SLUGS.get(method, method)


def get_method_label(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return method.replace("__", " + ").replace("_", " ")


def get_method_description(method: str) -> str:
    if method in METHOD_DESCRIPTIONS:
        return METHOD_DESCRIPTIONS[method]
    return f"Method family `{method}` from the GO-IC ranking input."


def short_run_label(run_id: str) -> str:
    label = run_id
    label = label.replace("whole_adaptive_diffusion", "whole adaptive diffusion")
    label = label.replace("adaptive_diffusion_kak__", "adaptive diffusion cosine subspace, ")
    label = label.replace("raw_kak__", "raw cosine subspace, ")
    label = label.replace("current__", "current, ")
    label = label.replace("__", ", ")
    label = label.replace("adaptive_common_mode_01", "common mode 01")
    label = label.replace("adaptive_modes_", "modes ")
    label = label.replace("_", " ")
    return label


def wrapped_label(value: str, width: int = 38) -> str:
    return "\n".join(textwrap.wrap(short_run_label(value), width=width))


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_banner_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def write_ranked_page(source_page: Path, target_page: Path, row: pd.Series, method_label: str) -> None:
    image = Image.open(source_page).convert("RGB")
    banner_h = 96
    output = Image.new("RGB", (image.width, image.height + banner_h), "white")
    output.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, banner_h, image.width, banner_h + 48), fill="white")
    font_title = load_banner_font(28)
    font_body = load_banner_font(20)
    title = (
        f"{method_label} | method rank {int(row['method_rank']):02d} | "
        f"display rank {int(row['display_rank']):02d} | raw GO-IC rank {int(row['raw_go_ic_rank']):02d}"
    )
    subtitle = (
        f"{short_run_label(str(row['run_id']))} | "
        f"GO-BIC/gene {float(row['go_bic_active_per_gene']):.2f} | "
        f"K={int(row['n_clusters'])} | "
        f"coherent={int(row['coherent_cluster_count'])}/{int(row['n_clusters'])} | "
        f"largest={float(row['largest_cluster_fraction']):.3f} | "
        f"singletons={float(row['singleton_gene_fraction']):.3f}"
    )
    draw.text((28, 18), title, fill="#111111", font=font_title)
    draw.text((28, 58), subtitle, fill="#333333", font=font_body)
    output.save(target_page)


def plot_method_top(method_ranking: pd.DataFrame, output_path: Path, method_label: str) -> None:
    frame = method_ranking.sort_values("method_rank", ascending=False)
    labels = [
        f"{int(row.method_rank):02d}  {wrapped_label(str(row.run_id))}"
        for row in frame.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(12.5, max(3.4, 0.68 * len(frame) + 1.3)))
    bars = ax.barh(labels, frame["go_bic_active_per_gene"], color="#4c78a8")
    ax.set_title(f"{method_label}: GO-IC ranking")
    ax.set_xlabel("GO-BIC active per gene (lower is better)")
    ax.set_ylabel("tree")
    ax.set_xlim(0, float(frame["go_bic_active_per_gene"].max()) + 420.0)
    ax.grid(axis="x", alpha=0.22)
    ax.tick_params(axis="y", labelsize=8)
    for bar, row in zip(bars, frame.itertuples(index=False), strict=False):
        ax.text(
            bar.get_width() + 22,
            bar.get_y() + bar.get_height() / 2,
            f"K={int(row.n_clusters)}, coherent={int(row.coherent_cluster_count)}/{int(row.n_clusters)}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_method_scatter(method_ranking: pd.DataFrame, output_path: Path, method_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(
        method_ranking["go_bic_active_per_gene"],
        method_ranking["coherent_cluster_fraction"],
        c=method_ranking["weighted_mean_within_tfidf_cosine"],
        s=60 + 150 * (1.0 - method_ranking["singleton_gene_fraction"].clip(0, 1)),
        cmap="viridis",
        alpha=0.84,
        linewidths=0.5,
        edgecolors="#333333",
    )
    for _, row in method_ranking.iterrows():
        ax.annotate(
            str(int(row["method_rank"])),
            (row["go_bic_active_per_gene"], row["coherent_cluster_fraction"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(f"{method_label}: GO-IC vs GO coherence")
    ax.set_xlabel("GO-BIC active per gene (lower is better)")
    ax.set_ylabel("coherent cluster fraction")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("weighted within-cluster TF-IDF cosine")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_pdf_cover(
    pdf: PdfPages,
    method_label: str,
    method_description: str,
    method_ranking: pd.DataFrame,
) -> None:
    best = method_ranking.iloc[0]
    sentence = (
        f"Method: {method_label}. Values shown on each tree page are: "
        "GO-BIC/gene, where lower is better within comparable non-degenerate trees; "
        "coherent, the fraction of clusters passing the GO-enrichment rule; "
        "within TF-IDF, the weighted mean within-cluster GO-vector cosine; "
        "largest, the fraction of genes in the largest cluster; and singletons, "
        "the singleton-gene fraction."
    )
    lines = [
        method_label,
        "",
        method_description,
        "",
        sentence,
        "",
        "Best tree in this method-specific PDF:",
        f"  {best['run_id']}",
        f"  GO-BIC/gene: {best['go_bic_active_per_gene']:.2f}",
        f"  clusters: {int(best['n_clusters'])}",
        (
            "  coherent clusters: "
            f"{int(best['coherent_cluster_count'])}/{int(best['n_clusters'])} "
            f"({best['coherent_cluster_fraction']:.3f})"
        ),
        f"  largest cluster fraction: {best['largest_cluster_fraction']:.3f}",
        f"  singleton-gene fraction: {best['singleton_gene_fraction']:.3f}",
    ]
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.axis("off")
    ax.text(
        0.05,
        0.92,
        lines[0],
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        0.05,
        0.80,
        "\n\n".join(textwrap.fill(line, width=105) for line in lines[2:]),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.32,
        family="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pages_pdf(
    page_paths: list[Path],
    output_path: Path,
    method_label: str,
    method_description: str,
    method_ranking: pd.DataFrame,
) -> None:
    with PdfPages(output_path) as pdf:
        write_pdf_cover(pdf, method_label, method_description, method_ranking)
        for page_path in page_paths:
            image = plt.imread(page_path)
            height, width = image.shape[:2]
            fig_width = 14.5
            fig_height = fig_width * height / width
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(image)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking = read_required_csv(args.input_dir / f"{args.prefix}_tree_ranking.csv")
    coherence = read_required_csv(args.input_dir / f"{args.prefix}_cluster_coherence_long.csv")
    tfidf_quality = read_required_csv(args.input_dir / f"{args.prefix}_tfidf_quality_long.csv")
    pages_dir = args.input_dir / "tree_pages"
    if not pages_dir.exists():
        raise FileNotFoundError(pages_dir)

    seen_methods = list(dict.fromkeys(ranking["family"].astype(str).to_list()))
    ordered_methods = [method for method in METHOD_ORDER if method in seen_methods]
    ordered_methods.extend(method for method in seen_methods if method not in ordered_methods)

    method_summaries: list[dict[str, object]] = []
    for method in ordered_methods:
        method_slug = get_method_slug(method)
        method_label = get_method_label(method)
        method_dir = args.output_dir / method_slug
        method_pages_dir = method_dir / "tree_pages"
        if method_dir.exists():
            shutil.rmtree(method_dir)
        method_pages_dir.mkdir(parents=True, exist_ok=True)

        method_ranking = ranking[ranking["family"].eq(method)].copy()
        if method_ranking.empty:
            continue
        method_ranking = method_ranking.sort_values("display_rank").reset_index(drop=True)
        method_ranking.insert(0, "method_rank", range(1, len(method_ranking) + 1))
        method_ranking["source_family"] = method_ranking["family"]
        method_ranking["family"] = method_slug
        method_ranking["method_label"] = method_label
        if "method_run_id" not in method_ranking.columns:
            method_ranking["method_run_id"] = (
                method_ranking["run_id"]
                .str.replace("adaptive_diffusion_kak__", "adaptive_diffusion_cosine_subspace__", regex=False)
                .str.replace("raw_kak__", "raw_cosine_subspace__", regex=False)
            )
        method_ranking["source_run_id"] = method_ranking["run_id"]
        source_run_ids = method_ranking["source_run_id"].copy()
        method_ranking["run_id"] = method_ranking["method_run_id"]

        method_prefix = f"{args.prefix}_{safe_name(method_slug)}"
        method_ranking_path = method_dir / f"{method_prefix}_tree_ranking.csv"
        method_coherence_path = method_dir / f"{method_prefix}_cluster_coherence_long.csv"
        method_tfidf_path = method_dir / f"{method_prefix}_tfidf_quality_long.csv"
        public_method_ranking = method_ranking.drop(
            columns=["source_family", "source_run_id", "assignments_path"],
            errors="ignore",
        )
        public_method_ranking.to_csv(method_ranking_path, index=False)
        method_coherence = coherence[coherence["run_id"].isin(source_run_ids)].copy()
        method_coherence["source_run_id"] = method_coherence["run_id"]
        method_coherence["run_id"] = method_coherence["run_id"].replace(
            dict(zip(method_ranking["source_run_id"], method_ranking["method_run_id"], strict=False))
        )
        method_coherence = method_coherence.drop(columns=["source_run_id"], errors="ignore")
        method_coherence.to_csv(method_coherence_path, index=False)
        method_tfidf = tfidf_quality[tfidf_quality["run_id"].isin(source_run_ids)].copy()
        method_tfidf["source_run_id"] = method_tfidf["run_id"]
        method_tfidf["run_id"] = method_tfidf["run_id"].replace(
            dict(zip(method_ranking["source_run_id"], method_ranking["method_run_id"], strict=False))
        )
        method_tfidf = method_tfidf.drop(columns=["source_run_id"], errors="ignore")
        method_tfidf.to_csv(method_tfidf_path, index=False)

        copied_pages: list[Path] = []
        for _, row in method_ranking.iterrows():
            source_candidates = [
                pages_dir / f"{int(row['display_rank']):02d}_{safe_name(str(row['method_run_id']))}.png",
                pages_dir / f"{int(row['display_rank']):02d}_{safe_name(str(row['source_run_id']))}.png",
            ]
            source_page = next((path for path in source_candidates if path.exists()), None)
            if source_page is None:
                raise FileNotFoundError(source_candidates[0])
            if not source_page.exists():
                raise FileNotFoundError(source_page)
            target_page = method_pages_dir / (
                f"{int(row['method_rank']):02d}_{safe_name(str(row['method_run_id']))}.png"
            )
            write_ranked_page(source_page, target_page, row, method_label)
            copied_pages.append(target_page)

        pdf_path = method_dir / f"{method_prefix}_tree_pages.pdf"
        write_pages_pdf(
            copied_pages,
            pdf_path,
            method_label,
            get_method_description(method),
            method_ranking,
        )
        plot_method_top(
            method_ranking,
            method_dir / f"{method_prefix}_top_trees.png",
            method_label,
        )
        plot_method_scatter(
            method_ranking,
            method_dir / f"{method_prefix}_quality_scatter.png",
            method_label,
        )

        best = method_ranking.iloc[0]
        method_summaries.append(
            {
                "method": method_slug,
                "method_label": method_label,
                "n_trees": len(method_ranking),
                "best_run_id": best["run_id"],
                "best_method_run_id": best["method_run_id"],
                "best_go_bic_active_per_gene": best["go_bic_active_per_gene"],
                "best_n_clusters": best["n_clusters"],
                "best_coherent_cluster_count": best["coherent_cluster_count"],
                "best_coherent_cluster_fraction": best["coherent_cluster_fraction"],
                "best_largest_cluster_fraction": best["largest_cluster_fraction"],
                "tree_pages_pdf": str(pdf_path),
                "tree_ranking_csv": str(method_ranking_path),
            }
        )

        readme = [
            f"# {method_label}",
            "",
            f"Method family: `{method_slug}`",
            f"Method description: {get_method_description(method)}",
            f"Trees in this folder: `{len(method_ranking)}`",
            "",
            "This folder is method-isolated. It contains no trees, CSV rows, or pages from other method families.",
            "PDF cover page defines the method and the plotted values.",
            "",
            "## Files",
            "",
            f"- `{method_prefix}_tree_ranking.csv`",
            f"- `{method_prefix}_cluster_coherence_long.csv`",
            f"- `{method_prefix}_tfidf_quality_long.csv`",
            f"- `{method_prefix}_tree_pages.pdf`",
            f"- `{method_prefix}_top_trees.png`",
            f"- `{method_prefix}_quality_scatter.png`",
            "- `tree_pages/*.png`",
            "",
            "## Best Tree In This Method",
            "",
            method_ranking[
                [
                    "method_rank",
                    "display_rank",
                    "raw_go_ic_rank",
                    "run_id",
                    "quality_tier_label",
                    "go_bic_active_per_gene",
                    "n_clusters",
                    "coherent_cluster_count",
                    "coherent_cluster_fraction",
                    "singleton_gene_fraction",
                    "largest_cluster_fraction",
                    "weighted_mean_within_tfidf_cosine",
                ]
            ]
            .head(8)
            .to_string(index=False),
            "",
        ]
        (method_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    summary = pd.DataFrame(method_summaries)
    summary.to_csv(args.output_dir / f"{args.prefix}_method_summary.csv", index=False)
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# allGO New Quality-Aware GO-IC Results By Method",
                "",
                "Each subdirectory contains only one method family. Use these method-specific",
                "PDFs, CSVs, and plots for interpretation.",
                "",
                "## Method Folders",
                "",
                *[f"- `{method}`" for method in summary["method"].tolist()],
                "",
                "## Summary",
                "",
                summary.to_string(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote method-separated GO-IC outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
