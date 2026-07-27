"""Matched adult/Goncalves NNLS report with identical page structure."""
# ruff: noqa: I001

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from applications.scrna.plots.selected_nnls_fit_summary import _fit_diagnostic, _plot_fit
from applications.scrna.plots.selected_nnls_report import (
    BASE,
    DATASETS,
    METHOD_TITLE,
    SELECTED,
    add_generated_at,
    build_full_tree_info,
    cluster_summary,
    make_cluster_palette,
    plot_full_radial_tree,
    plot_umap,
)

OUT_PDF = BASE / "scrna_selected_adaptive_diffusion_nnls_matched_report.pdf"
OUT_ADULT_PAGE = BASE / "scrna_selected_adaptive_diffusion_nnls_matched_adult_page.png"
OUT_GONCALVES_PAGE = BASE / "scrna_selected_adaptive_diffusion_nnls_matched_goncalves_page.png"
OUT_SUMMARY = BASE / "scrna_selected_adaptive_diffusion_nnls_matched_summary.csv"
OUT_MANIFEST = BASE / "scrna_selected_adaptive_diffusion_nnls_matched_manifest.json"


def _selected_metric_row(cfg: dict[str, object]) -> pd.Series:
    metrics = pd.read_csv(Path(cfg["dir"]) / "method_metrics.csv")
    mask = metrics["method"].astype(str).str.contains(
        "TBS adaptive diffusion branch-time recomputed-NNLS",
        regex=False,
        na=False,
    )
    rows = metrics.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected metric row for {cfg['key']}, found {len(rows)}")
    return rows.iloc[0]


def _summary_rows(
    cfg: dict[str, object],
    assignments: pd.DataFrame,
    diagnostic: dict[str, object],
) -> dict[str, object]:
    metric = _selected_metric_row(cfg)
    selected = str(cfg["selected"])
    return {
        "dataset": cfg["key"],
        "n_cells": len(assignments),
        "n_clusters": assignments[selected].nunique(),
        "ari": metric["ari"],
        "v_measure": metric["v_measure"],
        "weighted_cluster_purity": metric["weighted_cluster_purity"],
        "weighted_label_dominant_cluster_recall": metric[
            "weighted_label_dominant_cluster_recall"
        ],
        "nnls_fit_r2": diagnostic["r2"],
        "nnls_fit_correlation": diagnostic["correlation"],
        "nnls_fit_rmse": diagnostic["rmse"],
        "nnls_fit_mae": diagnostic["mae"],
        "zero_branch_fraction": diagnostic["zero_branch_fraction"],
        "branch_length_max": diagnostic["branch_length_max"],
    }


def _metric_text(row: dict[str, object]) -> str:
    return (
        f"cells / clusters: {row['n_cells']:,} / {row['n_clusters']}\n"
        f"ARI: {row['ari']:.3f}\n"
        f"V-measure: {row['v_measure']:.3f}\n"
        f"cluster purity: {row['weighted_cluster_purity']:.3f}\n"
        f"dominant-label recall: {row['weighted_label_dominant_cluster_recall']:.3f}\n"
        f"NNLS R2 / corr.: {row['nnls_fit_r2']:.3f} / {row['nnls_fit_correlation']:.3f}\n"
        f"NNLS RMSE / MAE: {row['nnls_fit_rmse']:.3f} / {row['nnls_fit_mae']:.3f}"
    )


def _add_metric_box(ax: plt.Axes, row: dict[str, object]) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        _metric_text(row),
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f9fafb", "edgecolor": "#d1d5db"},
    )


def _plot_clear_size_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    cluster_palette: dict[int, str],
    title: str,
    *,
    max_rows: int = 24,
) -> None:
    summary = summary.sort_values("n_cells_assignment", ascending=False).copy()
    if len(summary) > max_rows:
        head = summary.head(max_rows - 1).copy()
        tail = summary.iloc[max_rows - 1 :]
        other = {
            "cluster": -1,
            "n_cells_assignment": int(tail["n_cells_assignment"].sum()),
            "top_celltype_assignment": f"remaining {len(tail)} clusters",
            "top_celltype_fraction_assignment": 1.0,
        }
        summary = pd.concat([head, pd.DataFrame([other])], ignore_index=True)

    summary = summary.sort_values("n_cells_assignment", ascending=True)
    labels = []
    colors = []
    for row in summary.itertuples(index=False):
        if int(row.cluster) < 0:
            labels.append(str(row.top_celltype_assignment))
            colors.append("#9ca3af")
        else:
            labels.append(
                f"C{int(row.cluster)}  {str(row.top_celltype_assignment)[:22]} "
                f"({row.top_celltype_fraction_assignment:.0%})"
            )
            colors.append(cluster_palette[int(row.cluster)])

    y = range(len(summary))
    ax.barh(
        list(y),
        summary["n_cells_assignment"],
        color=colors,
        edgecolor="none",
        alpha=0.92,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("cells", fontsize=8)
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#e5e7eb", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _dataset_page(
    *,
    generated_at: str,
    record: dict[str, object],
    save_path: Path | None = None,
) -> plt.Figure:
    cfg = record["cfg"]
    assignments = record["assignments"]
    summary = record["cluster_summary"]
    palette = record["palette"]
    full_tree = record["full_tree"]

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.1, 1.25, 0.7],
        height_ratios=[1.0, 0.48],
        hspace=0.2,
        wspace=0.2,
    )
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_sizes = fig.add_subplot(gs[1, :])

    plot_umap(
        ax_umap,
        assignments,
        str(cfg["selected"]),
        palette,
        "selected NNLS-TBS clusters",
        annotate_clusters=True,
        point_size=7,
    )
    plot_full_radial_tree(ax_tree, full_tree, palette, "real full tree")
    _plot_clear_size_bars(ax_sizes, summary, palette, "largest clusters and dominant reference label")
    _add_metric_box(ax_metrics, record["summary_row"])

    fig.suptitle(
        f"{cfg['title']}\n{METHOD_TITLE}",
        fontsize=16,
        weight="bold",
        y=0.988,
    )
    fig.text(
        0.012,
        0.012,
        "This page structure is reused unchanged for both datasets: selected-cluster UMAP, "
        "real full tree, metric box, and cluster-size/dominant-label panel. "
        f"Generated at: {generated_at}",
        fontsize=9,
        color="#374151",
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def _fit_page(
    *,
    generated_at: str,
    dataset_records: list[dict[str, object]],
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    gs = GridSpec(1, len(dataset_records), figure=fig, wspace=0.2)
    for col_index, record in enumerate(dataset_records):
        ax = fig.add_subplot(gs[0, col_index])
        _plot_fit(ax, record["diagnostic"])
        ax.set_title(
            f"{record['cfg']['title']}\nNNLS branch-length fit",
            fontsize=11,
            weight="bold",
        )
    fig.suptitle("Same NNLS fit diagnostic for both analyses", fontsize=15, weight="bold", y=0.99)
    add_generated_at(fig, generated_at)
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(generated_at: str) -> None:
    artifacts = []
    for path, role in [
        (OUT_PDF, "matched adult/Goncalves report PDF"),
        (OUT_ADULT_PAGE, "matched adult report page PNG"),
        (OUT_GONCALVES_PAGE, "matched Goncalves report page PNG"),
        (OUT_SUMMARY, "matched adult/Goncalves summary table"),
    ]:
        record = {
            "path": str(path.relative_to(BASE)),
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            record["rows"] = len(table)
            record["columns"] = len(table.columns)
            record["generated_at"] = generated_at
            record["generated_at_values"] = sorted(table["generated_at"].dropna().unique().tolist())
        artifacts.append(record)

    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "manifest_schema_version": "static_artifact_provenance/v1",
                "generated_at": generated_at,
                "output_dir": str(BASE.relative_to(BASE.parents[1])),
                "source_scripts": [
                    "applications/scrna/plots/selected_nnls_matched_report.py",
                    "applications/scrna/plots/selected_nnls_report.py",
                    "applications/scrna/plots/selected_nnls_fit_summary.py",
                ],
                "selected_method_column": SELECTED,
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    dataset_records = []
    summary_rows = []
    for cfg in DATASETS:
        assignments = pd.read_csv(Path(cfg["dir"]) / "method_assignments.csv")
        palette = make_cluster_palette(assignments[str(cfg["selected"])].unique())
        diagnostic = _fit_diagnostic(cfg)
        summary_row = _summary_rows(cfg, assignments, diagnostic)
        summary_row["generated_at"] = generated_at
        summary = cluster_summary(assignments, str(cfg["selected"]))
        dataset_records.append(
            {
                "cfg": cfg,
                "assignments": assignments,
                "palette": palette,
                "diagnostic": diagnostic,
                "summary_row": summary_row,
                "cluster_summary": summary,
                "full_tree": build_full_tree_info(cfg, assignments),
            }
        )
        summary_rows.append(summary_row)

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    with PdfPages(
        OUT_PDF,
        metadata={
            "Title": "Matched adult/Goncalves scRNA NNLS-TBS report",
            "Subject": f"Generated at: {generated_at}",
        },
    ) as pdf:
        for record, save_path in zip(dataset_records, [OUT_ADULT_PAGE, OUT_GONCALVES_PAGE], strict=True):
            page = _dataset_page(generated_at=generated_at, record=record, save_path=save_path)
            pdf.savefig(page, bbox_inches="tight")
            plt.close(page)

        fit_page = _fit_page(generated_at=generated_at, dataset_records=dataset_records)
        pdf.savefig(fit_page, bbox_inches="tight")
        plt.close(fit_page)

    _write_manifest(generated_at)
    print(OUT_PDF)
    print(OUT_ADULT_PAGE)
    print(OUT_GONCALVES_PAGE)
    print(OUT_SUMMARY)
    print(OUT_MANIFEST)


if __name__ == "__main__":
    main()
