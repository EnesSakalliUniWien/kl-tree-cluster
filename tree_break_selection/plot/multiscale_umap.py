"""Reusable multi-scale traversal overlays for existing UMAP coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _infer_gene_column(frame: pd.DataFrame) -> str:
    for candidate in ("sample_id", "gene_symbol", "gene", "id"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def _infer_umap_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    gene_column = _infer_gene_column(frame)
    for x_col, y_col in (("UMAP-1", "UMAP-2"), ("umap_1", "umap_2"), ("x", "y")):
        if x_col in frame.columns and y_col in frame.columns:
            return gene_column, x_col, y_col
    raise ValueError("UMAP coordinates must contain UMAP-1/UMAP-2, umap_1/umap_2, or x/y.")


def load_overlay_data(
    *,
    gene_assignments_path: Path,
    umap_coordinates_path: Path,
    method_id: str | None = None,
    data_role: str | None = None,
    replicate: int | None = None,
) -> pd.DataFrame:
    assignments = pd.read_csv(gene_assignments_path)
    coordinates = pd.read_csv(umap_coordinates_path)
    assignment_gene_column = _infer_gene_column(assignments)
    coordinate_gene_column, x_col, y_col = _infer_umap_columns(coordinates)
    if method_id is not None and "method_id" in assignments:
        assignments = assignments[assignments["method_id"].astype(str).eq(str(method_id))]
    if data_role is not None and "data_role" in assignments:
        assignments = assignments[assignments["data_role"].astype(str).eq(str(data_role))]
    if replicate is not None and "replicate" in assignments:
        assignments = assignments[assignments["replicate"].astype(int).eq(int(replicate))]
    if assignments.empty:
        raise ValueError("No gene-assignment rows remain after filters.")
    assignment_view = assignments.rename(columns={assignment_gene_column: "gene_symbol"})
    coordinate_view = coordinates.rename(
        columns={
            coordinate_gene_column: "gene_symbol",
            x_col: "UMAP-1",
            y_col: "UMAP-2",
        }
    )
    merged = assignment_view.merge(
        coordinate_view[["gene_symbol", "UMAP-1", "UMAP-2"]],
        on="gene_symbol",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No genes overlap between assignments and UMAP coordinates.")
    merged["stable_region_id"] = merged["stable_region_id"].fillna("").astype(str)
    merged["zone_region_id"] = merged["zone_region_id"].fillna("").astype(str)
    merged["has_zone_overlay"] = merged["zone_region_id"].ne("")
    return merged


def _region_palette(values: pd.Series, top_regions: int) -> dict[str, object]:
    counts = values.value_counts()
    retained = list(counts.head(int(top_regions)).index)
    cmap = plt.get_cmap("tab20")
    palette = {region: cmap(index % cmap.N) for index, region in enumerate(retained)}
    palette["__other__"] = (0.75, 0.75, 0.75, 0.55)
    return palette


def render_multiscale_umap_overlay(
    overlay_data: pd.DataFrame,
    output_path: Path,
    *,
    top_regions: int = 20,
) -> Path:
    palette = _region_palette(overlay_data["stable_region_id"], top_regions)
    fig, ax = plt.subplots(figsize=(9, 7))
    plotted_regions = [
        region
        for region in palette
        if region != "__other__" and region in set(overlay_data["stable_region_id"])
    ]
    other = overlay_data[~overlay_data["stable_region_id"].isin(plotted_regions)]
    if not other.empty:
        ax.scatter(
            other["UMAP-1"],
            other["UMAP-2"],
            s=16,
            c=[palette["__other__"]],
            linewidths=0,
            label="other stable regions",
        )
    for region in plotted_regions:
        group = overlay_data[overlay_data["stable_region_id"].eq(region)]
        ax.scatter(
            group["UMAP-1"],
            group["UMAP-2"],
            s=18,
            c=[palette[region]],
            linewidths=0,
            label=region,
        )
    zones = overlay_data[overlay_data["has_zone_overlay"]]
    if not zones.empty:
        dominant_zone_label = ""
        zone_counts = zones["zone_region_id"].value_counts()
        if not zone_counts.empty:
            dominant_zone = str(zone_counts.index[0])
            dominant_fraction = float(zone_counts.iloc[0] / len(overlay_data))
            if dominant_fraction >= 0.8:
                dominant_zone_label = (
                    f"dominant guard zone: {dominant_zone} (n={int(zone_counts.iloc[0])})"
                )
                zones = zones[~zones["zone_region_id"].astype(str).eq(dominant_zone)]
        ax.scatter(
            zones["UMAP-1"],
            zones["UMAP-2"],
            s=55,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="pass-through/guard zone",
        )
        if dominant_zone_label:
            ax.text(
                0.01,
                0.01,
                dominant_zone_label,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
            )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Multi-scale traversal UMAP overlay")
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
