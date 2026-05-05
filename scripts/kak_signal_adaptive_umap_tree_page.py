#!/usr/bin/env python3
"""Create whole-UMAP -> KAK sub-UMAP -> tree pages for KAK signal-adaptive runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scripts.kak_mp_tree_method_test import (
    kak_mp_gene_scores,
    load_matrix,
    mp_bounds,
    standardize_active_features,
)

GLOBAL_X = "UMAP axis 1: main GO-pattern gradient"
GLOBAL_Y = "UMAP axis 2: secondary GO-pattern gradient"
GLOBAL_Z = "UMAP axis 3: fine GO-pattern gradient"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KAK signal-adaptive whole-UMAP/sub-UMAP/tree plots."
    )
    parser.add_argument("--input", type=Path, required=True, help="Full GO feature matrix TSV.")
    parser.add_argument("--global-umap", type=Path, required=True, help="Existing global UMAP TSV.")
    parser.add_argument("--kak-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_global_umap(path: Path) -> pd.DataFrame:
    coords = pd.read_csv(path, sep="\t")
    required = {"gene", GLOBAL_X, GLOBAL_Y, GLOBAL_Z}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(f"Global UMAP missing columns: {sorted(missing)}")
    return coords[["gene", GLOBAL_X, GLOBAL_Y, GLOBAL_Z]].copy()


def load_assignments(kak_results_dir: Path) -> dict[str, pd.DataFrame]:
    assignment_dir = kak_results_dir / "assignments"
    assignments: dict[str, pd.DataFrame] = {}
    for path in sorted(assignment_dir.glob("*_cluster_assignments.csv")):
        run_id = path.name.removesuffix("_cluster_assignments.csv")
        df = pd.read_csv(path)
        if "gene" not in df.columns:
            df = df.rename(columns={df.columns[0]: "gene"})
        assignments[run_id] = df
    if not assignments:
        raise ValueError(f"No assignment files found in {assignment_dir}")
    return assignments


def load_summary(kak_results_dir: Path) -> pd.DataFrame:
    summary_path = kak_results_dir / "kak_mp_tree_method_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return pd.read_csv(summary_path)


def color_values(labels: pd.Series) -> tuple[np.ndarray, dict[int, int]]:
    ordered = labels.astype(int)
    sizes = ordered.value_counts()
    rank_by_cluster = {int(cluster_id): rank for rank, cluster_id in enumerate(sizes.index)}
    return ordered.map(rank_by_cluster).to_numpy(), rank_by_cluster


def compute_kak_block_coordinates(
    data: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    standardized, n_active = standardize_active_features(data)
    eigenvalues, eigenvectors = kak_mp_gene_scores(standardized)
    mp_lower, mp_upper = mp_bounds(standardized.shape[0], n_active)

    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "correlation_eigenvalue": eigenvalues,
            "mp_lower": mp_lower,
            "mp_upper": mp_upper,
            "mp_regime": np.where(eigenvalues > mp_upper, "above-MP coupled", "inside-MP bulk"),
        }
    )

    raw_coords_by_run: dict[str, np.ndarray] = {}
    linkage_by_run: dict[str, np.ndarray] = {}
    for _, row in summary.iterrows():
        run_id = str(row["run_id"])
        start = int(row["component_start"]) - 1
        end = int(row["component_end"])
        indices = np.arange(start, end)
        coords = eigenvectors[:, indices] * np.sqrt(np.maximum(eigenvalues[indices], 0.0))
        coords = np.nan_to_num(coords)
        raw_coords_by_run[run_id] = coords
        linkage_by_run[run_id] = linkage(pdist(coords, metric="euclidean"), method="average")

    return spectrum, raw_coords_by_run, linkage_by_run


def fit_umap(coords: np.ndarray, *, n_components: int, random_state: int) -> np.ndarray:
    if coords.shape[1] == 1:
        # UMAP can handle this, but a tiny jitter prevents duplicate-neighbor degeneracy.
        rng = np.random.default_rng(random_state)
        coords = np.column_stack([coords[:, 0], rng.normal(0.0, 1e-6, size=coords.shape[0])])
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=18,
        min_dist=0.05,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(coords)


def draw_block_page(
    *,
    run_id: str,
    assignment: pd.DataFrame,
    global_umap: pd.DataFrame,
    sub_umap_2d: np.ndarray,
    linkage_matrix: np.ndarray,
    summary_row: pd.Series,
    path: Path,
) -> None:
    merged = global_umap.merge(assignment, on="gene", how="inner")
    colors, _ = color_values(merged["cluster_id"])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.6),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.3]},
    )

    axes[0].scatter(
        merged[GLOBAL_X],
        merged[GLOBAL_Y],
        c=colors,
        cmap="tab20",
        s=14,
        alpha=0.82,
        linewidths=0,
    )
    axes[0].set_title("Whole-data UMAP: same clusters mapped globally")
    axes[0].set_xlabel("Global UMAP axis 1: main GO-pattern gradient")
    axes[0].set_ylabel("Global UMAP axis 2: secondary GO-pattern gradient")

    axes[1].scatter(
        sub_umap_2d[:, 0],
        sub_umap_2d[:, 1],
        c=colors,
        cmap="tab20",
        s=14,
        alpha=0.82,
        linewidths=0,
    )
    axes[1].set_title("KAK-cosine sub-UMAP: within-block separation")
    axes[1].set_xlabel("Sub-UMAP axis 1: KAK-coupled GO direction")
    axes[1].set_ylabel("Sub-UMAP axis 2: KAK-coupled GO direction")

    dendrogram(
        linkage_matrix,
        ax=axes[2],
        no_labels=True,
        truncate_mode="lastp",
        p=42,
        color_threshold=None,
    )
    axes[2].set_title("Tree used by our gated decomposition")
    axes[2].set_xlabel("Collapsed gene branches")
    axes[2].set_ylabel("KAK subspace distance")

    title = (
        f"{run_id}: {int(summary_row['n_clusters'])} clusters, "
        f"largest {int(summary_row['largest_cluster_size'])}/703 genes, "
        f"modes {int(summary_row['component_start'])}-{int(summary_row['component_end'])}"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def top_cluster_sizes(labels: pd.Series, *, limit: int = 6) -> str:
    sizes = labels.astype(int).value_counts().head(limit)
    return ", ".join(f"C{int(cluster_id)}={int(size)}" for cluster_id, size in sizes.items())


def draw_consolidated_page(
    *,
    assignments: dict[str, pd.DataFrame],
    global_umap: pd.DataFrame,
    sub_umap_2d_by_run: dict[str, np.ndarray],
    linkage_by_run: dict[str, np.ndarray],
    summary: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Draw the same overview contract as the earlier adaptive KAK page.

    Every row is one KAK signal-adaptive block. The tree column includes a
    cluster strip in dendrogram leaf order, which is the critical mapping from
    TreeDecomposition assignments back onto the tree topology.
    """

    run_ids = [run_id for run_id in summary["run_id"].astype(str) if run_id in assignments]
    if not run_ids:
        raise ValueError("No overlapping run ids between summary and assignments")

    summary_by_run = summary.set_index("run_id")
    height = max(8.0, 3.25 * len(run_ids) + 1.0)
    fig = plt.figure(figsize=(20.0, height))
    grid = fig.add_gridspec(
        nrows=len(run_ids),
        ncols=4,
        width_ratios=[1.0, 1.0, 1.35, 0.95],
        hspace=0.55,
        wspace=0.28,
    )

    cmap = plt.get_cmap("tab20")

    for row_index, run_id in enumerate(run_ids):
        assignment = assignments[run_id]
        merged = global_umap.merge(assignment, on="gene", how="inner")
        colors, _ = color_values(merged["cluster_id"])
        row = summary_by_run.loc[run_id]

        ax_global = fig.add_subplot(grid[row_index, 0])
        ax_sub = fig.add_subplot(grid[row_index, 1])
        tree_grid = grid[row_index, 2].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[0.82, 0.18],
            hspace=0.03,
        )
        ax_tree = fig.add_subplot(tree_grid[0, 0])
        ax_strip = fig.add_subplot(tree_grid[1, 0])
        ax_text = fig.add_subplot(grid[row_index, 3])

        ax_global.scatter(
            merged[GLOBAL_X],
            merged[GLOBAL_Y],
            c=colors,
            cmap=cmap,
            s=13,
            alpha=0.84,
            linewidths=0,
        )
        ax_global.set_title("Whole-data UMAP", fontsize=10)
        ax_global.set_xlabel("Main full-data GO-pattern gradient", fontsize=9)
        ax_global.set_ylabel("Secondary full-data GO-pattern gradient", fontsize=9)
        ax_global.tick_params(labelsize=8)

        sub_umap_2d = sub_umap_2d_by_run[run_id]
        ax_sub.scatter(
            sub_umap_2d[:, 0],
            sub_umap_2d[:, 1],
            c=colors,
            cmap=cmap,
            s=13,
            alpha=0.84,
            linewidths=0,
        )
        ax_sub.set_title("KAK block sub-UMAP", fontsize=10)
        ax_sub.set_xlabel("Within-block GO-rotation gradient", fontsize=9)
        ax_sub.set_ylabel("Second within-block GO-rotation gradient", fontsize=9)
        ax_sub.tick_params(labelsize=8)

        dendrogram_result = dendrogram(
            linkage_by_run[run_id],
            ax=ax_tree,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#343434",
            link_color_func=lambda _: "#343434",
        )
        ax_tree.set_title("Tree with decomposition labels", fontsize=10)
        ax_tree.set_ylabel("KAK subspace distance", fontsize=9)
        ax_tree.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_tree.tick_params(axis="y", labelsize=8)

        leaf_order = np.asarray(dendrogram_result["leaves"], dtype=int)
        strip_values = colors[leaf_order]
        n_colors = max(int(strip_values.max()) + 1, 1)
        strip_cmap = ListedColormap([cmap(i % cmap.N) for i in range(n_colors)])
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
        ax_strip.set_xlabel("Genes ordered by tree leaves; color = our cluster assignment", fontsize=8)
        for spine in ax_strip.spines.values():
            spine.set_visible(False)

        silhouette = row["silhouette_in_kak_subspace"]
        silhouette_text = "not defined" if pd.isna(silhouette) else f"{float(silhouette):.3f}"
        metrics = "\n".join(
            [
                f"{run_id}",
                "",
                "Decomposition:",
                f"KAK/MP signal block {int(row['component_start'])}-{int(row['component_end'])}",
                f"Dimensions: {int(row['subspace_dimensions'])}",
                f"Energy in block: {float(row['energy_fraction']):.3f}",
                "",
                "TreeDecomposition result:",
                f"Clusters: {int(row['n_clusters'])}",
                (
                    "Largest cluster: "
                    f"{int(row['largest_cluster_size'])}/703 "
                    f"({100.0 * float(row['largest_cluster_fraction']):.1f}%)"
                ),
                f"Subspace silhouette: {silhouette_text}",
                "",
                "Largest groups:",
                top_cluster_sizes(merged["cluster_id"]),
            ]
        )
        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            metrics,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            linespacing=1.25,
        )

    fig.suptitle(
        "KAK signal-adaptive method: whole UMAP -> subspace UMAP -> assigned tree",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(top=0.94, bottom=0.035, left=0.055, right=0.985)
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def make_dropdown_html(
    *,
    global_umap: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    sub_umap_3d_by_run: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig = go.Figure()
    trace_groups: dict[str, list[int]] = {}

    for run_index, (run_id, assignment) in enumerate(assignments.items()):
        merged = global_umap.merge(assignment, on="gene", how="inner")
        coords = sub_umap_3d_by_run[run_id]
        trace_groups[run_id] = []
        for cluster_id, group in merged.groupby("cluster_id", sort=True):
            idx = group.index.to_numpy()
            visible = run_index == 0
            trace_index = len(fig.data)
            trace_groups[run_id].append(trace_index)
            fig.add_trace(
                go.Scatter3d(
                    x=coords[idx, 0],
                    y=coords[idx, 1],
                    z=coords[idx, 2],
                    mode="markers",
                    visible=visible,
                    name=f"{run_id}: cluster {cluster_id}",
                    text=group["gene"],
                    customdata=np.stack(
                        [
                            group["cluster_id"].to_numpy(),
                            group["cluster_root"].astype(str).to_numpy(),
                            group["cluster_size"].to_numpy(),
                        ],
                        axis=1,
                    ),
                    marker={"size": 4, "opacity": 0.82},
                    hovertemplate=(
                        "gene=%{text}<br>"
                        "cluster=%{customdata[0]}<br>"
                        "cluster root=%{customdata[1]}<br>"
                        "cluster size=%{customdata[2]}<br>"
                        "Sub-UMAP axis 1=%{x:.3f}<br>"
                        "Sub-UMAP axis 2=%{y:.3f}<br>"
                        "Sub-UMAP axis 3=%{z:.3f}"
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
                    {
                        "title": (
                            "KAK-cosine sub-UMAP with our TreeDecomposition clusters: "
                            f"{run_id}"
                        )
                    },
                ],
            }
        )

    fig.update_layout(
        title=(
            "KAK-cosine sub-UMAP with our TreeDecomposition clusters: "
            f"{next(iter(assignments))}"
        ),
        scene={
            "xaxis_title": "Sub-UMAP axis 1: KAK-coupled GO direction",
            "yaxis_title": "Sub-UMAP axis 2: KAK-coupled GO direction",
            "zaxis_title": "Sub-UMAP axis 3: fine KAK-coupled GO direction",
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
    global_umap = load_global_umap(args.global_umap)
    assignments = load_assignments(args.kak_results_dir)
    summary = load_summary(args.kak_results_dir)
    summary_by_run = summary.set_index("run_id")

    spectrum, raw_coords_by_run, linkage_by_run = compute_kak_block_coordinates(data, summary)
    spectrum.to_csv(out / "kak_signal_adaptive_spectrum_for_plot.csv", index=False)

    coordinate_frames: list[pd.DataFrame] = []
    sub_umap_2d_by_run: dict[str, np.ndarray] = {}
    sub_umap_3d_by_run: dict[str, np.ndarray] = {}
    png_paths: list[Path] = []

    for i, run_id in enumerate(assignments):
        coords = raw_coords_by_run[run_id]
        sub2 = fit_umap(coords, n_components=2, random_state=100 + i)
        sub3 = fit_umap(coords, n_components=3, random_state=200 + i)
        sub_umap_2d_by_run[run_id] = sub2
        sub_umap_3d_by_run[run_id] = sub3

        assignment = assignments[run_id]
        coord_df = pd.DataFrame(
            {
                "run_id": run_id,
                "gene": data.index.to_numpy(),
                "Sub-UMAP axis 1: KAK-coupled GO direction": sub2[:, 0],
                "Sub-UMAP axis 2: KAK-coupled GO direction": sub2[:, 1],
                "Sub-UMAP 3D axis 1: KAK-coupled GO direction": sub3[:, 0],
                "Sub-UMAP 3D axis 2: KAK-coupled GO direction": sub3[:, 1],
                "Sub-UMAP 3D axis 3: fine KAK-coupled GO direction": sub3[:, 2],
            }
        ).merge(assignment, on="gene", how="left")
        coordinate_frames.append(coord_df)

        png_path = out / "per_block_png" / f"{run_id}_umap_subumap_tree.png"
        draw_block_page(
            run_id=run_id,
            assignment=assignment,
            global_umap=global_umap,
            sub_umap_2d=sub2,
            linkage_matrix=linkage_by_run[run_id],
            summary_row=summary_by_run.loc[run_id],
            path=png_path,
        )
        png_paths.append(png_path)

    coords_long = pd.concat(coordinate_frames, ignore_index=True)
    coords_long.to_csv(out / "kak_signal_adaptive_subumap_coordinates_long.csv", index=False)

    with PdfPages(out / "kak_signal_adaptive_umap_subumap_tree_pages.pdf") as pdf:
        for png_path in png_paths:
            image = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(18, 5.6))
            ax.imshow(image)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig)
            plt.close(fig)

    # Make a compact first-page PNG from the best largest-core reduction.
    best_run = (
        summary.sort_values(["largest_cluster_fraction", "n_clusters"], ascending=[True, False])
        .iloc[0]["run_id"]
    )
    best_png = out / "kak_signal_adaptive_best_umap_subumap_tree_page.png"
    best_source = out / "per_block_png" / f"{best_run}_umap_subumap_tree.png"
    best_png.write_bytes(best_source.read_bytes())

    draw_consolidated_page(
        assignments=assignments,
        global_umap=global_umap,
        sub_umap_2d_by_run=sub_umap_2d_by_run,
        linkage_by_run=linkage_by_run,
        summary=summary,
        output_png=out / "kak_signal_adaptive_global_vs_subumap_tree_page.png",
        output_pdf=out / "kak_signal_adaptive_global_vs_subumap_tree_page.pdf",
    )

    make_dropdown_html(
        global_umap=global_umap,
        assignments=assignments,
        sub_umap_3d_by_run=sub_umap_3d_by_run,
        output_path=out / "kak_signal_adaptive_subumap_3d_dropdown.html",
    )

    readme = [
        "# KAK Signal-Adaptive UMAP -> Sub-UMAP -> Tree Pages",
        "",
        "This folder maps the KAK signal-adaptive TreeDecomposition results into the same visual story used by the previous analyses.",
        "",
        "Each page shows:",
        "",
        "1. Whole-data UMAP with that block's gated clusters.",
        "2. KAK-cosine block-specific sub-UMAP with the same clusters.",
        "3. The KAK block tree/dendrogram that was fed into `TreeDecomposition`.",
        "",
        "Files:",
        "",
        "- `kak_signal_adaptive_best_umap_subumap_tree_page.png`: best single-page summary by largest-core reduction.",
        "- `kak_signal_adaptive_global_vs_subumap_tree_page.png`: consolidated all-block page matching the previous global/subspace/tree layout.",
        "- `kak_signal_adaptive_global_vs_subumap_tree_page.pdf`: PDF version of the consolidated all-block page.",
        "- `kak_signal_adaptive_umap_subumap_tree_pages.pdf`: one page per KAK signal-adaptive block.",
        "- `kak_signal_adaptive_subumap_3d_dropdown.html`: interactive 3D sub-UMAP dropdown across blocks.",
        "- `kak_signal_adaptive_subumap_coordinates_long.csv`: sub-UMAP coordinates and cluster assignments.",
        "- `per_block_png/`: individual PNG pages.",
        "",
        "Axis names are written for non-experts: global axes describe full GO-pattern gradients, while sub-UMAP axes describe within-block KAK-coupled GO directions.",
    ]
    (out / "README.md").write_text("\n".join(readme))


if __name__ == "__main__":
    main()
