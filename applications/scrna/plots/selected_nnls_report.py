"""Build the selected scRNA NNLS report with shared UMAP/tree colors."""
# ruff: noqa: I001

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image

from applications.scrna.plots.report_helpers import plot_cluster_size_bars

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "raw/assets/benchmark-results"
SELECTED = (
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001"
)
METHOD_TITLE = "TBS adaptive diffusion recomputed-NNLS branch-time (k90, alpha=0.01, edge=0.001)"

DATASETS = [
    {
        "key": "adult",
        "title": "Adult pancreas scRNA benchmark",
        "dir": BASE / "pancreas_scrna_cluster_benchmark_20260623",
        "selected": SELECTED,
    },
    {
        "key": "goncalves",
        "title": "Goncalves fetal pancreas progenitor benchmark",
        "dir": BASE / "goncalves_fetal_pancreas_progenitor_benchmark_20260624",
        "selected": SELECTED,
    },
]

OUT_PDF = BASE / "scrna_selected_adaptive_diffusion_nnls_full_report.pdf"
OUT_ADULT = BASE / "scrna_selected_adaptive_diffusion_nnls_adult_colored_tree_page.png"
OUT_GONCALVES = BASE / "scrna_selected_adaptive_diffusion_nnls_goncalves_colored_tree_page.png"
OUT_ADULT_RADIAL = BASE / "scrna_selected_adaptive_diffusion_nnls_adult_radial_tree_page.png"
OUT_GONCALVES_RADIAL = (
    BASE / "scrna_selected_adaptive_diffusion_nnls_goncalves_radial_tree_page.png"
)
OUT_ADULT_FULL_RADIAL = (
    BASE / "scrna_selected_adaptive_diffusion_nnls_adult_full_radial_tree_page.png"
)
OUT_GONCALVES_FULL_RADIAL = (
    BASE / "scrna_selected_adaptive_diffusion_nnls_goncalves_full_radial_tree_page.png"
)
OUT_SUMMARY = BASE / "scrna_selected_adaptive_diffusion_nnls_cluster_boundary_summary.csv"

PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
    "#969696",
    "#9ecae1",
    "#a1d99b",
    "#bcbddc",
    "#fdae6b",
    "#c7e9c0",
    "#6baed6",
    "#fd8d3c",
    "#74c476",
    "#9e9ac8",
    "#fdd0a2",
    "#c994c7",
    "#c6dbef",
    "#e7ba52",
    "#ad494a",
    "#a55194",
    "#6b6ecf",
    "#b5cf6b",
    "#cedb9c",
    "#e7cb94",
    "#e7969c",
    "#de9ed6",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]
CELLTYPE_PALETTE = PALETTE[10:] + PALETTE[:10]


def sorted_labels(values):
    labels = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(labels, key=lambda x: int(x))
    except Exception:
        return sorted(labels, key=lambda x: str(x))


def make_cluster_palette(labels):
    labels = sorted_labels(labels)
    if len(labels) > len(PALETTE):
        cmap = plt.get_cmap("hsv")
        colors = [mcolors.to_hex(cmap(i / len(labels))) for i in range(len(labels))]
    else:
        colors = PALETTE
    return {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}


def make_label_palette(labels):
    labels = sorted(labels, key=lambda x: str(x))
    return {lab: CELLTYPE_PALETTE[i % len(CELLTYPE_PALETTE)] for i, lab in enumerate(labels)}


def assign_tree_coordinates(children, root, branch_of):
    x = {}
    leaf_counter = [0]

    def assign_x(node):
        if not children.get(node):
            x[node] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for child in children[node]:
                assign_x(child)
            x[node] = float(np.mean([x[child] for child in children[node]]))
        return x[node]

    assign_x(root)

    y = {root: 0.0}
    depth = {root: 0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            length = branch_of.get((node, child), 0.0)
            length = 0.0 if pd.isna(length) else max(float(length), 0.0)
            y[child] = y[node] + length
            depth[child] = depth[node] + 1
            stack.append(child)
    return x, y, depth, leaf_counter[0]


def strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_generated_at(fig, generated_at):
    fig.text(
        0.99,
        0.01,
        f"Generated at: {generated_at}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6b7280",
    )


def plot_umap(ax, df, color_col, palette, title, annotate_clusters=False, point_size=5):
    colors = df[color_col].map(palette).fillna("#999999")
    ax.scatter(
        df["umap1"],
        df["umap2"],
        c=colors,
        s=point_size,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11, weight="bold", pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    strip_axes(ax)
    if not annotate_clusters:
        return
    centroids = df.groupby(color_col, observed=True)[["umap1", "umap2"]].median()
    for lab, row in centroids.iterrows():
        ax.text(
            row["umap1"],
            row["umap2"],
            f"C{int(lab)}",
            ha="center",
            va="center",
            fontsize=5.5 if len(centroids) > 30 else 6.5,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            clip_on=True,
        )


def build_boundary_tree_info(cfg, assign_df):
    trace = pd.read_csv(cfg["dir"] / f"{SELECTED}_traversal_trace.csv")
    edges = pd.read_csv(cfg["dir"] / f"{SELECTED}_tree_edges.csv")
    selected = cfg["selected"]
    cell_to_cluster = assign_df.set_index("cell_id")[selected]
    cell_to_type = assign_df.set_index("cell_id")["celltype"]

    boundaries = trace.loc[
        trace["final_boundary"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    boundary_records = []
    for _, row in boundaries.iterrows():
        cells = json.loads(row["descendant_leaf_signature"])
        clusters = cell_to_cluster.reindex(cells).dropna()
        types = cell_to_type.reindex(cells).dropna()
        counts = clusters.value_counts()
        cluster = int(counts.index[0])
        purity = float(counts.iloc[0]) / float(len(clusters))
        type_counts = types.value_counts()
        top_celltype = str(type_counts.index[0]) if len(type_counts) else "unknown"
        top_celltype_fraction = (
            float(type_counts.iloc[0]) / float(len(types)) if len(type_counts) else np.nan
        )
        boundary_records.append(
            {
                "node_id": row["node_id"],
                "cluster": cluster,
                "n_cells_boundary": int(row["n_descendant_leaves"]),
                "cluster_purity_in_boundary": purity,
                "top_celltype_boundary": top_celltype,
                "top_celltype_fraction_boundary": top_celltype_fraction,
            }
        )
    boundary_df = pd.DataFrame(boundary_records)
    boundary_nodes = set(boundary_df["node_id"])
    node_to_cluster = dict(zip(boundary_df["node_id"], boundary_df["cluster"]))
    node_to_size = dict(zip(boundary_df["node_id"], boundary_df["n_cells_boundary"]))

    parent_of = dict(zip(edges["child"], edges["parent"]))
    branch_of = dict(zip(zip(edges["parent"], edges["child"]), edges["branch_length"].fillna(0.0)))
    roots = sorted(set(edges["parent"]) - set(edges["child"]))
    if not roots:
        raise RuntimeError(f"No root found for {cfg['key']}")
    root = roots[0]

    required = {root} | boundary_nodes
    for boundary in boundary_nodes:
        node = boundary
        seen = set()
        while node in parent_of and node not in seen:
            seen.add(node)
            parent = parent_of[node]
            required.add(parent)
            node = parent

    children = defaultdict(list)
    for _, row in edges.iterrows():
        parent, child = row["parent"], row["child"]
        if parent in required and child in required:
            children[parent].append(child)
    for boundary in boundary_nodes:
        children[boundary] = []

    desc_clusters = {}

    def collect_clusters(node):
        if node in desc_clusters:
            return desc_clusters[node]
        if node in node_to_cluster and not children[node]:
            vals = {node_to_cluster[node]}
        else:
            vals = set()
            for child in children.get(node, []):
                vals |= collect_clusters(child)
        desc_clusters[node] = vals
        return vals

    collect_clusters(root)
    order_index = {lab: i for i, lab in enumerate(sorted_labels(assign_df[selected].unique()))}

    def min_order(node):
        vals = collect_clusters(node)
        return min(order_index.get(v, 10**9) for v in vals) if vals else 10**9

    for node in list(children):
        children[node].sort(key=min_order)

    x, y, depth, _n_leaves = assign_tree_coordinates(children, root, branch_of)

    y_values = np.array(list(y.values()), dtype=float)
    if not np.isfinite(y_values).all() or (np.nanmax(y_values) - np.nanmin(y_values) < 1e-9):
        y = {node: float(depth.get(node, 0)) for node in required}
        y_label = "topological depth"
    else:
        y_label = "cumulative NNLS branch time"

    return {
        "root": root,
        "children": children,
        "x": x,
        "y": y,
        "boundary_df": boundary_df,
        "node_to_cluster": node_to_cluster,
        "node_to_size": node_to_size,
        "desc_clusters": desc_clusters,
        "y_label": y_label,
    }


def plot_compact_tree(ax, info, cluster_palette, title):
    children = info["children"]
    x = info["x"]
    y = info["y"]
    desc_clusters = info["desc_clusters"]
    node_to_cluster = info["node_to_cluster"]
    node_to_size = info["node_to_size"]

    def draw_edges(node):
        for child in children.get(node, []):
            child_clusters = desc_clusters.get(child, set())
            if len(child_clusters) == 1:
                color = cluster_palette[next(iter(child_clusters))]
                linewidth = 1.7
                alpha = 0.92
            else:
                color = "#9ca3af"
                linewidth = 0.75
                alpha = 0.72
            ax.plot(
                [x[node], x[node]], [-y[node], -y[child]], color=color, lw=linewidth, alpha=alpha
            )
            ax.plot(
                [x[node], x[child]], [-y[child], -y[child]], color=color, lw=linewidth, alpha=alpha
            )
            draw_edges(child)

    draw_edges(info["root"])

    internal = [node for node in x if node not in node_to_cluster]
    if internal:
        ax.scatter(
            [x[node] for node in internal],
            [-y[node] for node in internal],
            s=8,
            color="#4b5563",
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )

    bnodes = list(node_to_cluster)
    sizes = np.array([node_to_size[node] for node in bnodes], dtype=float)
    marker_sizes = 28 + 8.5 * np.sqrt(sizes)
    ax.scatter(
        [x[node] for node in bnodes],
        [-y[node] for node in bnodes],
        s=marker_sizes,
        c=[cluster_palette[node_to_cluster[node]] for node in bnodes],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    for node in bnodes:
        ax.text(
            x[node],
            -y[node],
            f"C{int(node_to_cluster[node])}",
            fontsize=5.7 if len(bnodes) > 30 else 6.5,
            ha="center",
            va="center",
            color="black",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )

    ax.set_title(title, fontsize=11, weight="bold", pad=8)
    ax.set_xlabel("selected terminal boundaries, ordered by cluster ID", fontsize=8)
    ax.set_ylabel(info["y_label"], fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.45)
    ax.margins(x=0.03, y=0.08)


def _polar(angle, radius):
    return radius * np.cos(angle), radius * np.sin(angle)


def _angle_arc(start, end, n=28):
    delta = (end - start + np.pi) % (2 * np.pi) - np.pi
    return start + np.linspace(0, delta, n)


def plot_radial_tree(ax, info, cluster_palette, title):
    children = info["children"]
    x = info["x"]
    y = info["y"]
    desc_clusters = info["desc_clusters"]
    node_to_cluster = info["node_to_cluster"]
    node_to_size = info["node_to_size"]
    max_x = max(x.values()) if x else 1.0
    max_y = max(y.values()) if y else 1.0
    max_y = max(max_y, 1e-12)

    angles = {}
    radii = {}
    for node, xpos in x.items():
        angles[node] = (2 * np.pi * (xpos / max(max_x + 1, 1))) - (np.pi / 2)
        radii[node] = y[node] / max_y

    def circular_mean(nodes):
        vals = np.array([angles[node] for node in nodes], dtype=float)
        return float(np.arctan2(np.sin(vals).mean(), np.cos(vals).mean()))

    for node in sorted(x, key=lambda key: y[key], reverse=True):
        if children.get(node):
            leaves = []
            stack = list(children[node])
            while stack:
                child = stack.pop()
                if children.get(child):
                    stack.extend(children[child])
                else:
                    leaves.append(child)
            if leaves:
                angles[node] = circular_mean(leaves)

    def draw_edges(node):
        for child in children.get(node, []):
            child_clusters = desc_clusters.get(child, set())
            if len(child_clusters) == 1:
                color = cluster_palette[next(iter(child_clusters))]
                linewidth = 1.55
                alpha = 0.92
            else:
                color = "#9ca3af"
                linewidth = 0.75
                alpha = 0.72

            parent_angle = angles[node]
            child_angle = angles[child]
            parent_radius = radii[node]
            child_radius = radii[child]

            arc_angles = _angle_arc(parent_angle, child_angle)
            arc_x, arc_y = _polar(arc_angles, parent_radius)
            ax.plot(arc_x, arc_y, color=color, lw=linewidth, alpha=alpha)
            radial_r = np.linspace(parent_radius, child_radius, 28)
            radial_x, radial_y = _polar(child_angle, radial_r)
            ax.plot(radial_x, radial_y, color=color, lw=linewidth, alpha=alpha)
            draw_edges(child)

    draw_edges(info["root"])

    internal = [node for node in x if node not in node_to_cluster]
    if internal:
        internal_xy = [_polar(angles[node], radii[node]) for node in internal]
        ax.scatter(
            [xy[0] for xy in internal_xy],
            [xy[1] for xy in internal_xy],
            s=8,
            color="#4b5563",
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )

    bnodes = list(node_to_cluster)
    sizes = np.array([node_to_size[node] for node in bnodes], dtype=float)
    marker_sizes = 28 + 8.5 * np.sqrt(sizes)
    boundary_xy = [_polar(angles[node], radii[node]) for node in bnodes]
    ax.scatter(
        [xy[0] for xy in boundary_xy],
        [xy[1] for xy in boundary_xy],
        s=marker_sizes,
        c=[cluster_palette[node_to_cluster[node]] for node in bnodes],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )

    for node in bnodes:
        angle = angles[node]
        label_x, label_y = _polar(angle, 1.075)
        degrees = np.degrees(angle)
        rotation = degrees
        ha = "left"
        if 90 < (degrees % 360) < 270:
            rotation = degrees + 180
            ha = "right"
        ax.text(
            label_x,
            label_y,
            f"C{int(node_to_cluster[node])}",
            fontsize=5.9 if len(bnodes) > 30 else 7,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            color="#111827",
        )

    ax.set_title(title, fontsize=12, weight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    strip_axes(ax)
    ax.text(
        0.01,
        0.01,
        f"radius = {info['y_label']}; colors = selected cluster IDs",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
        ha="left",
        va="bottom",
    )


def _leaf_index(node):
    if isinstance(node, str) and node.startswith("L"):
        try:
            return int(node[1:])
        except ValueError:
            return 10**12
    return 10**12


def build_full_tree_info(cfg, assign_df):
    edges = pd.read_csv(cfg["dir"] / f"{SELECTED}_tree_edges.csv")
    selected = cfg["selected"]
    children = defaultdict(list)
    branch_of = {}
    for _, row in edges.iterrows():
        parent, child = row["parent"], row["child"]
        children[parent].append(child)
        branch = row["branch_length"]
        branch_of[(parent, child)] = 0.0 if pd.isna(branch) else max(float(branch), 0.0)

    roots = sorted(set(edges["parent"]) - set(edges["child"]))
    if not roots:
        raise RuntimeError(f"No root found for {cfg['key']}")
    root = roots[0]

    leaf_cluster = {}
    for node in set(edges["child"]) - set(edges["parent"]):
        idx = _leaf_index(node)
        if idx < len(assign_df):
            leaf_cluster[node] = int(assign_df.iloc[idx][selected])

    cluster_counts = {}

    def count_clusters(node):
        if node in cluster_counts:
            return cluster_counts[node]
        if node in leaf_cluster:
            counts = {leaf_cluster[node]: 1}
        else:
            counts = defaultdict(int)
            for child in children.get(node, []):
                for cluster, count in count_clusters(child).items():
                    counts[cluster] += count
            counts = dict(counts)
        cluster_counts[node] = counts
        return counts

    count_clusters(root)

    def child_sort_key(node):
        counts = count_clusters(node)
        if not counts:
            return (10**12, _leaf_index(node))
        dominant, n = max(counts.items(), key=lambda item: item[1])
        purity = n / max(sum(counts.values()), 1)
        return (dominant, -purity, _leaf_index(node))

    for node in list(children):
        children[node].sort(key=child_sort_key)

    x, y, _depth, n_leaves = assign_tree_coordinates(children, root, branch_of)

    return {
        "root": root,
        "children": children,
        "x": x,
        "y": y,
        "leaf_cluster": leaf_cluster,
        "cluster_counts": cluster_counts,
        "n_leaves": n_leaves,
    }


def plot_full_radial_tree(ax, info, cluster_palette, title):
    children = info["children"]
    x = info["x"]
    y = info["y"]
    leaf_cluster = info["leaf_cluster"]
    cluster_counts = info["cluster_counts"]
    max_x = max(x.values()) if x else 1.0
    max_y = max(y.values()) if y else 1.0
    max_y = max(max_y, 1e-12)

    angles = {
        node: (2 * np.pi * (xpos / max(max_x + 1, 1))) - (np.pi / 2) for node, xpos in x.items()
    }
    radii = {node: y[node] / max_y for node in x}

    def edge_style(child):
        counts = cluster_counts.get(child, {})
        if len(counts) == 1:
            color = cluster_palette[next(iter(counts))]
            return color, 0.34, 0.9
        return "#9ca3af", 0.24, 0.28

    def draw_edges(node):
        for child in children.get(node, []):
            color, linewidth, alpha = edge_style(child)
            parent_angle = angles[node]
            child_angle = angles[child]
            parent_radius = radii[node]
            child_radius = radii[child]
            arc_angles = _angle_arc(parent_angle, child_angle, n=8)
            arc_x, arc_y = _polar(arc_angles, parent_radius)
            ax.plot(arc_x, arc_y, color=color, lw=linewidth, alpha=alpha)
            radial_r = np.linspace(parent_radius, child_radius, 8)
            radial_x, radial_y = _polar(child_angle, radial_r)
            ax.plot(radial_x, radial_y, color=color, lw=linewidth, alpha=alpha)
            draw_edges(child)

    draw_edges(info["root"])

    leaves = sorted(leaf_cluster, key=lambda node: x[node])
    leaf_xy = [_polar(angles[node], radii[node]) for node in leaves]
    ax.scatter(
        [xy[0] for xy in leaf_xy],
        [xy[1] for xy in leaf_xy],
        s=2.8,
        c=[cluster_palette[leaf_cluster[node]] for node in leaves],
        linewidths=0,
        alpha=0.95,
        zorder=4,
    )

    ax.set_title(title, fontsize=12, weight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    strip_axes(ax)
    ax.text(
        0.01,
        0.01,
        f"all {len(leaves):,} leaves; radius = cumulative NNLS branch time; colors match the labeled UMAP",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
        ha="left",
        va="bottom",
    )


def cluster_summary(assign_df, selected):
    size = assign_df[selected].value_counts().rename("n_cells_assignment")
    ct = assign_df.groupby([selected, "celltype"], observed=True).size().rename("n").reset_index()
    rows = []
    for label, group in ct.groupby(selected, observed=True):
        group = group.sort_values("n", ascending=False)
        total = int(size.loc[label])
        top = group.iloc[0]
        rows.append(
            {
                "cluster": int(label),
                "n_cells_assignment": total,
                "top_celltype_assignment": str(top["celltype"]),
                "top_celltype_fraction_assignment": float(top["n"]) / total,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def plot_size_bars(ax, summary, cluster_palette, title):
    plot_cluster_size_bars(ax, summary, cluster_palette, title)


def selected_dataset_page(cfg, save_path=None, generated_at=None):
    assign = pd.read_csv(cfg["dir"] / "method_assignments.csv")
    selected = cfg["selected"]
    cluster_palette = make_cluster_palette(assign[selected].unique())
    celltype_palette = make_label_palette(assign["celltype"].fillna("unknown").unique())
    tree_info = build_boundary_tree_info(cfg, assign)
    summary = cluster_summary(assign, selected)

    fig = plt.figure(figsize=(17, 11), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.38],
        height_ratios=[1.03, 0.97],
        wspace=0.26,
        hspace=0.28,
    )
    ax_celltype = fig.add_subplot(gs[0, 0])
    ax_cluster = fig.add_subplot(gs[0, 1])
    ax_tree = fig.add_subplot(gs[:, 2])
    ax_sizes = fig.add_subplot(gs[1, :2])

    plot_umap(ax_celltype, assign, "celltype", celltype_palette, "Reference labels on UMAP")
    plot_umap(
        ax_cluster, assign, selected, cluster_palette, "Selected NNLS-TBS clusters on UMAP", True
    )
    plot_compact_tree(
        ax_tree, tree_info, cluster_palette, "Compact selected-boundary tree, same colors"
    )
    plot_size_bars(
        ax_sizes, summary, cluster_palette, "Selected clusters: size and dominant reference label"
    )

    n_clusters = assign[selected].nunique()
    min_purity = tree_info["boundary_df"]["cluster_purity_in_boundary"].min()
    fig.suptitle(
        f"{cfg['title']}\n{METHOD_TITLE} | {len(assign):,} cells | {n_clusters} clusters | "
        f"{len(tree_info['boundary_df'])} terminal boundaries | min boundary purity {min_purity:.3f}",
        fontsize=14,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.012,
        0.012,
        "Colors are selected cluster IDs. The compact tree is reconstructed from traversal final_boundary nodes; "
        "shared ancestral branches are gray, single-cluster descendant branches inherit cluster color.",
        fontsize=8.5,
        color="#374151",
    )
    if generated_at is not None:
        add_generated_at(fig, generated_at)
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    combined_summary = summary.merge(tree_info["boundary_df"], on="cluster", how="left")
    combined_summary.insert(0, "dataset", cfg["key"])
    return fig, combined_summary


def _selected_radial_tree_page(
    cfg,
    *,
    tree_info_builder,
    tree_plotter,
    tree_title,
    page_title,
    save_path=None,
    generated_at=None,
):
    assign = pd.read_csv(cfg["dir"] / "method_assignments.csv")
    selected = cfg["selected"]
    cluster_palette = make_cluster_palette(assign[selected].unique())
    tree_info = tree_info_builder(cfg, assign)

    fig = plt.figure(figsize=(16, 9.2), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.92, 1.38], wspace=0.12)
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_radial = fig.add_subplot(gs[0, 1])

    plot_umap(
        ax_umap, assign, selected, cluster_palette, "Selected NNLS-TBS clusters on UMAP", True, 5
    )
    tree_plotter(ax_radial, tree_info, cluster_palette, tree_title)
    fig.suptitle(
        f"{cfg['title']} {page_title}\n{METHOD_TITLE}",
        fontsize=14,
        weight="bold",
        y=0.985,
    )
    if generated_at is not None:
        add_generated_at(fig, generated_at)
    if save_path is not None:
        fig.savefig(save_path, dpi=230, bbox_inches="tight")
    return fig


def selected_radial_dataset_page(cfg, save_path=None, generated_at=None):
    return _selected_radial_tree_page(
        cfg,
        tree_info_builder=build_boundary_tree_info,
        tree_plotter=plot_radial_tree,
        tree_title="Selected-boundary radial tree, same colors",
        page_title="radial tree",
        save_path=save_path,
        generated_at=generated_at,
    )


def selected_full_radial_dataset_page(cfg, save_path=None, generated_at=None):
    return _selected_radial_tree_page(
        cfg,
        tree_info_builder=build_full_tree_info,
        tree_plotter=plot_full_radial_tree,
        tree_title="Real full radial tree, same leaf colors",
        page_title="full selected tree",
        save_path=save_path,
        generated_at=generated_at,
    )


def add_image_grid_page(pdf, title, items, ncols=2, generated_at=None):
    existing = [(Path(path), caption) for path, caption in items if Path(path).exists()]
    if not existing:
        return
    ncols = min(ncols, len(existing))
    nrows = int(math.ceil(len(existing) / ncols))
    fig = plt.figure(figsize=(16.5, 10.8), facecolor="white")
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.16)
    fig.suptitle(title, fontsize=15, weight="bold", y=0.99)
    for i, (path, caption) in enumerate(existing):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(caption, fontsize=9, pad=5)
        strip_axes(ax)
    for j in range(len(existing), nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")
    if generated_at is not None:
        add_generated_at(fig, generated_at)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summaries = []
    with PdfPages(
        OUT_PDF,
        metadata={
            "Title": "Selected scRNA NNLS-TBS report",
            "Subject": f"Generated at: {generated_at}",
        },
    ) as pdf:
        for cfg, png_path, radial_path, full_radial_path in zip(
            DATASETS,
            [OUT_ADULT, OUT_GONCALVES],
            [OUT_ADULT_RADIAL, OUT_GONCALVES_RADIAL],
            [OUT_ADULT_FULL_RADIAL, OUT_GONCALVES_FULL_RADIAL],
        ):
            fig, summary = selected_dataset_page(cfg, save_path=png_path, generated_at=generated_at)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            summaries.append(summary)
            radial_fig = selected_radial_dataset_page(
                cfg, save_path=radial_path, generated_at=generated_at
            )
            pdf.savefig(radial_fig, bbox_inches="tight")
            plt.close(radial_fig)
            full_radial_fig = selected_full_radial_dataset_page(
                cfg,
                save_path=full_radial_path,
                generated_at=generated_at,
            )
            pdf.savefig(full_radial_fig, bbox_inches="tight")
            plt.close(full_radial_fig)

        adult_dir = DATASETS[0]["dir"]
        gon_dir = DATASETS[1]["dir"]
        audit_dir = BASE / "scrna_distributional_action_audit_20260624"
        effect_dir = BASE / "scrna_branch_length_effect_audit_20260624"

        add_image_grid_page(
            pdf,
            "Adult pancreas: selected NNLS-TBS tree diagnostics",
            [
                (
                    adult_dir / f"{SELECTED}_clusters_radial_tree_ggtree.png",
                    "Selected cluster radial tree",
                ),
                (adult_dir / f"{SELECTED}_radial_tree_ggtree.png", "Selected full radial tree"),
                (adult_dir / f"{SELECTED}_branch_lengths.png", "Selected NNLS branch lengths"),
                (adult_dir / f"{SELECTED}_tree_dendrogram.png", "Selected dendrogram"),
            ],
            generated_at=generated_at,
        )
        add_image_grid_page(
            pdf,
            "Adult pancreas: clustering and progenitor context",
            [
                (adult_dir / "method_split_merge_diagnostic.png", "Method split/merge diagnostic"),
                (adult_dir / "method_ari_barplot.png", "ARI benchmark summary"),
                (adult_dir / "pancreas_progenitor_signature_umap.png", "Progenitor signature UMAP"),
                (
                    adult_dir / "tbs_adaptive_inner_node_progenitor_comparison.png",
                    "Internal-node progenitor comparison",
                ),
                (
                    adult_dir / "tbs_adaptive_two_three_cluster_junction_mixed_umap.png",
                    "Mixed two/three cluster junction UMAP",
                ),
                (adult_dir / "tbs_branch_time_sensitivity.png", "Branch-time sensitivity"),
            ],
            generated_at=generated_at,
        )
        add_image_grid_page(
            pdf,
            "Goncalves: selected NNLS-TBS tree diagnostics",
            [
                (gon_dir / f"{SELECTED}_radial_tree_ggtree.png", "Selected radial tree"),
                (gon_dir / f"{SELECTED}_branch_lengths.png", "Selected NNLS branch lengths"),
                (gon_dir / f"{SELECTED}_tree_dendrogram.png", "Selected dendrogram"),
                (
                    gon_dir / "goncalves_tbs_cluster_umap_tree_page_ggtree.png",
                    "Cluster UMAP + tree page",
                ),
            ],
            generated_at=generated_at,
        )
        add_image_grid_page(
            pdf,
            "Goncalves progenitor UMAP and tree panels",
            [
                (gon_dir / "goncalves_progenitor_signature_umap.png", "Progenitor signature UMAP"),
                (
                    gon_dir / "goncalves_tbs_progenitor_umap_tree_panel_wide_ggtree.png",
                    "Progenitor UMAP + tree panel",
                ),
                (
                    gon_dir / "goncalves_tbs_progenitor_state_umap_tree_page_ggtree.png",
                    "Progenitor state UMAP + tree",
                ),
                (
                    gon_dir / "goncalves_tbs_progenitor_fraction_umap_tree_page_ggtree.png",
                    "Progenitor fraction UMAP + tree",
                ),
            ],
            generated_at=generated_at,
        )
        add_image_grid_page(
            pdf,
            "Goncalves relationship panels",
            [
                (gon_dir / "goncalves_tbs_relation_umap_grid_ggtree.png", "Relation UMAP grid"),
                (gon_dir / "goncalves_tbs_relation_tree_grid_ggtree.png", "Relation tree grid"),
                (
                    gon_dir / "goncalves_tbs_population_umap_tree_page_ggtree.png",
                    "Population UMAP + tree",
                ),
                (
                    gon_dir / "goncalves_population_tbs_cluster_signature_heatmap.png",
                    "Population cluster signature heatmap",
                ),
            ],
            generated_at=generated_at,
        )
        add_image_grid_page(
            pdf,
            "Branch-length and distributional-action audits",
            [
                (
                    audit_dir / "distributional_action_vs_branch_length.png",
                    "Distributional action vs branch length",
                ),
                (
                    audit_dir / "top_internal_distributional_action_edges.png",
                    "Top internal distributional-action edges",
                ),
                (
                    audit_dir / "distributional_action_vs_edge_statistic.png",
                    "Distributional action vs edge statistic",
                ),
                (
                    effect_dir / "branch_time_sensitivity_effects.png",
                    "Branch-time sensitivity effects",
                ),
                (effect_dir / "branch_length_cluster_effects.png", "Branch-length cluster effects"),
                (
                    effect_dir / "branch_length_assignment_similarity_heatmap.png",
                    "Assignment similarity heatmap",
                ),
            ],
            generated_at=generated_at,
        )

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary.insert(0, "generated_at", generated_at)
    combined_summary.to_csv(OUT_SUMMARY, index=False)
    for path in [
        OUT_PDF,
        OUT_ADULT,
        OUT_GONCALVES,
        OUT_ADULT_RADIAL,
        OUT_GONCALVES_RADIAL,
        OUT_ADULT_FULL_RADIAL,
        OUT_GONCALVES_FULL_RADIAL,
        OUT_SUMMARY,
    ]:
        print(path)


if __name__ == "__main__":
    main()
