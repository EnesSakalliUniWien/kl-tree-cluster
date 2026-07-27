"""Interactive Plotly MNIST TBS analysis pages from the saved PCA50 sweep."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from benchmarks.experiments.mnist.run import load_mnist_subset
from benchmarks.experiments.mnist.run_higher_categories import (
    create_plotly_higher_category_plot_3d,
)
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from tree_break_selection.tree.poset_tree import PosetTree

from applications.mnist._shared import best_rows, parse_digit_counts

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "benchmarks/results/experiments/mnist"
SWEEP_DIR = SOURCE_DIR / "alpha_sweep_continuous_pca50_20260605"
OUT_DIR = ROOT / "raw/assets/benchmark-results/mnist_tbs_analysis_20260624_plotly"

OUT_INDEX = OUT_DIR / "index.html"
OUT_CSV = OUT_DIR / "mnist_tbs_analysis_summary.csv"
OUT_UMAP_COORDS = OUT_DIR / "mnist_pca50_umap_coordinates.csv"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_UMAP3D_HTML = OUT_DIR / "00b_mnist_umap3d_true_digit_best_tbs_cluster.html"
OUT_UMAP3D_DIGIT_LABELS_HTML = OUT_DIR / "00c_mnist_umap3d_visible_digit_labels.html"
OUT_UMAP3D_DOCSTYLE_HTML = OUT_DIR / "00d_mnist_umap3d_docstyle_digit_colorbar.html"
OUT_UMAP_IMAGE_INSPECTOR_HTML = OUT_DIR / "00e_mnist_umap_point_image_inspector.html"
OUT_UMAP3D_IMAGE_INSPECTOR_HTML = OUT_DIR / "00f_mnist_umap3d_point_image_inspector.html"

DIGIT_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ab",
}


def _alpha_label(value: float) -> str:
    return f"{value:g}"


def _assignment_key(linkage: str, edge_alpha: float, sibling_alpha: float) -> str:
    return f"{linkage}_e{_alpha_label(edge_alpha)}_s{_alpha_label(sibling_alpha)}"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        margin={"l": 75, "r": 40, "t": 86, "b": 86},
        width=1050,
        height=720,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
    )
    return fig


def _write(fig: go.Figure, filename: str, title: str) -> Path:
    path = OUT_DIR / filename
    _base_layout(fig, title).write_html(path, include_plotlyjs=True, full_html=True)
    return path


def best_summary_figure(summary: pd.DataFrame) -> go.Figure:
    best = best_rows(summary)
    labels = [
        f"{row.linkage}<br>edge={row.edge_alpha:g}, sibling={row.sibling_alpha:g}<br>{int(row.n_clusters)} clusters"
        for row in best.itertuples()
    ]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=best["ARI"],
        name="ARI",
        marker_color="#2563eb",
        text=[f"{v:.3f}" for v in best["ARI"]],
        textposition="outside",
        customdata=best[["linkage", "edge_alpha", "sibling_alpha", "n_clusters", "NMI"]].to_numpy(),
        hovertemplate=(
            "linkage=%{customdata[0]}<br>edge=%{customdata[1]:g}<br>"
            "sibling=%{customdata[2]:g}<br>clusters=%{customdata[3]}<br>"
            "ARI=%{y:.4f}<br>NMI=%{customdata[4]:.4f}<extra></extra>"
        ),
    )
    fig.add_bar(
        x=labels,
        y=best["NMI"],
        name="NMI",
        marker_color="#059669",
        text=[f"{v:.3f}" for v in best["NMI"]],
        textposition="outside",
        customdata=best[["linkage", "edge_alpha", "sibling_alpha", "n_clusters", "ARI"]].to_numpy(),
        hovertemplate=(
            "linkage=%{customdata[0]}<br>edge=%{customdata[1]:g}<br>"
            "sibling=%{customdata[2]:g}<br>clusters=%{customdata[3]}<br>"
            "NMI=%{y:.4f}<br>ARI=%{customdata[4]:.4f}<extra></extra>"
        ),
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="score", range=[0, 0.76], gridcolor="#e5e7eb")
    fig.update_xaxes(title_text="best setting per linkage")
    return fig


def heatmap_figure(summary: pd.DataFrame, linkage: str, value: str, title: str) -> go.Figure:
    frame = summary[summary["linkage"] == linkage].copy()
    edges = sorted(frame["edge_alpha"].unique())
    siblings = sorted(frame["sibling_alpha"].unique())
    matrix: list[list[float]] = []
    text: list[list[str]] = []
    for sibling in siblings:
        row_values = []
        row_text = []
        for edge in edges:
            selected = frame[(frame["edge_alpha"] == edge) & (frame["sibling_alpha"] == sibling)]
            metric = float(selected.iloc[0][value])
            row_values.append(metric)
            row_text.append(f"{metric:.3f}" if value != "n_clusters" else str(int(metric)))
        matrix.append(row_values)
        text.append(row_text)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[_alpha_label(v) for v in edges],
            y=[_alpha_label(v) for v in siblings],
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis" if value != "n_clusters" else "Magma",
            colorbar={"title": value},
            hovertemplate="edge alpha=%{x}<br>sibling alpha=%{y}<br>" + value + "=%{z}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="edge alpha")
    fig.update_yaxes(title_text="sibling alpha")
    fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
    return fig


def composition_figure(composition: pd.DataFrame) -> go.Figure:
    frame = composition.sort_values("size", ascending=True).copy()
    y_labels = [
        f"C{int(row.cluster)} | n={int(row.size)} | top={int(row.dominant_digit)} ({row.purity:.0%})"
        for row in frame.itertuples()
    ]
    fig = go.Figure()
    parsed = frame["digit_counts"].map(parse_digit_counts)
    for digit in range(10):
        values = [counts.get(digit, 0) for counts in parsed]
        fig.add_bar(
            x=values,
            y=y_labels,
            orientation="h",
            name=str(digit),
            marker_color=DIGIT_COLORS[digit],
            hovertemplate=f"digit={digit}<br>samples=%{{x}}<br>%{{y}}<extra></extra>",
        )
    fig.update_layout(barmode="stack", height=760)
    fig.update_xaxes(title_text="samples", gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="best-run cluster")
    return fig


def baseline_figure(binary: pd.DataFrame, continuous: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    best_sweep = summary.sort_values("ARI", ascending=False).iloc[0]
    best_continuous = continuous.loc[continuous["ARI"].idxmax()]
    best_binary = binary.loc[binary["ARI"].idxmax()]
    rows = pd.DataFrame(
        [
            {
                "run": "best alpha sweep<br>continuous PCA50 Ward",
                "ARI": best_sweep["ARI"],
                "NMI": best_sweep["NMI"],
                "clusters": int(best_sweep["n_clusters"]),
            },
            {
                "run": "best fixed continuous<br>PCA50 complete",
                "ARI": best_continuous["ARI"],
                "NMI": best_continuous["NMI"],
                "clusters": int(best_continuous["n_clusters"]),
            },
            {
                "run": "old binary run<br>jaccard/dice single",
                "ARI": best_binary["ARI"],
                "NMI": best_binary["NMI"],
                "clusters": int(best_binary["n_clusters"]),
            },
        ]
    )
    labels = [f"{row.run}<br>{row.clusters} clusters" for row in rows.itertuples()]
    fig = go.Figure()
    fig.add_bar(x=labels, y=rows["ARI"], name="ARI", marker_color="#2563eb", text=[f"{v:.3f}" for v in rows["ARI"]], textposition="outside")
    fig.add_bar(x=labels, y=rows["NMI"], name="NMI", marker_color="#059669", text=[f"{v:.3f}" for v in rows["NMI"]], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="score", range=[0, 0.76], gridcolor="#e5e7eb")
    fig.update_xaxes(title_text="MNIST result source")
    return fig


def _write_index(pages: list[tuple[str, str]], generated_at: str) -> None:
    links = "\n".join(
        f'<li><a href="{html.escape(path)}">{html.escape(title)}</a></li>' for title, path in pages
    )
    OUT_INDEX.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MNIST TBS Plotly Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    li {{ margin: 10px 0; }}
    .note {{ color: #374151; max-width: 920px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>MNIST TBS Plotly Analysis</h1>
  <p class="note"><strong>Generated at:</strong> {html.escape(generated_at)}</p>
  <p class="note">Saved run: 2,000 normalized MNIST images, PCA50 representation
  with 83.13% variance retained, continuous TBS alpha sweep over complete,
  weighted, and Ward linkage. The UMAP pages use the same sample/PCA50 setup
  and join visualization coordinates to the saved best-run TBS labels;
  clustering labels and metrics are not rerun here. The 3D page is generated
  with the existing MNIST Plotly 3D writer. The visible-label 3D page draws
  each point with its true MNIST digit as text. The docs-style 3D page follows
  the UMAP basic-usage convention: one embedding, colored by true digit with a
  digit colorbar. The 2D and 3D image-inspector pages show the original 28x28
  MNIST bitmap for the hovered or clicked point, enlarged in a fixed side panel,
  with a radial TBS tree context panel showing the selected point's final cluster
  in the best-run hierarchy. UMAP hover labels start with the true digit number.
  Each linked page contains one Plotly figure unless it is an inspector page,
  where the UMAP and radial tree are coordinated.</p>
  <ol>
    {links}
  </ol>
</body>
</html>
""",
        encoding="utf-8",
    )


def _artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        record["rows"] = int(len(frame))
        record["columns"] = int(len(frame.columns))
        if "generated_at" in frame.columns:
            record["generated_at_values"] = sorted(frame["generated_at"].astype(str).unique())
    return record


def _write_manifest(
    pages: list[tuple[str, str]],
    *,
    generated_at: str,
    best_key: str,
) -> None:
    artifact_paths = [OUT_INDEX, OUT_CSV, OUT_UMAP_COORDS] + [OUT_DIR / page for _, page in pages]
    manifest = {
        "manifest_schema_version": "static_artifact_provenance/v1",
        "bundle": "mnist_tbs_analysis_20260624_plotly",
        "generated_at": generated_at,
        "source_script": "applications/mnist/plot_interactive.py",
        "source_inputs": [
            str((SWEEP_DIR / "alpha_sweep_summary.csv").relative_to(ROOT)),
            str((SWEEP_DIR / "alpha_sweep_assignments.csv").relative_to(ROOT)),
            str((SWEEP_DIR / f"{best_key}_top_cluster_digit_composition.csv").relative_to(ROOT)),
            str((SOURCE_DIR / "mnist_benchmark_summary.csv").relative_to(ROOT)),
            str((SOURCE_DIR / "mnist_continuous_pca50_summary.csv").relative_to(ROOT)),
        ],
        "artifacts": [_artifact_record(path) for path in artifact_paths],
        "notes": [
            "The HTML pages were generated by the existing MNIST Plotly report script.",
            "The UMAP coordinate CSV may be reused from cache when its sample, digit, and TBS assignment columns match the saved alpha-sweep assignments.",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_or_create_umap(assignments: pd.DataFrame, best_key: str) -> pd.DataFrame:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    if OUT_UMAP_COORDS.exists():
        cached = pd.read_csv(OUT_UMAP_COORDS)
        required_columns = {"sample", "true_digit", "umap1", "umap2", "best_tbs_cluster"}
        if required_columns.issubset(cached.columns) and len(cached) == len(assignments):
            cache_matches_assignments = (
                cached["sample"].tolist() == assignments["sample"].tolist()
                and cached["true_digit"].astype(int).tolist()
                == assignments["true_digit"].astype(int).tolist()
                and cached["best_tbs_cluster"].astype(int).tolist()
                == assignments[best_key].astype(int).tolist()
            )
            if cache_matches_assignments:
                return cached

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pca50)
    frame = pd.DataFrame(
        {
            "sample": [f"Sample_{index}" for index in range(len(pca50))],
            "true_digit": y_subset.astype(int),
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
        }
    )
    frame = frame.merge(assignments[["sample", best_key]], on="sample", how="left")
    frame = frame.rename(columns={best_key: "best_tbs_cluster"})
    if frame["best_tbs_cluster"].isna().any():
        raise ValueError("UMAP samples did not all join to saved TBS assignments")
    frame.to_csv(OUT_UMAP_COORDS, index=False)
    return frame


def _load_mnist_raw_images(expected_digits: pd.Series) -> np.ndarray:
    raw_images, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=False,
    )
    expected_digit_values = expected_digits.astype(int).to_numpy()
    if len(raw_images) != len(expected_digit_values):
        raise ValueError("Raw MNIST image count does not match the UMAP frame")
    if y_subset.astype(int).tolist() != expected_digit_values.tolist():
        raise ValueError("Raw MNIST labels do not match the saved assignment order")
    return np.asarray(raw_images, dtype=np.float64)


def _mnist_pixel_payload(raw_images: np.ndarray) -> tuple[list[list[int]], int]:
    image_matrix = np.asarray(raw_images, dtype=np.float64)
    if image_matrix.ndim != 2:
        raise ValueError(f"MNIST image payload must be a 2D matrix; got {image_matrix.shape}.")
    image_side = int(round(np.sqrt(image_matrix.shape[1])))
    if image_side * image_side != image_matrix.shape[1]:
        raise ValueError(
            "Flattened MNIST image width must be a square number of pixels; "
            f"got {image_matrix.shape[1]}."
        )
    if image_matrix.size == 0:
        return [], image_side
    max_value = float(np.nanmax(image_matrix))
    scale = 255.0 if max_value <= 1.0 else 1.0
    pixels = np.clip(np.rint(image_matrix * scale), 0, 255).astype(np.uint8)
    return pixels.tolist(), image_side


def _build_tbs_radial_tree_context(
    assignments: pd.DataFrame,
    best_key: str,
    feature_matrix: np.ndarray,
    linkage_method: str,
    *,
    div_id: str = "mnist-radial-tbs-tree",
    full_tree: bool = False,
) -> dict[str, object]:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")
    if len(assignments) != int(feature_matrix.shape[0]):
        raise ValueError("Saved TBS assignments do not match the MNIST feature matrix")

    sample_names = assignments["sample"].astype(str).tolist()
    cluster_labels = assignments[best_key].astype(int).to_numpy()
    digit_labels = assignments["true_digit"].astype(int).to_numpy()

    linkage_matrix = linkage(pdist(feature_matrix), method=str(linkage_method))
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=sample_names)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    node_by_leaf_set = {frozenset(leaves): node for node, leaves in descendant_sets.items()}
    leaf_node_by_label = {
        str(attrs["label"]): node
        for node, attrs in tree.nodes(data=True)
        if bool(attrs.get("is_leaf", False))
    }
    parent_by_child = {child: parent for parent, child in tree.edges()}
    leaf_cluster_by_sample = dict(
        zip(assignments["sample"].astype(str), cluster_labels.astype(int), strict=True)
    )
    digit_by_sample = dict(
        zip(assignments["sample"].astype(str), digit_labels.astype(int), strict=True)
    )

    cluster_roots: dict[int, object] = {}
    cluster_records: dict[int, dict[str, object]] = {}
    for cluster_id in sorted(np.unique(cluster_labels).astype(int).tolist()):
        is_member = cluster_labels == cluster_id
        cluster_samples = assignments.loc[is_member, "sample"].astype(str).tolist()
        cluster_leaf_set = frozenset(cluster_samples)
        root_node = node_by_leaf_set.get(cluster_leaf_set)
        exact_tree_boundary = root_node is not None
        if root_node is None:
            root_node = tree.find_lca_for_set(leaf_node_by_label[sample] for sample in cluster_samples)
        cluster_roots[cluster_id] = root_node

        cluster_digits = digit_labels[is_member]
        digit_values, digit_counts = np.unique(cluster_digits, return_counts=True)
        dominant_index = int(np.argmax(digit_counts))
        dominant_digit = int(digit_values[dominant_index])
        dominant_count = int(digit_counts[dominant_index])
        size = int(is_member.sum())
        cluster_records[cluster_id] = {
            "cluster_id": cluster_id,
            "node_id": str(root_node),
            "size": size,
            "dominant_digit": dominant_digit,
            "purity": dominant_count / size,
            "exact_tree_boundary": exact_tree_boundary,
        }

    root_node = tree.root()
    cluster_root_set = set(cluster_roots.values())

    def min_sample_index(node: object) -> int:
        sample_indices = [
            int(str(sample).split("_")[-1]) for sample in descendant_sets[node] if "_" in str(sample)
        ]
        return min(sample_indices) if sample_indices else 0

    if full_tree:
        included_nodes = set(tree.nodes)
        display_children = {
            node: sorted(list(tree.successors(node)), key=min_sample_index)
            for node in included_nodes
        }
        ordered_layout_leaves: list[object] = []

        def visit_full(node: object) -> None:
            children = display_children.get(node, [])
            if not children:
                ordered_layout_leaves.append(node)
                return
            for child in children:
                visit_full(child)

        visit_full(root_node)
    else:
        included_nodes = {root_node}
        for cluster_root in cluster_roots.values():
            node = cluster_root
            included_nodes.add(node)
            while node in parent_by_child:
                node = parent_by_child[node]
                included_nodes.add(node)

        display_children = {node: [] for node in included_nodes}
        for node in included_nodes:
            if node == root_node:
                continue
            parent = parent_by_child[node]
            while parent not in included_nodes:
                parent = parent_by_child[parent]
            display_children[parent].append(node)

        for children in display_children.values():
            children.sort(key=min_sample_index)

        ordered_layout_leaves = []

        def visit_boundary(node: object) -> None:
            if node in cluster_root_set:
                ordered_layout_leaves.append(node)
                return
            for child in display_children.get(node, []):
                visit_boundary(child)

        visit_boundary(root_node)
        if len(ordered_layout_leaves) != len(cluster_roots):
            ordered_layout_leaves = sorted(cluster_root_set, key=min_sample_index)

    for children in display_children.values():
        children.sort(key=min_sample_index)

    leaf_ordinals = {
        node: float(index)
        for index, node in enumerate(ordered_layout_leaves)
    }
    depths = {root_node: 0}
    stack = [root_node]
    while stack:
        node = stack.pop()
        for child in display_children.get(node, []):
            depths[child] = depths[node] + 1
            stack.append(child)
    max_depth = max(depths.values()) if depths else 1

    ordinals: dict[object, float] = {}

    def assign_ordinal(node: object) -> float:
        if node in leaf_ordinals:
            ordinals[node] = leaf_ordinals[node]
            return ordinals[node]
        child_values = [assign_ordinal(child) for child in display_children.get(node, [])]
        ordinals[node] = float(np.mean(child_values)) if child_values else 0.0
        return ordinals[node]

    assign_ordinal(root_node)
    angular_slots = max(len(ordered_layout_leaves), 1)
    coordinates: dict[object, tuple[float, float]] = {}
    for node in included_nodes:
        angle = 2.0 * np.pi * (ordinals.get(node, 0.0) / angular_slots) - (np.pi / 2.0)
        radius = 0.0 if max_depth == 0 else depths[node] / max_depth
        coordinates[node] = (float(radius * np.cos(angle)), float(radius * np.sin(angle)))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for parent, children in display_children.items():
        parent_x, parent_y = coordinates[parent]
        for child in children:
            child_x, child_y = coordinates[child]
            edge_x.extend([parent_x, child_x, None])
            edge_y.extend([parent_y, child_y, None])

    cluster_id_by_root = {root: cluster_id for cluster_id, root in cluster_roots.items()}
    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_hover: list[str] = []
    node_color: list[float] = []
    node_size: list[int] = []
    for node in sorted(included_nodes, key=lambda item: (depths[item], ordinals.get(item, 0.0))):
        x_value, y_value = coordinates[node]
        node_x.append(x_value)
        node_y.append(y_value)
        cluster_id = cluster_id_by_root.get(node)
        if cluster_id is None:
            descendant_cluster_count = sum(
                descendant_sets[cluster_root].issubset(descendant_sets[node])
                for cluster_root in cluster_root_set
            )
            node_text.append("")
            node_hover.append(
                f"Tree node: {node}<br>Depth: {depths[node]}<br>"
                f"Descendant final clusters: {descendant_cluster_count}"
            )
            node_color.append(-1.0)
            node_size.append(8)
        else:
            record = cluster_records[cluster_id]
            node_text.append(f"C{cluster_id}")
            node_hover.append(
                f"Final TBS cluster C{cluster_id}<br>Tree boundary: {node}<br>"
                f"Samples: {record['size']}<br>Dominant digit: {record['dominant_digit']} "
                f"({record['purity']:.0%})"
            )
            node_color.append(float(cluster_id))
            node_size.append(14 if full_tree else 15)
            continue

        if full_tree and bool(tree.nodes[node].get("is_leaf", False)):
            sample = str(tree.nodes[node]["label"])
            node_text[-1] = ""
            node_hover[-1] = (
                f"Leaf: {sample}<br>Digit number: {digit_by_sample[sample]}<br>"
                f"Final TBS cluster: C{leaf_cluster_by_sample[sample]}"
            )
            node_color[-1] = float(leaf_cluster_by_sample[sample])
            node_size[-1] = 2

    initial_cluster = sorted(cluster_records)[0]
    initial_node = cluster_roots[initial_cluster]
    initial_x, initial_y = coordinates[initial_node]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#94a3b8", "width": 1},
            hoverinfo="skip",
            showlegend=False,
            name="tree edge",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont={"size": 10, "color": "#111827"},
            marker={
                "size": node_size,
                "color": node_color,
                "colorscale": "Turbo",
                "cmin": -1,
                "showscale": False,
                "line": {"color": "#ffffff", "width": 0.5 if full_tree else 1},
            },
            hovertext=node_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
            name="tree node",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[initial_x],
            y=[initial_y],
            mode="markers",
            marker={
                "size": 26,
                "color": "rgba(0,0,0,0)",
                "line": {"color": "#dc2626", "width": 3},
            },
            hoverinfo="skip",
            showlegend=False,
            name="selected cluster",
        )
    )
    fig.update_layout(
        title={
            "text": "Full best-run TBS radial tree" if full_tree else "Best-run TBS radial tree",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 12},
        height=820 if full_tree else 360,
        margin={"l": 8, "r": 8, "t": 46, "b": 8},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        plot_bgcolor="#ffffff",
    )

    cluster_payload: list[dict[str, object]] = []
    for cluster_id, record in cluster_records.items():
        x_value, y_value = coordinates[cluster_roots[cluster_id]]
        payload = dict(record)
        payload["x"] = x_value
        payload["y"] = y_value
        cluster_payload.append(payload)

    return {
        "html": fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id),
        "div_id": div_id,
        "cluster_records": cluster_payload,
        "n_clusters": len(cluster_records),
        "n_nodes": len(included_nodes),
        "n_edges": sum(len(children) for children in display_children.values()),
        "tree_mode": "full" if full_tree else "cluster_boundary",
    }


def _write_image_inspector_page(
    fig: go.Figure,
    plot_frame: pd.DataFrame,
    raw_images: np.ndarray,
    *,
    output_path: Path,
    plot_div_id: str,
    page_title: str,
    tree_context: dict[str, object] | None = None,
) -> Path:
    image_pixels, image_side = _mnist_pixel_payload(raw_images)
    if len(image_pixels) != len(plot_frame):
        raise ValueError("MNIST image payload length does not match the UMAP frame")

    plot_frame = plot_frame.reset_index(drop=True).copy()
    if "point_index" not in plot_frame.columns:
        plot_frame["point_index"] = np.arange(len(plot_frame), dtype=int)
    sample_count_text = f"{len(plot_frame):,}"
    tree_html = ""
    tree_records: list[dict[str, object]] = []
    tree_div_id = ""
    tree_cluster_count = 0
    tree_node_count = 0
    tree_edge_count = 0
    tree_mode = "cluster_boundary"
    if tree_context is not None:
        tree_html = str(tree_context["html"])
        tree_records = list(tree_context["cluster_records"])
        tree_div_id = str(tree_context["div_id"])
        tree_cluster_count = int(tree_context["n_clusters"])
        tree_node_count = int(tree_context.get("n_nodes", 0))
        tree_edge_count = int(tree_context.get("n_edges", 0))
        tree_mode = str(tree_context.get("tree_mode", "cluster_boundary"))
    tree_note = (
        f"Full radial tree uses the same best-run TBS hierarchy: {tree_node_count:,} nodes, "
        f"{tree_edge_count:,} edges, and {tree_cluster_count} highlighted final clusters."
        if tree_mode == "full"
        else (
            "Radial tree uses the same best-run TBS hierarchy and highlights the selected "
            f"point's final cluster among {tree_cluster_count} clusters."
        )
    )

    plot_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id=plot_div_id,
    )
    records = plot_frame[["sample", "true_digit", "best_tbs_cluster", "point_index"]].to_dict(
        orient="records"
    )
    inspector_html = f"""
      <aside class="inspector">
        <canvas id="digit-canvas" width="{image_side}" height="{image_side}"></canvas>
        <div class="meta">
          <div><span class="label">Sample</span><span id="sample-value">Sample_0</span></div>
          <div><span class="label">True digit</span><span id="digit-value">-</span></div>
          <div><span class="label">TBS cluster</span><span id="cluster-value">-</span></div>
          <div><span class="label">Tree node</span><span id="tree-node-value">-</span></div>
        </div>
        <p class="hint">Hover or click a UMAP point to show the original MNIST image used for that point. Contains {sample_count_text} MNIST examples. Source images are {image_side}x{image_side} pixels; this canvas enlarges them for inspection.</p>
      </aside>
""".strip()
    tree_panel_html = f"""
      <aside class="tree-panel">
        {tree_html}
        <p class="tree-meta">{tree_note}</p>
      </aside>
""".strip()
    shell_class = "shell shell-with-under-tree" if tree_mode == "full" else "shell"
    sidebar_tree_panel_html = "" if tree_mode == "full" else tree_panel_html
    side_panel_inner_html = "\n".join(
        panel for panel in (inspector_html, sidebar_tree_panel_html) if panel
    )
    under_plot_tree_panel_html = (
        f"""
    <section class="tree-panel tree-under-plot">
      {tree_html}
      <p class="tree-meta">{tree_note}</p>
    </section>
""".strip()
        if tree_mode == "full"
        else ""
    )
    under_plot_tree_section_html = (
        f"\n    {under_plot_tree_panel_html}" if under_plot_tree_panel_html else ""
    )
    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(page_title)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      color: #111827;
      background: #f8fafc;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(680px, 1fr) 430px;
      gap: 18px;
      padding: 18px;
      align-items: start;
    }}
    .shell-with-under-tree {{
      grid-template-columns: minmax(760px, 1fr) 300px;
    }}
    .plot-panel, .inspector, .tree-panel {{
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    .inspector {{
      padding: 16px;
    }}
    .side-panel {{
      position: sticky;
      top: 18px;
      display: grid;
      gap: 14px;
    }}
    #digit-canvas {{
      width: 224px;
      height: 224px;
      image-rendering: pixelated;
      border: 1px solid #111827;
      background: #000000;
      display: block;
      margin: 0 auto 14px;
    }}
    .tree-panel {{
      padding: 10px 10px 12px;
    }}
    .tree-under-plot {{
      grid-column: 1 / -1;
      padding: 12px 12px 14px;
    }}
    .meta {{
      font-size: 14px;
      line-height: 1.55;
    }}
    .label {{
      color: #6b7280;
      display: inline-block;
      min-width: 92px;
    }}
    .hint {{
      color: #4b5563;
      font-size: 13px;
      line-height: 1.45;
      margin: 14px 0 0;
    }}
    .tree-meta {{
      color: #4b5563;
      font-size: 13px;
      line-height: 1.45;
      margin: 8px 4px 0;
    }}
  </style>
</head>
<body>
  <div class="{shell_class}">
    <div class="plot-panel">{plot_html}</div>
    <div class="side-panel">
      {side_panel_inner_html}
    </div>{under_plot_tree_section_html}
  </div>
  <script>
    const imagePixels = {json.dumps(image_pixels, separators=(",", ":"))};
    const pointRecords = {json.dumps(records, separators=(",", ":"))};
    const imageSide = {image_side};
    const treeRecords = {json.dumps(tree_records, separators=(",", ":"))};
    const treeRecordByCluster = new Map(treeRecords.map((record) => [Number(record.cluster_id), record]));
    const canvas = document.getElementById("digit-canvas");
    const context = canvas.getContext("2d");
    const sampleValue = document.getElementById("sample-value");
    const digitValue = document.getElementById("digit-value");
    const clusterValue = document.getElementById("cluster-value");
    const treeNodeValue = document.getElementById("tree-node-value");

    function updateTreeHighlight(clusterId) {{
      const treeRecord = treeRecordByCluster.get(Number(clusterId));
      const treePlot = document.getElementById("{tree_div_id}");
      if (!treeRecord) {{
        treeNodeValue.textContent = "-";
        return;
      }}
      treeNodeValue.textContent = treeRecord.node_id;
      if (treePlot && window.Plotly) {{
        Plotly.restyle(treePlot, {{x: [[treeRecord.x]], y: [[treeRecord.y]]}}, [2]);
      }}
    }}

    function drawDigitImage(pointIndex) {{
      const pixels = imagePixels[pointIndex];
      const record = pointRecords[pointIndex];
      if (!pixels || !record) {{
        return;
      }}
      const imageData = context.createImageData(imageSide, imageSide);
      for (let index = 0; index < pixels.length; index += 1) {{
        const value = pixels[index];
        const offset = index * 4;
        imageData.data[offset] = value;
        imageData.data[offset + 1] = value;
        imageData.data[offset + 2] = value;
        imageData.data[offset + 3] = 255;
      }}
      context.putImageData(imageData, 0, 0);
      sampleValue.textContent = record.sample;
      digitValue.textContent = record.true_digit;
      clusterValue.textContent = record.best_tbs_cluster;
      updateTreeHighlight(record.best_tbs_cluster);
    }}

    const plot = document.getElementById("{plot_div_id}");
    plot.on("plotly_hover", function(eventData) {{
      const point = eventData.points[0];
      drawDigitImage(Number(point.customdata[3]));
    }});
    plot.on("plotly_click", function(eventData) {{
      const point = eventData.points[0];
      drawDigitImage(Number(point.customdata[3]));
    }});
    drawDigitImage(0);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _write_umap_image_inspector(
    frame: pd.DataFrame,
    raw_images: np.ndarray,
    *,
    output_path: Path = OUT_UMAP_IMAGE_INSPECTOR_HTML,
    tree_context: dict[str, object] | None = None,
) -> Path:
    plot_frame = frame.reset_index(drop=True).copy()
    plot_frame["point_index"] = np.arange(len(plot_frame), dtype=int)
    hovertext = [
        f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
        f"Best TBS cluster: {int(row.best_tbs_cluster)}"
        for row in plot_frame.itertuples(index=False)
    ]
    fig = go.Figure(
        go.Scattergl(
            x=plot_frame["umap1"],
            y=plot_frame["umap2"],
            mode="markers",
            name="MNIST sample",
            marker={
                "size": 7,
                "opacity": 0.76,
                "color": plot_frame["true_digit"].astype(int),
                "colorscale": "Spectral",
                "cmin": -0.5,
                "cmax": 9.5,
                "colorbar": {
                    "title": "true digit",
                    "tickmode": "array",
                    "tickvals": list(range(10)),
                    "ticktext": [str(digit) for digit in range(10)],
                },
            },
            customdata=plot_frame[
                ["sample", "true_digit", "best_tbs_cluster", "point_index"]
            ].to_numpy(),
            hovertext=hovertext,
            hovertemplate=(
                "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={"text": "MNIST PCA50 UMAP: point image inspector", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        width=960,
        height=720,
        margin={"l": 70, "r": 24, "t": 76, "b": 70},
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2", scaleanchor="x", scaleratio=1)

    return _write_image_inspector_page(
        fig,
        plot_frame,
        raw_images,
        output_path=output_path,
        plot_div_id="mnist-image-inspector-plot",
        page_title="MNIST UMAP Point Image Inspector",
        tree_context=tree_context,
    )


def _write_umap3d_image_inspector(
    assignments: pd.DataFrame,
    best_key: str,
    raw_images: np.ndarray,
    *,
    output_path: Path = OUT_UMAP3D_IMAGE_INSPECTOR_HTML,
    tree_context: dict[str, object] | None = None,
    feature_matrix: np.ndarray | None = None,
    digit_labels: np.ndarray | None = None,
) -> Path:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    if feature_matrix is None or digit_labels is None:
        feature_matrix, digit_labels = load_mnist_subset(
            n_samples=2000,
            seed=42,
            use_pca=True,
            n_components=50,
        )
    cluster_labels = assignments[best_key].astype(int).to_numpy()
    digit_labels = np.asarray(digit_labels, dtype=int)
    if len(cluster_labels) != len(digit_labels):
        raise ValueError("Saved TBS assignments do not match the MNIST sample size")
    if len(raw_images) != len(digit_labels):
        raise ValueError("Raw MNIST image payload does not match the 3D UMAP sample size")

    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(feature_matrix)
    plot_frame = pd.DataFrame(
        {
            "sample": assignments["sample"].astype(str).to_numpy(),
            "true_digit": digit_labels,
            "best_tbs_cluster": cluster_labels,
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "umap3": embedding[:, 2],
            "point_index": np.arange(len(digit_labels), dtype=int),
        }
    )
    hovertext = [
        f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
        f"Best TBS cluster: {int(row.best_tbs_cluster)}"
        for row in plot_frame.itertuples(index=False)
    ]
    customdata = plot_frame[
        ["sample", "true_digit", "best_tbs_cluster", "point_index"]
    ].to_numpy()
    hovertemplate = (
        "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<br>"
        "UMAP3=%{z:.3f}<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=plot_frame["umap1"],
            y=plot_frame["umap2"],
            z=plot_frame["umap3"],
            mode="markers",
            name="MNIST sample",
            marker={
                "size": 5,
                "opacity": 0.62,
                "color": digit_labels,
                "colorscale": "Spectral",
                "cmin": -0.5,
                "cmax": 9.5,
                "colorbar": {
                    "title": "true digit",
                    "tickmode": "array",
                    "tickvals": list(range(10)),
                    "ticktext": [str(digit) for digit in range(10)],
                },
            },
            customdata=customdata,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=plot_frame["umap1"],
            y=plot_frame["umap2"],
            z=plot_frame["umap3"],
            mode="markers",
            name="hover target",
            showlegend=False,
            marker={"size": 22, "opacity": 0.01, "color": "#111827"},
            customdata=customdata,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title={"text": "MNIST PCA50 3D UMAP: point image inspector", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        width=960,
        height=760,
        margin={"l": 4, "r": 4, "t": 76, "b": 4},
        scene={
            "xaxis_title": "UMAP-1",
            "yaxis_title": "UMAP-2",
            "zaxis_title": "UMAP-3",
        },
    )
    return _write_image_inspector_page(
        fig,
        plot_frame,
        raw_images,
        output_path=output_path,
        plot_div_id="mnist-3d-image-inspector-plot",
        page_title="MNIST 3D UMAP Point Image Inspector",
        tree_context=tree_context,
    )


def _write_umap3d_with_existing_writer(
    assignments: pd.DataFrame,
    best_key: str,
    *,
    output_path: Path,
    visible_digit_labels: bool = False,
) -> Path:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    cluster_labels = assignments[best_key].to_numpy()
    if len(cluster_labels) != len(y_subset):
        raise ValueError("Saved TBS assignments do not match the MNIST sample size")

    create_plotly_higher_category_plot_3d(
        feature_matrix=pca50,
        digit_labels=y_subset,
        cluster_labels=cluster_labels,
        true_higher_category_names=np.array([f"digit {int(digit)}" for digit in y_subset]),
        predicted_higher_category_names=np.array(
            [f"TBS C{int(cluster)}" for cluster in cluster_labels]
        ),
        output_path=output_path,
        visible_digit_labels=visible_digit_labels,
    )
    replacements = {
        "MNIST Higher Categories - Interactive 3D UMAP": "MNIST PCA50 saved TBS labels - Interactive 3D UMAP",
        "3D UMAP - True Higher Categories": "3D UMAP - True digits",
        "3D UMAP - Predicted Higher Categories": "3D UMAP - Best TBS clusters",
        "Raw Tree Clusters": "Saved TBS clusters",
        "True Higher Category": "True digit",
        "Pred Higher Category": "Best TBS cluster",
        "Tree Cluster": "TBS cluster",
    }
    html_text = output_path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        html_text = html_text.replace(old, new)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _write_umap3d_docstyle(assignments: pd.DataFrame, best_key: str) -> Path:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    cluster_labels = assignments[best_key].to_numpy()
    if len(cluster_labels) != len(y_subset):
        raise ValueError("Saved TBS assignments do not match the MNIST sample size")

    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pca50)
    customdata = np.column_stack(
        [
            [f"Sample_{index}" for index in range(len(y_subset))],
            y_subset.astype(int),
            cluster_labels.astype(int),
        ]
    )
    hovertext = [
        f"Digit number: {int(digit)}<br>Sample: Sample_{index}<br>"
        f"Best TBS cluster: {int(cluster)}"
        for index, (digit, cluster) in enumerate(zip(y_subset, cluster_labels, strict=True))
    ]
    hovertemplate = "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<br>UMAP3=%{z:.3f}<extra></extra>"
    fig = go.Figure(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers+text",
            text=[str(int(digit)) for digit in y_subset],
            textposition="middle center",
            textfont={"size": 8, "color": "#111827"},
            customdata=customdata,
            marker={
                "size": 7,
                "opacity": 0.42,
                "color": y_subset.astype(int),
                "colorscale": "Spectral",
                "cmin": -0.5,
                "cmax": 9.5,
                "colorbar": {
                    "title": "true digit",
                    "tickmode": "array",
                    "tickvals": list(range(10)),
                    "ticktext": [str(digit) for digit in range(10)],
                },
            },
            hovertext=hovertext,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            name="hover target",
            showlegend=False,
            customdata=customdata,
            hovertext=hovertext,
            marker={"size": 24, "opacity": 0.01, "color": "#111827"},
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title={"text": "MNIST PCA50 3D UMAP: digits as labels", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        width=1050,
        height=760,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        scene={
            "xaxis_title": "UMAP-1",
            "yaxis_title": "UMAP-2",
            "zaxis_title": "UMAP-3",
        },
    )
    fig.write_html(OUT_UMAP3D_DOCSTYLE_HTML, include_plotlyjs=True, full_html=True)
    return OUT_UMAP3D_DOCSTYLE_HTML


def umap_figure(frame: pd.DataFrame) -> go.Figure:
    true_digits = sorted(frame["true_digit"].unique())
    clusters = sorted(frame["best_tbs_cluster"].unique())
    fig = go.Figure()
    for digit in true_digits:
        group = frame[frame["true_digit"] == digit]
        hovertext = [
            f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
            f"Best TBS cluster: {int(row.best_tbs_cluster)}"
            for row in group.itertuples(index=False)
        ]
        fig.add_scattergl(
            x=group["umap1"],
            y=group["umap2"],
            mode="markers",
            name=f"digit {digit}",
            marker={"size": 5, "opacity": 0.78},
            customdata=group[["sample", "true_digit", "best_tbs_cluster"]].to_numpy(),
            hovertext=hovertext,
            hovertemplate=(
                "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
            ),
            visible=True,
            legendgroup="true",
        )
    for cluster in clusters:
        group = frame[frame["best_tbs_cluster"] == cluster]
        hovertext = [
            f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
            f"Best TBS cluster: {int(row.best_tbs_cluster)}"
            for row in group.itertuples(index=False)
        ]
        fig.add_scattergl(
            x=group["umap1"],
            y=group["umap2"],
            mode="markers",
            name=f"C{int(cluster)}",
            marker={"size": 5, "opacity": 0.78},
            customdata=group[["sample", "true_digit", "best_tbs_cluster"]].to_numpy(),
            hovertext=hovertext,
            hovertemplate=(
                "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
            ),
            visible=False,
            legendgroup="cluster",
        )

    n_true = len(true_digits)
    n_cluster = len(clusters)
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Color by true digit",
                        "method": "update",
                        "args": [
                            {"visible": [True] * n_true + [False] * n_cluster},
                            {"legend": {"title": {"text": "true digit"}}},
                        ],
                    },
                    {
                        "label": "Color by best TBS cluster",
                        "method": "update",
                        "args": [
                            {"visible": [False] * n_true + [True] * n_cluster},
                            {"legend": {"title": {"text": "TBS cluster"}}},
                        ],
                    },
                ],
            }
        ],
        legend={"title": {"text": "true digit"}},
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2", scaleanchor="x", scaleratio=1)
    return fig


def main() -> None:
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = pd.read_csv(SWEEP_DIR / "alpha_sweep_summary.csv")
    binary = pd.read_csv(SOURCE_DIR / "mnist_benchmark_summary.csv")
    continuous = pd.read_csv(SOURCE_DIR / "mnist_continuous_pca50_summary.csv")
    assignments = pd.read_csv(SWEEP_DIR / "alpha_sweep_assignments.csv")
    best = summary.sort_values("ARI", ascending=False).iloc[0]
    best_key = _assignment_key(str(best["linkage"]), float(best["edge_alpha"]), float(best["sibling_alpha"]))
    composition = pd.read_csv(SWEEP_DIR / f"{best_key}_top_cluster_digit_composition.csv")
    umap_frame = _load_or_create_umap(assignments, best_key)
    umap3d_path = _write_umap3d_with_existing_writer(
        assignments,
        best_key,
        output_path=OUT_UMAP3D_HTML,
    )
    umap3d_digit_labels_path = _write_umap3d_with_existing_writer(
        assignments,
        best_key,
        output_path=OUT_UMAP3D_DIGIT_LABELS_HTML,
        visible_digit_labels=True,
    )
    umap3d_docstyle_path = _write_umap3d_docstyle(assignments, best_key)
    raw_mnist_images = _load_mnist_raw_images(umap_frame["true_digit"])
    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    boundary_tree_context = _build_tbs_radial_tree_context(
        assignments,
        best_key,
        pca50,
        str(best["linkage"]),
    )
    full_tree_context = _build_tbs_radial_tree_context(
        assignments,
        best_key,
        pca50,
        str(best["linkage"]),
        div_id="mnist-full-radial-tbs-tree",
        full_tree=True,
    )
    umap_image_inspector_path = _write_umap_image_inspector(
        umap_frame,
        raw_mnist_images,
        tree_context=boundary_tree_context,
    )
    umap3d_image_inspector_path = _write_umap3d_image_inspector(
        assignments,
        best_key,
        raw_mnist_images,
        tree_context=full_tree_context,
        feature_matrix=pca50,
        digit_labels=y_subset,
    )

    analysis_summary = best_rows(summary)
    analysis_summary.insert(0, "generated_at", generated_at)
    analysis_summary.to_csv(OUT_CSV, index=False)
    pages = [
        (
            "MNIST UMAP: true digit and best TBS cluster",
            _write(
                umap_figure(umap_frame),
                "00_mnist_umap_true_digit_best_tbs_cluster.html",
                "MNIST PCA50 UMAP: true digit and best TBS cluster",
            ).name,
        ),
        (
            "MNIST 3D UMAP: true digit and best TBS cluster",
            umap3d_path.name,
        ),
        (
            "MNIST 3D UMAP: visible digit labels",
            umap3d_digit_labels_path.name,
        ),
        (
            "MNIST 3D UMAP: docs-style digit colorbar",
            umap3d_docstyle_path.name,
        ),
        (
            "MNIST UMAP: point image inspector",
            umap_image_inspector_path.name,
        ),
        (
            "MNIST 3D UMAP: point image inspector",
            umap3d_image_inspector_path.name,
        ),
        (
            "Best alpha setting per linkage",
            _write(
                best_summary_figure(summary),
                "01_best_alpha_setting_per_linkage.html",
                "MNIST continuous PCA50 TBS: best alpha setting per linkage",
            ).name,
        ),
        (
            "Ward ARI reaction to alpha",
            _write(
                heatmap_figure(summary, "ward", "ARI", "Ward linkage: ARI reaction to alpha"),
                "02_ward_ari_alpha_heatmap.html",
                "Ward linkage: ARI reaction to alpha",
            ).name,
        ),
        (
            "Ward cluster-count reaction to alpha",
            _write(
                heatmap_figure(summary, "ward", "n_clusters", "Ward linkage: cluster-count reaction to alpha"),
                "03_ward_cluster_count_alpha_heatmap.html",
                "Ward linkage: cluster-count reaction to alpha",
            ).name,
        ),
        (
            "Best-run cluster composition by true digit",
            _write(
                composition_figure(composition),
                "04_best_ward_cluster_digit_composition.html",
                "Best MNIST TBS run: cluster composition by true digit",
            ).name,
        ),
        (
            "Result source comparison",
            _write(
                baseline_figure(binary, continuous, summary),
                "05_result_source_comparison.html",
                "Which MNIST result should be trusted?",
            ).name,
        ),
    ]
    _write_index(pages, generated_at)
    _write_manifest(pages, generated_at=generated_at, best_key=best_key)
    print(OUT_INDEX)
    print(OUT_CSV)
    print(OUT_UMAP_COORDS)
    print(OUT_MANIFEST)
    print(OUT_UMAP3D_HTML)
    print(OUT_UMAP3D_DIGIT_LABELS_HTML)
    print(OUT_UMAP3D_DOCSTYLE_HTML)
    print(OUT_UMAP_IMAGE_INSPECTOR_HTML)
    print(OUT_UMAP3D_IMAGE_INSPECTOR_HTML)
    for _title, page in pages:
        print(OUT_DIR / page)


def run_cli() -> None:
    """Validate command-line arguments before generating retained figures."""

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()


if __name__ == "__main__":
    run_cli()
