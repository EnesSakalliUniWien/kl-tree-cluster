#!/usr/bin/env python3
"""Plot the manuscript 16-leaf right-block example in 3D feature space.

The underlying toy feature matrix has six features. These plots show the
direct 3D projection onto ``(f1, f2, f3)``:

- ``right_block_3d_leaf_points.png`` contains only the 16 leaf rows.
- ``right_block_3d_points_with_inner_nodes.png`` adds internal-node means.

No tree edges or distance labels are drawn.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_right_block_toytree import FEATURE_MATRIX, GROUPS

MAIN_INTERNAL_NODES = ("Root", "C1-C8", "C9-C16", "C1-C4", "C5-C8", "C9-C12", "C13-C16")
PAIR_INTERNAL_NODES = (
    "C1-C2",
    "C3-C4",
    "C5-C6",
    "C7-C8",
    "C9-C10",
    "C11-C12",
    "C13-C14",
    "C15-C16",
)


def projected_leaf_points() -> dict[str, np.ndarray]:
    return {
        label: np.array(values[:3], dtype=float)
        for label, values in FEATURE_MATRIX.items()
    }


def group_mean(labels: tuple[str, ...]) -> np.ndarray:
    rows = np.array([FEATURE_MATRIX[label][:3] for label in labels], dtype=float)
    return rows.mean(axis=0)


def projected_points_with_inner_nodes() -> dict[str, np.ndarray]:
    points = projected_leaf_points()
    for node, labels in GROUPS.items():
        points[node] = group_mean(labels)
    return points


def set_equal_axes(ax, points: dict[str, np.ndarray]) -> None:
    xyz = np.vstack(list(points.values()))
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2 + 0.55)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_projected_points(output_dir: Path, points: dict[str, np.ndarray]) -> None:
    with (output_dir / "right_block_3d_projected_points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "kind", "f1", "f2", "f3"])
        for idx in range(1, 17):
            label = f"C{idx}"
            writer.writerow([label, "leaf", *[f"{value:.6g}" for value in points[label]]])
        for label in (*PAIR_INTERNAL_NODES, *MAIN_INTERNAL_NODES):
            writer.writerow([label, "internal", *[f"{value:.6g}" for value in points[label]]])


def draw_leaf_points(ax, points: dict[str, np.ndarray], label_points: bool = True) -> None:
    for idx in range(1, 17):
        label = f"C{idx}"
        xyz = points[label]
        is_special = idx >= 13
        color = "#d62728" if is_special else "#2ca02c"
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=70,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            depthshade=False,
        )
        if label_points:
            ax.text(
                xyz[0] + 0.05,
                xyz[1] + 0.05,
                xyz[2] + 0.05,
                label,
                fontsize=7,
                color=color,
            )


def draw_inner_nodes(ax, points: dict[str, np.ndarray]) -> None:
    for label in PAIR_INTERNAL_NODES:
        xyz = points[label]
        is_special = label in {"C13-C14", "C15-C16"}
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=72,
            marker="s",
            color="#ffb3b3" if is_special else "#c7e9c0",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            alpha=0.95,
        )

    colors = {
        "Root": "#222222",
        "C1-C8": "#74c476",
        "C9-C16": "#f3d77a",
        "C1-C4": "#74c476",
        "C5-C8": "#74c476",
        "C9-C12": "#74c476",
        "C13-C16": "#d62728",
    }
    labeled_nodes = {"Root", "C1-C8", "C9-C16", "C13-C16"}
    for label in MAIN_INTERNAL_NODES:
        xyz = points[label]
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=150 if label != "Root" else 180,
            marker="s",
            color=colors[label],
            edgecolor="black",
            linewidth=1.0,
            depthshade=False,
        )
        if label in labeled_nodes:
            ax.text(
                xyz[0] + 0.07,
                xyz[1] + 0.07,
                xyz[2] + 0.07,
                label,
                fontsize=8,
                color="black",
            )


def style_axes(ax, points: dict[str, np.ndarray]) -> None:
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=23, azim=-52)
    ax.grid(True, alpha=0.25)
    set_equal_axes(ax, points)
    ax.set_box_aspect((1, 1, 1))


def save_leaf_plot(output_dir: Path) -> Path:
    points = projected_leaf_points()
    fig = plt.figure(figsize=(8.2, 6.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    draw_leaf_points(ax, points)
    style_axes(ax, points)
    fig.tight_layout()
    path = output_dir / "right_block_3d_leaf_points.png"
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return path


def save_inner_node_plot(output_dir: Path) -> Path:
    points = projected_points_with_inner_nodes()
    fig = plt.figure(figsize=(8.2, 6.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    draw_leaf_points(ax, points, label_points=False)
    draw_inner_nodes(ax, points)
    style_axes(ax, points)
    fig.tight_layout()
    path = output_dir / "right_block_3d_points_with_inner_nodes.png"
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    write_projected_points(output_dir, points)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/toytrees/right_block_example"),
        help="Directory for generated plots and CSV file.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Generate only the leaf-point plot.",
    )
    parser.add_argument(
        "--inner-nodes",
        action="store_true",
        help="Generate only the leaf-plus-internal-node plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.leaf_only:
        print(save_leaf_plot(args.output_dir))
    elif args.inner_nodes:
        print(save_inner_node_plot(args.output_dir))
    else:
        print(save_leaf_plot(args.output_dir))
        print(save_inner_node_plot(args.output_dir))


if __name__ == "__main__":
    main()
