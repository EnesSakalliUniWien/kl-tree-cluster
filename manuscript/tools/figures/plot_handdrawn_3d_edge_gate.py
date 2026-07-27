#!/usr/bin/env python3
"""Plot the manuscript handdrawn five-cell example in 3D feature space.

The plot uses the feature matrix from the handdrawn note:

    C1 = (0, 1, 2)
    C2 = (0, 2, 1)
    C3 = (1, 3, 3)
    C4 = (2, 0, 4)
    C5 = (3, 1, 5)

Internal node distributions are leaf-count-weighted descendant means.
The visible handdrawn subtree is ``((C3, C4), C5)``. To place all five
rows in one 3D tree, the script also includes ``(C1, C2)`` as the other
branch under the root.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEAF_POINTS: dict[str, tuple[float, float, float]] = {
    "C1": (0.0, 1.0, 2.0),
    "C2": (0.0, 2.0, 1.0),
    "C3": (1.0, 3.0, 3.0),
    "C4": (2.0, 0.0, 4.0),
    "C5": (3.0, 1.0, 5.0),
}

GROUPS: dict[str, tuple[str, ...]] = {
    "C1-C2": ("C1", "C2"),
    "C3-C4": ("C3", "C4"),
    "C3-C5": ("C3", "C4", "C5"),
    "Root": ("C1", "C2", "C3", "C4", "C5"),
}

EDGES: tuple[tuple[str, str], ...] = (
    ("Root", "C1-C2"),
    ("Root", "C3-C5"),
    ("C1-C2", "C1"),
    ("C1-C2", "C2"),
    ("C3-C5", "C3-C4"),
    ("C3-C5", "C5"),
    ("C3-C4", "C3"),
    ("C3-C4", "C4"),
)


def node_mean(labels: tuple[str, ...]) -> np.ndarray:
    rows = np.array([LEAF_POINTS[label] for label in labels], dtype=float)
    return rows.mean(axis=0)


def build_points() -> dict[str, np.ndarray]:
    points = {label: np.array(value, dtype=float) for label, value in LEAF_POINTS.items()}
    for node, labels in GROUPS.items():
        points[node] = node_mean(labels)
    return points


def write_tables(output_dir: Path, points: dict[str, np.ndarray]) -> None:
    with (output_dir / "handdrawn_3d_points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "kind", "f1", "f2", "f3"])
        for label in ("C1", "C2", "C3", "C4", "C5"):
            writer.writerow([label, "leaf", *[f"{x:.6g}" for x in points[label]]])
        for label in ("C1-C2", "C3-C4", "C3-C5", "Root"):
            writer.writerow([label, "internal", *[f"{x:.6g}" for x in points[label]]])

    with (output_dir / "handdrawn_3d_edge_distances.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "distance"])
        for parent, child in EDGES:
            distance = float(np.linalg.norm(points[child] - points[parent]))
            writer.writerow([parent, child, f"{distance:.6g}"])


def set_equal_axes(ax, points: dict[str, np.ndarray]) -> None:
    xyz = np.vstack(list(points.values()))
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2 + 0.35)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot(output_dir: Path, points: dict[str, np.ndarray]) -> Path:
    fig = plt.figure(figsize=(9.5, 7.5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    leaf_colors = {
        "C1": "#d62728",
        "C2": "#d62728",
        "C3": "#1f77b4",
        "C4": "#1f77b4",
        "C5": "#2ca02c",
    }
    internal_colors = {
        "C1-C2": "#ffb3b3",
        "C3-C4": "#9ecae1",
        "C3-C5": "#9467bd",
        "Root": "#222222",
    }

    for parent, child in EDGES:
        start = points[parent]
        end = points[child]
        color = "#d62728" if child in {"C3-C5", "C5", "C13-C16"} else "#5b6b73"
        if child in {"C3-C4", "C3", "C4"}:
            color = "#1f77b4"
        if child == "C5":
            color = "#2ca02c"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=2.5,
            alpha=0.9,
        )

        midpoint = (start + end) / 2
        distance = float(np.linalg.norm(end - start))
        if parent in {"C3-C5", "C3-C4", "Root"}:
            ax.text(
                midpoint[0],
                midpoint[1],
                midpoint[2] + 0.08,
                f"{distance:.2f}",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
            )

    for label in ("C1", "C2", "C3", "C4", "C5"):
        xyz = points[label]
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=80,
            color=leaf_colors[label],
            edgecolor="white",
            linewidth=1.2,
            depthshade=False,
        )
        ax.text(xyz[0] + 0.06, xyz[1] + 0.06, xyz[2] + 0.06, label, fontsize=9)

    for label in ("C1-C2", "C3-C4", "C3-C5", "Root"):
        xyz = points[label]
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=145 if label != "Root" else 175,
            marker="s",
            color=internal_colors[label],
            edgecolor="black",
            linewidth=1.0,
            depthshade=False,
        )
        ax.text(xyz[0] + 0.06, xyz[1] + 0.06, xyz[2] + 0.08, label, fontsize=9)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=23, azim=-52)
    ax.grid(True, alpha=0.25)
    set_equal_axes(ax, points)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()

    path = output_dir / "handdrawn_3d_edge_gate.png"
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return path


def plot_leaf_points_only(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(8.0, 6.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "C1": "#d62728",
        "C2": "#d62728",
        "C3": "#1f77b4",
        "C4": "#1f77b4",
        "C5": "#2ca02c",
    }
    points = {label: np.array(value, dtype=float) for label, value in LEAF_POINTS.items()}
    for label, xyz in points.items():
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=100,
            color=colors[label],
            edgecolor="white",
            linewidth=1.2,
            depthshade=False,
        )
        ax.text(xyz[0] + 0.06, xyz[1] + 0.06, xyz[2] + 0.06, label, fontsize=10)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=23, azim=-52)
    ax.grid(True, alpha=0.25)
    set_equal_axes(ax, points)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()

    path = output_dir / "handdrawn_3d_points_only.png"
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return path


def plot_points_with_inner_nodes(output_dir: Path, points: dict[str, np.ndarray]) -> Path:
    fig = plt.figure(figsize=(8.0, 6.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    leaf_colors = {
        "C1": "#d62728",
        "C2": "#d62728",
        "C3": "#1f77b4",
        "C4": "#1f77b4",
        "C5": "#2ca02c",
    }
    internal_colors = {
        "C1-C2": "#ffb3b3",
        "C3-C4": "#9ecae1",
        "C3-C5": "#9467bd",
        "Root": "#222222",
    }

    for label in ("C1", "C2", "C3", "C4", "C5"):
        xyz = points[label]
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=92,
            color=leaf_colors[label],
            edgecolor="white",
            linewidth=1.2,
            depthshade=False,
        )
        ax.text(xyz[0] + 0.06, xyz[1] + 0.06, xyz[2] + 0.06, label, fontsize=10)

    for label in ("C1-C2", "C3-C4", "C3-C5", "Root"):
        xyz = points[label]
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=155 if label != "Root" else 180,
            marker="s",
            color=internal_colors[label],
            edgecolor="black",
            linewidth=1.0,
            depthshade=False,
        )
        ax.text(xyz[0] + 0.06, xyz[1] + 0.06, xyz[2] + 0.08, label, fontsize=9)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=23, azim=-52)
    ax.grid(True, alpha=0.25)
    set_equal_axes(ax, points)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()

    path = output_dir / "handdrawn_3d_points_with_inner_nodes.png"
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/toytrees/handdrawn_3d_example"),
        help="Directory for generated image and CSV files.",
    )
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Plot only the five leaf points, without internal nodes or edges.",
    )
    parser.add_argument(
        "--inner-nodes",
        action="store_true",
        help="Plot leaf points plus internal-node means, without edges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = build_points()
    write_tables(args.output_dir, points)
    if args.points_only:
        print(plot_leaf_points_only(args.output_dir))
    elif args.inner_nodes:
        print(plot_points_with_inner_nodes(args.output_dir, points))
    else:
        print(plot(args.output_dir, points))


if __name__ == "__main__":
    main()
