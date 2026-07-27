#!/usr/bin/env python3
"""Plot the manuscript distance-matrix construction for the handdrawn example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = ("C1", "C2", "C3", "C4", "C5")
FEATURES = ("f1", "f2", "f3")
X = np.array(
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
        [2, 0, 4],
        [3, 1, 5],
    ],
    dtype=float,
)


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diffs = x[:, None, :] - x[None, :, :]
    return np.sqrt((diffs * diffs).sum(axis=2))


def draw_table(
    ax: plt.Axes,
    cell_text: list[list[str]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    scale: tuple[float, float] = (1.25, 1.8),
) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(*scale)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        if row == 0:
            cell.set_facecolor("#f0f3f6")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=20, weight="bold", pad=24)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/toytrees/handdrawn_3d_example"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distances = pairwise_distances(X)

    fig = plt.figure(figsize=(21, 6.4), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1.2], wspace=0.32)
    ax_matrix = fig.add_subplot(gs[0, 0])
    ax_formula = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[0, 2])

    draw_table(
        ax_matrix,
        [[f"{value:.0f}" for value in row] for row in X],
        list(LABELS),
        list(FEATURES),
        "1. Feature matrix",
    )

    ax_formula.axis("off")
    ax_formula.set_title("2. Pairwise distance", fontsize=20, weight="bold", pad=24)
    formula = (
        r"$X_i=(f_{i1},f_{i2},f_{i3})$"
        "\n\n"
        r"$d(C_i,C_j)=\|X_i-X_j\|_2$"
        "\n\n"
        r"$=\sqrt{(f_{i1}-f_{j1})^2+(f_{i2}-f_{j2})^2+(f_{i3}-f_{j3})^2}$"
        "\n\n"
        r"$d(C1,C2)=\sqrt{0^2+(-1)^2+1^2}=\sqrt{2}=1.41$"
    )
    ax_formula.text(0.0, 0.58, formula, fontsize=17, va="center")

    im = ax_heat.imshow(distances, cmap="viridis", vmin=0)
    ax_heat.set_title("3. Distance matrix", fontsize=20, weight="bold", pad=24)
    ax_heat.set_xticks(np.arange(len(LABELS)), LABELS, fontsize=13)
    ax_heat.set_yticks(np.arange(len(LABELS)), LABELS, fontsize=13)
    ax_heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if distances[i, j] > distances.max() * 0.55 else "black"
            ax_heat.text(
                j,
                i,
                f"{distances[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                weight="bold",
            )
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=2)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Euclidean distance", rotation=90, labelpad=12, fontsize=12)

    fig.patch.set_facecolor("white")
    path = args.output_dir / "handdrawn_distance_matrix_steps.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
