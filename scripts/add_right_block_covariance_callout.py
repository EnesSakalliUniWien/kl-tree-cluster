#!/usr/bin/env python3
"""Append a covariance explanation to the existing right-block tree plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_right_block_toytree import FEATURE_MATRIX

EDGE_PARENT = "C9-C16"
EDGE_CHILD = "C13-C16"
PARENT_LABELS = tuple(f"C{i}" for i in range(9, 17))
CHILD_LABELS = tuple(f"C{i}" for i in range(13, 17))
FEATURE_NAMES_3D = ("f1", "f2", "f3")
FEATURE_NAMES_ALL = ("f1", "f2", "f3", "f4", "f5", "f6")


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    path = f"/System/Library/Fonts/Supplemental/{name}"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def edge_covariances(
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_parent = np.array([FEATURE_MATRIX[label][:feature_count] for label in PARENT_LABELS])
    parent_cov = np.cov(x_parent, rowvar=False, ddof=1)
    scale = 1 / len(CHILD_LABELS) - 1 / len(PARENT_LABELS)
    edge_cov = scale * parent_cov
    child_cov = np.cov(
        np.array([FEATURE_MATRIX[label][:feature_count] for label in CHILD_LABELS]),
        rowvar=False,
        ddof=1,
    )
    return parent_cov, edge_cov, child_cov, scale


def write_matrix(path: Path, matrix: np.ndarray, feature_names: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *feature_names])
        for name, row in zip(feature_names, matrix):
            writer.writerow([name, *[f"{value:.6f}" for value in row]])


def matrix_lines(matrix: np.ndarray) -> list[str]:
    return [
        "[" + "  ".join(f"{value:8.3f}" for value in row) + "]"
        for row in matrix
    ]


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = xy
    draw.text(
        (
            (x0 + x1 - bbox[2] + bbox[0]) / 2,
            (y0 + y1 - bbox[3] + bbox[1]) / 2,
        ),
        text,
        font=font,
        fill=fill,
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int],
) -> None:
    draw.line([start, end], fill=fill, width=6)
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 18
    tip = (x1, y1)
    left = (x1 - ux * size + px * size * 0.55, y1 - uy * size + py * size * 0.55)
    right = (x1 - ux * size - px * size * 0.55, y1 - uy * size - py * size * 0.55)
    draw.polygon([tip, left, right], fill=fill)


def draw_callout(input_png: Path, output_png: Path, output_dir: Path) -> Path:
    base = Image.open(input_png).convert("RGB")
    parent_cov, edge_cov, child_cov, scale = edge_covariances(3)
    parent_cov_all, edge_cov_all, child_cov_all, _ = edge_covariances(6)

    write_matrix(
        output_dir / "right_block_parent_covariance_C9_C16_f1_f3.csv",
        parent_cov,
        FEATURE_NAMES_3D,
    )
    write_matrix(
        output_dir / "right_block_edge_covariance_C9_C16_to_C13_C16_f1_f3.csv",
        edge_cov,
        FEATURE_NAMES_3D,
    )
    write_matrix(
        output_dir / "right_block_child_covariance_C13_C16_f1_f3.csv",
        child_cov,
        FEATURE_NAMES_3D,
    )
    write_matrix(
        output_dir / "right_block_parent_covariance_C9_C16_f1_f6.csv",
        parent_cov_all,
        FEATURE_NAMES_ALL,
    )
    write_matrix(
        output_dir / "right_block_edge_covariance_C9_C16_to_C13_C16_f1_f6.csv",
        edge_cov_all,
        FEATURE_NAMES_ALL,
    )
    write_matrix(
        output_dir / "right_block_child_covariance_C13_C16_f1_f6.csv",
        child_cov_all,
        FEATURE_NAMES_ALL,
    )

    width, height = base.size
    panel_height = 500
    canvas = Image.new("RGB", (width, height + panel_height), "white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    red = (217, 54, 54)
    black = (20, 20, 20)
    gray = (90, 90, 90)
    light_gray = (246, 247, 248)
    pale_red = (255, 235, 235)

    font_title = load_font("Arial Bold.ttf", 33)
    font_sub = load_font("Arial.ttf", 25)
    font_bold = load_font("Arial Bold.ttf", 25)
    font_mono = load_font("Courier New.ttf", 24)
    font_small = load_font("Arial.ttf", 21)

    # Highlight the actual edge in the existing block-tree plot.
    draw_arrow(draw, (955, 315), (1085, 425), red)
    draw.rounded_rectangle([930, 330, 1135, 510], radius=16, outline=red, width=6)
    draw.rounded_rectangle([860, 285, 1170, 335], radius=12, fill="white", outline=red, width=4)
    center_text(
        draw,
        (860, 285, 1170, 335),
        "covariance shown below",
        font_bold,
        red,
    )

    y0 = height + 25
    draw.rounded_rectangle(
        [55, y0, width - 55, height + panel_height - 35],
        radius=18,
        fill=light_gray,
        outline=(205, 205, 205),
        width=3,
    )
    draw.text(
        (95, y0 + 30),
        f"Covariance for one edge: {EDGE_PARENT} -> {EDGE_CHILD}",
        font=font_title,
        fill=black,
    )
    draw.text(
        (95, y0 + 86),
        "The rows come from the feature matrix in the plot. For the parent/null covariance, use all descendant leaves under C9-C16: C9,...,C16. The child is the subset C13,...,C16.",
        font=font_sub,
        fill=black,
    )
    draw.text(
        (95, y0 + 128),
        "Shown for the plotted 3D features f1, f2, f3. Continuous empirical-Gaussian covariance is used because these toy features are real-valued.",
        font=font_small,
        fill=gray,
    )

    box_y = y0 + 180
    box_h = 170
    box_w = 690
    boxes = [
        (95, "parent covariance  Sigma_hat_u", parent_cov),
        (880, "edge contrast covariance", edge_cov),
        (1665, "child covariance, context only", child_cov),
    ]
    for x, title, matrix in boxes:
        fill = pale_red if "edge" in title else "white"
        outline = red if "edge" in title else (190, 190, 190)
        draw.rounded_rectangle([x, box_y, x + box_w, box_y + box_h], radius=12, fill=fill, outline=outline, width=3)
        draw.text((x + 24, box_y + 18), title, font=font_bold, fill=black)
        for idx, line in enumerate(matrix_lines(matrix)):
            draw.text((x + 24, box_y + 62 + idx * 32), line, font=font_mono, fill=black)

    formula_y = box_y + box_h + 28
    formula = (
        "Sigma_edge = (1/n_c - 1/n_u) Sigma_hat_u,   "
        f"n_u = 8, n_c = 4, scale = {scale:.3f}"
    )
    draw.text((95, formula_y), formula, font=font_bold, fill=black)
    draw.text(
        (95, formula_y + 38),
        "This covariance whitens the edge contrast theta_c - theta_u. The child covariance is shown only to explain the data block; it is not the null edge covariance.",
        font=font_small,
        fill=gray,
    )
    draw.text(
        (95, formula_y + 72),
        "The PNG shows the f1-f3 projection for readability. The full f1-f6 covariance matrices are written as CSV files in the same output folder.",
        font=font_small,
        fill=gray,
    )

    canvas.save(output_png)
    return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/toytrees/right_block_example"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_png = args.output_dir / "right_block_realistic_distances_clean.png"
    output_png = args.output_dir / "right_block_realistic_distances_with_covariance.png"
    path = draw_callout(input_png, output_png, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
