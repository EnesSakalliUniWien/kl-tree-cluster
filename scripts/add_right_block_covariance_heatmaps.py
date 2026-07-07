#!/usr/bin/env python3
"""Add covariance heatmaps beside the tested right-block edge."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_right_block_toytree import FEATURE_MATRIX

EDGE_PARENT = "C9-C16"
EDGE_CHILD = "C13-C16"
PARENT_LABELS = tuple(f"C{i}" for i in range(9, 17))
CHILD_LABELS = tuple(f"C{i}" for i in range(13, 17))
FEATURE_NAMES = ("f1", "f2", "f3")


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    path = f"/System/Library/Fonts/Supplemental/{name}"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def covariances() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_parent = np.array([FEATURE_MATRIX[label][:3] for label in PARENT_LABELS])
    x_child = np.array([FEATURE_MATRIX[label][:3] for label in CHILD_LABELS])
    parent_cov = np.cov(x_parent, rowvar=False, ddof=1)
    child_cov = np.cov(x_child, rowvar=False, ddof=1)
    scale = 1 / len(CHILD_LABELS) - 1 / len(PARENT_LABELS)
    edge_cov = scale * parent_cov
    return parent_cov, child_cov, edge_cov, scale


def color_for_value(value: float, limit: float) -> tuple[int, int, int]:
    """Diverging blue-white-red color map for covariance values."""
    t = max(-1.0, min(1.0, value / limit))
    if t >= 0:
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 211, 47, 47
        a = t
    else:
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 25, 118, 210
        a = -t
    return (
        int(r0 + (r1 - r0) * a),
        int(g0 + (g1 - g0) * a),
        int(b0 + (b1 - b0) * a),
    )


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | str,
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


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    x0: int,
    y0: int,
    title: str,
    limit: float,
    font_label: ImageFont.ImageFont,
    font_title: ImageFont.ImageFont,
    cell: int = 50,
) -> None:
    black = (20, 20, 20)
    label = 30
    draw.text((x0, y0), title, font=font_title, fill=black)
    y_grid = y0 + 34

    for idx, name in enumerate(FEATURE_NAMES):
        center_text(
            draw,
            (x0 + label + idx * cell, y_grid, x0 + label + (idx + 1) * cell, y_grid + label),
            name,
            font_label,
            black,
        )
        center_text(
            draw,
            (x0, y_grid + label + idx * cell, x0 + label, y_grid + label + (idx + 1) * cell),
            name,
            font_label,
            black,
        )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            x = x0 + label + col * cell
            y = y_grid + label + row * cell
            draw.rectangle(
                [x, y, x + cell, y + cell],
                fill=color_for_value(float(matrix[row, col]), limit),
                outline="white",
                width=3,
            )
    draw.rectangle(
        [x0 + label, y_grid + label, x0 + label + cell * 3, y_grid + label + cell * 3],
        outline=black,
        width=2,
    )


def render_formula(text: str, width: float, height: float, fontsize: int = 23) -> Image.Image:
    fig = plt.figure(figsize=(width, height), dpi=180)
    fig.patch.set_alpha(0)
    fig.text(0.0, 0.92, text, fontsize=fontsize, va="top", color="#141414")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def draw_colorbar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    height: int,
    limit: float,
    font: ImageFont.ImageFont,
) -> None:
    for idx in range(width):
        t = -1 + 2 * idx / max(width - 1, 1)
        draw.line(
            [(x0 + idx, y0), (x0 + idx, y0 + height)],
            fill=color_for_value(t * limit, limit),
        )
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=(20, 20, 20), width=1)
    draw.text((x0, y0 + height + 5), "-", font=font, fill=(20, 20, 20))
    center_text(
        draw,
        (x0 + width // 2 - 25, y0 + height + 5, x0 + width // 2 + 25, y0 + height + 25),
        "0",
        font,
        (20, 20, 20),
    )
    draw.text((x0 + width - 12, y0 + height + 5), "+", font=font, fill=(20, 20, 20))


def draw_heatmap_callout(input_png: Path, output_png: Path) -> Path:
    base = Image.open(input_png).convert("RGB")
    parent_cov, child_cov, edge_cov, scale = covariances()

    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)

    dark_gray = (80, 80, 80)
    font_small = load_font("Arial.ttf", 17)
    font_note = load_font("Arial.ttf", 20)
    font_heat_title = load_font("Arial Bold.ttf", 18)
    font_label = load_font("Arial Bold.ttf", 16)

    # Remove the feature matrix from the base plot for this explanatory version.
    draw.rectangle([1325, 45, 1900, 840], fill="white")

    parent_limit = max(float(abs(parent_cov).max()), 1e-12)
    child_limit = max(float(abs(child_cov).max()), 1e-12)
    draw_heatmap(
        draw,
        parent_cov,
        1325,
        105,
        "parent feature covariance",
        parent_limit,
        font_label,
        font_heat_title,
    )
    draw.text((1325, 328), "from leaves C9,...,C16", font=font_small, fill=dark_gray)

    draw_heatmap(
        draw,
        child_cov,
        1325,
        410,
        "child feature covariance",
        child_limit,
        font_label,
        font_heat_title,
    )
    draw.text((1325, 633), "from leaves C13,...,C16", font=font_small, fill=dark_gray)

    draw.text(
        (1645, 180),
        "feature-feature covariance",
        font=font_note,
        fill=(20, 20, 20),
    )
    draw.text(
        (1645, 215),
        "computed from descendant leaf vectors",
        font=font_note,
        fill=dark_gray,
    )
    formula = (
        r"$\theta_u=\frac{1}{n_u}\sum_{i\in L(u)}X_i$"
        "\n"
        r"$\delta_i=X_i-\theta_u$"
        "\n"
        r"$\widehat{\Sigma}_u=\frac{1}{n_u-1}\sum_{i\in L(u)}\delta_i\delta_i^\top$"
    )
    formula_image = render_formula(formula, width=3.6, height=1.5, fontsize=20)
    canvas.paste(formula_image, (1645, 270), formula_image)

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
    input_png = args.output_dir / "right_block_realistic_distances_clean.png"
    output_png = args.output_dir / "right_block_realistic_distances_with_heatmaps.png"
    print(draw_heatmap_callout(input_png, output_png))


if __name__ == "__main__":
    main()
