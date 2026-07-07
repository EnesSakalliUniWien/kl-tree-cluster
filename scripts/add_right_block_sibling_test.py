#!/usr/bin/env python3
"""Create a simple sibling-gate explanation from the right-block tree plot."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_right_block_toytree import FEATURE_MATRIX

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
    }
)

PARENT = "C9-C16"
LEFT_CHILD = "C9-C12"
RIGHT_CHILD = "C13-C16"
PARENT_LABELS = tuple(f"C{i}" for i in range(9, 17))
LEFT_LABELS = tuple(f"C{i}" for i in range(9, 13))
RIGHT_LABELS = tuple(f"C{i}" for i in range(13, 17))
FEATURE_NAMES = ("f1", "f2", "f3")


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    path = f"/System/Library/Fonts/Supplemental/{name}"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render_formula(lines: tuple[str, ...], width: float, height: float, fontsize: int = 22) -> Image.Image:
    fig = plt.figure(figsize=(width, height), dpi=180)
    fig.patch.set_alpha(0)
    y_positions = (0.86, 0.56, 0.26)
    for line, y in zip(lines, y_positions):
        fig.text(0.0, y, line, fontsize=fontsize, va="center", color="#141414")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


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


def sibling_covariances() -> tuple[np.ndarray, np.ndarray, float]:
    x_parent = np.array([FEATURE_MATRIX[label][:3] for label in PARENT_LABELS])
    parent_cov = np.cov(x_parent, rowvar=False, ddof=1)
    scale = 1 / len(LEFT_LABELS) + 1 / len(RIGHT_LABELS)
    return parent_cov, scale * parent_cov, scale


def color_for_value(value: float, limit: float) -> tuple[int, int, int]:
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


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    x0: int,
    y0: int,
    title: str,
    limit: float,
    font_label: ImageFont.ImageFont,
    font_title: ImageFont.ImageFont,
    cell: int = 40,
) -> None:
    black = (20, 20, 20)
    label = 26
    draw.text((x0, y0), title, font=font_title, fill=black)
    y_grid = y0 + 30

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
                width=2,
            )
    draw.rectangle(
        [x0 + label, y_grid + label, x0 + label + cell * 3, y_grid + label + cell * 3],
        outline=black,
        width=2,
    )


def draw_sibling_callout(input_png: Path, output_png: Path) -> Path:
    canvas = Image.open(input_png).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    parent_cov, contrast_cov, covariance_scale = sibling_covariances()

    black = (20, 20, 20)
    blue = (30, 117, 204)
    dark_blue = (24, 75, 135)
    dark_gray = (80, 80, 80)

    font_title = load_font("Arial Bold.ttf", 34)
    font_subtitle = load_font("Arial Bold.ttf", 24)
    font_body = load_font("Arial.ttf", 22)
    font_small = load_font("Arial.ttf", 18)
    font_heat_title = load_font("Arial Bold.ttf", 17)
    font_label = load_font("Arial Bold.ttf", 14)

    # Remove the feature matrix from the base plot for this explanatory version.
    draw.rectangle([1325, 45, 1900, 840], fill="white")

    # Highlight the shared parent and the two sibling children.
    draw.rounded_rectangle([870, 215, 1035, 315], radius=14, outline=dark_blue, width=5)
    draw.rounded_rectangle([700, 405, 860, 500], radius=12, outline=blue, width=4)
    draw.rounded_rectangle([1040, 405, 1210, 500], radius=12, outline=blue, width=4)
    draw.rounded_rectangle([870, 168, 1035, 208], radius=10, fill="white", outline=dark_blue, width=3)
    center_text(draw, (870, 168, 1035, 208), "shared parent", font_small, dark_blue)

    # Explanation in the cleared area.
    draw.text((1375, 115), "Sibling gate", font=font_title, fill=black)
    draw.text(
        (1375, 165),
        f"Shared parent: {PARENT}",
        font=font_body,
        fill=black,
    )
    draw.text(
        (1375, 205),
        f"Compare children: {LEFT_CHILD} and {RIGHT_CHILD}",
        font=font_body,
        fill=black,
    )
    draw.text(
        (1375, 252),
        "The sibling gate asks whether the two child subtrees",
        font=font_small,
        fill=dark_gray,
    )
    draw.text(
        (1375, 282),
        "have different internal distributions.",
        font=font_small,
        fill=dark_gray,
    )

    formula = (
        r"$H_0^{\mathrm{sib}}(u):\theta_l=\theta_r$"
        ,
        r"$\Delta_u^{\mathrm{sib}}=\theta_l-\theta_r$"
        ,
        r"$\theta_v=\frac{1}{n_v}\sum_{i\in L(v)}X_i$"
    )
    formula_image = render_formula(formula, width=4.7, height=2.05, fontsize=22)
    canvas.paste(formula_image, (1375, 320), formula_image)

    covariance_formula = (
        r"$\widehat{\Sigma}_u=\operatorname{Cov}\{X_i:i\in L(u)\}$",
        r"$\Sigma_u^{\mathrm{sib}}=\left(\frac{1}{n_l}+\frac{1}{n_r}\right)\widehat{\Sigma}_u$",
        rf"$n_l=4,\ n_r=4,\ \mathrm{{scale}}={covariance_scale:.2f}$",
    )
    covariance_formula_image = render_formula(
        covariance_formula,
        width=4.8,
        height=1.55,
        fontsize=16,
    )
    canvas.paste(covariance_formula_image, (1815, 555), covariance_formula_image)

    heatmap_limit = max(float(abs(parent_cov).max()), 1e-12)
    draw_heatmap(
        draw,
        parent_cov,
        1820,
        115,
        "parent/null covariance",
        heatmap_limit,
        font_label,
        font_heat_title,
    )
    draw.text((1820, 305), "from leaves C9,...,C16", font=font_small, fill=dark_gray)
    draw_heatmap(
        draw,
        contrast_cov,
        1820,
        350,
        "sibling contrast covariance",
        heatmap_limit,
        font_label,
        font_heat_title,
    )
    draw.text((1820, 540), "used for theta_l - theta_r", font=font_small, fill=dark_gray)

    draw.text((1375, 700), "Here:", font=font_subtitle, fill=black)
    draw.text((1375, 740), r"u = C9-C16", font=font_body, fill=black)
    draw.text((1375, 777), r"l = C9-C12", font=font_body, fill=blue)
    draw.text((1375, 814), r"r = C13-C16", font=font_body, fill=blue)
    draw.text(
        (1375, 860),
        "The signal is the difference between the two child means.",
        font=font_small,
        fill=dark_gray,
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
    input_png = args.output_dir / "right_block_realistic_distances_clean.png"
    output_png = args.output_dir / "right_block_realistic_distances_sibling_test.png"
    print(draw_sibling_callout(input_png, output_png))


if __name__ == "__main__":
    main()
