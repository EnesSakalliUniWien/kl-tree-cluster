#!/usr/bin/env python3
"""Generate a 16-leaf toy tree with one right-side changed block.

The figure is intended as a compact explanatory visual:

- leaves C1-C12 are noisy samples from one shared/background profile;
- leaves C13-C16 are noisy samples from a different profile;
- internal node barplots show descendant-leaf mean feature vectors;
- edge labels show Euclidean distances between parent and child means.

Requires ``toytree``, ``toyplot``, and ``Pillow``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import toyplot.png
import toytree
from PIL import Image, ImageDraw, ImageFont

TOPOLOGY_NEWICK = (
    "((((C1,C2),(C3,C4)),((C5,C6),(C7,C8))),"
    "(((C9,C10),(C11,C12)),((C13,C14),(C15,C16))));"
)

FEATURES = ("f1", "f2", "f3", "f4", "f5", "f6")

FEATURE_MATRIX: dict[str, tuple[float, ...]] = {
    "C1": (1.0, 1.1, 5.0, 4.9, 5.2, 4.8),
    "C2": (1.2, 0.9, 4.8, 5.1, 5.0, 5.1),
    "C3": (0.9, 1.0, 5.1, 5.0, 4.9, 5.0),
    "C4": (1.1, 1.2, 4.9, 5.2, 5.1, 4.9),
    "C5": (1.0, 0.8, 5.2, 5.0, 4.8, 5.0),
    "C6": (1.3, 1.1, 4.7, 5.1, 5.2, 4.9),
    "C7": (0.8, 1.0, 5.0, 4.8, 5.1, 5.2),
    "C8": (1.1, 0.9, 5.1, 5.2, 4.9, 4.8),
    "C9": (1.0, 1.2, 5.0, 5.1, 5.0, 4.9),
    "C10": (0.9, 1.0, 5.2, 4.9, 4.8, 5.1),
    "C11": (1.2, 0.8, 4.9, 5.0, 5.2, 5.0),
    "C12": (1.1, 1.1, 4.8, 5.2, 5.1, 4.9),
    "C13": (8.8, 9.1, 1.0, 1.2, 0.9, 1.1),
    "C14": (9.2, 8.9, 1.1, 0.8, 1.2, 1.0),
    "C15": (9.0, 9.2, 0.9, 1.0, 1.1, 0.8),
    "C16": (8.9, 8.8, 1.2, 1.1, 1.0, 1.2),
}

GROUPS: dict[str, tuple[str, ...]] = {
    "C1-C2": ("C1", "C2"),
    "C3-C4": ("C3", "C4"),
    "C5-C6": ("C5", "C6"),
    "C7-C8": ("C7", "C8"),
    "C9-C10": ("C9", "C10"),
    "C11-C12": ("C11", "C12"),
    "C13-C14": ("C13", "C14"),
    "C15-C16": ("C15", "C16"),
    "C1-C4": ("C1", "C2", "C3", "C4"),
    "C5-C8": ("C5", "C6", "C7", "C8"),
    "C9-C12": ("C9", "C10", "C11", "C12"),
    "C13-C16": ("C13", "C14", "C15", "C16"),
    "C1-C8": ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
    "C9-C16": (
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
    ),
    "Root": tuple(f"C{i}" for i in range(1, 17)),
}

EDGES: tuple[tuple[str, str], ...] = (
    ("Root", "C1-C8"),
    ("Root", "C9-C16"),
    ("C1-C8", "C1-C4"),
    ("C1-C8", "C5-C8"),
    ("C9-C16", "C9-C12"),
    ("C9-C16", "C13-C16"),
    ("C1-C4", "C1-C2"),
    ("C1-C4", "C3-C4"),
    ("C5-C8", "C5-C6"),
    ("C5-C8", "C7-C8"),
    ("C9-C12", "C9-C10"),
    ("C9-C12", "C11-C12"),
    ("C13-C16", "C13-C14"),
    ("C13-C16", "C15-C16"),
    ("C1-C2", "C1"),
    ("C1-C2", "C2"),
    ("C3-C4", "C3"),
    ("C3-C4", "C4"),
    ("C5-C6", "C5"),
    ("C5-C6", "C6"),
    ("C7-C8", "C7"),
    ("C7-C8", "C8"),
    ("C9-C10", "C9"),
    ("C9-C10", "C10"),
    ("C11-C12", "C11"),
    ("C11-C12", "C12"),
    ("C13-C14", "C13"),
    ("C13-C14", "C14"),
    ("C15-C16", "C15"),
    ("C15-C16", "C16"),
)


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    path = f"/System/Library/Fonts/Supplemental/{name}"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def mean(labels: Iterable[str]) -> tuple[float, ...]:
    label_list = tuple(labels)
    return tuple(
        sum(FEATURE_MATRIX[label][feature_idx] for label in label_list)
        / len(label_list)
        for feature_idx in range(len(FEATURES))
    )


def euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right)))


def build_node_means() -> dict[str, tuple[float, ...]]:
    node_means = {name: mean(labels) for name, labels in GROUPS.items()}
    node_means.update(FEATURE_MATRIX)
    return node_means


def build_edge_distances(
    node_means: dict[str, tuple[float, ...]],
) -> dict[tuple[str, str], float]:
    return {
        (parent, child): euclidean(node_means[parent], node_means[child])
        for parent, child in EDGES
    }


def edge_distance_newick(edge_distances: dict[tuple[str, str], float]) -> str:
    def bl(parent: str, child: str) -> str:
        return f"{edge_distances[(parent, child)]:.4f}"

    return (
        f'((((C1:{bl("C1-C2", "C1")},C2:{bl("C1-C2", "C2")}):'
        f'{bl("C1-C4", "C1-C2")},'
        f'(C3:{bl("C3-C4", "C3")},C4:{bl("C3-C4", "C4")}):'
        f'{bl("C1-C4", "C3-C4")}):{bl("C1-C8", "C1-C4")},'
        f'((C5:{bl("C5-C6", "C5")},C6:{bl("C5-C6", "C6")}):'
        f'{bl("C5-C8", "C5-C6")},'
        f'(C7:{bl("C7-C8", "C7")},C8:{bl("C7-C8", "C8")}):'
        f'{bl("C5-C8", "C7-C8")}):{bl("C1-C8", "C5-C8")}):'
        f'{bl("Root", "C1-C8")},'
        f'(((C9:{bl("C9-C10", "C9")},C10:{bl("C9-C10", "C10")}):'
        f'{bl("C9-C12", "C9-C10")},'
        f'(C11:{bl("C11-C12", "C11")},C12:{bl("C11-C12", "C12")}):'
        f'{bl("C9-C12", "C11-C12")}):{bl("C9-C16", "C9-C12")},'
        f'((C13:{bl("C13-C14", "C13")},C14:{bl("C13-C14", "C14")}):'
        f'{bl("C13-C16", "C13-C14")},'
        f'(C15:{bl("C15-C16", "C15")},C16:{bl("C15-C16", "C16")}):'
        f'{bl("C13-C16", "C15-C16")}):{bl("C9-C16", "C13-C16")}):'
        f'{bl("Root", "C9-C16")});'
    )


def write_tables(
    output_dir: Path,
    node_means: dict[str, tuple[float, ...]],
    edge_distances: dict[tuple[str, str], float],
) -> None:
    with (output_dir / "right_block_realistic_features.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", *FEATURES])
        for idx in range(1, 17):
            writer.writerow(
                [f"C{idx}", *[f"{value:.1f}" for value in FEATURE_MATRIX[f"C{idx}"]]]
            )

    with (output_dir / "right_block_realistic_edge_distances.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "distance"])
        for parent, child in EDGES:
            writer.writerow([parent, child, f"{edge_distances[(parent, child)]:.4f}"])

    with (output_dir / "right_block_realistic_inner_means.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["node", *FEATURES])
        for node in (
            "Root",
            "C1-C8",
            "C9-C16",
            "C1-C4",
            "C5-C8",
            "C9-C12",
            "C13-C16",
        ):
            writer.writerow([node, *[f"{value:.4f}" for value in node_means[node]]])

    (output_dir / "right_block_realistic_distance_tree.nwk").write_text(
        edge_distance_newick(edge_distances) + "\n"
    )


def center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    rect: tuple[float, float, float, float],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | str,
) -> None:
    x1, y1, x2, y2 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((x1 + x2) / 2 - (bbox[2] - bbox[0]) / 2, (y1 + y2) / 2 - (bbox[3] - bbox[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


def render_toytree_png(output_dir: Path) -> Path:
    special = {"C13", "C14", "C15", "C16"}
    tree = toytree.tree(TOPOLOGY_NEWICK)
    node_colors = []
    edge_colors = []
    node_sizes = []
    for node in tree:
        leaves = {leaf.name for leaf in node.get_leaves()}
        color = "#d93636" if leaves and leaves.issubset(special) else "#5b6b73"
        node_colors.append(color)
        edge_colors.append(color)
        node_sizes.append(12 if leaves == special else 7)
    tip_colors = [
        "#d93636" if name in special else "#2a9d5b" for name in tree.get_tip_labels()
    ]

    canvas, _, _ = tree.draw(
        width=1000,
        height=760,
        layout="d",
        tip_labels=True,
        tip_labels_colors=tip_colors,
        tip_labels_style={"font-size": "16px", "font-weight": "bold"},
        node_colors=node_colors,
        node_sizes=node_sizes,
        edge_colors=edge_colors,
        edge_widths=3.5,
        edge_type="c",
        use_edge_lengths=False,
        scale_bar=False,
        padding=45,
    )
    tree_png = output_dir / "right_block_realistic_toytree_base.png"
    toyplot.png.render(canvas, str(tree_png), scale=2)
    return tree_png


def make_white_tree(tree_png: Path) -> Image.Image:
    tree_image = Image.open(tree_png).convert("RGBA")
    alpha = tree_image.getchannel("A")
    tree_image = tree_image.crop(alpha.getbbox())
    white_tree = Image.new("RGBA", tree_image.size, "white")
    white_tree.alpha_composite(tree_image)
    return white_tree.convert("RGB").resize((1280, 890), Image.Resampling.LANCZOS)


def draw_mini_barplot(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    label: str,
    values: tuple[float, ...],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    font_node: ImageFont.ImageFont,
    width: int = 150,
    height: int = 84,
) -> None:
    black = (20, 20, 20)
    feature_colors = (
        (217, 54, 54),
        (235, 96, 56),
        (42, 157, 91),
        (42, 157, 91),
        (42, 157, 91),
        (42, 157, 91),
    )
    x0 = int(center_x - width / 2)
    y0 = int(center_y - height / 2)
    draw.rounded_rectangle(
        [x0, y0, x0 + width, y0 + height],
        radius=7,
        fill=fill,
        outline=outline,
        width=2,
    )
    center_text(draw, label, (x0 + 4, y0 + 4, x0 + width - 4, y0 + 22), font_node, outline)
    plot_x = x0 + 15
    plot_y = y0 + 28
    plot_width = width - 30
    plot_height = 42
    base_y = plot_y + plot_height
    draw.line([plot_x, base_y, plot_x + plot_width, base_y], fill=(80, 80, 80), width=1)
    gap = 5
    bar_width = int((plot_width - gap * 5) / 6)
    for feature_idx, value in enumerate(values):
        bar_height = int((value / 9.5) * plot_height)
        bar_x = plot_x + feature_idx * (bar_width + gap)
        bar_y = base_y - bar_height
        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width, base_y],
            fill=feature_colors[feature_idx],
            outline=black,
            width=1,
        )


def draw_distance_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    value: float,
    outline: tuple[int, int, int],
    font: ImageFont.ImageFont,
    width: int = 66,
) -> None:
    draw.rounded_rectangle(
        [x - width / 2, y - 14, x + width / 2, y + 14],
        radius=5,
        fill="white",
        outline=outline,
        width=2,
    )
    center_text(draw, f"{value:.2f}", (x - width / 2, y - 14, x + width / 2, y + 14), font, outline)


def draw_feature_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    font_cell: ImageFont.ImageFont,
) -> None:
    black = (20, 20, 20)
    red = (217, 54, 54)
    green = (42, 157, 91)
    light_red = (255, 231, 231)
    light_green = (229, 247, 236)
    light_grey = (245, 247, 248)
    feature_colors = (
        (217, 54, 54),
        (235, 96, 56),
        (42, 157, 91),
        (42, 157, 91),
        (42, 157, 91),
        (42, 157, 91),
    )
    leaf_width = 72
    cell_width = 58
    row_height = 40
    widths = [leaf_width] + [cell_width] * len(FEATURES)
    headers = ["cell", *FEATURES]
    for idx, header in enumerate(headers):
        x = x0 + sum(widths[:idx])
        draw.rectangle(
            [x, y0, x + widths[idx], y0 + row_height],
            fill=light_grey,
            outline=black,
            width=2,
        )
        center_text(draw, header, (x, y0, x + widths[idx], y0 + row_height), font_cell, black)

    for idx in range(1, 17):
        row_y = y0 + row_height * idx
        is_special = idx >= 13
        base_fill = light_red if is_special else light_green
        label_color = red if is_special else green
        draw.rectangle(
            [x0, row_y, x0 + leaf_width, row_y + row_height],
            fill=base_fill,
            outline=black,
            width=1,
        )
        center_text(
            draw,
            f"C{idx}",
            (x0, row_y, x0 + leaf_width, row_y + row_height),
            font_cell,
            label_color,
        )
        for feature_idx, value in enumerate(FEATURE_MATRIX[f"C{idx}"]):
            x = x0 + leaf_width + cell_width * feature_idx
            if is_special and feature_idx in (0, 1):
                fill = feature_colors[feature_idx]
                text_fill: tuple[int, int, int] | str = "white"
            elif not is_special and feature_idx >= 2:
                fill = green
                text_fill = "white"
            else:
                fill = base_fill
                text_fill = black
            draw.rectangle(
                [x, row_y, x + cell_width, row_y + row_height],
                fill=fill,
                outline=black,
                width=1,
            )
            center_text(
                draw,
                f"{value:.1f}",
                (x, row_y, x + cell_width, row_y + row_height),
                font_cell,
                text_fill,
            )


def draw_figure(
    output_dir: Path,
    node_means: dict[str, tuple[float, ...]],
    edge_distances: dict[tuple[str, str], float],
) -> Path:
    black = (20, 20, 20)
    red = (217, 54, 54)
    green = (42, 157, 91)
    light_red = (255, 231, 231)
    light_green = (229, 247, 236)
    light_grey = (245, 247, 248)
    yellow = (255, 246, 212)

    font_node = load_font("Arial Bold.ttf", 17)
    font_dist = load_font("Arial Bold.ttf", 17)
    font_cell = load_font("Arial Bold.ttf", 20)

    tree_png = render_toytree_png(output_dir)
    tree_image = make_white_tree(tree_png)

    canvas = Image.new("RGB", (2450, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(tree_image, (35, 70))

    draw_mini_barplot(draw, 635, 75, "Root", node_means["Root"], light_grey, black, font_node)
    draw_mini_barplot(draw, 335, 265, "C1-C8", node_means["C1-C8"], light_green, green, font_node)
    draw_mini_barplot(draw, 950, 265, "C9-C16", node_means["C9-C16"], yellow, black, font_node)
    draw_mini_barplot(draw, 180, 455, "C1-C4", node_means["C1-C4"], light_green, green, font_node, 140, 80)
    draw_mini_barplot(draw, 490, 455, "C5-C8", node_means["C5-C8"], light_green, green, font_node, 140, 80)
    draw_mini_barplot(draw, 775, 455, "C9-C12", node_means["C9-C12"], light_green, green, font_node, 140, 80)
    draw_mini_barplot(draw, 1110, 455, "C13-C16", node_means["C13-C16"], light_red, red, font_node, 148, 80)

    draw_distance_label(draw, 450, 180, edge_distances[("Root", "C1-C8")], black, font_dist)
    draw_distance_label(draw, 815, 180, edge_distances[("Root", "C9-C16")], black, font_dist)
    draw_distance_label(draw, 245, 365, edge_distances[("C1-C8", "C1-C4")], green, font_dist, 58)
    draw_distance_label(draw, 425, 365, edge_distances[("C1-C8", "C5-C8")], green, font_dist, 58)
    draw_distance_label(draw, 760, 365, edge_distances[("C9-C16", "C9-C12")], black, font_dist)
    draw_distance_label(draw, 1030, 365, edge_distances[("C9-C16", "C13-C16")], red, font_dist)
    draw_distance_label(draw, 180, 620, edge_distances[("C1-C4", "C1-C2")], green, font_dist, 52)
    draw_distance_label(draw, 490, 620, edge_distances[("C5-C8", "C5-C6")], green, font_dist, 52)
    draw_distance_label(draw, 775, 620, edge_distances[("C9-C12", "C9-C10")], green, font_dist, 52)
    draw_distance_label(draw, 1110, 620, edge_distances[("C13-C16", "C13-C14")], red, font_dist, 52)

    draw_feature_matrix(draw, 1380, 95, font_cell)

    figure_path = output_dir / "right_block_realistic_distances_clean.png"
    canvas.save(figure_path)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean right-block toy tree figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/toytrees/right_block_example"),
        help="Directory for the figure and generated data files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_means = build_node_means()
    edge_distances = build_edge_distances(node_means)
    write_tables(args.output_dir, node_means, edge_distances)
    figure_path = draw_figure(args.output_dir, node_means, edge_distances)
    print(figure_path)


if __name__ == "__main__":
    main()
