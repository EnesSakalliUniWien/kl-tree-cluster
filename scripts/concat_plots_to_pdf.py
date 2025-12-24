#!/usr/bin/env python3
"""Concatenate multiple plot images into a single multi-page PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

# Headless backend for CLI usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402


def collect_images(
    directory: Path, pattern: str = "*.png", limit: int | None = None
) -> List[Path]:
    files = sorted(directory.glob(pattern))
    return files[:limit] if limit else files


def save_as_pdf(images: Iterable[Path], output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for path in images:
            with Image.open(path) as img:
                fig = plt.figure(figsize=(img.width / 100, img.height / 100))
                plt.axis("off")
                plt.imshow(img)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
                plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate plot images into a single PDF."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("cluster_tree_plots"),
        help="Directory containing plot images (default: cluster_tree_plots).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern to select images (default: *.png).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cluster_tree_plots/all_plots.pdf"),
        help="Output PDF path (default: cluster_tree_plots/all_plots.pdf).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to include (after sorting).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = collect_images(args.input_dir, args.pattern, args.limit)
    if not images:
        raise SystemExit(
            f"No images found in {args.input_dir} matching pattern {args.pattern!r}"
        )
    save_as_pdf(images, args.output)
    print(f"Wrote {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
