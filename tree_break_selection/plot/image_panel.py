"""Small image-panel rendering primitive shared by report applications."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def draw_image_panel(
    ax: Axes,
    path: object,
    title: str,
    *,
    font_size: int = 9,
    missing_text: str = "not available",
) -> None:
    """Draw an image into an axes, or a consistent missing-artifact placeholder."""

    ax.axis("off")
    ax.set_title(title, fontsize=font_size, loc="left")
    path_text = "" if path is None else str(path).strip()
    if path_text.lower() in {"", "<na>", "nan", "none"} or not Path(path_text).exists():
        ax.text(
            0.5,
            0.5,
            missing_text,
            ha="center",
            va="center",
            fontsize=font_size,
        )
        return
    ax.imshow(plt.imread(path_text))
