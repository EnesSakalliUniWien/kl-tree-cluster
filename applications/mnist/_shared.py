"""Shared data-selection helpers for MNIST report engines."""

import re

import pandas as pd

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


def alpha_label(value: float) -> str:
    """Return the compact alpha label used in MNIST artifact keys."""

    return f"{value:g}"


def assignment_key(linkage: str, edge_alpha: float, sibling_alpha: float) -> str:
    """Return the summary-column key for one MNIST alpha setting."""

    return f"{linkage}_e{alpha_label(edge_alpha)}_s{alpha_label(sibling_alpha)}"


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-ARI configuration for each linkage."""

    indices = summary.groupby("linkage")["ARI"].idxmax()
    return summary.loc[indices].sort_values("ARI", ascending=False).reset_index(drop=True)


def parse_digit_counts(value: object) -> dict[int, int]:
    """Parse compact ``digit:count`` fields from benchmark summaries."""

    return {int(digit): int(count) for digit, count in re.findall(r"(\d+):(\d+)", str(value))}
