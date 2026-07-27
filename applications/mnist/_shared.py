"""Shared data-selection helpers for MNIST report engines."""

import re

import pandas as pd


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-ARI configuration for each linkage."""

    indices = summary.groupby("linkage")["ARI"].idxmax()
    return summary.loc[indices].sort_values("ARI", ascending=False).reset_index(drop=True)


def parse_digit_counts(value: object) -> dict[int, int]:
    """Parse compact ``digit:count`` fields from benchmark summaries."""

    return {int(digit): int(count) for digit, count in re.findall(r"(\d+):(\d+)", str(value))}
