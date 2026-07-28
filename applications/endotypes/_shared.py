"""Shared naming helpers for endotype application entry points."""

from pathlib import Path

import pandas as pd


def safe_name(value: object) -> str:
    """Return a filesystem-safe representation without changing case."""

    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def matrix_slug(input_path: Path, dataset_label: str | None = None) -> str:
    """Return the stable application slug for an input feature matrix."""

    raw = dataset_label or input_path.stem
    if raw.startswith("feature_matrix_"):
        raw = raw[len("feature_matrix_") :]
    return safe_name(raw).strip("_").lower() or "feature_matrix"


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load a numeric feature matrix and enforce cosine-space row support."""

    data = pd.read_csv(path, sep="\t", index_col=0)
    zero_columns = data.columns[(data.sum(axis=0) == 0).to_numpy()]
    if len(zero_columns):
        data = data.drop(columns=zero_columns)
    zero_rows = data.index[(data.sum(axis=1) == 0).to_numpy()]
    if len(zero_rows):
        raise ValueError(
            f"Rows with zero feature mass cannot enter cosine analysis: {list(zero_rows[:10])!r}"
        )
    return data.astype(float)
