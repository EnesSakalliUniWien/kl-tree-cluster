"""Shared helpers for scRNA application reports and audits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    """Serialize standard scRNA report values without changing their schema."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def top_counts_text(values: pd.Series, n: int = 8) -> str:
    """Return the most frequent string values in stable report form."""
    counts = values.astype(str).value_counts()
    return "; ".join(f"{label}:{count}" for label, count in counts.head(n).items())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path,
    *,
    relative_to: Path,
    role: str | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return a static-artifact provenance record."""

    record: dict[str, object] = {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        record["role"] = role
    if path.suffix == ".csv":
        table = pd.read_csv(path, low_memory=False)
        record["rows"] = int(len(table))
        record["columns"] = int(len(table.columns))
        if generated_at is not None:
            record["generated_at"] = generated_at
        if "generated_at" in table.columns:
            record["generated_at_values"] = sorted(
                table["generated_at"].dropna().unique().tolist()
            )
    return record
