"""Output bundle writer for calibration and diagnostic panels."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def print_diagnostic_output_paths(outputs: Mapping[str, Path]) -> None:
    """Print diagnostic output paths as a JSON object for CLI callers."""
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


def diagnostic_json_default(value: object) -> object:
    """Serialize the value types admitted by diagnostic manifests."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_diagnostic_bundle(
    *,
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    filenames: Mapping[str, str],
    manifest: Mapping[str, object],
    manifest_filename: str = "manifest.json",
) -> dict[str, Path]:
    """Write named tables and their common manifest envelope."""
    unknown_tables = sorted(set(tables).difference(filenames))
    missing_tables = sorted(set(filenames).difference(tables))
    if unknown_tables or missing_tables:
        raise ValueError(
            "Diagnostic table and filename keys must match; "
            f"missing filenames={unknown_tables!r}, missing tables={missing_tables!r}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / filenames[key] for key in tables}
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)

    manifest_path = output_dir / manifest_filename
    payload = {
        **dict(manifest),
        "generated_at": datetime.now(UTC).isoformat(),
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, default=diagnostic_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


__all__ = [
    "diagnostic_json_default",
    "print_diagnostic_output_paths",
    "write_diagnostic_bundle",
]
