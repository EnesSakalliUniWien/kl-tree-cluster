"""Sibling projection-dimension rule grid diagnostic.

This diagnostic compares predeclared sibling projection-dimension rules on
row-level selected-geometry records. It reports dimension distributions only;
it does not validate false-split control or install a production rule.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_sibling_projection_dimension_rule_grid_not_calibration"

REQUIRED_COLUMNS = {
    "sibling_projection_dimension",
    "parent_test_projection_dimension",
    "raw_mp_signal_count",
}

RULE_IDS = (
    "current_edge_derived_rule",
    "parent_test_projection_dimension",
    "raw_mp_parent_signal_count",
    "raw_mp_parent_signal_count_floor1",
    "raw_mp_parent_signal_count_floor2",
)


def add_projection_dimension_rule_columns(records: pd.DataFrame) -> pd.DataFrame:
    """Add candidate projection-dimension rule columns."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(
            f"Projection-dimension records are missing columns: {sorted(missing)!r}."
        )
    table = records.copy()
    current = pd.to_numeric(
        table["sibling_projection_dimension"],
        errors="raise",
    ).astype(int)
    parent = pd.to_numeric(
        table["parent_test_projection_dimension"],
        errors="raise",
    ).astype(int)
    raw_mp = pd.to_numeric(table["raw_mp_signal_count"], errors="raise").astype(int)
    if bool((current < 0).any() or (parent < 0).any() or (raw_mp < 0).any()):
        raise ValueError("Projection dimensions and raw MP counts must be non-negative.")
    table["current_edge_derived_rule"] = current
    table["parent_test_projection_dimension"] = parent
    table["raw_mp_parent_signal_count"] = raw_mp
    table["raw_mp_parent_signal_count_floor1"] = np.minimum(np.maximum(raw_mp, 1), parent)
    table["raw_mp_parent_signal_count_floor2"] = np.minimum(np.maximum(raw_mp, 2), parent)
    return table


def evaluate_projection_dimension_rule_grid(records: pd.DataFrame) -> pd.DataFrame:
    """Return distribution summaries for candidate dimension rules."""
    table = add_projection_dimension_rule_columns(records)
    rows: list[dict[str, object]] = []
    for rule_id in RULE_IDS:
        dimensions = table[rule_id].to_numpy(dtype=int)
        rows.append(
            {
                "rule_id": rule_id,
                "n_records": int(dimensions.shape[0]),
                "dimension_mean": float(np.mean(dimensions)),
                "dimension_median": float(np.median(dimensions)),
                "dimension_min": int(np.min(dimensions)),
                "dimension_max": int(np.max(dimensions)),
                "frequency_k0": float(np.mean(dimensions == 0)),
                "frequency_k1": float(np.mean(dimensions == 1)),
                "frequency_k2": float(np.mean(dimensions == 2)),
                "frequency_k_ge3": float(np.mean(dimensions >= 3)),
                "differs_from_current_fraction": float(
                    np.mean(dimensions != table["current_edge_derived_rule"].to_numpy(dtype=int))
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def run_projection_dimension_rule_grid(
    *,
    records_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the projection-dimension rule-grid diagnostic from a CSV."""
    records = pd.read_csv(records_path)
    summary = evaluate_projection_dimension_rule_grid(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sibling_projection_dimension_rule_grid.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "outputs": {"summary": str(summary_path)},
        "interpretation": (
            "Diagnostic dimension-distribution grid for sibling projection rules. "
            "This is not a calibration or power validation."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary_path, "manifest": manifest_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_projection_dimension_rule_grid(
        records_path=args.records,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RULE_IDS",
    "STUDY_ROLE",
    "add_projection_dimension_rule_columns",
    "evaluate_projection_dimension_rule_grid",
    "run_projection_dimension_rule_grid",
]
