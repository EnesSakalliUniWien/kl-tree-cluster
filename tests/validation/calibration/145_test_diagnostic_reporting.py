import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.reporting import write_diagnostic_bundle


@dataclass(frozen=True)
class _ExampleConfig:
    input_path: Path
    seed: np.integer


def test_write_diagnostic_bundle_writes_tables_and_manifest(tmp_path: Path) -> None:
    paths = write_diagnostic_bundle(
        output_dir=tmp_path,
        tables={"rows": pd.DataFrame({"value": [1, 2]})},
        filenames={"rows": "rows.csv"},
        manifest={
            "schema_version": "test/v1",
            "config": _ExampleConfig(tmp_path / "input.csv", np.int64(7)),
        },
    )

    assert paths["rows"] == tmp_path / "rows.csv"
    assert paths["manifest"] == tmp_path / "manifest.json"
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["row_counts"] == {"rows": 2}
    assert payload["config"]["seed"] == 7
    assert payload["outputs"]["rows"] == str(tmp_path / "rows.csv")
    assert payload["generated_at"]


def test_write_diagnostic_bundle_rejects_mismatched_table_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keys must match"):
        write_diagnostic_bundle(
            output_dir=tmp_path,
            tables={"rows": pd.DataFrame()},
            filenames={"summary": "summary.csv"},
            manifest={},
        )
