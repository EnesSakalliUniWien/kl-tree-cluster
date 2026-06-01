"""Benchmark case-data generation for preloaded matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
)


def generate_preloaded_case(test_case: dict, _seed: int | None) -> CaseDataResult:
    """Load a pre-existing TSV/CSV matrix as a benchmark case."""
    file_path = test_case["file_path"]
    sep = test_case["sep"]

    path = Path(file_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f"Preloaded data file not found: {path}")

    data_df = pd.read_csv(path, sep=sep, index_col=0)
    n_samples, n_features = data_df.shape
    labels = np.full(n_samples, np.nan)
    x_original = data_df.values.copy()

    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=int(test_case["n_clusters"]),
        noise=np.nan,
        generator="preloaded",
        source_family="preloaded_matrix",
        feature_representation="preloaded_matrix",
        requires_precomputed_kl_distance=False,
        extra={
            "source_file": str(path),
            "sparsity": float(1 - data_df.values.mean()),
        },
    )
    return data_df, labels, x_original, metadata
