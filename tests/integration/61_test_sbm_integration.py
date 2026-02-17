import pytest
import numpy as np

from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.cases import get_default_test_cases


def test_sbm_cases_run_through_pipeline_minimal():
    # Select SBM cases from the default config
    all_cases = get_default_test_cases()
    sbm_cases = [c for c in all_cases if c.get("generator") == "sbm"]
    assert sbm_cases, "No SBM test cases found in default config"

    # Run pipeline with graph methods (runners will skip if optional deps are missing)
    df, fig = benchmark_cluster_algorithm(
        test_cases=sbm_cases, methods=["louvain", "leiden"], verbose=False
    )

    assert "Method" in df.columns
    assert "ARI" in df.columns
    assert df.shape[0] >= len(sbm_cases)
    # Ensure status contract stays normalized.
    assert set(df["Status"]).issubset({"ok", "skip"})
