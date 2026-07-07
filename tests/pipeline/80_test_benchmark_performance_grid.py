from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.shared.performance_grid import write_benchmark_performance_grid


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_case": 1,
                "case_id": "case_a",
                "case_category": "binary",
                "source_family": "binary_template",
                "feature_representation": "binary",
                "samples": 40,
                "features": 8,
                "true_clusters": 2,
                "found_clusters": 2,
                "method": "tbs",
                "run_id": "tbs::default",
                "benchmark_class": "canonical",
                "benchmark_grid": "default_methods",
                "benchmark_repeat": 0,
                "params": "tree_linkage_method=average",
                "status": "ok",
                "skip_reason": "",
                "ari": 0.9,
                "nmi": 0.8,
                "purity": 1.0,
                "cluster_count_abs_error": 0.0,
            },
            {
                "test_case": 1,
                "case_id": "case_a",
                "case_category": "binary",
                "source_family": "binary_template",
                "feature_representation": "binary",
                "samples": 40,
                "features": 8,
                "true_clusters": 2,
                "found_clusters": 3,
                "method": "kmeans",
                "run_id": "kmeans::default",
                "benchmark_class": "canonical",
                "benchmark_grid": "default_methods",
                "benchmark_repeat": 0,
                "params": "n_clusters=true",
                "status": "ok",
                "skip_reason": "",
                "ari": 0.7,
                "nmi": 0.6,
                "purity": 0.8,
                "cluster_count_abs_error": 1.0,
            },
            {
                "test_case": 2,
                "case_id": "case_b",
                "case_category": "continuous",
                "source_family": "gaussian_blobs",
                "feature_representation": "continuous",
                "samples": 50,
                "features": 4,
                "true_clusters": 3,
                "found_clusters": 0,
                "method": "tbs",
                "run_id": "tbs::default",
                "benchmark_class": "canonical",
                "benchmark_grid": "default_methods",
                "benchmark_repeat": 0,
                "params": "tree_linkage_method=average",
                "status": "skip",
                "skip_reason": "unsupported",
                "ari": float("nan"),
                "nmi": float("nan"),
                "purity": float("nan"),
                "cluster_count_abs_error": float("nan"),
            },
            {
                "test_case": 2,
                "case_id": "case_b",
                "case_category": "continuous",
                "source_family": "gaussian_blobs",
                "feature_representation": "continuous",
                "samples": 50,
                "features": 4,
                "true_clusters": 3,
                "found_clusters": 3,
                "method": "kmeans",
                "run_id": "kmeans::default",
                "benchmark_class": "canonical",
                "benchmark_grid": "default_methods",
                "benchmark_repeat": 0,
                "params": "n_clusters=true",
                "status": "ok",
                "skip_reason": "",
                "ari": 0.95,
                "nmi": 0.9,
                "purity": 1.0,
                "cluster_count_abs_error": 0.0,
            },
        ]
    )


def test_write_benchmark_performance_grid_outputs_case_grids(tmp_path: Path):
    artifacts = write_benchmark_performance_grid(
        _rows(),
        tmp_path,
        source_path=Path("benchmark.csv"),
    )

    assert artifacts.report_md.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.ari_grid_csv.exists()
    assert artifacts.status_grid_csv.exists()

    summary = pd.read_csv(artifacts.summary_csv)
    assert list(summary["run_id"]) == ["kmeans::default", "tbs::default"]
    assert summary.loc[0, "mean_ari"] == pytest.approx(0.825)
    assert summary.loc[0, "exact_k_rate"] == pytest.approx(0.5)
    assert summary.loc[1, "skip_rate"] == pytest.approx(0.5)

    ari_grid = pd.read_csv(artifacts.ari_grid_csv)
    assert list(ari_grid["case_id"]) == ["case_a", "case_b"]
    assert ari_grid.loc[0, "tbs::default"] == pytest.approx(0.9)
    assert ari_grid.loc[1, "kmeans::default"] == pytest.approx(0.95)

    status_grid = pd.read_csv(artifacts.status_grid_csv)
    assert status_grid.loc[1, "tbs::default"] == "skip"

    report = artifacts.report_md.read_text()
    assert "# Benchmark Performance Grid" in report
    assert "benchmark_performance_grid_ari.csv" in report
    assert "kmeans::default" in report


def test_write_benchmark_performance_grid_rejects_duplicate_case_run(tmp_path: Path):
    rows = pd.concat([_rows(), _rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="unique case_id/run_id"):
        write_benchmark_performance_grid(rows, tmp_path)
