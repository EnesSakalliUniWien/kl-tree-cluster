"""Tests for the incremental temporal clustering benchmark."""

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.benchmarking.temporal_benchmark import (
    run_incremental_temporal_benchmark,
    run_temporal_benchmark_suite,
)


class TestIncrementalTemporalBenchmark:
    """Tests for run_incremental_temporal_benchmark."""

    def test_basic_run(self):
        """Test basic benchmark execution."""
        df = run_incremental_temporal_benchmark(
            n_time_points=4,
            n_features=30,
            n_categories=4,
            samples_per_time=10,
            mutation_rate=0.3,
            random_seed=42,
            method="kl",
            min_time_points=2,
            verbose=False,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # time steps 2, 3, 4
        assert "time_step" in df.columns
        assert "ari" in df.columns
        assert "nmi" in df.columns
        assert "n_clusters_found" in df.columns
        assert "divergence_from_ancestor" in df.columns

    def test_time_steps_correct(self):
        """Test that time steps are correctly recorded."""
        df = run_incremental_temporal_benchmark(
            n_time_points=5,
            n_features=20,
            samples_per_time=8,
            random_seed=123,
            min_time_points=2,
            verbose=False,
        )
        
        assert list(df["time_step"]) == [2, 3, 4, 5]
        assert list(df["n_clusters_true"]) == [2, 3, 4, 5]

    def test_sample_counts_correct(self):
        """Test that sample counts increase correctly."""
        samples_per_time = 15
        df = run_incremental_temporal_benchmark(
            n_time_points=4,
            n_features=20,
            samples_per_time=samples_per_time,
            random_seed=456,
            min_time_points=2,
            verbose=False,
        )
        
        expected_samples = [2 * samples_per_time, 3 * samples_per_time, 4 * samples_per_time]
        assert list(df["n_samples"]) == expected_samples

    def test_divergence_increases(self):
        """Test that divergence from ancestor generally increases over time."""
        df = run_incremental_temporal_benchmark(
            n_time_points=6,
            n_features=50,
            samples_per_time=10,
            mutation_rate=0.4,
            random_seed=789,
            min_time_points=2,
            verbose=False,
        )
        
        divergences = df["divergence_from_ancestor"].values
        # Divergence should generally increase (allow some noise)
        assert divergences[-1] > divergences[0]

    def test_metrics_in_valid_range(self):
        """Test that ARI and NMI are in valid ranges."""
        df = run_incremental_temporal_benchmark(
            n_time_points=4,
            n_features=30,
            samples_per_time=10,
            random_seed=111,
            min_time_points=2,
            verbose=False,
        )
        
        # ARI can be negative but typically in [-1, 1]
        assert df["ari"].min() >= -1.0
        assert df["ari"].max() <= 1.0
        
        # NMI is in [0, 1]
        valid_nmi = df[df["nmi"].notna()]
        assert valid_nmi["nmi"].min() >= 0.0
        assert valid_nmi["nmi"].max() <= 1.0

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        kwargs = dict(
            n_time_points=4,
            n_features=30,
            samples_per_time=10,
            mutation_rate=0.3,
            random_seed=999,
            min_time_points=2,
            verbose=False,
        )
        
        df1 = run_incremental_temporal_benchmark(**kwargs)
        df2 = run_incremental_temporal_benchmark(**kwargs)
        
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            run_incremental_temporal_benchmark(
                n_time_points=3,
                n_features=20,
                method="invalid_method",
                verbose=False,
            )


class TestTemporalBenchmarkSuite:
    """Tests for run_temporal_benchmark_suite."""

    def test_basic_suite(self):
        """Test basic suite execution with default configs."""
        df = run_temporal_benchmark_suite(
            configs=[
                {"name": "test1", "n_time_points": 3, "n_features": 20, "samples_per_time": 8, "seed": 1},
                {"name": "test2", "n_time_points": 3, "n_features": 20, "samples_per_time": 8, "seed": 2},
            ],
            methods=["kl"],
            verbose=False,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert "config" in df.columns
        assert "method" in df.columns
        assert set(df["config"].unique()) == {"test1", "test2"}

    def test_multiple_methods(self):
        """Test suite with multiple methods."""
        df = run_temporal_benchmark_suite(
            configs=[
                {"name": "test", "n_time_points": 3, "n_features": 20, "samples_per_time": 8, "seed": 1},
            ],
            methods=["kl", "hdbscan"],
            verbose=False,
        )
        
        assert set(df["method"].unique()) == {"kl", "hdbscan"}
