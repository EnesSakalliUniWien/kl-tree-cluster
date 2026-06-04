from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.diagnostics.analysis import analyze_relationships


def _write_run_csv(results_dir: Path, run_name: str, csv_name: str) -> Path:
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / csv_name
    csv_path.write_text(
        "test_case,case_id,case_category,method,true_clusters,found_clusters,"
        "samples,features,noise,ari,nmi,purity,status\n"
        "1,case_a,improved_gaussian,kl,2,2,10,5,0.1,1.0,1.0,1.0,ok\n"
    )
    return run_dir


def test_latest_run_resolver_finds_suite_suffixed_run_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmarks_root = tmp_path / "benchmarks"
    results_dir = benchmarks_root / "results"
    _write_run_csv(
        results_dir,
        "run_20250101_000000Z_full",
        "full_benchmark_comparison.csv",
    )
    latest = _write_run_csv(
        results_dir,
        "run_20250102_000000Z_binary",
        "binary_benchmark_comparison.csv",
    )
    (results_dir / "run_20250103_000000Z_empty").mkdir()

    monkeypatch.setattr(analyze_relationships, "BENCHMARKS_ROOT", benchmarks_root)

    assert analyze_relationships._resolve_run_dir(None) == latest


def test_comparison_csv_resolver_prefers_suite_specific_csv(tmp_path: Path) -> None:
    run_dir = _write_run_csv(
        tmp_path,
        "run_20250102_000000Z_discretized_gaussian",
        "discretized_gaussian_benchmark_comparison.csv",
    )

    assert (
        analyze_relationships._resolve_comparison_csv(run_dir).name
        == "discretized_gaussian_benchmark_comparison.csv"
    )


def test_comparison_csv_resolver_rejects_ambiguous_csvs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20250102_000000Z_custom"
    run_dir.mkdir()
    (run_dir / "alpha_benchmark_comparison.csv").write_text("test_case\n1\n")
    (run_dir / "beta_benchmark_comparison.csv").write_text("test_case\n1\n")

    with pytest.raises(ValueError, match="Multiple benchmark CSVs"):
        analyze_relationships._resolve_comparison_csv(run_dir)
