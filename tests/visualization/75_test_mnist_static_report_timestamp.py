"""Regression tests for static MNIST report timestamps."""

import importlib.util
import re
from pathlib import Path

import pandas as pd


def _load_static_report_module():
    script_path = Path(__file__).resolve().parents[2] / "applications/mnist/plot_report.py"
    spec = importlib.util.spec_from_file_location("plot_mnist_tbs_analysis_report", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_mnist_report_records_generated_timestamp(monkeypatch, tmp_path):
    report = _load_static_report_module()
    source_dir = tmp_path / "source"
    sweep_dir = source_dir / "alpha_sweep"
    out_dir = tmp_path / "out"
    sweep_dir.mkdir(parents=True)
    out_dir.mkdir()

    pd.DataFrame(
        [
            {"linkage": "ward", "edge_alpha": 0.0001, "sibling_alpha": 0.0001, "ARI": 0.5, "NMI": 0.6, "n_clusters": 11},
            {"linkage": "ward", "edge_alpha": 0.001, "sibling_alpha": 0.0001, "ARI": 0.4, "NMI": 0.55, "n_clusters": 9},
            {"linkage": "complete", "edge_alpha": 0.0001, "sibling_alpha": 0.0001, "ARI": 0.3, "NMI": 0.45, "n_clusters": 7},
        ]
    ).to_csv(sweep_dir / "alpha_sweep_summary.csv", index=False)
    pd.DataFrame([{"ARI": 0.2, "NMI": 0.3, "n_clusters": 4}]).to_csv(
        source_dir / "mnist_benchmark_summary.csv",
        index=False,
    )
    pd.DataFrame([{"ARI": 0.35, "NMI": 0.5, "n_clusters": 6}]).to_csv(
        source_dir / "mnist_continuous_pca50_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"cluster": 0, "size": 2, "dominant_digit": 0, "purity": 1.0, "digit_counts": "0:2"},
            {"cluster": 1, "size": 3, "dominant_digit": 1, "purity": 0.67, "digit_counts": "1:2,2:1"},
        ]
    ).to_csv(sweep_dir / "ward_e0.0001_s0.0001_top_cluster_digit_composition.csv", index=False)

    monkeypatch.setattr(report, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(report, "SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(report, "OUT_PDF", out_dir / "report.pdf")
    monkeypatch.setattr(report, "OUT_MAIN", out_dir / "summary.png")
    monkeypatch.setattr(report, "OUT_CSV", out_dir / "summary.csv")

    report.main()

    summary = pd.read_csv(out_dir / "summary.csv")
    assert "generated_at" in summary.columns
    assert summary["generated_at"].astype(str).str.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"
    ).all()
    assert (out_dir / "summary.png").exists()
    pdf_bytes = (out_dir / "report.pdf").read_bytes()
    assert re.search(rb"Generated at", pdf_bytes)
