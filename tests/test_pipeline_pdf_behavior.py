import pytest
from pathlib import Path

from kl_clustering_analysis.benchmarking import pipeline


def test_concat_pdf_forces_no_pngs_and_collects(monkeypatch, tmp_path: Path):
    called = {}

    def fake_generate_benchmark_plots(
        df_results,
        computed_results,
        plots_root,
        verbose,
        plot_umap,
        plot_manifold,
        save_png=True,
        collect_figs=False,
        *,
        pdf=None,
    ):
        # Record what was passed
        called["save_png"] = save_png
        called["collect_figs"] = collect_figs
        called["pdf"] = pdf
        # Return empty figures mapping
        return None, {"validation": [], "trees": [], "umap": [], "manifold": []}

    monkeypatch.setattr(
        pipeline, "generate_benchmark_plots", fake_generate_benchmark_plots
    )

    # With save_individual_plots=True, pipeline should NOT stream and should not collect figures.
    df, fig = pipeline.benchmark_cluster_algorithm(
        test_cases=[],
        verbose=True,
        concat_plots_pdf=True,
        save_individual_plots=True,
        methods=[],
    )

    assert called.get("save_png") is True
    assert called.get("collect_figs") is False
    assert called.get("pdf") is None


def test_concat_pdf_without_request_collects_and_no_pngs(monkeypatch, tmp_path: Path):
    called = {}

    def fake_generate_benchmark_plots(
        df_results,
        computed_results,
        plots_root,
        verbose,
        plot_umap,
        plot_manifold,
        save_png=True,
        collect_figs=False,
        *,
        pdf=None,
    ):
        called["save_png"] = save_png
        called["collect_figs"] = collect_figs
        called["pdf"] = pdf
        return None, {"validation": [], "trees": [], "umap": [], "manifold": []}

    monkeypatch.setattr(
        pipeline, "generate_benchmark_plots", fake_generate_benchmark_plots
    )

    # With concat_plots_pdf=True and save_individual_plots=False (default), pipeline streams to PdfPages.
    df, fig = pipeline.benchmark_cluster_algorithm(
        test_cases=[],
        verbose=True,
        concat_plots_pdf=True,
        methods=[],
    )

    assert called.get("save_png") is False
    assert called.get("collect_figs") is False
    assert called.get("pdf") is not None
