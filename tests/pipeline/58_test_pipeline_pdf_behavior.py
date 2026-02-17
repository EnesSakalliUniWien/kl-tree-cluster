import pytest
from pathlib import Path

from benchmarks.shared import pipeline


def test_removed_legacy_plot_kwargs_raise_type_error():
    with pytest.raises(TypeError):
        pipeline.benchmark_cluster_algorithm(
            test_cases=[],
            verbose=True,
            concat_plots_pdf=True,
            concat_pattern="tree_case_*.png",
            methods=[],
        )

    with pytest.raises(TypeError):
        pipeline.benchmark_cluster_algorithm(
            test_cases=[],
            verbose=True,
            concat_plots_pdf=True,
            save_individual_plots=True,
            methods=[],
        )


def test_concat_pdf_streams_and_no_pngs(monkeypatch, tmp_path: Path):
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

    # With concat_plots_pdf=True, pipeline streams to PdfPages and never emits PNGs.
    df, fig = pipeline.benchmark_cluster_algorithm(
        test_cases=[],
        verbose=True,
        concat_plots_pdf=True,
        methods=[],
    )

    assert called.get("save_png") is False
    assert called.get("collect_figs") is False
    assert called.get("pdf") is not None
