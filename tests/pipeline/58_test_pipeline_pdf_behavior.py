from pathlib import Path

import pytest
from benchmarks.shared import pipeline


def test_removed_plot_kwargs_raise_type_error():
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
        include_cover_pages=True,
        include_validation_page=True,
        *,
        pdf=None,
    ):
        del computed_results, plot_umap, plot_manifold
        called["save_png"] = save_png
        called["collect_figs"] = collect_figs
        called["include_cover_pages"] = include_cover_pages
        called["include_validation_page"] = include_validation_page
        called["pdf"] = pdf
        return None, {"validation": [], "trees": [], "umap": [], "manifold": []}

    monkeypatch.setattr(
        pipeline, "generate_benchmark_plots", fake_generate_benchmark_plots
    )

    # With concat_plots_pdf=True, pipeline streams to PdfPages and never emits PNGs.
    _df, _fig = pipeline.benchmark_cluster_algorithm(
        test_cases=[],
        verbose=True,
        concat_plots_pdf=True,
        methods=["kl"],
    )

    assert called.get("save_png") is False
    assert called.get("collect_figs") is False
    assert called.get("include_cover_pages") is True
    assert called.get("include_validation_page") is True
    assert called.get("pdf") is not None


def test_pipeline_preserves_explicit_case_identity(monkeypatch):
    seen_case_numbers: list[int] = []

    def fake_run_single_case(**kwargs):
        seen_case_numbers.append(kwargs["tc"]["test_case_num"])
        return [], []

    monkeypatch.setattr(pipeline, "run_single_case", fake_run_single_case)
    monkeypatch.setattr(
        pipeline,
        "generate_benchmark_plots",
        lambda *args, **kwargs: (None, {"validation": [], "trees": [], "umap": [], "manifold": []}),
    )

    pipeline.benchmark_cluster_algorithm(
        test_cases=[
            {"name": "case_a", "seed": 1, "test_case_num": 41},
            {"name": "case_b", "seed": 2, "test_case_num": 42},
        ],
        verbose=False,
        methods=["kl"],
    )

    assert seen_case_numbers == [41, 42]
