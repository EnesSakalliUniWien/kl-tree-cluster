"""Tests for generated timestamps in selected scRNA NNLS report figures."""

import importlib

import pandas as pd
from matplotlib.figure import Figure


def _load_report_module():
    return importlib.import_module("applications.scrna.plots.selected_nnls_report")


def test_standalone_png_pages_have_timestamp_before_save(monkeypatch, tmp_path):
    report = _load_report_module()
    generated_at = "2026-06-24T19:45:00+02:00"
    selected = "selected_method"
    assignments = pd.DataFrame(
        {
            "cell_id": ["L0", "L1"],
            selected: [0, 1],
            "celltype": ["alpha", "beta"],
            "umap1": [0.0, 1.0],
            "umap2": [1.0, 0.0],
        }
    )
    boundary = pd.DataFrame(
        {
            "cluster": [0, 1],
            "cluster_purity_in_boundary": [1.0, 1.0],
        }
    )
    summary = pd.DataFrame(
        {
            "cluster": [0, 1],
            "n_cells_assignment": [1, 1],
            "top_celltype_assignment": ["alpha", "beta"],
            "top_celltype_fraction_assignment": [1.0, 1.0],
        }
    )

    monkeypatch.setattr(report.pd, "read_csv", lambda *_args, **_kwargs: assignments.copy())
    monkeypatch.setattr(
        report,
        "build_boundary_tree_info",
        lambda *_args, **_kwargs: {"boundary_df": boundary.copy()},
    )
    monkeypatch.setattr(report, "build_full_tree_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(report, "cluster_summary", lambda *_args, **_kwargs: summary.copy())
    monkeypatch.setattr(report, "plot_umap", lambda ax, *_args, **_kwargs: ax.set_title("UMAP"))
    monkeypatch.setattr(
        report, "plot_compact_tree", lambda ax, *_args, **_kwargs: ax.set_title("Tree")
    )
    monkeypatch.setattr(
        report, "plot_size_bars", lambda ax, *_args, **_kwargs: ax.set_title("Sizes")
    )
    monkeypatch.setattr(
        report, "plot_radial_tree", lambda ax, *_args, **_kwargs: ax.set_title("Radial")
    )
    monkeypatch.setattr(
        report,
        "plot_full_radial_tree",
        lambda ax, *_args, **_kwargs: ax.set_title("Full radial"),
    )

    saved_figure_texts = []

    def savefig_spy(self, *_args, **_kwargs):
        saved_figure_texts.append([text.get_text() for text in self.texts])

    monkeypatch.setattr(Figure, "savefig", savefig_spy)
    cfg = {"dir": tmp_path, "selected": selected, "title": "Test dataset", "key": "test"}

    report.selected_dataset_page(cfg, save_path=tmp_path / "dataset.png", generated_at=generated_at)
    report.selected_radial_dataset_page(
        cfg, save_path=tmp_path / "radial.png", generated_at=generated_at
    )
    report.selected_full_radial_dataset_page(
        cfg,
        save_path=tmp_path / "full_radial.png",
        generated_at=generated_at,
    )

    assert len(saved_figure_texts) == 3
    assert all(
        [f"Generated at: {generated_at}"]
        == [text for text in texts if text.startswith("Generated at:")]
        for texts in saved_figure_texts
    )
