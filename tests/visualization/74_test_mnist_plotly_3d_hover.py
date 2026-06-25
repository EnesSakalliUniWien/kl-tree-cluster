"""Regression tests for MNIST Plotly 3D hover labels."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from benchmarks.experiments.mnist.run_higher_categories import (
    create_plotly_higher_category_plot_3d,
)


class _DummyUMAP:
    def __init__(self, **_kwargs):
        pass

    def fit_transform(self, feature_matrix):
        base = np.arange(feature_matrix.shape[0], dtype=float)
        return np.column_stack([base, base + 1.0, base + 2.0])


def test_3d_plotly_writer_shows_number_on_hover_and_visible_text(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=_DummyUMAP))
    output_path = tmp_path / "mnist_3d.html"

    create_plotly_higher_category_plot_3d(
        feature_matrix=np.ones((4, 3)),
        digit_labels=np.array([0, 1, 2, 3]),
        cluster_labels=np.array([10, 10, 20, 20]),
        true_higher_category_names=np.array(["digit 0", "digit 1", "digit 2", "digit 3"]),
        predicted_higher_category_names=np.array(["TBS C10", "TBS C10", "TBS C20", "TBS C20"]),
        output_path=output_path,
        visible_digit_labels=True,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Digit number: 0" in html
    assert "Sample: Sample_0" in html
    assert "%{hovertext}" in html
    assert '"Sample_0"' in html
    assert "Digit: %{customdata[0]}" not in html
    assert '"mode":"markers+text"' in html
    assert '"size":5' in html
    assert '"name":"digit 0 hover target"' in html
    assert '"size":24' in html
    assert '"opacity":0.01' in html
    assert '"textposition":"middle center"' in html


def _load_mnist_report_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts/plot_mnist_tbs_analysis_plotly.py"
    spec = importlib.util.spec_from_file_location("plot_mnist_tbs_analysis_plotly", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_2d_report_hover_shows_number_before_sample():
    report = _load_mnist_report_module()
    frame = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1"],
            "true_digit": [3, 8],
            "umap1": [0.0, 1.0],
            "umap2": [1.0, 0.0],
            "best_tbs_cluster": [7, 9],
        }
    )

    html = report.umap_figure(frame).to_html(include_plotlyjs=False, full_html=False)

    assert "Digit number: 3" in html
    assert "Sample: Sample_0" in html
    assert "Best TBS cluster: 7" in html
    assert "%{hovertext}" in html
    assert "Digit: %{customdata[0]}" not in html


def test_2d_image_inspector_embeds_digit_pixels(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    output_path = tmp_path / "mnist_image_inspector.html"
    frame = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1"],
            "true_digit": [3, 8],
            "umap1": [0.0, 1.0],
            "umap2": [1.0, 0.0],
            "best_tbs_cluster": [7, 9],
        }
    )
    raw_images = np.array(
        [
            [0.0, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.0],
        ],
        dtype=float,
    )

    report._write_umap_image_inspector(frame, raw_images, output_path=output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "mnist-image-inspector-plot" in html
    assert "const imagePixels = [[0,128,191,255],[255,191,128,0]];" in html
    assert "const imageSide = 2;" in html
    assert "function drawDigitImage(pointIndex)" in html
    assert 'plot.on("plotly_hover"' in html
    assert 'plot.on("plotly_click"' in html
    assert "Digit number: 3" in html
    assert '"sample":"Sample_0"' in html
    assert '"best_tbs_cluster":7' in html
    assert "Contains 2 MNIST examples" in html
    assert "Source images are 2x2 pixels" in html


def test_radial_tree_context_recovers_final_cluster_boundaries():
    report = _load_mnist_report_module()
    assignments = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1", "Sample_2", "Sample_3"],
            "true_digit": [0, 0, 9, 9],
            "best": [0, 0, 1, 1],
        }
    )
    feature_matrix = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=float)

    context = report._build_tbs_radial_tree_context(
        assignments,
        "best",
        feature_matrix,
        "ward",
    )

    assert context["n_clusters"] == 2
    assert context["tree_mode"] == "cluster_boundary"
    assert "mnist-radial-tbs-tree" in context["html"]
    records = {record["cluster_id"]: record for record in context["cluster_records"]}
    assert sorted(records) == [0, 1]
    assert records[0]["size"] == 2
    assert records[0]["dominant_digit"] == 0
    assert records[1]["dominant_digit"] == 9
    assert records[0]["exact_tree_boundary"] is True


def test_full_radial_tree_context_includes_all_linkage_nodes():
    report = _load_mnist_report_module()
    assignments = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1", "Sample_2", "Sample_3"],
            "true_digit": [0, 0, 9, 9],
            "best": [0, 0, 1, 1],
        }
    )
    feature_matrix = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=float)

    context = report._build_tbs_radial_tree_context(
        assignments,
        "best",
        feature_matrix,
        "ward",
        full_tree=True,
    )

    assert context["tree_mode"] == "full"
    assert context["n_nodes"] == 7
    assert context["n_edges"] == 6
    assert context["n_clusters"] == 2
    assert "Full best-run TBS radial tree" in context["html"]
    assert "Leaf: Sample_0" in context["html"]
    assert "Final TBS cluster C0" in context["html"]


def test_3d_image_inspector_embeds_digit_pixels_and_tree(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    output_path = tmp_path / "mnist_3d_image_inspector.html"
    monkeypatch.setattr(report.umap, "UMAP", _DummyUMAP)
    assignments = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1"],
            "true_digit": [3, 8],
            "best": [7, 9],
        }
    )
    raw_images = np.array(
        [
            [0.0, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.0],
        ],
        dtype=float,
    )
    tree_context = {
        "html": '<div id="mnist-radial-tbs-tree"></div>',
        "div_id": "mnist-radial-tbs-tree",
        "cluster_records": [
            {"cluster_id": 7, "node_id": "N2", "x": 0.0, "y": 1.0},
            {"cluster_id": 9, "node_id": "N3", "x": 1.0, "y": 0.0},
        ],
        "n_clusters": 2,
        "n_nodes": 7,
        "n_edges": 6,
        "tree_mode": "full",
    }

    report._write_umap3d_image_inspector(
        assignments,
        "best",
        raw_images,
        output_path=output_path,
        tree_context=tree_context,
        feature_matrix=np.ones((2, 3)),
        digit_labels=np.array([3, 8]),
    )

    html = output_path.read_text(encoding="utf-8")
    assert "mnist-3d-image-inspector-plot" in html
    assert "mnist-radial-tbs-tree" in html
    assert "const imagePixels = [[0,128,191,255],[255,191,128,0]];" in html
    assert "const imageSide = 2;" in html
    assert "function drawDigitImage(pointIndex)" in html
    assert "function updateTreeHighlight(clusterId)" in html
    assert "UMAP3=%{z:.3f}" in html
    assert "Digit number: 3" in html
    assert '"node_id":"N2"' in html
    assert "Full radial tree uses the same best-run TBS hierarchy: 7 nodes, 6 edges" in html


def test_report_index_writes_timestamp(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    output_path = tmp_path / "index.html"
    monkeypatch.setattr(report, "OUT_INDEX", output_path)

    report._write_index(
        [("MNIST page", "00_mnist.html")],
        "2026-06-24T19:10:56+02:00",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Generated at:</strong> 2026-06-24T19:10:56+02:00" in html
    assert '<a href="00_mnist.html">MNIST page</a>' in html


def test_report_manifest_records_generated_timestamp(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    output_dir = tmp_path / "plotly"
    output_dir.mkdir()
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "OUT_DIR", output_dir)
    monkeypatch.setattr(report, "OUT_INDEX", output_dir / "index.html")
    monkeypatch.setattr(report, "OUT_CSV", output_dir / "mnist_tbs_analysis_summary.csv")
    monkeypatch.setattr(report, "OUT_UMAP_COORDS", output_dir / "mnist_pca50_umap_coordinates.csv")
    monkeypatch.setattr(report, "OUT_MANIFEST", output_dir / "manifest.json")
    monkeypatch.setattr(report, "SOURCE_DIR", tmp_path / "benchmarks/results/experiments/mnist")
    monkeypatch.setattr(
        report,
        "SWEEP_DIR",
        tmp_path / "benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605",
    )

    report.OUT_INDEX.write_text("<html>index</html>", encoding="utf-8")
    pd.DataFrame(
        {
            "generated_at": ["2026-06-24T20:38:05+02:00"],
            "linkage": ["ward"],
        }
    ).to_csv(report.OUT_CSV, index=False)
    pd.DataFrame(
        {
            "sample": ["Sample_0"],
            "true_digit": [7],
            "umap1": [0.0],
            "umap2": [1.0],
            "best_tbs_cluster": [3],
        }
    ).to_csv(report.OUT_UMAP_COORDS, index=False)
    page_path = output_dir / "00_mnist.html"
    page_path.write_text("<html>page</html>", encoding="utf-8")

    report._write_manifest(
        [("MNIST page", "00_mnist.html")],
        generated_at="2026-06-24T20:38:05+02:00",
        best_key="ward_e0.001_s0.05",
    )

    manifest = json.loads(report.OUT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["generated_at"] == "2026-06-24T20:38:05+02:00"
    assert manifest["source_script"] == "scripts/plot_mnist_tbs_analysis_plotly.py"
    assert len(manifest["artifacts"]) == 4
    summary = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"].endswith("mnist_tbs_analysis_summary.csv")
    )
    assert summary["rows"] == 1
    assert summary["columns"] == 2
    assert summary["generated_at_values"] == ["2026-06-24T20:38:05+02:00"]


def test_stale_2d_umap_cache_is_rebuilt(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    cache_path = tmp_path / "mnist_pca50_umap_coordinates.csv"
    pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1"],
            "true_digit": [9, 9],
            "umap1": [10.0, 11.0],
            "umap2": [12.0, 13.0],
            "best_tbs_cluster": [99, 99],
        }
    ).to_csv(cache_path, index=False)

    class DummyUMAP:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, feature_matrix):
            return np.column_stack(
                [
                    np.arange(feature_matrix.shape[0], dtype=float),
                    np.arange(feature_matrix.shape[0], dtype=float) + 10.0,
                ]
            )

    monkeypatch.setattr(report, "OUT_UMAP_COORDS", cache_path)
    monkeypatch.setattr(
        report,
        "load_mnist_subset",
        lambda **_kwargs: (np.ones((2, 3)), np.array([3, 4])),
    )
    monkeypatch.setattr(report.umap, "UMAP", DummyUMAP)
    assignments = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1"],
            "true_digit": [3, 4],
            "best": [7, 8],
        }
    )

    frame = report._load_or_create_umap(assignments, "best")

    assert frame["true_digit"].tolist() == [3, 4]
    assert frame["best_tbs_cluster"].tolist() == [7, 8]
    assert frame["umap1"].tolist() == [0.0, 1.0]


def test_docstyle_3d_plot_has_centered_digit_hover_targets(monkeypatch, tmp_path):
    report = _load_mnist_report_module()
    output_path = tmp_path / "docstyle_3d.html"

    monkeypatch.setattr(report, "OUT_UMAP3D_DOCSTYLE_HTML", output_path)
    monkeypatch.setattr(
        report,
        "load_mnist_subset",
        lambda **_kwargs: (np.ones((4, 3)), np.array([4, 1, 9, 0])),
    )
    monkeypatch.setattr(report.umap, "UMAP", _DummyUMAP)
    assignments = pd.DataFrame(
        {
            "sample": ["Sample_0", "Sample_1", "Sample_2", "Sample_3"],
            "best": [2, 2, 7, 8],
        }
    )

    report._write_umap3d_docstyle(assignments, "best")

    html = output_path.read_text(encoding="utf-8")
    assert "Digit number: 4" in html
    assert "Sample: Sample_0" in html
    assert "Best TBS cluster: 2" in html
    assert "%{hovertext}" in html
    assert '"mode":"markers+text"' in html
    assert '"size":7' in html
    assert '"name":"hover target"' in html
    assert '"size":24' in html
    assert '"opacity":0.01' in html
    assert '"textposition":"middle center"' in html
