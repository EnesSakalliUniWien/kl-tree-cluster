from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from benchmarks.shared.plots.cover_page import write_case_manifest_pages_to_pdf
from benchmarks.shared.util.pdf.figure_split import (
    split_collected_figs_to_pdfs,
)
from benchmarks.shared.util.pdf.merge import merge_existing_pdfs
from matplotlib import pyplot as plt
from PIL import Image
from pypdf import PdfReader


def _write_dummy_png(path: Path, color=(255, 0, 0)):
    img = Image.new("RGB", (100, 80), color=color)
    img.save(path)


def test_split_collected_figs_to_pdfs(tmp_path: Path):
    figs = []
    # Tree fig
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_title("Hierarchical Tree with KL Divergence Clusters\nTest Case 1")
    figs.append(f2)

    # UMAP fig (uses suptitle)
    f3 = plt.figure()
    f3.suptitle("Test Case 1: expected 3 clusters\nSome meta text")
    figs.append(f3)

    # Manifold fig (UMAP vs Isomap) should be grouped with UMAP plots
    f4 = plt.figure()
    f4.suptitle("Manifold diagnostics (UMAP vs Isomap) - Test Case 1")
    figs.append(f4)

    results = split_collected_figs_to_pdfs(figs, output_dir=tmp_path, verbose=True)
    # Ensure PDFs for categories exist
    assert "tree" in results and results["tree"].exists()
    assert "umap" in results and results["umap"].exists()

    # Files are non-empty
    assert results["tree"].stat().st_size > 0
    assert results["umap"].stat().st_size > 0

    # The manifold figure should have been classified into the UMAP PDF
    # (we don't expose counts from the helper, but the presence of the 'umap'
    # PDF guarantees classification occurred for at least one UMAP/manifold fig)


def test_merge_existing_pdfs_uses_declared_python_dependency(tmp_path: Path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    output = tmp_path / "merged.pdf"

    for path, title in ((first, "First"), (second, "Second")):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        fig.savefig(path)
        plt.close(fig)

    assert merge_existing_pdfs([first, second], output, verbose=False)
    assert output.exists()
    assert len(PdfReader(str(output)).pages) == 2


def test_case_manifest_pages_record_generation_contract(tmp_path: Path):
    output = tmp_path / "manifest.pdf"
    cases = [
        {
            "name": "gaussian_case",
            "category": "improved_gaussian",
            "generator": "blobs",
            "n_samples": 12,
            "n_features": 8,
            "n_clusters": 3,
            "cluster_std": 0.5,
        },
        {
            "name": "binary_case",
            "category": "improved_binary_low_noise",
            "generator": "binary",
            "n_samples": 20,
            "n_features": 10,
            "noise_features": 2,
            "n_clusters": 4,
            "entropy_param": 0.1,
        },
    ]

    write_case_manifest_pages_to_pdf(cases, output, rows_per_page=20)

    assert output.exists()
    assert len(PdfReader(str(output)).pages) == 1
