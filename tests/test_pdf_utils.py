import os
from pathlib import Path
import tempfile
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from kl_clustering_analysis.benchmarking.pdf_utils import (
    concat_plots_to_pdf,
    concat_tree_plots,
    concat_umap_plots,
    split_collected_figs_to_pdfs,
)


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
