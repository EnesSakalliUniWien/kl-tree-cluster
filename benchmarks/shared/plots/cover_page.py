"""Generate benchmark cover / experiment-setup pages for PDF reports.

Each page is a matplotlib figure rendered as formatted text.  The public API
returns both an overall cover page and per-category section pages so the
caller can interleave them into the merged PDF at the right positions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

import matplotlib.pyplot as plt

from benchmarks.shared.util.pdf.layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure

# ---------------------------------------------------------------------------
# Category key → parent group mapping
# ---------------------------------------------------------------------------
# Each subcategory key from ALL_CASE_CATEGORIES maps to a top-level group.
# When a new subcategory is added, register it here.

_CATEGORY_GROUP: dict[str, str] = {}


def _register(group: str, *subcategories: str) -> None:
    for sub in subcategories:
        _CATEGORY_GROUP[sub] = group


_register(
    "gaussian",
    "gaussian_clear",
    "gaussian_mixed",
    "gaussian_extreme_noise",
    "improved_gaussian",
    "gaussian_null",
)
_register(
    "binary",
    "binary_balanced_low_noise",
    "binary_sparse_features",
    "improved_binary_perfect",
    "improved_binary_low_noise",
    "improved_binary_moderate",
    "improved_binary_hard",
    "improved_binary_unbalanced",
    "improved_binary_edge_cases",
    "binary_null",
    "binary_multiscale",
    "binary_noise_features",
    # Legacy names kept for backward compat
    "binary_perfect_separation",
    "binary_unbalanced",
    "binary_large_matrices",
    "binary_high_noise",
)
_register("sbm", "sbm", "sbm_graphs")
_register(
    "categorical",
    "cat_clear",
    "cat_moderate",
    "cat_high_cardinality",
    "cat_unbalanced",
    "cat_overlapping",
    "cat_high_dimensional",
    "categorical_clear",
    "categorical_moderate",
    "categorical_high_cardinality",
    "categorical_unbalanced",
    "categorical_overlapping",
    "categorical_high_dimensional",
)
_register(
    "phylogenetic",
    "phylo_clear",
    "phylo_moderate",
    "phylo_divergent",
    "phylo_high_mutation",
    "phylo_many_taxa",
    "phylogenetic_dna",
    "phylogenetic_protein",
    "phylogenetic_divergent",
    "phylogenetic_conserved",
    "phylogenetic_large",
)
_register(
    "overlapping",
    "overlapping_binary_heavy",
    "overlapping_binary_moderate",
    "overlapping_binary_partial",
    "overlapping_binary_highd",
    "overlapping_binary_unbalanced",
    "overlapping_gaussian",
)
_register("real_data", "real_data")


def category_group(subcategory: str) -> str:
    """Return the parent group name for a subcategory key.

    Falls back to the subcategory itself if not registered.
    """
    return _CATEGORY_GROUP.get(subcategory, subcategory)


# Text content for each section
# ---------------------------------------------------------------------------

_OVERVIEW_TEXT = dedent(
    """\
    KL-Divergence Hierarchical Clustering -- Benchmark Report

    This report evaluates the KL-divergence hierarchical clustering algorithm
    across {n_cases} synthetic and real-world test cases drawn from 7 data
    categories.  Each case is generated with known ground-truth cluster labels
    so that recovery accuracy can be measured objectively.

    Pipeline:
      binary matrix -> Hamming distance -> average linkage -> PosetTree
      -> populate_node_divergences() -> three statistical gates
      -> decompose() -> cluster assignments

    Statistical gates applied at each internal node:
      Gate 1 -- Binary structure:
        The parent node must have exactly 2 children.
      Gate 2 -- Child-parent divergence:
        Projected Wald chi-squared test on KL(child || parent).
      Gate 3 -- Sibling divergence:
        Calibrated Wald test (cousin-weighted) on KL(left || right)
        with Benjamini-Hochberg false discovery rate correction.

    Default configuration:
      Distance metric:              Hamming
      Linkage method:               average
      Significance level (alpha):   0.05  (edge and sibling)
      Sibling test method:          cousin_weighted_wald
      Post-hoc merge:               enabled
      Felsenstein branch-length adjustment: enabled
"""
)

_GAUSSIAN_TEXT = dedent(
    """\
    Gaussian Cases  (12 cases)

    Data generation:
      sklearn.make_blobs with configurable cluster standard deviation, then
      median-binarized per feature: (X > median(X, axis=0)).astype(int).

    Parameter ranges:
      number of samples       30 to 300
      number of features      20 to 20 000
      number of clusters      1 to 30
      cluster standard dev    0.5 to 7.5

    Subcategories:

      1. improved_gaussian (7 cases)
         Progressive difficulty from well-separated (standard deviation 0.5)
         through moderate overlap (1.2 to 1.5) to challenging noise (2.0
         to 2.5).  Targets statistical power edge cases: small sample
         sizes, moderate cluster counts.

      2. gaussian_extreme_noise (3 cases)
         High noise (standard deviation 2.0 to 7.5) and/or very high
         dimensionality (up to 20 000 features, up to 30 clusters).
         Probes failure modes where signal is buried in noise.

      3. gaussian_null (2 cases)
         Single-cluster data (K=1) with no structure.  Tests the
         algorithm's ability to correctly return K=1.

    Design notes:
      Median binarization forces theta close to 0.5, maximizing Wald
      variance and reducing test power.  Signal survives only when
      per-cluster theta deviates sufficiently from the global median.
"""
)

_BINARY_TEXT = dedent(
    """\
    Binary Cases  (28 cases)

    Data generation:
      generate_random_feature_matrix produces a {0,1} matrix directly.
      All cases use sparse (block-diagonal) templates where each cluster
      owns a distinct subset of features (feature_sparsity = 0.05).
      The entropy parameter controls bit-flip noise (0 = perfect, 0.5 = random).

    Parameter ranges:
      number of rows        40 to 300
      number of columns     50 to 2 000
      number of clusters    1 to 15
      entropy               0.00 to 0.25

    Subcategories:
      balanced low noise (2)       sparse features (2)
      perfect separation (3)       low noise (4)
      moderate noise (3)           hard (2)
      unbalanced clusters (2)      edge cases (3)
      null (2  K=1)                multi-scale (2)
      noise features (3)

    Noise-feature cases append 200 to 500 uninformative Bernoulli(0.5)
    columns to test robustness to irrelevant features.
"""
)

_SBM_TEXT = dedent(
    """\
    Stochastic Block Model Cases  (3 cases)

    Data generation:
      generate_sbm creates a random graph with planted community structure.
      The adjacency matrix is converted to a modularity-based distance:
      distance = 1 - B_norm,  where B = A - dd^T / 2m.

    Parameter ranges:
      block sizes                [30, 30] to [50, 40, 30]
      within-block probability   0.05 to 0.12
      between-block probability  0.005 to 0.04
      number of clusters         2 to 3

    The pre-computed modularity distance is passed directly to the
    KL runner.  There is no fallback to pdist() on raw adjacency data.
"""
)

_CATEGORICAL_TEXT = dedent(
    """\
    Categorical Cases  (11 cases)

    Data generation:
      generate_categorical_feature_matrix produces an (n x p) matrix of
      integer category indices (0 to C-1).  Before clustering, each column
      is one-hot encoded into C binary indicators, expanding the feature
      space to n x (p x C).

    Parameter ranges:
      number of rows           100 to 300
      number of columns        40 to 1 000
      number of clusters       4 to 6
      number of categories     3 to 20
      entropy                  0.05 to 0.35

    Subcategories:
      clear (3)                moderate (2)
      high cardinality (2)     unbalanced (1)
      overlapping (1)          high dimensional (2)
"""
)

_PHYLOGENETIC_TEXT = dedent(
    """\
    Phylogenetic Cases  (13 cases)

    Data generation:
      generate_phylogenetic_data simulates trait evolution along a random
      phylogenetic tree using a Jukes-Cantor-like substitution model.
      Each taxon acts as a cluster; samples_per_taxon sequences are drawn
      from the evolved distribution at each leaf.  The resulting category
      matrix is one-hot encoded before the KL pipeline.

    Parameter ranges:
      number of taxa (= clusters)   4 to 64
      number of features            50 to 1 000
      number of categories          4 to 20
      samples per taxon             8 to 30
      mutation rate                 0.05 to 0.80

    Known weakness:  severe over-splitting due to near-zero branch lengths
    in the constructed clustering tree.
"""
)

_OVERLAPPING_TEXT = dedent(
    """\
    Overlapping Cases  (29 cases)

    Data generation:
      Binary subcategories use generate_random_feature_matrix with high
      entropy parameter (0.22 to 0.48) so cluster feature profiles
      significantly overlap.  Gaussian subcategories use make_blobs with
      cluster standard deviation 3.0 to 6.0.

    Parameter ranges:
      number of samples     300 to 1 000
      number of features    30 to 5 000
      number of clusters    3 to 10

    Subcategories:
      heavy overlap (5)            moderate overlap (4)
      partial overlap (4)          high dimensional (4)
      unbalanced (4)               Gaussian overlap (8)

    Purpose:  tests the algorithm's ability to correctly merge overlapping
    groups rather than over-split.
"""
)

_REALDATA_TEXT = dedent(
    """\
    Real Data  (1 case)

    Source:  feature_matrix.tsv -- a binary Gene Ontology term annotation
    matrix from the repository root.

    No ground-truth cluster labels are available.  This case is included
    for qualitative evaluation only (tree structure, cluster sizes).
"""
)

_EVALUATION_TEXT = dedent(
    """\
    Evaluation Metrics

    All cases with ground-truth labels are evaluated with:

      Adjusted Rand Index            [-1, 1]   1 = perfect, 0 = random
      Normalized Mutual Information  [ 0, 1]   1 = perfect label correspondence
      Purity                         [ 0, 1]   fraction in dominant true class
      Exact K                        count     found K equals true K

    Primary method:
      KL-divergence hierarchical clustering with Hamming distance and
      average linkage.

    Competing methods (when included):
      Community detection (Leiden, Louvain)
      K-Means, Spectral -- oracle baselines given true number of clusters
      Density-based (DBSCAN, OPTICS, HDBSCAN)
"""
)


# ---------------------------------------------------------------------------
# Section text mapped by group name
# ---------------------------------------------------------------------------

_SECTION_TEXTS: dict[str, str] = {
    "gaussian": _GAUSSIAN_TEXT,
    "binary": _BINARY_TEXT,
    "sbm": _SBM_TEXT,
    "categorical": _CATEGORICAL_TEXT,
    "phylogenetic": _PHYLOGENETIC_TEXT,
    "overlapping": _OVERLAPPING_TEXT,
    "real_data": _REALDATA_TEXT,
}

# Canonical group order (controls insertion order in merged PDF)
GROUP_ORDER: list[str] = [
    "gaussian",
    "binary",
    "sbm",
    "categorical",
    "phylogenetic",
    "overlapping",
    "real_data",
]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _text_page(
    text: str,
    *,
    fontsize: float = 10.0,
    title_fontsize: float = 14.0,
) -> plt.Figure:
    """Render *text* as a single PDF page figure.

    The first line is treated as the title (rendered slightly larger and bold).
    """
    lines = text.strip().split("\n")
    title = lines[0] if lines else ""
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    fig = plt.figure(figsize=PDF_PAGE_SIZE_INCHES)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Title
    ax.text(
        0.0,
        1.0,
        title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Body (monospace for alignment)
    ax.text(
        0.0,
        0.93,
        body,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        fontfamily="monospace",
        linespacing=1.35,
    )

    prepare_pdf_figure(fig)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_overview_page(
    n_cases: int = 97,
    timestamp: str | None = None,
) -> plt.Figure:
    """Return a single overview/title page figure."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    overview = _OVERVIEW_TEXT.format(n_cases=n_cases) + f"\n    Report generated: {timestamp}\n"
    return _text_page(overview, fontsize=10.5, title_fontsize=16.0)


def generate_section_page(group: str) -> plt.Figure | None:
    """Return a section description page for *group*, or None if unknown."""
    text = _SECTION_TEXTS.get(group)
    if text is None:
        return None
    # Evaluation metrics appended to the last section page (real_data)
    if group == "real_data":
        text = text + "\n" + _EVALUATION_TEXT
    return _text_page(text, fontsize=9.5)


def generate_cover_pages(
    n_cases: int = 97,
    timestamp: str | None = None,
) -> list[plt.Figure]:
    """Return ALL cover/section pages as a flat list (legacy API).

    Order: overview, then one page per group in GROUP_ORDER, with the
    evaluation metrics appended to the last page.
    """
    figs: list[plt.Figure] = [generate_overview_page(n_cases, timestamp)]
    for group in GROUP_ORDER:
        fig = generate_section_page(group)
        if fig is not None:
            figs.append(fig)
    return figs


def write_cover_pages_to_pdf(
    pdf_path: str | None = None,
    *,
    n_cases: int = 97,
    timestamp: str | None = None,
) -> str | None:
    """Write the cover pages to a standalone PDF file.

    Returns the path written, or None if *pdf_path* was None.
    """
    if pdf_path is None:
        return None

    from matplotlib.backends.backend_pdf import PdfPages

    pages = generate_cover_pages(n_cases=n_cases, timestamp=timestamp)
    with PdfPages(pdf_path) as pp:
        for fig in pages:
            pp.savefig(fig)
            plt.close(fig)
    return pdf_path


def write_section_page_to_pdf(group: str, pdf_path: str) -> str | None:
    """Write a single section page to a standalone PDF.

    Returns the path written, or None if group is unknown.
    """
    fig = generate_section_page(group)
    if fig is None:
        return None

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pp:
        pp.savefig(fig)
    plt.close(fig)
    return pdf_path


__all__ = [
    "category_group",
    "generate_cover_pages",
    "generate_overview_page",
    "generate_section_page",
    "write_cover_pages_to_pdf",
    "write_section_page_to_pdf",
    "GROUP_ORDER",
]
