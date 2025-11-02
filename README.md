# KL-Divergence Hierarchical Clustering Toolkit

Pipeline utilities for exploring hierarchical cluster structure with KL-divergence based scoring,
statistical significance testing, and tree decomposition helpers.

## Highlights
- Build a hierarchy from binary data using SciPy linkage and NetworkX-backed `PosetTree`.
- Quantify KL-divergence at every internal node to detect informative feature splits.
- Run multiple statistical tests to flag significant child-parent and sibling relationships.
- Decompose the resulting tree into cluster assignments you can validate against ground truth.

## Getting Started
### Prerequisites
- Python `>=3.11`
- A virtual environment tool such as `uv` or `venv`
- Optional: SageMath if you want the Sage-specific tooling (`uv sync --extra sage`)

### Install Dependencies
Using `uv` (recommended):
```bash
uv sync
```

Using `pip` inside a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Quick Start Pipeline
`quick_start.py` wires together the full analysis pipeline on a synthetic dataset so you can see each
stage in action.

```bash
python quick_start.py
```

What the script does:
1. **Generate data** – creates a binary feature matrix by thresholding Gaussian blobs so you can
   reproduceable demo data with known clusters.
2. **Build the hierarchy** – computes pairwise Hamming distances, runs SciPy `linkage`, and wraps the
   result in a `PosetTree`.
3. **Score nodes** – applies `calculate_hierarchy_kl_divergence` to quantify how informative each
   split is for the generated features.
4. **Annotate significance** – runs multiple hypothesis tests to identify statistically significant
   branches (`annotate_nodes_with_statistical_significance_tests`,
   `annotate_child_parent_divergence`, and `annotate_sibling_independence_cmi`).
5. **Decompose clusters** – uses `ClusterDecomposer` to turn significant nodes into cluster
   assignments and prints a concise report.
6. **Validate results** – compares discovered clusters with the synthetic ground truth using
   Adjusted Rand Index (ARI) so you know how well the decomposition performed.

You should see console output describing each step, a summary of discovered clusters, and the final
ARI score (`1.0` is a perfect match; `0.0` indicates random assignment). No files are written by this
demo; it is safe to rerun repeatedly.

## Working With Your Own Data
- Replace the synthetic data block in `quick_start.py` with your dataframe (binary feature matrix).
- Keep sample names as the index so the reporting remains readable.
- If your data is not binary, adapt the preprocessing section to binarize or adjust the distance
  metric in `pdist`.
- Preserve the overall pipeline order so the statistical annotations stay in sync with the
  calculated metrics.

## Project Layout
```
.
├── quick_start.py                 # End-to-end reference pipeline
├── tree/                          # PosetTree utilities and graph adapters
├── hierarchy_analysis/            # KL divergence, statistical tests, decomposition
├── simulation/                    # Alternative data generators and helpers
├── notebooks/                     # Exploratory notebooks (see clustering_pipe_line.ipynb)
├── tests/                         # Pytest suite (validation helpers, decomposition checks)
└── docs/                          # Additional conceptual documentation and figures
```

## Validation & Testing
- Run the automated tests with `pytest`.
- Inspect `tests/test_cluster_validation.py` for examples of how to assert cluster quality in
  custom scenarios.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## License
MIT
