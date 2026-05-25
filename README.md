# KL-TE Clustering Toolkit

This repository develops the KL-TE hierarchy decomposition method and its
validation software. The active method builds a hierarchy, represents node
distributions in an explicit feature space, tests child-parent and sibling
contrasts with projected-Wald statistics, and uses strict calibration contracts
to decide where the tree should stop splitting.

## Overview

- Analyse binary, categorical, and explicitly supported continuous benchmark
  inputs through a typed `FeatureSpace`.
- Build a `PosetTree` hierarchy and annotate it with child-parent edge tests and
  sibling split tests.
- Use projected-Wald geometry, spectral projection dimensions, empirical-null
  inflation, and multiple-testing control to select the final cluster cut.
- Record unsupported statistical contexts as explicit skipped statuses instead
  of silently falling back to neutral corrections.
- Produce benchmark tables, diagnostic traces, manuscript figures, and wiki
  records that connect results back to source files.

The package code lives under `kl_clustering_analysis/`. Benchmarks,
diagnostics, manuscript material, and wiki synthesis are separate repository
surfaces with their own contracts.

## Key Concepts

- **Hierarchical tree** – the analysis revolves around a `PosetTree`, a directed structure that records parent/child
  relationships alongside per-node distributions.
- **Feature space** – observed columns are interpreted through one explicit
  feature-space object. Binary/categorical blocks define their covariance and
  contrast coordinates; unsupported continuous contracts fail clearly until the
  required covariance math exists.
- **Projected-Wald statistic** – edge and sibling tests compare distributions in
  a projection basis with an explicit covariance model.
- **Empirical-null inflation** – sibling split p-values are corrected only when
  the calibration support is valid. Invalid support produces an unsupported
  status, not a neutral fallback.
- **Top-down decomposition** – cluster boundaries appear at the first node where
  the traversal contract says the split is not supported.

### Implementation Map

- Core tree structure, node distributions, and I/O: `kl_clustering_analysis/tree/`.
- Decomposition entrypoint and traversal: `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py`.
- Gate orchestration and split/merge evaluation: `kl_clustering_analysis/hierarchy_analysis/decomposition/gates/`.
- Statistical tests, projection helpers, and FDR correction: `kl_clustering_analysis/hierarchy_analysis/statistics/`.
- Benchmark harness and report generation: `benchmarks/`.
- User-facing analysis commands: `scripts/analysis/`.
- Scientific manuscript workspace: `manuscript/`.
- Durable project memory and open questions: `wiki/`.

### Repository Path Policy

- `kl_clustering_analysis/`: importable package code only.
- `benchmarks/`: benchmark runners, reusable benchmark infrastructure, and
  benchmark diagnostics. Generated benchmark outputs belong under
  `benchmarks/results/`.
- `scripts/analysis/`: user-facing analysis commands for tracked feature
  matrices and real-data workflows.
- `scripts/wiki/`: wiki maintenance tools.
- `data/feature_matrices/`: canonical tracked feature matrices.
- `data/reference/`: external reference tables used for interpretation.
- `reports/`: policy marker for deliberately retained evidence. Routine logs,
  profiling outputs, scan reports, and notebook image exports should stay local
  unless they are intentionally cited.
- `notebooks/`: notebooks only; reusable Python code belongs in package,
  benchmark, or script directories.
- `raw/`: captured primary material before synthesis into the wiki.
- `local_data/`: ignored local-only data.

### Pipeline Workflow

Starting from a data matrix and an explicit feature-space contract, the pipeline
proceeds through four checkpoints:

1. **Pairwise linkage** – compute or receive the distance representation used to
   build the hierarchy.

2. **Node distributions** – aggregate descendant leaves into node-level
   distribution parameters in the declared feature space.

3. **Child-parent tests** – evaluate whether each child differs from its parent
   in the projected-Wald geometry, then apply tree-aware multiplicity control.

4. **Sibling split tests** – evaluate whether sibling subtrees should remain
   separated after projected-Wald testing, empirical-null inflation, and sibling
   FDR control.

## Statistical Gates and Independence Checks

The `TreeDecomposition` treats every internal node as a checkpoint—called a gate—that decides whether the tree is
allowed to split at that spot. Each gate represents a statistical question about the parent/child relationship; if the
answer is “yes,” the walk continues into the children, and if the answer is “no,” the branch stays merged and forms a
cluster boundary.

- **Gate 1 – child-parent edge check**: both children must have supported and
  significant child-parent evidence.
- **Gate 2 – sibling split check**: the parent must have a supported and
  significant sibling split after empirical-null inflation and sibling FDR.
- **Optional parent gate**: Setting `parent_gate="strict"` adds one more requirement—only parents already marked
  significant can split. Leave it `off` to ignore this extra guard.

If any gate fails, the algorithm labels the parent node as the cluster boundary and stops there. When every active gate
passes, it continues the walk into each child so the process can repeat deeper in the tree.

The walk follows a depth-first rule:

1. If $u$ is a leaf, record its cluster label and return.
2. If both child-parent gates and the sibling split gate pass, recurse on the children.
3. Otherwise, stop at $u$ and assign all leaves beneath $u$ to the same cluster.

This recursion ensures that every branch of the tree either terminates at the
earliest unsupported split or keeps splitting while all active gates support the
children.

### Worked Example

See the worked examples in the manuscript method sections for step-by-step
edge-test and sibling-test calculations.

### Documentation

The repository keeps the durable entrypoint docs in a small set of files:

- `README.md` for installation and the end-to-end workflow.
- `docs/onboarding.md` for the first 30 minutes and contributor route through
  the repository.
- `manuscript/sections/method/edge_test.tex` and `manuscript/sections/method/sibling_test.tex` for numeric walk-throughs.
- `tests/README.md` and `benchmarks/README.md` for the validation and benchmark harnesses.
- Package READMEs under `kl_clustering_analysis/` for module-level maps.
- `manuscript/README.md` for the paper workspace and build tooling.

## Highlights

- Build a hierarchy using SciPy linkage and NetworkX-backed `PosetTree`.
- Annotate child-parent and sibling evidence with projected-Wald tests.
- Keep unsupported calibration contexts explicit in benchmark outputs.
- Decompose the resulting tree into cluster assignments you can validate against ground truth.

## Getting Started

### Prerequisites

- Python `>=3.11`
- A virtual environment tool such as `uv` or `venv`

New contributors should read `docs/onboarding.md` after this README. It gives a
short route through the package, benchmarks, wiki, manuscript, and tests.

### Install Dependencies

Using `uv` (recommended):

```bash
uv venv --python 3.11 .venv
uv sync --extra dev --extra benchmark --extra viz --locked
```

Using `pip` inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Quick Start Pipeline

`quick_start.py` wires together the full analysis pipeline on a synthetic dataset to illustrate each stage.

```bash
python quick_start.py
```

What the script does:

1. **Generate data** – creates a binary feature matrix by thresholding Gaussian blobs so you can reproduce demo data
   with known clusters.
2. **Build the hierarchy** – computes pairwise Hamming distances, runs SciPy `linkage`, and wraps the result in a
   `PosetTree` so each node keeps track of its distribution, significance markers, and children.
3. **Annotate node distributions** – populates node-level distribution summaries
   used by the statistical gates.
4. **Decompose clusters** – runs the child-parent and sibling gate pipeline to
   turn supported split decisions into cluster assignments and prints a
   concise report.
5. **Validate results** – compares discovered clusters with the synthetic ground truth using Adjusted Rand Index (ARI)
   so you know how well the decomposition performed.

The script prints console output summarizing each step, reports the discovered clusters, and ends with the ARI score
(`1.0` denotes a perfect match; `0.0` indicates random assignment). The demo does not create files, so reruns can be
performed without cleanup.

## Working With Your Own Data

- Replace the synthetic data block in `quick_start.py` with your dataframe and
  an explicit feature-space contract when the data are not simple binary
  indicators.
- Keep sample names as the index so the reporting remains readable.
- Preserve the overall pipeline order so the statistical annotations stay in sync with the calculated metrics.

## Validation & Testing

- Run the automated tests with `pytest`.
- Use `tests/README.md` for the current suite layout and staged execution order.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## Benchmark Methods (Optional)

The benchmarking suite can run additional clustering baselines side-by-side with the KL pipeline:

- Graph community detection: Leiden, Louvain
- Density-based clustering: DBSCAN, OPTICS, HDBSCAN (optional)

Optional dependencies (skipped automatically if missing):

```bash
pip install leidenalg igraph python-louvain hdbscan
```

## License

MIT
