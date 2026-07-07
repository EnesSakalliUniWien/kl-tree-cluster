# Tree-Break Selection Clustering Toolkit

This repository develops the Tree-Break Selection hierarchy decomposition method and its
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

The package code lives under `tree_break_selection/`. Benchmarks,
diagnostics, manuscript material, and wiki synthesis are separate repository
surfaces with their own contracts.

## Key Concepts

- **Hierarchical tree** – the analysis revolves around a `PosetTree`, a directed structure that records parent/child
  relationships alongside per-node distributions.
- **Feature space** – observed columns are interpreted through one explicit
  feature-space object. Bernoulli coordinates, categorical drop-last simplex
  blocks, and continuous empirical-Gaussian blocks define their covariance and
  contrast coordinates before any projected-Wald test is evaluated.
- **Projected-Wald statistic** – edge and sibling tests compare distributions in
  a projection basis with an explicit covariance model.
- **Empirical-null inflation** – sibling split p-values are corrected only when
  the calibration support is valid. Invalid support produces an unsupported
  status, not a neutral fallback.
- **Top-down decomposition** – cluster boundaries appear at the first node where
  the traversal contract says the split is not supported.

### Implementation Map

- Core tree structure, node distributions, and I/O: `tree_break_selection/tree/`.
- Decomposition entrypoint and traversal: `tree_break_selection/hierarchy_analysis/tree_decomposition.py`.
- Gate orchestration and split/merge evaluation: `tree_break_selection/hierarchy_analysis/decomposition/gates/`.
- Statistical tests, projection helpers, and FDR correction: `tree_break_selection/hierarchy_analysis/statistics/`.
- Benchmark harness and report generation: `benchmarks/`.
- User-facing analysis commands: `scripts/analysis/`.
- Scientific manuscript workspace: `manuscript/`.
- Durable project memory and open questions: `wiki/`.

### Repository Path Policy

- `tree_break_selection/`: importable package code only.
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
   distribution parameters in the declared feature space. Internal node
   distributions are empirical subtree barycenters: leaf-count-weighted means
   of their child distributions.

3. **Child-parent tests** – evaluate whether each child differs from its parent
   in the projected-Wald geometry, then apply tree-aware multiplicity control.

4. **Sibling split tests** – evaluate whether sibling subtrees should remain
   separated after projected-Wald testing, empirical-null inflation, and sibling
   FDR control.

## Statistical Gates and Traversal

The `TreeDecomposition` treats every internal node as a split checkpoint. A
node can split only when one structural prerequisite and both statistical gates
are satisfied.

- **Binary structure prerequisite**: the node must have exactly two children.
- **Edge divergence gate**: at least one child-parent edge must have supported
  and significant child-parent divergence evidence.
- **Sibling divergence gate**: the sibling pair must have a supported and
  significant sibling split after empirical-null inflation and sibling FDR.

If the split is not supported, the algorithm labels the parent node as a
cluster boundary and stops there. When the split is supported, it continues the
walk into each child so the process can repeat deeper in the tree. In
pass-through mode, a closed sibling-divergence gate may still allow traversal to
descendants when a deeper split is already supported; pass-through is a
traversal policy, not a third statistical test.

The walk follows a depth-first rule:

1. If $u$ is a leaf, record its cluster label and return.
2. If the binary prerequisite, edge-divergence gate, and sibling-divergence gate pass, recurse on the children.
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
- Package READMEs under `tree_break_selection/` for module-level maps.
- `manuscript/README.md` for the paper workspace and build tooling.

## Highlights

- Build a hierarchy using SciPy linkage and NetworkX-backed `PosetTree`.
- Annotate child-parent and sibling evidence with projected-Wald tests.
- Keep unsupported calibration contexts explicit in benchmark outputs.
- Decompose the resulting tree into cluster assignments you can validate against ground truth.

## Getting Started

### Prerequisites

- Python `>=3.11`
- `uv`

New contributors should read `docs/onboarding.md` after this README. It gives a
short route through the package, benchmarks, wiki, manuscript, and tests.

### Install Dependencies

Use the locked `uv` environment. This is the lean install path for ordinary
development, benchmarks, visualization, and manuscript-adjacent checks.

```bash
uv venv --python 3.11 .venv
uv sync --extra dev --extra benchmark --extra viz --locked
```

For the full repository test suite, include the scRNA and optional GPL extras.
Several full-suite tests import or exercise Scanpy/AnnData and graphtools
paths that are intentionally outside the lean development environment.

```bash
uv sync --extra all --extra experimental-gpl --locked
```

## Run the Quick Start Pipeline

`quick_start.py` wires together the full analysis pipeline on a synthetic dataset to illustrate each stage.

```bash
uv run python quick_start.py
```

What the script does:

1. **Generate data** – creates a binary feature matrix by thresholding Gaussian blobs so you can reproduce demo data
   with known clusters.
2. **Build the hierarchy** – computes pairwise Hamming distances, runs SciPy `linkage`, and wraps the result in a
   `PosetTree` so each node keeps track of its distribution, significance markers, and children.
3. **Annotate node distributions** – populates node-level distribution summaries
   used by the statistical gates.
4. **Decompose clusters** – runs the child-parent and sibling-divergence pipeline to
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

- After the full test environment sync above, run the automated tests with
  `uv run pytest`.
- In the lean development environment, run targeted tests for the code you
  changed. scRNA-only tests are skipped when their optional dependencies are
  absent, but full-suite parity expects the full environment.
- Use `tests/README.md` for the current suite layout and staged execution order.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## Benchmark Methods (Optional)

The benchmarking suite can run additional clustering baselines side-by-side with the TBS pipeline:

- Graph community detection: Leiden, Louvain
- Density-based clustering: DBSCAN, OPTICS, HDBSCAN (optional)

The optional benchmark dependencies are included by the canonical `uv sync`
command above through the `benchmark` extra.

## License

MIT
