---
title: scRNA Space Decomposition Rerun 2026-06-27
type: source
status: reviewed
updated: 2026-06-27
sources:
  - applications/scrna/analyze_space_decomposition.py
  - applications/scrna/pancreas_benchmark.py
  - applications/scrna/goncalves_benchmark.py
  - pyproject.toml
  - uv.lock
  - raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/adult_pancreas_benchmark/method_metrics.csv
  - raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/goncalves_fetal_pancreas_benchmark/method_metrics.csv
  - raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/manifest.json
  - raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/scrna_space_decomposition_summary.csv
tags:
  - scrna
  - pancreas
  - clustering
  - space-decomposition
---

# scRNA Space Decomposition Rerun 2026-06-27

## Summary

The adult pancreas and Goncalves fetal pancreas scRNA benchmarks were rerun
from their benchmark entry points, then analyzed with
`applications/scrna/analyze_space_decomposition.py`. The diagnostic standardizes the
saved benchmark PCA coordinates, decomposes the centered cell-by-PC matrix by
SVD, treats axis 1 as the common/invariant axis, and treats axes 2 through 6 as
orthogonal/equivariant axes. This is a descriptive PCA-space diagnostic, not a
formal Cartan decomposition claim.

## Key Points

- The scRNA optional dependency extra now installs `scanpy` and `anndata`;
  `uv sync --all-extras` completed before the reruns.
- The adult pancreas benchmark rerun used `max_cells=2500`, `n_pcs=30`, and
  seed `0`, writing to
  `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/adult_pancreas_benchmark/`.
- The Goncalves fetal pancreas rerun used the local UCSC inputs with
  `--skip-download`, `max_cells=2500`, `n_pcs=30`, `n_hvgs=2000`, and seed `0`,
  writing to
  `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/goncalves_fetal_pancreas_benchmark/`.
- Adult pancreas celltype structure is stronger in the equivariant radius than
  in the invariant axis: celltype between-fraction is `0.484671` for
  equivariant radius versus `0.277534` for the invariant axis.
- Goncalves fetal pancreas shows weak celltype separation in this diagnostic:
  celltype between-fraction is `0.016764` for equivariant radius and
  `0.090909` for the invariant axis.
- Adult adaptive-diffusion TBS rows explain the most cluster variance in the
  equivariant radius among the method partitions: topology and NNLS branch-time
  each have `48` clusters, ARI `0.388616`, V-measure `0.648785`, and
  cluster-between equivariant-radius fraction `0.605833`.
- Adult adaptive-diffusion raw-linkage branch-time has the strongest V-measure
  among the TBS rows in this rerun: `40` clusters, ARI `0.415248`, V-measure
  `0.665991`, and cluster-between equivariant-radius fraction `0.601216`.
- Goncalves standardized-PCA topology TBS has the largest method-level
  equivariant-radius separation (`0.186878`) but remains fragmented at `36`
  clusters with ARI `0.248570`; the adaptive-diffusion TBS rows return `24`
  clusters with ARI `0.205837` and V-measure `0.409991`.
- Goncalves standardized-PCA NNLS and raw-linkage branch-time rows collapse to
  one cluster in this rerun, matching the earlier branch-length-effect audit.

## Evidence

- `applications/scrna/analyze_space_decomposition.py` defines the
  `scrna_space_decomposition/v1` output contract, writes per-cell geometry,
  celltype summaries, method-cluster summaries, method summaries, axis
  loadings, plots, and a manifest.
- `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/manifest.json`
  records `equivariant_dim=5`, the two input benchmark directories, generated
  CSV/PNG paths, and `formal_cartan_claim=false`.
- `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/scrna_space_decomposition_summary.csv`
  records the adult and Goncalves invariant/equivariant energy and
  between-celltype fractions.
- `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/adult_pancreas/method_space_summary.csv`
  records the adult method-level invariant-axis and equivariant-radius
  between-cluster fractions with method labels joined by canonical assignment
  keys.
- `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/goncalves_fetal_pancreas/method_space_summary.csv`
  records the corresponding Goncalves method-level summaries.
- `uv run --all-extras ruff check applications/scrna/analyze_space_decomposition.py
  applications/scrna/pancreas_benchmark.py
  applications/scrna/goncalves_benchmark.py` passed.
- `uv run --all-extras python -m py_compile
  applications/scrna/analyze_space_decomposition.py
  applications/scrna/pancreas_benchmark.py
  applications/scrna/goncalves_benchmark.py` passed.

## Links

- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]
- [[scrna-branch-length-effect-audit-20260624]]
